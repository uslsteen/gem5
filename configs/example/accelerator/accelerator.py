#!/usr/bin/env python3
"""System topology:

    DRAM 64 GiB (4 x DDR4_2400_8x8, [0x80000000, 0x1080000000))
      |
    SystemXBar 64B)
      |-- boot/DRAM only (boot bridges, AccelDma.port, mem_ctrls)

    ctl_bus (NoncoherentXBar 128B — the control CPU's private plane)
      |-- Control CPU (RiscvTimingSimpleCPU, hart 0, icache+dcache)
      |-- ctl_mem     (SimpleMemory, the control hart's local memory)
      |-- bridge_dma  (1 ns; the 200-cycle config latency is charged
      |                 once per transfer inside AccelDma)
      |-- bridge_arr  (1 ns; the 100-cycle latency is charged once per
      |                 launch inside SystolicArray)
      |-- bridge_vec  (150 ns -> control->vector sync interface, charged
      |                 per access = per flag message)
      |-- ctl_boot_bridge (first DRAM page)
      |
    iobus (NoncoherentXBar 128B, the control/SPM plane)
      |-- AccelDma     PIO @ 0x10000000 (flags/src/dst/len)
      |-- SystolicArray PIO @ 0x10000040 (a/b/c/k_len)
      |-- SPM tiles (SimpleMemory): spm_qp/spm_kv/spm_so
      |-- AccelDma.sram_port + SystolicArray.local 

    sram_bus (NoncoherentXBar 128B — the vector CPU's private SRAM port, 1024 bit/cycle)
      |-- Vector CPU (RiscvO3CPU + RVV, hart 1)
      |-- bridge_vec
      |-- boot_bridge
      |-- sram_workspace (SimpleMemory @ 0x11000000..0x11010000)
      |-- sram_stack     (SimpleMemory @ 0x110F0000..0x11100000,
      |
      |-- spm_bridge -> iobus
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_CFG = Path(__file__).resolve().parent
sys.path.insert(0, str(_CFG))

import op_class_config as lp

import m5
from m5.objects import (
    AccelDma,
    AddrRange,
    Bridge,
    DDR4_2400_8x8,
    FUPool,
    NoncoherentXBar,
    RiscvBareMetal,
    RiscvO3CPU,
    RiscvTimingSimpleCPU,
    Root,
    SimpleMemory,
    SrcClockDomain,
    SystolicArray,
    System,
    SystemXBar,
    VoltageDomain,
)

log = logging.getLogger("accelerator")

# The parameters shared with the vector-CPU-only model
from accel_params import (CLK, DRAM_BASE, EXIT_CODE_MASK, ISSUE_WIDTH,
                          MEM_BANDWIDTH, MEM_LATENCY, VLEN)

DRAM_SIZE = "64GiB"
DRAM_CHANNELS = 4
CHANNEL_SIZE = "16GiB"

SRAM_BASE = 0x10000000
WORKSPACE_BASE = 0x11000000     # vector-CPU workspace (1 MiB, below DRAM)
WORKSPACE_SIZE = "1MiB"

LAT_DMA = 200     # control -> DMA configuration latency (cycles @ 1 GHz)
LAT_ARR = 100     # control -> systolic array latency
LAT_VEC = 300     # control -> vector CPU communication latency

# Fixed device map 
DMA_PIO = 0x10000000
MATMUL_PIO = 0x10000040
SPM_A = 0x10000080          # spm_qp: Q + P = 2304 B
SPM_B = 0x10000980          # spm_kv: double-buffered KV slots: 2 x 2048 B
SPM_C = 0x10001980          # spm_so: S_j + 8 O_j chunks = 4608 B

DMA_PIO_END = SRAM_BASE + 0x40
ARR_PIO_END = SRAM_BASE + 0x500
SPM_WINDOW_END = SRAM_BASE + 0x3000
CTL_REGION_BASE = WORKSPACE_BASE + 0x10000
CTL_REGION_END = WORKSPACE_BASE + 0xF0000
BOOT_PAGE_SIZE = 0x1000

# NOTE: # 64 GiB total DRAM
CHANNEL_BYTES = 0x1000000000
EXIT_CODE_MASK = 0xFFFFFFFF

assert SPM_A + 2304 == SPM_B and SPM_B + 4096 == SPM_C, \
    "SPM tiles must be contiguous"
assert SPM_C + 4608 == 0x10002B80, "SPM window must end at 0x10002b80"


class VectorCpuPool(FUPool):
    FUList = [unit() for unit in lp.FU_POOL]


def make_system(elf, issue_width=ISSUE_WIDTH, vlen=VLEN):
    system = System()
    system.mem_mode = "timing"

    system.mem_ranges = [
        AddrRange(DRAM_BASE, size=DRAM_SIZE),
        AddrRange(WORKSPACE_BASE, size=WORKSPACE_SIZE),
    ]

    system.mmap_using_noreserve = True
    system.clk_domain = SrcClockDomain(
        clock=CLK, voltage_domain=VoltageDomain()
    )
    system.workload = RiscvBareMetal()
    system.workload.bootloader = elf
    system.workload.bare_metal = True
    system.workload.auto_reset_vect = True

    # Memory system
    system.membus = SystemXBar(width=64)

    system.iobus = NoncoherentXBar(
        width=128, frontend_latency=1, forward_latency=0,
        response_latency=1)

    system.sram_bus = NoncoherentXBar(
        width=128, frontend_latency=1, forward_latency=0,
        response_latency=1)

    # The control CPU's private plane: local code/data memory and the
    # device bridges hang off this 1-stage bus
    system.ctl_bus = NoncoherentXBar(
        width=128, frontend_latency=1, forward_latency=0,
        response_latency=1)

    system.system_port = system.membus.cpu_side_ports

    mem_ctrls = []
    for i in range(DRAM_CHANNELS):
        intf = DDR4_2400_8x8()
        intf.range = AddrRange(
            DRAM_BASE + i * (CHANNEL_BYTES // DRAM_CHANNELS),
            size=CHANNEL_SIZE,
        )
        ctrl = intf.controller()
        ctrl.port = system.membus.mem_side_ports
        mem_ctrls.append(ctrl)
    system.mem_ctrls = mem_ctrls

    # Control-plane bridges
    # The DMA and array latencies are charged once per operation inside
    # the devices, so those bridges are plain routing
    def _bridge(start, end, delay_ns, target_bus):
        bridge = Bridge(delay=f"{delay_ns}ns",
                        ranges=[AddrRange(start, end)])
        bridge.cpu_side_port = system.ctl_bus.mem_side_ports
        bridge.mem_side_port = target_bus
        return bridge

    system.bridge_dma = _bridge(SRAM_BASE, DMA_PIO_END,
                                1, system.iobus.cpu_side_ports)
    system.bridge_arr = _bridge(SRAM_BASE + 0x40, ARR_PIO_END,
                                1, system.iobus.cpu_side_ports)

    # The control's sync interface to the vector CPU
    system.bridge_vec = _bridge(WORKSPACE_BASE,
                                CTL_REGION_BASE, LAT_VEC // 2,
                                system.sram_bus.cpu_side_ports)

    # The Control hart's own code/data/stack: local memory on its own path.
    system.ctl_mem = SimpleMemory(
        range=AddrRange(CTL_REGION_BASE, size=CTL_REGION_END -
                        CTL_REGION_BASE),
        latency=MEM_LATENCY, bandwidth=MEM_BANDWIDTH)
    system.ctl_mem.port = system.ctl_bus.mem_side_ports

    system.spm_bridge = Bridge(
        delay="1ns",
        ranges=[AddrRange(SRAM_BASE, SPM_WINDOW_END)])
    system.spm_bridge.cpu_side_port = system.sram_bus.mem_side_ports
    system.spm_bridge.mem_side_port = system.iobus.cpu_side_ports

    # The vector hart's boot path
    system.boot_bridge = Bridge(delay="10ns",
                                ranges=[AddrRange(DRAM_BASE,
                                                  DRAM_BASE +
                                                  BOOT_PAGE_SIZE)])
    system.boot_bridge.cpu_side_port = system.sram_bus.mem_side_ports
    system.boot_bridge.mem_side_port = system.membus.cpu_side_ports

    # The control hart's DRAM window
    system.ctl_boot_bridge = Bridge(delay="10ns",
                                    ranges=[AddrRange(DRAM_BASE,
                                                      size=DRAM_SIZE)])
    system.ctl_boot_bridge.cpu_side_port = system.ctl_bus.mem_side_ports
    system.ctl_boot_bridge.mem_side_port = system.membus.cpu_side_ports

    # Functional-only routes so the bare-metal loader
    system.load_ws_bridge = Bridge(
        delay="1ns", ranges=[AddrRange(WORKSPACE_BASE, CTL_REGION_BASE)])
    system.load_ws_bridge.cpu_side_port = system.membus.mem_side_ports
    system.load_ws_bridge.mem_side_port = system.sram_bus.cpu_side_ports
    system.load_ctl_bridge = Bridge(
        delay="1ns", ranges=[AddrRange(CTL_REGION_BASE, CTL_REGION_END)])
    system.load_ctl_bridge.cpu_side_port = system.membus.mem_side_ports
    system.load_ctl_bridge.mem_side_port = system.ctl_bus.cpu_side_ports

    system.dma = AccelDma(pio_addr=DMA_PIO)
    system.dma.pio = system.iobus.mem_side_ports
    system.dma.port = system.membus.cpu_side_ports
    system.dma.sram_port = system.iobus.cpu_side_ports
    system.dma.sram_window = [AddrRange(SRAM_BASE, size="12KiB")]
    system.dma.dram_base = DRAM_BASE

    # The PIO register access itself is free
    system.dma.pio_latency = "1ns"

    # The 16x16 systolic array: PIO-visible from the control side
    system.matmul = SystolicArray(pio_addr=MATMUL_PIO)
    system.matmul.pio = system.iobus.mem_side_ports
    system.matmul.local = system.iobus.cpu_side_ports
    system.matmul.pio_latency = "1ns"

    # The tile scratchpads - one SimpleMemory per SPM tile
    system.spm_qp = SimpleMemory(
        range=AddrRange(SPM_A, size="2304B"), latency=MEM_LATENCY,
        bandwidth=MEM_BANDWIDTH)

    system.spm_qp.port = system.iobus.mem_side_ports
    system.spm_kv = SimpleMemory(
        range=AddrRange(SPM_B, size="4096B"), latency=MEM_LATENCY,
        bandwidth=MEM_BANDWIDTH)

    system.spm_kv.port = system.iobus.mem_side_ports
    system.spm_so = SimpleMemory(
        range=AddrRange(SPM_C, size="4608B"), latency=MEM_LATENCY,
        bandwidth=MEM_BANDWIDTH)

    system.spm_so.port = system.iobus.mem_side_ports

    # Control CPU (hart 0)
    control_cpu = RiscvTimingSimpleCPU(cpu_id=0)
    control_cpu.createThreads()
    control_cpu.createInterruptController()
    control_cpu.clk_domain = SrcClockDomain(
        clock=CLK, voltage_domain=VoltageDomain()
    )
    control_cpu.icache_port = system.ctl_bus.cpu_side_ports
    control_cpu.dcache_port = system.ctl_bus.cpu_side_ports

    # Vector CPU (hart 1)
    vector_cpu = RiscvO3CPU(
        cpu_id=1,
        fuPool=VectorCpuPool(),
        fetchWidth=issue_width,
        decodeWidth=issue_width,
        renameWidth=issue_width,
        issueWidth=issue_width,
        wbWidth=issue_width,
        commitWidth=issue_width,
    )
    vector_cpu.createThreads()
    vector_cpu.isa[0].vlen = vlen
    vector_cpu.createInterruptController()
    vector_cpu.clk_domain = SrcClockDomain(
        clock=CLK, voltage_domain=VoltageDomain()
    )

    # The vector hart fetches its code from the SRAM (.sram.text) and does
    # all its data accesses through the SRAM arbiter
    vector_cpu.icache_port = system.sram_bus.cpu_side_ports
    vector_cpu.dcache_port = system.sram_bus.cpu_side_ports

    system.cpu = [control_cpu, vector_cpu]

    # The vector-CPU workspace: the SRAM's working-set partition, split
    # around the control hart's private CTL region (ctl_mem above).
    # The lo piece holds the sync flags, the vector's code/data and the
    # fp32 accumulator; the hi piece is the vector's stack (the linker
    # puts it at the workspace top).
    system.sram_workspace = SimpleMemory(
        range=AddrRange(WORKSPACE_BASE, CTL_REGION_BASE),
        latency=MEM_LATENCY, bandwidth=MEM_BANDWIDTH,
    )
    system.sram_workspace.port = system.sram_bus.mem_side_ports
    system.sram_stack = SimpleMemory(
        range=AddrRange(CTL_REGION_END, size="64KiB"),
        latency=MEM_LATENCY, bandwidth=MEM_BANDWIDTH,
    )
    system.sram_stack.port = system.sram_bus.mem_side_ports

    return system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", required=True,
                        help="bare-metal ELF (control+vector firmware)")
    parser.add_argument("--data", default="",
                        help="input image for m5_read_file (e.g. "
                             "flash_data.bin)")
    parser.add_argument("--issue-width", type=int, default=ISSUE_WIDTH)
    parser.add_argument("--vlen", type=int, default=VLEN)
    args = parser.parse_args()

    _max_tick = os.environ.get("MAX_TICK")
    if _max_tick:
        m5.setMaxTick(int(_max_tick))

    system = make_system(args.elf, issue_width=args.issue_width,
                         vlen=args.vlen)
    if args.data:
        system.readfile = args.data

    root = Root(full_system=True, system=system)

    m5.instantiate()

    log.warning("Accelerator: control=%s vector=%s(issue=%d, vlen=%d)",
                type(system.cpu[0]).__name__,
                type(system.cpu[1]).__name__,
                args.issue_width, args.vlen)
    log.warning("Accelerator: dram=%s x%d, sram window=0x%x, workspace=0x%x",
                DRAM_SIZE, DRAM_CHANNELS, SRAM_BASE,
                WORKSPACE_BASE)
    log.warning("Accelerator: elf=%s data=%s", args.elf,
                args.data or "(none)")
    log.warning("Accelerator: starting simulation")

    exit_event = m5.simulate()

    cause = exit_event.getCause()
    code = exit_event.getCode() & EXIT_CODE_MASK
    log.warning("Accelerator: exit cause: %s", cause)
    log.warning("Accelerator: exit code : 0x%08x (%d)", code, code)

    m5.stats.dump()

    stats_path = os.path.join(m5.options.outdir, "stats.txt")
    wanted = ("simInsts", "simTicks", "numCycles")
    with open(stats_path, "r", encoding="utf-8") as stats_file:
        for line in stats_file:
            key = line.split()[0] if line.strip() else ""
            if key in wanted:
                log.warning("Accelerator: stat: %s", line.rstrip())
    log.warning("Accelerator: done")


if __name__ == "__m5_main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    main()
