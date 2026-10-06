from m5.objects.Device import BasicPioDevice
from m5.params import *
from m5.proxy import *


class AccelDma(BasicPioDevice):
    type = "AccelDma"
    cxx_header = "dev/accel_dma.hh"

    pio_size = Param.Addr(
        28, "MMR size (flags byte + 8B-aligned src/dst/len registers)"
    )
    port = RequestPort("Master port to the DRAM side")
    sram_port = RequestPort("Master port to the SRAM side")

    clock_period = Param.Int(
        1, "Clock period in ns (1 GHz domain)"
    )
    dram_read_cycles = Param.UInt64(
        200, "DRAM read latency, cycles (time to first byte)"
    )
    dram_write_cycles = Param.UInt64(
        150, "DRAM write latency, cycles (time to first byte)"
    )
    dma_bw_bytes_per_cycle = Param.UInt64(
        64, "DRAM<->DMA bandwidth, bytes/cycle (512 bit/cycle)"
    )
    config_latency_cycles = Param.UInt64(
        200, "Control CPU to DMA configuration latency, cycles "
        "(charged once per transfer)"
    )
    dram_base = Param.Addr(
        0x80000000, "Latency class: addresses >= dram_base pay the DRAM "
        "read/write latencies; below is SRAM"
    )
    sram_window = VectorParam.AddrRange(
        [AddrRange(0x10000000, size="12KiB")],
        "Addresses served through the sram_port (the SPM window); "
        "everything else goes through the DRAM-side port"
    )
