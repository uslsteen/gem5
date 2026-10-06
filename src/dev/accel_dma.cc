/*
 * AccelDma: the DMA engine of the custom accelerator
 */

#include "dev/accel_dma.hh"

#include <cstring>
#include <vector>

#include "base/trace.hh"
#include "debug/DmaExec.hh"
#include "mem/packet_access.hh"
#include "mem/request.hh"
#include "sim/core.hh"
#include "sim/system.hh"

namespace {

constexpr uint8_t kDmaRun = 0x01;
constexpr uint8_t kDmaDone = 0x04;

} // anonymous namespace

AccelDma::AccelDma(const AccelDmaParams &p)
    : BasicPioDevice(p, p.pio_size), memPort(name() + ".port", this),
      sramPort(name() + ".sram_port", this), ioAddr(p.pio_addr),
      ioSize(p.pio_size), running(false), startTick(0), activeSrc(0),
      activeDst(0), requestorId(sys->getRequestorId(this)),
      clockPeriod(p.clock_period), dramReadCycles(p.dram_read_cycles),
      dramWriteCycles(p.dram_write_cycles),
      bwBytesPerCycle(p.dma_bw_bytes_per_cycle),
      configLatencyCycles(p.config_latency_cycles), dramBase(p.dram_base),
      sramWindow(p.sram_window),
      doneEvent([this] { finishTransfer(); }, name() + ".done") {
  mmreg = new uint8_t[ioSize];
  std::memset(mmreg, 0, ioSize);
  FLAGS = mmreg;
  SRC = reinterpret_cast<uint64_t *>(mmreg + 8);
  DST = reinterpret_cast<uint64_t *>(mmreg + 16);
  LEN = reinterpret_cast<uint32_t *>(mmreg + 24);
}

AccelDma::~AccelDma() { delete[] mmreg; }

bool AccelDma::MemPort::recvTimingResp(PacketPtr pkt) {
  panic("%s: unexpected timing response; the accelerator DMA issues "
        "functional transfers only\n",
        owner->name());
  return true;
}

void AccelDma::functionalCopy(Addr src, Addr dst, size_t len) {
  std::vector<uint8_t> data(len);
  {
    RequestPtr req = std::make_shared<Request>(src, len, 0, requestorId);
    Packet pkt(req, MemCmd::ReadReq);
    pkt.dataStatic(data.data());
    (usesSramPort(src) ? sramPort : memPort).sendFunctional(&pkt);
  }
  {
    RequestPtr req = std::make_shared<Request>(dst, len, 0, requestorId);
    Packet pkt(req, MemCmd::WriteReq);
    pkt.dataStatic(data.data());
    (usesSramPort(dst) ? sramPort : memPort).sendFunctional(&pkt);
  }
}

void AccelDma::startTransfer() {
  running = true;
  *FLAGS &= ~kDmaRun;
  activeSrc = *SRC;
  activeDst = *DST;
  const uint32_t len = *LEN;
  startTick = curTick();

  DPRINTF(DmaExec, "%lu: k: %s(0x%016x) -> %s(0x%016x), %d B\n", curTick(),
          isDram(activeSrc) ? "DRAM" : "SRAM", activeSrc,
          isDram(activeDst) ? "DRAM" : "SRAM", activeDst, len);

  // The transfer's physical timing is analytic formula:
  // configuration latency + DRAM read/write latency + 512-bit/cycle
  // streaming; the config latency is charged once per transfer.
  functionalCopy(activeSrc, activeDst, len);
  const uint64_t cycles = configLatencyCycles +
                          (isDram(activeSrc) ? dramReadCycles : 0) +
                          (isDram(activeDst) ? dramWriteCycles : 0) +
                          (len + bwBytesPerCycle - 1) / bwBytesPerCycle;
  schedule(doneEvent, curTick() + cycles * clockPeriod * sim_clock::as_int::ps);
}

void AccelDma::finishTransfer() {
  running = false;
  *FLAGS |= kDmaDone;
  DPRINTF(DmaExec, "%lu: DMA end: %d B in %lu ticks\n", curTick(), (int)*LEN,
          (unsigned long)(curTick() - startTick));
}

Tick AccelDma::read(PacketPtr pkt) {
  const Addr offset = pkt->req->getPaddr() - ioAddr;
  if (offset >= ioSize) {
    panic("%s: PIO read at offset %#x is outside the %#x-byte MMR\n", name(),
          offset, ioSize);
  }
  uint64_t data = 0;
  switch (pkt->getSize()) {
  case 1:
    data = *(mmreg + offset);
    break;
  case 4:
    data = *(uint32_t *)(mmreg + offset);
    break;
  case 8:
    data = *(uint64_t *)(mmreg + offset);
    break;
  default:
    panic("%s: PIO read of %d bytes is not supported\n", name(),
          pkt->getSize());
  }
  switch (pkt->getSize()) {
  case 1:
    pkt->setLE<uint8_t>(data);
    break;
  case 4:
    pkt->setLE<uint32_t>(data);
    break;
  case 8:
    pkt->setLE<uint64_t>(data);
    break;
  }
  pkt->makeAtomicResponse();
  return pioDelay;
}

Tick AccelDma::write(PacketPtr pkt) {
  const Addr offset = pkt->req->getPaddr() - ioAddr;
  if (offset >= ioSize) {
    panic("%s: PIO write at offset %#x is outside the %#x-byte MMR\n", name(),
          offset, ioSize);
  }
  pkt->writeData(mmreg + offset);

  // The flags byte at offset 0: a write with the run bit set starts
  // the transfer (the SRC/DST/LEN registers are already latched); a
  // plain write (the firmware's post-done clear) is a no-op.
  if (offset == 0 && (*FLAGS & kDmaRun) && !running) {
    startTransfer();
  }

  pkt->makeAtomicResponse();
  return pioDelay;
}

Port &AccelDma::getPort(const std::string &if_name, PortID idx) {
  if (if_name == "port") {
    return memPort;
  } else if (if_name == "sram_port") {
    return sramPort;
  }
  return PioDevice::getPort(if_name, idx);
}
