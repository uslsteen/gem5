/*
 * Cycle-accurate model of the 16x16 systolic array
 */

#include "dev/systolic_array.hh"

#include <algorithm>
#include <cstring>

#include "base/trace.hh"
#include "debug/MatMulExec.hh"
#include "mem/packet_access.hh"
#include "mem/request.hh"
#include "sim/core.hh"
#include "sim/system.hh"

namespace
{

constexpr uint8_t kDevInit = 0x01;  // the firmware's run bit
constexpr uint8_t kDevIntr = 0x04;  // the firmware's done bit

// The C tile writeout: 16x16 x 16 bit = 4096 bit
// 64 bit/cycle output link.
constexpr unsigned kDrainCycles = (16 * 16 * 2 * 8 + 63) / 64;

}  // anonymous namespace

SystolicArray::SystolicArray(const SystolicArrayParams &p)
    : BasicPioDevice(p, p.pio_size),
      spmPort(name() + ".local", this),
      ioAddr(p.pio_addr),
      ioSize(p.pio_size),
      running(false),
      lastK(0),
      runStart(0),
      busyUntil(0),
      requestorId(sys->getRequestorId(this)),
      clockPeriod(p.clock_period),
      doneEvent([this] { finishRun(); }, name() + ".done")
{
    mmreg = new uint8_t[ioSize];
    std::memset(mmreg, 0, ioSize);
    FLAGS = mmreg;
    A_BASE = reinterpret_cast<uint64_t *>(mmreg + 8);
    B_BASE = reinterpret_cast<uint64_t *>(mmreg + 16);
    C_BASE = reinterpret_cast<uint64_t *>(mmreg + 24);
    K_LEN = reinterpret_cast<uint64_t *>(mmreg + 32);
}

SystolicArray::~SystolicArray()
{
    delete[] mmreg;
}

bool
SystolicArray::SpmPort::recvTimingResp(PacketPtr pkt)
{
    panic("%s: unexpected timing response; the systolic array issues "
          "functional scratchpad accesses only\n",
          owner->name());
    return true;
}

void
SystolicArray::spmRead(Addr addr, uint8_t *buf, size_t size)
{
    RequestPtr req = std::make_shared<Request>(addr, size, 0, requestorId);
    Packet pkt(req, MemCmd::ReadReq);
    pkt.dataStatic(buf);
    spmPort.sendFunctional(&pkt);
}

void
SystolicArray::spmWrite(Addr addr, const uint8_t *buf, size_t size)
{
    RequestPtr req = std::make_shared<Request>(addr, size, 0, requestorId);
    Packet pkt(req, MemCmd::WriteReq);
    pkt.dataStatic(const_cast<uint8_t *>(buf));
    spmPort.sendFunctional(&pkt);
}

void
SystolicArray::startRun()
{
    const uint64_t a = *A_BASE;
    const uint64_t b = *B_BASE;
    const uint64_t c = *C_BASE;
    const uint64_t k = *K_LEN;

    running = true;
    *FLAGS &= ~kDevInit;
    lastK = k;

    if (k == 0 || k > kMaxK || (k % 8) != 0) {
        panic("%s: unsupported K=%llu (must be a multiple of 8, <= %u)\n",
              name(), static_cast<unsigned long long>(k), kMaxK);
    }

    // 16xK int8 operands (k-major) + the 16x16 int16 result.
    const size_t tileBytes = kMatrix * static_cast<size_t>(k);
    std::vector<uint8_t> aBuf(tileBytes);
    std::vector<uint8_t> bBuf(tileBytes);
    std::vector<uint8_t> cBuf(kMatrix * kMatrix * 2);
    spmRead(a, aBuf.data(), tileBytes);
    spmRead(b, bBuf.data(), tileBytes);

    // C[i][j] = sum_k A[i][k] * B[j][k], int8 -> int16 with 64-bit
    // k-major word loads and an int16 accumulator
    for (unsigned i = 0; i < kMatrix; ++i) {
        for (unsigned j = 0; j < kMatrix; ++j) {
            int16_t acc = 0;
            for (uint64_t kk = 0; kk < k; kk += 8) {
                uint64_t av, bv;
                std::memcpy(&av, &aBuf[i * k + kk], sizeof(av));
                std::memcpy(&bv, &bBuf[j * k + kk], sizeof(bv));
                for (unsigned e = 0; e < 8; ++e) {
                    const int16_t aElem =
                        static_cast<int16_t>(static_cast<int8_t>(
                            (av >> (8 * e)) & 0xFFu));
                    const int16_t bElem =
                        static_cast<int16_t>(static_cast<int8_t>(
                            (bv >> (8 * e)) & 0xFFu));
                    acc = static_cast<int16_t>(acc + aElem * bElem);
                }
            }
            int16_t accLe = acc;
            std::memcpy(&cBuf[(i * kMatrix + j) * 2], &accLe, 2);
        }
    }
    spmWrite(c, cBuf.data(), cBuf.size());

    // K MAC cycles + pipeline fill/drain (2*(16-1));
    // the C writeout (64 cycles) is pipelined into the next run, so a
    // launch inside the drain window of the previous run stalls until
    // then.
    runStart = curTick();
    const Tick computeDone =
        runStart + (k + kFillDrain) * clockPeriod * sim_clock::as_int::ps;
    const Tick ready = std::max(computeDone, busyUntil);
    DPRINTF(MatMulExec, "%lu: Kernel Run Start (K=%llu, %lu ticks)\n",
            curTick(), static_cast<unsigned long long>(k),
            static_cast<unsigned long>(ready - runStart));
    schedule(doneEvent, ready);
    busyUntil = ready + kDrainCycles * clockPeriod * sim_clock::as_int::ps;
}

void
SystolicArray::finishRun()
{
    running = false;
    *FLAGS |= kDevIntr;
    DPRINTF(MatMulExec, "%lu: Kernel Run End: K=%llu ticks=%lu\n",
            curTick(), static_cast<unsigned long long>(lastK),
            static_cast<unsigned long>(curTick() - runStart));
}

Tick
SystolicArray::read(PacketPtr pkt)
{
    const Addr offset = pkt->req->getPaddr() - ioAddr;
    if (offset >= ioSize) {
        panic("%s: PIO read at offset %#x is outside the %#x-byte MMR\n",
              name(), offset, ioSize);
    }
    uint64_t data = 0;
    switch (pkt->getSize()) {
        case 1:
            data = *(mmreg + offset);
            break;
        case 2:
            data = *(uint16_t *)(mmreg + offset);
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
        case 2:
            pkt->setLE<uint16_t>(data);
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

Tick
SystolicArray::write(PacketPtr pkt)
{
    const Addr offset = pkt->req->getPaddr() - ioAddr;
    if (offset >= ioSize) {
        panic("%s: PIO write at offset %#x is outside the %#x-byte MMR\n",
              name(), offset, ioSize);
    }
    pkt->writeData(mmreg + offset);

    // The flags byte at offset 0: a write with the run bit set starts
    // the array (the a/b/c/k_len registers are already latched); a
    // plain write (the firmware's post-done clear) is a no-op.
    if (offset == 0 && (*FLAGS & kDevInit) && !running) {
        startRun();
    }

    pkt->makeAtomicResponse();
    return pioDelay;
}

Port &
SystolicArray::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "local") {
        return spmPort;
    }
    return PioDevice::getPort(if_name, idx);
}
