/*
 * Cycle-accurate model of the 16x16 systolic array as dedicated PIO device.
 * *
 * The device performs the arithmetic functionally at launch (int8 ->
 * int16, 64-bit k-major word loads, int32 accumulation with int16
 * saturation) and signals
 * completion through the PIO flag protocol the firmware polls:
 *
 *   +0  flags (1 B): 0x01 run (kDevInit), 0x04 done (kDevIntr)
 *   +8  a_base (8 B)
 *   +16 b_base (8 B)
 *   +24 c_base (8 B)
 *   +32 k_len  (8 B)
 *
 * The control CPU's 100-cycle communication latency is charged once
 * per launch inside the busy window below (configLatencyCycles).
 */

#ifndef __DEV_SYSTOLIC_ARRAY_HH__
#define __DEV_SYSTOLIC_ARRAY_HH__

#include <cstdint>
#include <string>
#include <vector>

#include "base/types.hh"
#include "dev/io_device.hh"
#include "mem/packet.hh"
#include "mem/port.hh"
#include "params/SystolicArray.hh"
#include "sim/eventq.hh"

using namespace gem5;

class SystolicArray : public BasicPioDevice
{
  private:
    static constexpr unsigned kMatrix = 16;
    static constexpr unsigned kFillDrain = 2 * (kMatrix - 1);  // 30 cycles
    static constexpr unsigned kMaxK = 256;

    class SpmPort : public RequestPort
    {
      private:
        SystolicArray *owner;

      public:
        SpmPort(const std::string &name, SystolicArray *owner,
                PortID id = InvalidPortID)
            : RequestPort(name, id), owner(owner)
        {}

      protected:
        bool recvTimingResp(PacketPtr pkt) override;
        void recvReqRetry() override {}
    };

    SpmPort spmPort;

    Addr ioAddr;
    Addr ioSize;

    uint8_t *mmreg;
    uint8_t *FLAGS;
    uint64_t *A_BASE;
    uint64_t *B_BASE;
    uint64_t *C_BASE;
    uint64_t *K_LEN;

    bool running;
    uint64_t lastK;
    Tick runStart;

    // Output-drain reservation: the C writeout (64 cycles at 64 bit/
    // cycle) overlaps the next run; a back-to-back launch therefore
    // stalls until the previous drain completes.
    Tick busyUntil;

    // The system-side requestor identity for the functional traffic
    // (must be obtained at construction, before regStats()).
    RequestorID requestorId;

    // Cycle -> tick conversion (clock_period is in ns).
    int clockPeriod;
    uint64_t configLatencyCycles;

    EventFunctionWrapper doneEvent;

    void startRun();
    void finishRun();
    void spmRead(Addr addr, uint8_t *buf, size_t size);
    void spmWrite(Addr addr, const uint8_t *buf, size_t size);

  public:
    PARAMS(SystolicArray);
    SystolicArray(const SystolicArrayParams &p);
    ~SystolicArray();

    Tick read(PacketPtr pkt) override;
    Tick write(PacketPtr pkt) override;

    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;
};

#endif  // __DEV_SYSTOLIC_ARRAY_HH__
