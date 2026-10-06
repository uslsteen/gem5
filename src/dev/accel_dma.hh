/*
 * AccelDma: the DMA engine as a plain PIO device.
 */

#ifndef __DEV_ACCEL_DMA_HH__
#define __DEV_ACCEL_DMA_HH__

#include <cstddef>
#include <cstdint>
#include <string>

#include "base/addr_range.hh"
#include "base/types.hh"
#include "dev/io_device.hh"
#include "mem/packet.hh"
#include "mem/port.hh"
#include "params/AccelDma.hh"
#include "sim/eventq.hh"

using namespace gem5;

class AccelDma : public BasicPioDevice
{
  private:
    class MemPort : public RequestPort
    {
      private:
        AccelDma *owner;

      public:
        MemPort(const std::string &name, AccelDma *owner,
                PortID id = InvalidPortID)
            : RequestPort(name, id), owner(owner)
        {}

      protected:
        bool recvTimingResp(PacketPtr pkt) override;
        void recvReqRetry() override {}
    };

    MemPort memPort;
    MemPort sramPort;

    Addr ioAddr;
    Addr ioSize;

    uint8_t *mmreg;
    uint8_t *FLAGS;
    uint64_t *SRC;
    uint64_t *DST;
    uint32_t *LEN;

    bool running;
    Tick startTick;
    Addr activeSrc;
    Addr activeDst;

    RequestorID requestorId;

    int clockPeriod;
    uint64_t dramReadCycles;
    uint64_t dramWriteCycles;
    uint64_t bwBytesPerCycle;
    uint64_t configLatencyCycles;
    Addr dramBase;
    AddrRange sramWindow;

    EventFunctionWrapper doneEvent;

    // The port (routing) and the latency (timing) are decoupled
    bool isDram(Addr addr) const { return addr >= dramBase; }
    bool usesSramPort(Addr addr) const { return sramWindow.contains(addr); }
    void functionalCopy(Addr src, Addr dst, size_t len);
    void startTransfer();
    void finishTransfer();

  public:
    PARAMS(AccelDma);
    AccelDma(const AccelDmaParams &p);
    ~AccelDma();

    Tick read(PacketPtr pkt) override;
    Tick write(PacketPtr pkt) override;

    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;
};

#endif  // __DEV_ACCEL_DMA_HH__
