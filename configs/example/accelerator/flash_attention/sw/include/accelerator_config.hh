#ifndef ACCELERATOR_CONFIG_HH
#define ACCELERATOR_CONFIG_HH

#include <cstddef>
#include <cstdint>

#include "config.h"

static_assert(config::kBR == config::kMatrix, "kBR must match kMatrix");
static_assert(config::kBC == config::kMatrix, "kBC must match kMatrix");
static_assert(config::kTileElems == config::kMatrix * config::kMatrix,
              "kTileElems must match kMatrix*kMatrix");

namespace accel {

// Device MMIO map
inline constexpr std::uintptr_t DMA_Flags = 0x10000000;
inline constexpr std::uintptr_t DMA_RdAddr = 0x10000008;
inline constexpr std::uintptr_t DMA_WrAddr = 0x10000010;
inline constexpr std::uintptr_t DMA_CopyLen = 0x10000018;
inline constexpr std::uintptr_t MATMUL = 0x10000040;
inline constexpr std::uintptr_t MATMUL_A = 0x10000048;
inline constexpr std::uintptr_t MATMUL_B = 0x10000050;
inline constexpr std::uintptr_t MATMUL_C = 0x10000058;
inline constexpr std::uintptr_t MATMUL_K = 0x10000060;
inline constexpr std::uintptr_t SPM_A = 0x10000080;

// SPM_B is double-buffered (2 x 2048 B): K_{j+1} streams in while V_j
// feeds the PV runs (the pipelined schedule).
inline constexpr std::uintptr_t SPM_B = 0x10000980;
inline constexpr std::uintptr_t SPM_C = 0x10001980;

// Hart roles (mhartid)
inline constexpr int kControlHart = 0;
inline constexpr int kVectorHart = 1;

// Workload geometry
inline constexpr std::size_t kBR = config::kBR;        // 16 query rows
inline constexpr std::size_t kBC = config::kBC;        // 16 keys per tile
inline constexpr std::size_t kHeadDim = 128;           // contraction of QK^T
inline constexpr std::size_t kTileI16Bytes =           // S_j / one O chunk
    2 * config::kTileElems;                            //   512 B
inline constexpr std::size_t kQBytes = kBR * kHeadDim; // 2048 B
inline constexpr std::size_t kKvTileBytes = kBC * kHeadDim;      // 2048 B
inline constexpr std::size_t kKvChunkBytes = config::kTileElems; // 256 B
inline constexpr std::size_t kVChunks = kHeadDim / kBC; // 8 PV runs/tile
inline constexpr std::size_t kOAccBytes =               // final O (fp32)
    kBR * kHeadDim * sizeof(float);                     //   8192 B

// Device MMR flags
inline constexpr std::uint8_t kDevInit = 0x01;
inline constexpr std::uint8_t kDevIntr = 0x04;

inline constexpr int kPollTries = 1000000;
inline constexpr int kExitPass = 0;
inline constexpr int kExitFail = 1;
inline constexpr int kExitBase = 2;
inline constexpr int kExitStageShift = 16;

// Stage codes
inline constexpr int kStageQ = 0;     // DMA: Q  -> SPM_A
inline constexpr int kStageK = 1;     // DMA: K_j -> SPM_B
inline constexpr int kStageS = 2;     // array: S_j = Q K_j^T
inline constexpr int kStageSync = 3;  // vector softmax handshake
inline constexpr int kStageV = 5;     // DMA: V_j -> SPM_B
inline constexpr int kStagePV = 6;    // array: O_j = P_j V_j (8 chunks)
inline constexpr int kStageFinal = 7; // final O/l handshake + DMA out

// Quantization: P fp16 -> int8 (array input), truncate toward zero;
// host's s_q/s_k/s_v scales in the generated flash_data.hh.

// Tensor DRAM addresses (the packed Q|K|V image from flash_data.bin)
inline constexpr std::uint64_t kFlashDataAddr = 0x80100000;

inline constexpr std::uint64_t kMaxImageBytes = 1u << 20; // 1 MiB

// DRAM staging for the final O/l dump (the m5_write_file source)
inline constexpr std::uint64_t kDumpAddr = 0x80200000;

// --- SPM region map
// SPM_A @ 0x10000080, 2304 B:
//   kSpmQ  Q (16x128 int8, resident), 2048 B
//   kSpmP  the quantized P tile (16x16 int8), 256 B
//
// SPM_B @ 0x10000980, 4096 B — the double-buffered K/V slots; slot p
//   holds K_{j+1} during the prefetch or V_j during the PV runs, the
//   roles alternate between tiles.
//   kSpmKv0  KV slot 0, 2048 B
//   kSpmKv1  KV slot 1, 2048 B
//
// SPM_C @ 0x10001980, 4608 B:
//   kSpmS  S_j (16x16 int16, the S run's output), 512 B
//   kSpmO(r)  the O_j chunk r (16x16 int16), 8 x 512 B
inline constexpr std::uintptr_t kSpmQ = SPM_A;
inline constexpr std::uintptr_t kSpmP = SPM_A + kQBytes;
inline constexpr std::size_t kPBytes = kBR * kBC; // 256 B
inline constexpr std::uintptr_t kSpmKv0 = SPM_B;
inline constexpr std::uintptr_t kSpmKv1 = SPM_B + kKvTileBytes;
inline constexpr std::uintptr_t kSpmS = SPM_C;
inline constexpr std::uintptr_t kSpmO0 = SPM_C + kTileI16Bytes;
inline constexpr std::uintptr_t kSpmO(std::size_t r) {
  return kSpmO0 + r * kTileI16Bytes;
}

static_assert(kSpmP == SPM_A + kQBytes, "P must follow Q in SPM_A");
static_assert(kSpmP + kPBytes == SPM_B, "SPM_A must be exactly 2304 B");
static_assert(SPM_B + 2 * kKvTileBytes == SPM_C,
              "SPM_B must be exactly 2 KV tiles (4096 B)");
static_assert(kSpmO0 == SPM_C + kTileI16Bytes, "O must follow S in SPM_C");
static_assert(kSpmO(kVChunks) == 0x10002b80,
              "the SPM window must end at 0x10002b80");

// Vector-hart workspace -- SimpleMemory at the SRAM base
inline constexpr std::uintptr_t kSramBase = 0x11000000;
inline constexpr std::uintptr_t kSyncSoftmax = kSramBase + 0x00;

// One flag per O_j output slice (kSramBase + 0x01 + r, r = 0..7):
inline constexpr std::uintptr_t kSyncO = kSramBase + 0x01;
inline constexpr std::uintptr_t kSyncFinalize = kSramBase + 0x09;
inline constexpr std::uintptr_t kSyncDone = kSramBase + 0x0A;

// The final O/l (fp32 16x128 row-major, 8 KiB) written by the vector
inline constexpr std::uintptr_t kOAccAddr = 0x11006000;

// Volatile access helpers for the fixed addresses ---
[[gnu::always_inline]] inline volatile std::uint8_t &
sync_flag(std::uintptr_t addr) noexcept {
  return *reinterpret_cast<volatile std::uint8_t *>(addr);
}

} // namespace accel

#endif // ACCELERATOR_CONFIG_HH
