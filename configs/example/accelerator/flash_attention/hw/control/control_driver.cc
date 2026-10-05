/*
 * Control CPU driver (0): orchestrator of the flash-attention
 * pipeline across the DMA engine, the systolic array and the Vector CPU.
 *
 * Kernel launch: the boot itself is the kernel-launch command (the host
 * interface writes the Q|K|V image into the DRAM via m5_read_file
 *
 * Pipelined schedule:
 *   - SPM_B is double-buffered,
 *   - DMA engine is a separate agent and the vector CPU overlaps the array
 *
 */

#include "accelerator_config.hh"

#include <cstddef>
#include <cstdint>

#include <gem5/m5ops.h>

namespace {

using accel::DMA_CopyLen;
using accel::DMA_Flags;
using accel::DMA_RdAddr;
using accel::DMA_WrAddr;
using accel::kDevInit;
using accel::kDevIntr;
using accel::kExitBase;
using accel::kExitFail;
using accel::kExitPass;
using accel::kExitStageShift;
using accel::kPollTries;
using accel::MATMUL;
using accel::MATMUL_A;
using accel::MATMUL_B;
using accel::MATMUL_C;
using accel::MATMUL_K;

void fail_stage(int stage) { m5_exit(kExitBase + (kExitStageShift << stage)); }

// Program the DMA engine
void dma_copy(std::uint64_t src, std::uint64_t dst, std::uint32_t len,
              int stage) {
  auto *const flags = reinterpret_cast<volatile std::uint8_t *>(DMA_Flags);
  *reinterpret_cast<volatile std::uint64_t *>(DMA_RdAddr) = src;
  *reinterpret_cast<volatile std::uint64_t *>(DMA_WrAddr) = dst;
  *reinterpret_cast<volatile std::uint32_t *>(DMA_CopyLen) = len;
  *flags = kDevInit;

  int tries = 0;
  while ((*flags & kDevIntr) == 0) {
    if (++tries > kPollTries) {
      fail_stage(stage);
    }
  }
}

// Launch the systolic array with the full register set: the A/B/C SPM
// base addresses and the K contraction length
void matmul_exec(std::uint64_t a_base, std::uint64_t b_base,
                 std::uint64_t c_base, std::uint64_t k_len, int stage) {
  auto *const mm_flags = reinterpret_cast<volatile std::uint8_t *>(MATMUL);
  auto *const mm_a = reinterpret_cast<volatile std::uint64_t *>(MATMUL_A);
  auto *const mm_b = reinterpret_cast<volatile std::uint64_t *>(MATMUL_B);
  auto *const mm_c = reinterpret_cast<volatile std::uint64_t *>(MATMUL_C);
  auto *const mm_k = reinterpret_cast<volatile std::uint64_t *>(MATMUL_K);

  *mm_a = a_base;
  *mm_b = b_base;
  *mm_c = c_base;
  *mm_k = k_len;
  *mm_flags = kDevInit;

  int tries = 0;
  while ((*mm_flags & kDevIntr) == 0) {
    if (++tries > kPollTries) {
      fail_stage(stage);
    }
  }
}

// the scores matmul of the flash-attention: S_j = Q * (K_j)^T
void matmul_qk(std::uint64_t q_base, std::uint64_t k_base, std::uint64_t s_base,
               int stage) {
  matmul_exec(q_base, k_base, s_base, accel::kHeadDim, stage);
}

// Output matmul: O_j slice = P_j * V_j
void matmul_pv(std::uint64_t p_base, std::uint64_t v_base, std::uint64_t o_base,
               int stage) {
  matmul_exec(p_base, v_base, o_base, accel::kBC, stage);
}

// Wait for the Vector CPU to raise sync_done
void wait_vector(int stage) {
  int tries = 0;
  while (accel::sync_flag(accel::kSyncDone) == 0) {
    if (++tries > kPollTries) {
      fail_stage(stage);
    }
  }
  accel::sync_flag(accel::kSyncDone) = 0;
}

// PV_{1..7}: relaunch the systolic array rewriting only the B/C base
void matmul_pv_reconfig(std::uint64_t v_base, std::uint64_t o_base, int stage) {
  auto *const mm_flags = reinterpret_cast<volatile std::uint8_t *>(MATMUL);
  auto *const mm_b = reinterpret_cast<volatile std::uint64_t *>(MATMUL_B);
  auto *const mm_c = reinterpret_cast<volatile std::uint64_t *>(MATMUL_C);

  *mm_b = v_base;
  *mm_c = o_base;
  *mm_flags = kDevInit;

  int tries = 0;
  while ((*mm_flags & kDevIntr) == 0) {
    if (++tries > kPollTries) {
      fail_stage(stage);
    }
  }
}

} // namespace

extern "C" void control_main() {
  auto *const data = reinterpret_cast<void *>(
      static_cast<std::uintptr_t>(accel::kFlashDataAddr));
  const std::uint64_t image_bytes =
      m5_read_file(data, accel::kMaxImageBytes, 0);
  if (image_bytes < accel::kQBytes) {
    m5_exit(kExitFail);
  }
  const std::uint64_t tail = image_bytes - accel::kQBytes;
  const std::uint64_t n_tiles = tail / (2 * accel::kKvTileBytes);
  if (n_tiles == 0 || tail % (2 * accel::kKvTileBytes) != 0) {
    m5_exit(kExitFail);
  }

  const std::uint64_t q_addr = accel::kFlashDataAddr;
  const std::uint64_t k_addr = q_addr + accel::kQBytes;
  const std::uint64_t v_addr = k_addr + n_tiles * accel::kKvTileBytes;

  // The two K/V scratchpad buffers, selected by tile parity.
  const std::uint64_t b_buf[2] = {accel::kSpmKv0, accel::kSpmKv1};

  // Q (16x128, 2 KiB) is resident in SPM_A for the whole run.
  dma_copy(q_addr, accel::kSpmQ, accel::kQBytes, accel::kStageQ);

  // Prologue: stage K_0 and launch the first S run
  // The loop body launches S_{j+1} at the end of iteration j.
  dma_copy(k_addr, b_buf[0], accel::kKvTileBytes, accel::kStageK);
  matmul_qk(accel::kSpmQ, b_buf[0], accel::kSpmS, accel::kStageS);

  std::size_t p = 0;
  for (std::uint64_t j = 0; j < n_tiles; ++j) {
    // Softmax of S_j -> P
    // Vector CPU writes the quantized P to SPM_A+kQBytes).
    accel::sync_flag(accel::kSyncSoftmax) = 1;

    // Both DMAs hide under the softmax: K_{j+1} prefetches into the
    // parity buffer (its V_{j-1} was consumed by the previous PV
    // runs), V_j stages into the buffer K_j has just vacated.
    if (j + 1 < n_tiles) {
      dma_copy(k_addr + (j + 1) * accel::kKvTileBytes, b_buf[p ^ 1],
               accel::kKvTileBytes, accel::kStageK);
    }

    dma_copy(v_addr + j * accel::kKvTileBytes, b_buf[p], accel::kKvTileBytes,
             accel::kStageV);
    wait_vector(accel::kStageSync);

    // O_j = P_j V_j
    matmul_pv(accel::kSpmP, b_buf[p], accel::kSpmO(0), accel::kStagePV);

    accel::sync_flag(accel::kSyncO) = 1;
    for (std::size_t r = 1; r < accel::kVChunks; ++r) {
      matmul_pv_reconfig(b_buf[p] + r * accel::kKvChunkBytes, accel::kSpmO(r),
                         accel::kStagePV);
      accel::sync_flag(accel::kSyncO + r) = 1;
    }

    // S_{j+1} overlaps the Vector CPU accumulation
    // S run writes [0..511] while the vector reads the O chunks at [512..]
    if (j + 1 < n_tiles) {
      matmul_qk(accel::kSpmQ, b_buf[p ^ 1], accel::kSpmS, accel::kStageS);
    }
    wait_vector(accel::kStageSync);
    p ^= 1;
  }

  // Final O = O / l -- normalizes
  accel::sync_flag(accel::kSyncFinalize) = 1;
  wait_vector(accel::kStageFinal);
  dma_copy(accel::kOAccAddr, accel::kDumpAddr, accel::kOAccBytes,
           accel::kStageFinal);

  m5_write_file(
      reinterpret_cast<void *>(static_cast<std::uintptr_t>(accel::kDumpAddr)),
      accel::kOAccBytes, 0, "O_dump.bin");
  m5_dump_stats(0, 0);
  m5_exit(kExitPass);

  for (;;) {
  }
}
