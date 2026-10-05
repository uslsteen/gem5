/*
 * Vector CPU processor (hart 1): the softmax microkernel of the
 * flash-attention pipeline, with direct scratchpad access.
 *
 * Per tile:
 *   1. wait sync_softmax — read S_j (int16) from SPM_C, convert to the
 *      fp16 tile (bit-exact float_to_half),
 *      run the RVV online softmax (processTile), quantize P to int8 and store
 * it to SPM_A+kQBytes — the array's A operand for the PV runs;
 *   2. wait the per-slice sync_O flags — read the O_j chunks (int16)
 *      from SPM_C one slice at a time and accumulate O = c * O + O_j
 *      (fp32, 16x128);
 *   3. wait sync_finalize — normalize O = O / l and write it to the
 *      fixed O/l address in the workspace for the control's DMA.
 *
 * No DRAM traffic on the data path
 */

#include "accelerator_config.hh"
#include "rvv_softmax.hh"
#include "vector_state.hh"

#include <array>
#include <cstddef>
#include <cstdint>

namespace {

using kernel::vector_state;

[[gnu::section(".sram_text"), gnu::always_inline]] inline void
wait_flag(std::uintptr_t addr) {
  while (accel::sync_flag(addr) == 0) {
    __asm__ volatile("" ::: "memory");
  }
  accel::sync_flag(addr) = 0;
}

[[gnu::section(".sram_text"), gnu::always_inline]] inline void
raise_flag(std::uintptr_t addr) {
  accel::sync_flag(addr) = 1;
  __asm__ volatile("" ::: "memory");
}

} // namespace

extern "C" [[gnu::section(".sram_text"), noreturn]] void vector_main() {
  auto &s = vector_state;
  kernel::reset_softmax_state(s.m, s.l, s.c);
  const kernel::RvvSoftmax softmax{};

  auto *const spm_s = reinterpret_cast<std::int16_t *>(accel::kSpmS);
  auto *const spm_p = reinterpret_cast<std::uint8_t *>(accel::kSpmP);

  for (;;) {
    // Phase 1: the online softmax of the S_j tile
    while (accel::sync_flag(accel::kSyncSoftmax) == 0 &&
           accel::sync_flag(accel::kSyncFinalize) == 0) {
      __asm__ volatile("" ::: "memory");
    }
    if (accel::sync_flag(accel::kSyncFinalize) != 0) {
      // Phase 3 (once, at the end): O = O / l -> the
      // workspace, then idle until the Control hart exits. ---
      accel::sync_flag(accel::kSyncFinalize) = 0;
      kernel::finalize_o(s.o_acc.data(), s.l,
                         reinterpret_cast<float *>(accel::kOAccAddr));
      raise_flag(accel::kSyncDone);
      for (;;) {
        __asm__ volatile("" ::: "memory");
      }
    }
    accel::sync_flag(accel::kSyncSoftmax) = 0;

    // S_j: int16 (the array output) -> fp16 (bit-exact conversion).
    kernel::s16_tile_to_f16(spm_s, s.s_f16);

    softmax.processTile(s.s_f16, s.p_f16, s.m, s.l, s.c);

    // Quantize P to int8 (127 * P, truncated)
    // A operand is stored k-major (A[i][k] = P[i][k]),
    // so no transpose here.
    kernel::quantize_f16_tile_to_q8(s.p_f16, spm_p);

    raise_flag(accel::kSyncDone);

    // Phase 2: O_j accumulation: e^{m_{j-1}-m_j} rescale of
    // the running accumulator + the P_j V_j product
    // an elementwise fp32 update, hence on the vector CPU
    for (std::size_t chunk = 0; chunk < accel::kVChunks; ++chunk) {
      wait_flag(accel::kSyncO + chunk);
      const auto *const o_chunk =
          reinterpret_cast<std::int16_t *>(accel::kSpmO(chunk));
      kernel::accumulate_o_chunk(o_chunk, s.o_acc.data(), s.c, chunk);
    }

    raise_flag(accel::kSyncDone);
  }
}
