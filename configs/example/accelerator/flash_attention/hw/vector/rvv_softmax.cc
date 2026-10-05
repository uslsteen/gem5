#include "rvv_softmax.hh"

#include "exp_coeffs.h"

#include <cstddef>
#include <cstdint>
#include <riscv_vector.h>

namespace kernel {

namespace cfg = config;

namespace {

// The vexp_f32-only fp constants
// The 2^r polynomial coefficients come from the generated exp_coeffs.h.
constexpr std::uint32_t kHalf = 0x3F000000u;
constexpr std::uint32_t kLog2E = 0x3FB8AA3Bu;
constexpr std::uint32_t kExpLo = 0xC2AE999Au;
constexpr std::uint32_t kExpHi = 0x42B16666u;

// ---------------------------------------------------------------------------
// vexp_f32: vectorised e^x for x in [-inf, +inf].
//
// Numeric recipe
//   * lanes with x < kExpLo are set to exactly 0 (reference semantics),
//   * x is clamped to [kExpLo, kExpHi],
//   * t   = x * log2(e),
//   * k   = round-half-away-from-zero(t),  r = t - k  in [-0.5, +0.5],
//   * 2^r ~= P(r)  (degree-6 near-minimax fit, Horner + FMA),
//   * 2^k  = bit reconstruction of the fp32 exponent field,
//   * e^x ~= P(r) * 2^k.
// NaN lanes are NOT handled here; the caller masks them out.
// ---------------------------------------------------------------------------
[[gnu::section(".sram_text"), gnu::always_inline]] inline vfloat32m1_t
vexp_f32(vfloat32m1_t input, std::size_t lanes) noexcept {
  const auto lo_mask = __riscv_vmflt_vf_f32m1_b32(input, as_f32(kExpLo), lanes);
  auto clamped = __riscv_vfmax_vf_f32m1(input, as_f32(kExpLo), lanes);
  clamped = __riscv_vfmin_vf_f32m1(clamped, as_f32(kExpHi), lanes);

  const auto log2e_vec = __riscv_vfmv_v_f_f32m1(as_f32(kLog2E), lanes);
  const auto scaled = __riscv_vfmul_vv_f32m1(clamped, log2e_vec, lanes);

  // Round-half-away-from-zero: k = trunc(t + copysign(0.5, t)).
  const auto half_vec = __riscv_vfmv_v_f_f32m1(as_f32(kHalf), lanes);
  const auto bias = __riscv_vfsgnj_vv_f32m1(half_vec, scaled, lanes);
  const auto rounded = __riscv_vfadd_vv_f32m1(scaled, bias, lanes);
  const auto int_part = __riscv_vfcvt_rtz_x_f_v_i32m1(rounded, lanes);
  const auto int_float = __riscv_vfcvt_f_x_v_f32m1(int_part, lanes);
  const auto frac_part = __riscv_vfsub_vv_f32m1(scaled, int_float, lanes);

  auto poly = __riscv_vfmv_v_f_f32m1(as_f32(exp_coeffs::kC6), lanes);
  poly = __riscv_vfmadd_vv_f32m1(
      poly, frac_part, __riscv_vfmv_v_f_f32m1(as_f32(exp_coeffs::kC5), lanes),
      lanes);
  poly = __riscv_vfmadd_vv_f32m1(
      poly, frac_part, __riscv_vfmv_v_f_f32m1(as_f32(exp_coeffs::kC4), lanes),
      lanes);
  poly = __riscv_vfmadd_vv_f32m1(
      poly, frac_part, __riscv_vfmv_v_f_f32m1(as_f32(exp_coeffs::kC3), lanes),
      lanes);
  poly = __riscv_vfmadd_vv_f32m1(
      poly, frac_part, __riscv_vfmv_v_f_f32m1(as_f32(exp_coeffs::kC2), lanes),
      lanes);
  poly = __riscv_vfmadd_vv_f32m1(
      poly, frac_part, __riscv_vfmv_v_f_f32m1(as_f32(exp_coeffs::kC1), lanes),
      lanes);
  poly = __riscv_vfmadd_vv_f32m1(
      poly, frac_part, __riscv_vfmv_v_f_f32m1(as_f32(exp_coeffs::kC0), lanes),
      lanes);

  const auto biased_exp =
      __riscv_vadd_vx_i32m1(int_part, cfg::kF32ExpBias, lanes);
  const auto exp_bits =
      __riscv_vsll_vx_i32m1(biased_exp, cfg::kF32MantBits, lanes);
  const auto scale_vec = __riscv_vreinterpret_v_i32m1_f32m1(exp_bits);

  auto result = __riscv_vfmul_vv_f32m1(poly, scale_vec, lanes);
  const auto zero_vec = __riscv_vfmv_v_f_f32m1(as_f32(vconst::kZero), lanes);
  return __riscv_vmerge_vvm_f32m1(result, zero_vec, lo_mask, lanes);
}

[[gnu::section(".sram_text"), gnu::always_inline]] inline float
exp_rescale(float value) noexcept {
  const auto input = __riscv_vfmv_v_f_f32m1(value, 1);
  const auto expv = vexp_f32(input, 1);
  return __riscv_vfmv_f_s_f32m1_f32(expv);
}

} // namespace

[[gnu::section(".sram_text")]] void
RvvSoftmax::processTile(const Tile &S, Tile &P, StateVec &m, StateVec &l,
                        StateVec &c) const noexcept {
  const std::size_t lanes = __riscv_vsetvl_e32m1(cfg::kBC);
  const auto zero_vec = __riscv_vfmv_v_f_f32m1(as_f32(vconst::kZero), lanes);
  const auto neg_inf_vec =
      __riscv_vfmv_v_f_f32m1(as_f32(vconst::kNegInf), lanes);

  for (std::size_t row = 0; row < cfg::kBR; ++row) {
    // The S tile was produced by s16_tile_to_f16, so its fp16
    const auto score_vec = vload_f32_from_f16(S[row].data(), lanes);

    const auto max_vec =
        __riscv_vfredmax_vs_f32m1_f32m1(score_vec, neg_inf_vec, lanes);
    const float row_max = __riscv_vfmv_f_s_f32m1_f32(max_vec);

    const float max_prev = m[row];
    const float max_new = row_max > max_prev ? row_max : max_prev;

    const auto max_broadcast = __riscv_vfmv_v_f_f32m1(max_new, lanes);
    const auto shifted_vec =
        __riscv_vfsub_vv_f32m1(score_vec, max_broadcast, lanes);
    const auto nan_mask =
        __riscv_vmfne_vv_f32m1_b32(shifted_vec, shifted_vec, lanes);

    auto exp_vec = vexp_f32(shifted_vec, lanes);
    exp_vec = __riscv_vmerge_vvm_f32m1(exp_vec, zero_vec, nan_mask, lanes);

    const auto sum_vec =
        __riscv_vfredusum_vs_f32m1_f32m1(exp_vec, zero_vec, lanes);
    const float row_sum = __riscv_vfmv_f_s_f32m1_f32(sum_vec);

    const float scale = (max_prev == max_new) ? as_f32(vconst::kOne)
                                              : exp_rescale(max_prev - max_new);

    m[row] = max_new;
    l[row] = scale * l[row] + row_sum;
    c[row] = scale;

    vstore_f16_from_f32(P[row].data(), exp_vec, lanes);
  }
}

} // namespace kernel
