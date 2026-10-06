#ifndef RVV_SOFTMAX_HH
#define RVV_SOFTMAX_HH

#include "accelerator_config.hh"
#include "config.h"
#include "fp_types.hh"
#include "matrix.hh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <riscv_vector.h>

namespace kernel {

namespace cfg = config;

// A 16x16 fp16 tile:
// score tile S_j on input
// probability tile P_j on output
using Tile = Matrix<std::uint16_t, cfg::kBR, cfg::kBC>;

// Per-row online-softmax state (m, l, c): one float per tile row.
using StateVec = std::array<float, cfg::kBR>;

// ---------------------------------------------------------------------------
// fp constants as uint32 bit patterns.
//
// The Vector CPU's data path reaches the SRAM scratchpad only,
// so the kernel's fp constants MUST NOT land in the .rodata literal pool
//
// The 2^r polynomial coefficients live in the generated exp_coeffs.h
// ---------------------------------------------------------------------------
namespace vconst {
constexpr std::uint32_t kZero = 0x00000000u;
constexpr std::uint32_t kOne = 0x3F800000u;
constexpr std::uint32_t kNegInf = 0xFF800000u;
constexpr std::uint32_t kSubscale = 0x33800000u;   // 2^-24
constexpr std::uint32_t kQuantScale = 0x42FE0000u; // 127.0f
} // namespace vconst

// ---------------------------------------------------------------------------
// vload_f32_from_f16: widen fp16 (u16) to fp32 bit-exactly, replicating
// half_to_float() from fp_types.hh for the kernel's domain: 
// NORMAL, ZERO and +-INF inputs.
// Subnormals are NOT supported here — the kernel's loads are the S tiles:
// the int16 pipeline produces fp16-normal or zero
// ---------------------------------------------------------------------------
[[gnu::section(".sram_text"), gnu::always_inline]] inline vfloat32m1_t
vload_f32_from_f16(const std::uint16_t *ptr, std::size_t lanes) noexcept {
  const auto raw = __riscv_vle16_v_u16mf2(ptr, lanes);

  const auto val32 = __riscv_vzext_vf2_u32m1(raw, lanes);

  const auto low15 = __riscv_vand_vx_u32m1(val32, cfg::kF16NoSignMask, lanes);
  const auto sign_bits = __riscv_vsll_vx_u32m1(
      __riscv_vand_vx_u32m1(val32, cfg::kF16SignMask, lanes),
      cfg::kF16ToF32SignShift, lanes);

  const auto body_norm = __riscv_vsll_vx_u32m1(
      __riscv_vadd_vx_u32m1(
          low15,
          static_cast<std::uint32_t>(cfg::kF32ExpBias - cfg::kF16ExpBias)
              << cfg::kF16ExpShift,
          lanes),
      cfg::kF32ToF16Shift, lanes);

  const auto body_inf = __riscv_vor_vx_u32m1(
      __riscv_vsll_vx_u32m1(
          __riscv_vand_vx_u32m1(val32, cfg::kF16MantMask, lanes),
          cfg::kF32ToF16Shift, lanes),
      cfg::kF32ExpFieldMask, lanes);

  const auto exp_field =
      __riscv_vand_vx_u32m1(low15, cfg::kF16ExpFieldMask, lanes);
  const auto m_inf =
      __riscv_vmseq_vx_u32m1_b32(exp_field, cfg::kF16ExpFieldMask, lanes);
  const auto m_zero = __riscv_vmseq_vx_u32m1_b32(low15, 0u, lanes);

  auto body = __riscv_vmerge_vvm_u32m1(body_norm, body_inf, m_inf, lanes);
  body = __riscv_vmerge_vxm_u32m1(body, 0u, m_zero, lanes);

  const auto bits = __riscv_vor_vv_u32m1(body, sign_bits, lanes);
  return __riscv_vreinterpret_v_u32m1_f32m1(bits);
}

// ---------------------------------------------------------------------------
// vstore_f16_from_f32: narrow fp32 to fp16 (u16) bit-exactly, replicating
// float_to_half() from fp_types.hh (round toward zero) for the kernel's
// domain: NORMAL, ZERO and SUBNORMAL values.
// ---------------------------------------------------------------------------
[[gnu::section(".sram_text"), gnu::always_inline]] inline void
vstore_f16_from_f32(std::uint16_t *ptr, vfloat32m1_t value,
                    std::size_t lanes) noexcept {
  const auto bits = __riscv_vreinterpret_v_f32m1_u32m1(value);

  const auto sign16 = __riscv_vand_vx_u32m1(
      __riscv_vsrl_vx_u32m1(bits, cfg::kF16ToF32SignShift, lanes),
      cfg::kF16SignMask, lanes);

  const auto exp_s = __riscv_vsub_vx_i32m1(
      __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vand_vx_u32m1(
          __riscv_vsrl_vx_u32m1(bits, cfg::kF32MantBits, lanes),
          cfg::kF32ExpMask, lanes)),
      cfg::kF32ExpBias, lanes);
  const auto mant = __riscv_vand_vx_u32m1(bits, cfg::kF32MantMask, lanes);

  // Normal: (exp_s + 15) << 10 | mant >> 13.
  const auto half_exp = __riscv_vadd_vx_i32m1(exp_s, cfg::kF16ExpBias, lanes);
  const auto body_norm = __riscv_vor_vv_u32m1(
      __riscv_vsll_vx_u32m1(__riscv_vreinterpret_v_i32m1_u32m1(half_exp),
                            cfg::kF16ExpShift, lanes),
      __riscv_vsrl_vx_u32m1(mant, cfg::kF32ToF16Shift, lanes), lanes);

  // Subnormal: (mant | implicit) >> (-1 - exp_s).
  const auto shift_amt =
      __riscv_vreinterpret_v_i32m1_u32m1(__riscv_vnot_v_i32m1(exp_s, lanes));
  const auto body_sub = __riscv_vsrl_vv_u32m1(
      __riscv_vor_vx_u32m1(mant, cfg::kF32Implicit, lanes), shift_amt, lanes);

  const auto m_sub =
      __riscv_vmsle_vx_i32m1_b32(exp_s, cfg::kF16ExpMin - 1, lanes);
  const auto m_zero = __riscv_vmslt_vx_i32m1_b32(
      exp_s, cfg::kF16ExpMin - cfg::kF16UflowGuard, lanes);

  auto body = body_norm;
  body = __riscv_vmerge_vvm_u32m1(body, body_sub, m_sub, lanes);
  body = __riscv_vmerge_vxm_u32m1(body, 0u, m_zero, lanes);

  const auto result = __riscv_vor_vv_u32m1(body, sign16, lanes);

  const auto narrow = __riscv_vnsrl_wx_u16mf2(result, 0, lanes);
  __riscv_vse16_v_u16mf2(ptr, narrow, lanes);
}

// ---------------------------------------------------------------------------
// vstore_f16_from_f32_pos: the P-store variant of vstore_f16_from_f32.
// P = exp(S - m) >= 0, so the sign field is known to be zero and the
// sign extraction (vand + vsrl) plus the final sign merge (vor) are
// dropped. The normal|subnormal|zero branches are unchanged.
// ---------------------------------------------------------------------------
[[gnu::section(".sram_text"), gnu::always_inline]] inline void
vstore_f16_from_f32_pos(std::uint16_t *ptr, vfloat32m1_t value,
                        std::size_t lanes) noexcept {
  const auto bits = __riscv_vreinterpret_v_f32m1_u32m1(value);

  const auto exp_s = __riscv_vsub_vx_i32m1(
      __riscv_vreinterpret_v_u32m1_i32m1(__riscv_vand_vx_u32m1(
          __riscv_vsrl_vx_u32m1(bits, cfg::kF32MantBits, lanes),
          cfg::kF32ExpMask, lanes)),
      cfg::kF32ExpBias, lanes);
  const auto mant = __riscv_vand_vx_u32m1(bits, cfg::kF32MantMask, lanes);

  // Normal: (exp_s + 15) << 10 | mant >> 13.
  const auto half_exp = __riscv_vadd_vx_i32m1(exp_s, cfg::kF16ExpBias, lanes);
  const auto body_norm = __riscv_vor_vv_u32m1(
      __riscv_vsll_vx_u32m1(__riscv_vreinterpret_v_i32m1_u32m1(half_exp),
                            cfg::kF16ExpShift, lanes),
      __riscv_vsrl_vx_u32m1(mant, cfg::kF32ToF16Shift, lanes), lanes);

  // Subnormal: (mant | implicit) >> (-1 - exp_s).
  const auto shift_amt =
      __riscv_vreinterpret_v_i32m1_u32m1(__riscv_vnot_v_i32m1(exp_s, lanes));
  const auto body_sub = __riscv_vsrl_vv_u32m1(
      __riscv_vor_vx_u32m1(mant, cfg::kF32Implicit, lanes), shift_amt, lanes);

  const auto m_sub =
      __riscv_vmsle_vx_i32m1_b32(exp_s, cfg::kF16ExpMin - 1, lanes);
  const auto m_zero = __riscv_vmslt_vx_i32m1_b32(
      exp_s, cfg::kF16ExpMin - cfg::kF16UflowGuard, lanes);

  auto body = body_norm;
  body = __riscv_vmerge_vvm_u32m1(body, body_sub, m_sub, lanes);
  body = __riscv_vmerge_vxm_u32m1(body, 0u, m_zero, lanes);

  const auto narrow = __riscv_vnsrl_wx_u16mf2(body, 0, lanes);
  __riscv_vse16_v_u16mf2(ptr, narrow, lanes);
}

// ---------------------------------------------------------------------------
// The tile-staging helpers of the flash-attention data path:
// the S/P/O transfers between the scratchpad and the vector CPU vectorised with
// standard RVV 1.0. Each lane performs exactly the scalar reference's single
// fp32 operation — no reductions no reassociation — so the results stay
// bit-identical to the scalar reference loops.
// ---------------------------------------------------------------------------

// S_j: int16 (the array's C output) -> the fp16 score tile
[[gnu::section(".sram_text"), gnu::always_inline]] inline void
s16_tile_to_f16(const std::int16_t *src, Tile &dst) noexcept {
  const std::size_t lanes = __riscv_vsetvl_e16mf2(cfg::kBC);
  for (std::size_t row = 0; row < cfg::kBR; ++row) {
    const auto raw = __riscv_vle16_v_i16mf2(src + row * cfg::kBC, lanes);
    const auto f32 = __riscv_vfwcvt_f_x_v_f32m1(raw, lanes);
    vstore_f16_from_f32(dst[row].data(), f32, lanes);
  }
}

// P_j: fp16 -> int8
[[gnu::section(".sram_text"), gnu::always_inline]] inline void
quantize_f16_tile_to_q8(const Tile &p, std::uint8_t *dst) noexcept {
  const std::size_t lanes = __riscv_vsetvl_e16m4(cfg::kTileElems);
  const auto raw = __riscv_vle16_v_u16m4(p.data(), lanes);
  const auto val32 = __riscv_vzext_vf2_u32m8(raw, lanes);

  const std::size_t lanes32 = __riscv_vsetvl_e32m8(cfg::kTileElems);
  const auto low15 = __riscv_vand_vx_u32m8(val32, cfg::kF16NoSignMask, lanes32);
  const auto body_norm = __riscv_vsll_vx_u32m8(
      __riscv_vadd_vx_u32m8(
          low15,
          static_cast<std::uint32_t>(cfg::kF32ExpBias - cfg::kF16ExpBias)
              << cfg::kF16ExpShift,
          lanes32),
      cfg::kF32ToF16Shift, lanes32);

  const auto mant32 = __riscv_vand_vx_u32m8(val32, cfg::kF16MantMask, lanes32);
  const auto body_sub = __riscv_vreinterpret_v_f32m8_u32m8(
      __riscv_vfmul_vf_f32m8(__riscv_vfcvt_f_xu_v_f32m8(mant32, lanes32),
                             as_f32(vconst::kSubscale), lanes32));

  const auto exp_field =
      __riscv_vand_vx_u32m8(low15, cfg::kF16ExpFieldMask, lanes32);
  const auto m_exp0 = __riscv_vmseq_vx_u32m8_b4(exp_field, 0u, lanes32);
  const auto m_zero = __riscv_vmseq_vx_u32m8_b4(low15, 0u, lanes32);
  const auto m_sub =
      __riscv_vmand_mm_b4(m_exp0, __riscv_vmnot_m_b4(m_zero, lanes32), lanes32);

  auto body = body_norm;
  body = __riscv_vmerge_vvm_u32m8(body, body_sub, m_sub, lanes32);
  body = __riscv_vmerge_vxm_u32m8(body, 0u, m_zero, lanes32);
  const auto f32 = __riscv_vreinterpret_v_u32m8_f32m8(body);

  const auto scale_vec =
      __riscv_vfmv_v_f_f32m8(as_f32(vconst::kQuantScale), lanes32);
  const auto scaled = __riscv_vfmul_vv_f32m8(f32, scale_vec, lanes32);
  auto i32 = __riscv_vfcvt_rtz_x_f_v_i32m8(scaled, lanes32);
  auto u32 = __riscv_vreinterpret_v_i32m8_u32m8(i32);
  const auto u16 = __riscv_vnsrl_wx_u16m4(u32, 0, lanes32);
  const auto u8 = __riscv_vnsrl_wx_u8m2(u16, 0, lanes32);
  __riscv_vse8_v_u8m2(dst, u8, lanes32);
}

// One O_j chunk: o_acc[row][chunk*kBC + col] = c[row] * o_acc[...] +
// O_j[row][col].
[[gnu::section(".sram_text"), gnu::always_inline]] inline void
accumulate_o_chunk(const std::int16_t *o_chunk, float *o_acc, const StateVec &c,
                   std::size_t chunk) noexcept {
  const std::size_t lanes = __riscv_vsetvl_e32m1(cfg::kBC);
  for (std::size_t row = 0; row < cfg::kBR; ++row) {
    const auto raw = __riscv_vle16_v_i16mf2(o_chunk + row * cfg::kBC, lanes);
    const auto o_f32 = __riscv_vfwcvt_f_x_v_f32m1(raw, lanes);
    float *acc = o_acc + row * accel::kHeadDim + chunk * cfg::kBC;
    const auto acc_vec = __riscv_vle32_v_f32m1(acc, lanes);
    const auto upd = __riscv_vfmacc_vf_f32m1(o_f32, c[row], acc_vec, lanes);
    __riscv_vse32_v_f32m1(acc, upd, lanes);
  }
}

// O = O / l row by row
[[gnu::section(".sram_text"), gnu::always_inline]] inline void
finalize_o(const float *o_acc, const StateVec &l, float *out) noexcept {
  const std::size_t lanes = __riscv_vsetvl_e32m4(accel::kHeadDim);
  for (std::size_t row = 0; row < cfg::kBR; ++row) {
    const float *row_acc = o_acc + row * accel::kHeadDim;
    float *row_out = out + row * accel::kHeadDim;
    const auto v = __riscv_vle32_v_f32m4(row_acc, lanes);
    const auto d = __riscv_vfdiv_vf_f32m4(v, l[row], lanes);
    __riscv_vse32_v_f32m4(row_out, d, lanes);
  }
}

// Reset the online-softmax row state: m = -inf, l = 0, c = 1.
inline void reset_softmax_state(StateVec &m, StateVec &l,
                                StateVec &c) noexcept {
  m.fill(as_f32(vconst::kNegInf)); // -inf
  l.fill(as_f32(vconst::kZero));   // 0.0
  c.fill(as_f32(vconst::kOne));    // 1.0
}

// RVV online-softmax kernel (one 16x16 score tile at a time).
//
// Parameters and state notation:
//   S  -- the incoming score tile S_j  (fp16),
//   P  -- the output probability tile P_j = exp(S_j - m_j)  (fp16),
//   m  -- the running max m_j  (fp32),
//   l  -- the running denominator l_j  (fp32),
//   c  -- the per-tile rescale factor c_j = exp(m_{j-1} - m_j)  (fp32).
//
class RvvSoftmax final {
public:
  RvvSoftmax() noexcept = default;

  void processTile(const Tile &S, Tile &P, StateVec &m, StateVec &l,
                   StateVec &c) const noexcept;
};

} // namespace kernel

#endif
