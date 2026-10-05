#ifndef FP_TYPES_HH
#define FP_TYPES_HH

#include "config.h"

#include <cstdint>

namespace kernel {

namespace cfg = config;

// The bit-pattern bitcasts
[[gnu::always_inline]] inline std::uint32_t as_u32(float value) noexcept {
  std::uint32_t bits;
  __asm__("fmv.x.w %0, %1" : "=r"(bits) : "f"(value));
  return bits;
}

[[gnu::always_inline]] inline float as_f32(std::uint32_t bits) noexcept {
  float value;
  __asm__("fmv.w.x %0, %1" : "=f"(value) : "r"(bits));
  return value;
}

[[gnu::always_inline]] inline float half_to_float(std::uint16_t half) noexcept {
  const auto sign = (half >> cfg::kF16SignShift) & 1u;
  const auto exponent = (half >> cfg::kF16ExpShift) & cfg::kF16ExpMask;
  const auto mantissa = half & cfg::kF16MantMask;

  std::uint32_t bits = sign << cfg::kF32SignShift;
  if (exponent == 0u) {
    if (mantissa != 0u) {
      std::uint32_t shifted = mantissa;
      std::int32_t sub_exp = cfg::kF16ExpMin;
      while ((shifted & cfg::kF16Implicit) == 0u) {
        shifted <<= 1;
        --sub_exp;
      }
      shifted &= ~cfg::kF16Implicit;
      bits |= static_cast<std::uint32_t>(sub_exp + cfg::kF32ExpBias)
              << cfg::kF32MantBits;
      bits |= shifted << cfg::kF32ToF16Shift;
    }
  } else if (exponent == cfg::kF16ExpMask) {
    bits |= cfg::kF32ExpMask << cfg::kF32MantBits;
    bits |= mantissa << cfg::kF32ToF16Shift;
  } else {
    bits |= static_cast<std::uint32_t>(static_cast<std::int32_t>(exponent) -
                                       cfg::kF16ExpBias + cfg::kF32ExpBias)
            << cfg::kF32MantBits;
    bits |= mantissa << cfg::kF32ToF16Shift;
  }
  return as_f32(bits);
}

[[gnu::always_inline]] inline std::uint16_t
float_to_half(float value) noexcept {
  const auto bits = as_u32(value);
  const auto sign = (bits >> cfg::kF32SignShift) & 1u;
  const auto exponent = static_cast<std::int32_t>((bits >> cfg::kF32MantBits) &
                                                  cfg::kF32ExpMask) -
                        cfg::kF32ExpBias;
  const auto mantissa = bits & cfg::kF32MantMask;

  if (exponent > cfg::kF16ExpMax)
    return static_cast<std::uint16_t>((sign << cfg::kF16SignShift) |
                                      cfg::kF16InfBits);
  if (exponent < cfg::kF16ExpMin - cfg::kF16UflowGuard)
    return static_cast<std::uint16_t>(sign << cfg::kF16SignShift);
  if (exponent < cfg::kF16ExpMin) {
    const std::uint32_t implicit_one = mantissa | cfg::kF32Implicit;
    const auto shift =
        static_cast<unsigned>(cfg::kF16ExpMin - exponent +
                              static_cast<std::int32_t>(cfg::kF32ToF16Shift));
    return static_cast<std::uint16_t>((sign << cfg::kF16SignShift) |
                                      (implicit_one >> shift));
  }
  const auto half_exp = static_cast<std::uint32_t>(exponent + cfg::kF16ExpBias);
  return static_cast<std::uint16_t>((sign << cfg::kF16SignShift) |
                                    (half_exp << cfg::kF16ExpShift) |
                                    (mantissa >> cfg::kF32ToF16Shift));
}

} // namespace kernel

#endif
