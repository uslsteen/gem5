#ifndef SOFTMAX_CONFIG_H
#define SOFTMAX_CONFIG_H

#include <cstddef>
#include <cstdint>

namespace config {

// Tile geometry
inline constexpr std::size_t kMatrix = 16;
inline constexpr std::size_t kBR = 16;
inline constexpr std::size_t kBC = 16;
inline constexpr std::size_t kTileElems = kBR * kBC;

// fp32 bit layout
inline constexpr std::uint32_t kF32SignShift = 31u;
inline constexpr std::uint32_t kF32MantBits = 23u;
inline constexpr std::uint32_t kF32ExpMask = 0xFFu;
inline constexpr std::uint32_t kF32ExpFieldMask = 0x7F800000u; // 0xFF << 23
inline constexpr std::uint32_t kF32MantMask = 0x7FFFFFu;
inline constexpr std::uint32_t kF32Implicit = 0x800000u;
inline constexpr std::int32_t kF32ExpBias = 127;

// fp16 bit layout
inline constexpr std::uint32_t kF16SignShift = 15u;
inline constexpr std::uint32_t kF16ExpShift = 10u;
inline constexpr std::uint32_t kF16SignMask = 0x8000u;
inline constexpr std::uint32_t kF16NoSignMask = 0x7FFFu; // exp | mant
inline constexpr std::uint32_t kF16ExpMask = 0x1Fu;
inline constexpr std::uint32_t kF16ExpFieldMask = 0x7C00u; // = the Inf bits
inline constexpr std::uint32_t kF16MantMask = 0x3FFu;
inline constexpr std::uint32_t kF16Implicit = 0x400u;
inline constexpr std::uint32_t kF16InfBits = 0x7C00u;
inline constexpr std::int32_t kF16ExpBias = 15;
inline constexpr std::int32_t kF16ExpMin = -14;
inline constexpr std::int32_t kF16ExpMax = 15;
inline constexpr std::int32_t kF16UflowGuard = 10;
inline constexpr std::uint32_t kF32ToF16Shift = 13u;

inline constexpr std::uint32_t kF16ToF32SignShift =
    kF32SignShift - kF16SignShift;

} // namespace config

#endif
