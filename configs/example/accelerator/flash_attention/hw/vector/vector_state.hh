#ifndef VECTOR_STATE_HH
#define VECTOR_STATE_HH

#include "accelerator_config.hh"
#include "rvv_softmax.hh"

#include <array>
#include <cstddef>
#include <cstdint>

namespace kernel {

struct VectorState {
  StateVec m;
  StateVec l;
  StateVec c;
  Tile s_f16;
  Tile p_f16;
  std::array<float, accel::kBR * accel::kHeadDim> o_acc;
};

inline VectorState vector_state [[gnu::section(".sram_bss")]];

} // namespace kernel

#endif // VECTOR_STATE_HH
