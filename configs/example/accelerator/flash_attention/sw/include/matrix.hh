#ifndef MATRIX_HH
#define MATRIX_HH

#include <array>
#include <cstddef>

namespace kernel {

// Generic row-major 2D matrix on top of a contiguous std::array storage.
//
//   matrix[row]        -> lightweight row view (ConstRow on a const matrix),
//   matrix[row][col]   -> element reference,
//   matrix[row].data() -> pointer to the row start (for vector loads/stores).
//
template <typename T, std::size_t Rows, std::size_t Cols>
class Matrix : public std::array<T, Rows * Cols> {
public:
  using Base = std::array<T, Rows * Cols>;

  static constexpr std::size_t kRows = Rows;
  static constexpr std::size_t kCols = Cols;

  class Row {
  public:
    constexpr explicit Row(T *ptr) noexcept : ptr_(ptr) {}
    constexpr T &operator[](std::size_t col) const noexcept {
      return ptr_[col];
    }
    constexpr T *data() const noexcept { return ptr_; }
    constexpr T *begin() const noexcept { return ptr_; }
    constexpr T *end() const noexcept { return ptr_ + Cols; }
    constexpr std::size_t size() const noexcept { return Cols; }

  private:
    T *ptr_;
  };

  class ConstRow {
  public:
    constexpr explicit ConstRow(const T *ptr) noexcept : ptr_(ptr) {}
    constexpr const T &operator[](std::size_t col) const noexcept {
      return ptr_[col];
    }
    constexpr const T *data() const noexcept { return ptr_; }
    constexpr const T *begin() const noexcept { return ptr_; }
    constexpr const T *end() const noexcept { return ptr_ + Cols; }
    constexpr std::size_t size() const noexcept { return Cols; }

  private:
    const T *ptr_;
  };

  constexpr Matrix() noexcept = default;
  constexpr explicit Matrix(const Base &base) noexcept : Base(base) {}

  constexpr Row operator[](std::size_t row) noexcept {
    return Row(Base::data() + row * Cols);
  }
  constexpr ConstRow operator[](std::size_t row) const noexcept {
    return ConstRow(Base::data() + row * Cols);
  }
};

} // namespace kernel

#endif
