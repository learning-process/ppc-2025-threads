#pragma once

#include <cstddef>
#include <vector>

namespace sarafanov_m_canon_mat_mul_seq {
class CanonMatrix {
  std::vector<double> matrix_;
  size_t size_ = 0;
  int shift_counts_ = 0;

  void CalculateSize(size_t s);
  void FullShift();
  void StairShift();

 public:
  CanonMatrix() = default;
  CanonMatrix(const std::vector<double>& initial_vector);
  void SetBaseMatrix(const std::vector<double>& initial_vector);
  void Shift();
  void Transpose();
  [[nodiscard]] const std::vector<double>& GetMatrix() const;
  [[nodiscard]] size_t GetSize() const;
  CanonMatrix operator*(const CanonMatrix& canon_matrix) const;
  CanonMatrix operator+(const CanonMatrix& canon_matrix);
  void ClearMatrix();
};
}  // namespace sarafanov_m_canon_mat_mul_seq