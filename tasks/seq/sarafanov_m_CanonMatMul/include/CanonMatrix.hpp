#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace sarafanov_m_CanonMatMul_seq {
class CanonMatrix {
  std::vector<double> matrix;
  size_t size = 0;
  int shift_counts = 0;

  void CalculateSize(size_t s);
  void FullShift();
  void StairShift();

 public:
  CanonMatrix() = default;
  CanonMatrix(const std::vector<double>& initial_vector);
  void SetBaseMatrix(const std::vector<double>& initial_vector);
  void Shift();
  void Transpose();
  const std::vector<double>& GetMatrix() const;
  size_t GetSize() const;
  CanonMatrix operator*(const CanonMatrix& canon_matrix) const;
  CanonMatrix operator+(const CanonMatrix& canon_matrix);
  void ClearMatrix();
};
}  // namespace sarafanov_m_CanonMatMul_seq