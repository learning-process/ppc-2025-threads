#include "seq/sarafanov_m_CanonMatMul/include/CanonMatrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sarafanov_m_CanonMatMul_seq {
CanonMatrix::CanonMatrix(const std::vector<double>& initial_vector) : matrix(initial_vector) {
  CalculateSize(initial_vector.size());
}
void CanonMatrix::CalculateSize(size_t s) { size = static_cast<int>(std::sqrt(s)); }

void CanonMatrix::SetBaseMatrix(const std::vector<double>& initial_vector) {
  CalculateSize(initial_vector.size());
  matrix = initial_vector;
}

void CanonMatrix::StairShift() {
  std::vector<double> new_matrix(matrix.size());
  std::copy(matrix.begin(), matrix.begin() + size, new_matrix.begin());
  for (size_t i = 1; i < size; ++i) {
    std::copy(matrix.begin() + size * i + i, matrix.begin() + size * (i + 1), new_matrix.begin() + size * i);
    for (size_t j = size * i; j < size * i + i; ++j) {
      new_matrix[j + size - i] = matrix[j];
    }
  }
  matrix = std::move(new_matrix);
}

void CanonMatrix::FullShift() {
  std::vector<double> new_matrix(matrix.size());
  double value = 0.0;
  bool swap_flag = false;
  for (size_t i = 0; i < matrix.size(); ++i) {
    if (i % size == 0) {
      if (swap_flag == true) {
        new_matrix[i - 1] = value;
      }
      value = matrix[i];
      swap_flag = true;
    } else {
      new_matrix[i - 1] = matrix[i];
    }
  }
  matrix = std::move(new_matrix);
  matrix.back() = value;
}

void CanonMatrix::Shift() {
  if (shift_counts == 0) {
    StairShift();
  } else if (shift_counts < size) {
    FullShift();
  }
  shift_counts++;
}

const std::vector<double>& CanonMatrix::GetMatrix() const { return matrix; }

size_t CanonMatrix::GetSize() const { return size; }
CanonMatrix CanonMatrix::operator*(const CanonMatrix& canon_matrix) const {
  std::vector<double> c_matrix(size * size);
  auto b_matrix = canon_matrix.GetMatrix();
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      c_matrix[i * size + j] = matrix[i * size + j] * b_matrix[j * size + i];
    }
  }
  return {c_matrix};
}

CanonMatrix CanonMatrix::operator+(const CanonMatrix& canon_matrix) {
  std::vector<double> c_matrix(canon_matrix.GetSize() * canon_matrix.GetSize());
  if (this->GetMatrix().empty()) {
    SetBaseMatrix(c_matrix);
  }
  auto b_matrix = canon_matrix.GetMatrix();
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      c_matrix[i * size + j] = matrix[i * size + j] + b_matrix[j * size + i];
    }
  }
  return {c_matrix};
}

void CanonMatrix::Transpose() {
  std::vector<double> new_matrix(size * size);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      new_matrix[i * size + j] = matrix[j * size + i];
    }
  }
  matrix = std::move(new_matrix);
}

void CanonMatrix::ClearMatrix() {
  matrix.clear();
  size = 0;
  shift_counts = 0;
}
}  // namespace sarafanov_m_CanonMatMul_seq