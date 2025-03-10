#include "seq/sarafanov_m_CanonMatMul/include/CanonMatrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "utility"

namespace sarafanov_m_canon_mat_mul_seq {
CanonMatrix::CanonMatrix(const std::vector<double>& initial_vector) : matrix_(initial_vector) {
  CalculateSize(initial_vector.size());
}
void CanonMatrix::CalculateSize(size_t s) { size_ = static_cast<int>(std::sqrt(s)); }

void CanonMatrix::SetBaseMatrix(const std::vector<double>& initial_vector) {
  if (matrix_.empty()) {
    CalculateSize(initial_vector.size());
    matrix_ = initial_vector;
  }
}

void CanonMatrix::StairShift() {
  std::vector<double> new_matrix(matrix_.size());
  std::copy(matrix_.begin(), matrix_.begin() + static_cast<int>(size_), new_matrix.begin());
  int s_size = static_cast<int>(size_);
  for (int i = 1; i < s_size; ++i) {
    std::copy(matrix_.begin() + s_size * i + i, matrix_.begin() + s_size * (i + 1), new_matrix.begin() + s_size * i);
    for (int j = s_size * i; j < s_size * i + i; ++j) {
      new_matrix[j + s_size - i] = matrix_[j];
    }
  }
  matrix_ = std::move(new_matrix);
}

void CanonMatrix::FullShift() {
  std::vector<double> new_matrix(matrix_.size());
  double value = 0.0;
  bool swap_flag = false;
  for (size_t i = 0; i < matrix_.size(); ++i) {
    if (i % size_ == 0) {
      if (swap_flag) {
        new_matrix[i - 1] = value;
      }
      value = matrix_[i];
      swap_flag = true;
    } else {
      new_matrix[i - 1] = matrix_[i];
    }
  }
  matrix_ = std::move(new_matrix);
  matrix_.back() = value;
}

void CanonMatrix::Shift() {
  if (shift_counts_ == 0) {
    StairShift();
  } else if (shift_counts_ < static_cast<int>(size_)) {
    FullShift();
  }
  shift_counts_++;
}

const std::vector<double>& CanonMatrix::GetMatrix() const { return matrix_; }

size_t CanonMatrix::GetSize() const { return size_; }
CanonMatrix CanonMatrix::operator*(const CanonMatrix& canon_matrix) const {
  std::vector<double> c_matrix(size_ * size_);
  const auto& b_matrix = canon_matrix.GetMatrix();
  for (size_t i = 0; i < size_; ++i) {
    for (size_t j = 0; j < size_; ++j) {
      c_matrix[(i * size_) + j] = matrix_[(i * size_) + j] * b_matrix[(j * size_) + i];
    }
  }
  return {c_matrix};
}

CanonMatrix CanonMatrix::operator+(const CanonMatrix& canon_matrix) {
  std::vector<double> c_matrix(canon_matrix.GetSize() * canon_matrix.GetSize());
  if (this->GetMatrix().empty()) {
    SetBaseMatrix(c_matrix);
  }
  const auto& b_matrix = canon_matrix.GetMatrix();
  for (size_t i = 0; i < size_; ++i) {
    for (size_t j = 0; j < size_; ++j) {
      c_matrix[(i * size_) + j] = matrix_[(i * size_) + j] + b_matrix[(j * size_) + i];
    }
  }
  return {c_matrix};
}

void CanonMatrix::Transpose() {
  std::vector<double> new_matrix(size_ * size_);
  for (size_t i = 0; i < size_; ++i) {
    for (size_t j = 0; j < size_; ++j) {
      new_matrix[(i * size_) + j] = matrix_[(j * size_) + i];
    }
  }
  matrix_ = std::move(new_matrix);
}

void CanonMatrix::ClearMatrix() {
  matrix_.clear();
  size_ = 0;
  shift_counts_ = 0;
}
}  // namespace sarafanov_m_canon_mat_mul_seq