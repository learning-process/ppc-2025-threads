#include "seq/borisov_s_strassen_seq/include/ops_seq.hpp"

#include <cmath>
#include <iostream>

namespace borisov_s_strassen_seq {

namespace {

std::vector<double> MultiplyNaive(const std::vector<double> &a, const std::vector<double> &b, int n) {
  std::vector<double> c(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += a[(i * n) + k] * b[(k * n) + j];
      }
      c[(i * n) + j] = sum;
    }
  }
  return c;
}

std::vector<double> AddMatr(const std::vector<double> &a, const std::vector<double> &b, int n) {
  std::vector<double> c(n * n);
  for (int i = 0; i < n * n; ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

std::vector<double> SubMatr(const std::vector<double> &a, const std::vector<double> &b, int n) {
  std::vector<double> c(n * n);
  for (int i = 0; i < n * n; ++i) {
    c[i] = a[i] - b[i];
  }
  return c;
}

std::vector<double> SubMatrix(const std::vector<double> &m, int n, int row, int col, int size) {
  std::vector<double> sub(size * size);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      sub[(i * size) + j] = m[((row + i) * n) + (col + j)];
    }
  }
  return sub;
}

void SetSubMatrix(std::vector<double> &m, const std::vector<double> &sub, int n, int row, int col, int size) {
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      m[((row + i) * n) + (col + j)] = sub[(i * size) + j];
    }
  }
}

std::vector<double> StrassenRecursive(const std::vector<double> &A, const std::vector<double> &B, int n) {
  if (n <= 16) {
    return MultiplyNaive(A, B, n);
  }
  int k = n / 2;
  auto A11 = SubMatrix(A, n, 0, 0, k);
  auto A12 = SubMatrix(A, n, 0, k, k);
  auto A21 = SubMatrix(A, n, k, 0, k);
  auto A22 = SubMatrix(A, n, k, k, k);

  auto B11 = SubMatrix(B, n, 0, 0, k);
  auto B12 = SubMatrix(B, n, 0, k, k);
  auto B21 = SubMatrix(B, n, k, 0, k);
  auto B22 = SubMatrix(B, n, k, k, k);

  auto M1 = StrassenRecursive(AddMatr(A11, A22, k), AddMatr(B11, B22, k), k);
  auto M2 = StrassenRecursive(AddMatr(A21, A22, k), B11, k);
  auto M3 = StrassenRecursive(A11, SubMatr(B12, B22, k), k);
  auto M4 = StrassenRecursive(A22, SubMatr(B21, B11, k), k);
  auto M5 = StrassenRecursive(AddMatr(A11, A12, k), B22, k);
  auto M6 = StrassenRecursive(SubMatr(A21, A11, k), AddMatr(B11, B12, k), k);
  auto M7 = StrassenRecursive(SubMatr(A12, A22, k), AddMatr(B21, B22, k), k);

  std::vector<double> C(n * n, 0.0);

  auto C11 = AddMatr(SubMatr(AddMatr(M1, M4, k), M5, k), M7, k);
  auto C12 = AddMatr(M3, M5, k);
  auto C21 = AddMatr(M2, M4, k);
  auto C22 = AddMatr(AddMatr(SubMatr(M1, M2, k), M3, k), M6, k);

  SetSubMatrix(C, C11, n, 0, 0, k);
  SetSubMatrix(C, C12, n, 0, k, k);
  SetSubMatrix(C, C21, n, k, 0, k);
  SetSubMatrix(C, C22, n, k, k, k);

  return C;
}

int NextPowerOfTwo(int n) {
  int r = 1;
  while (r < n) {
    r <<= 1;
  }
  return r;
}

}  // namespace

bool SequentialStrassenSeq::PreProcessingImpl() {
  size_t input_count = task_data->inputs_count[0];
  auto *double_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_.assign(double_ptr, double_ptr + input_count);

  //  for (int i = 0; i < input_count; i++) {
  //    std::cout << input_[i] << " " << input_[2];
  //  }

  size_t output_count = task_data->outputs_count[0];
  output_.resize(output_count, 0.0);

  if (input_.size() < 4) {
    return false;
  }

  rowsA_ = static_cast<int>(input_[0]);
  colsA_ = static_cast<int>(input_[1]);
  rowsB_ = static_cast<int>(input_[2]);
  colsB_ = static_cast<int>(input_[3]);

  return true;
}

bool SequentialStrassenSeq::ValidationImpl() {
  if (colsA_ != rowsB_) {
    return false;
  }

  size_t needed = 4 + (static_cast<size_t>(rowsA_) * colsA_) + (static_cast<size_t>(rowsB_) * colsB_);

  if (input_.size() < needed) {
    return false;
  }
  return true;
}

bool SequentialStrassenSeq::RunImpl() {
  size_t offset = 4;
  std::vector<double> A(rowsA_ * colsA_);
  for (int i = 0; i < rowsA_ * colsA_; ++i) {
    A[i] = input_[offset + i];
  }
  offset += static_cast<size_t>(rowsA_ * colsA_);

  std::vector<double> B(rowsB_ * colsB_);
  for (int i = 0; i < rowsB_ * colsB_; ++i) {
    B[i] = input_[offset + i];
  }

  int maxDim = std::max({rowsA_, colsA_, colsB_});
  int M = NextPowerOfTwo(maxDim);

  std::vector<double> Aexp(M * M, 0.0);
  std::vector<double> Bexp(M * M, 0.0);

  for (int i = 0; i < rowsA_; ++i) {
    for (int j = 0; j < colsA_; ++j) {
      Aexp[(i * M) + j] = A[(i * colsA_) + j];
    }
  }
  for (int i = 0; i < rowsB_; ++i) {
    for (int j = 0; j < colsB_; ++j) {
      Bexp[(i * M) + j] = B[(i * colsB_) + j];
    }
  }

  auto Cexp = StrassenRecursive(Aexp, Bexp, M);

  std::vector<double> C(rowsA_ * colsB_, 0.0);
  for (int i = 0; i < rowsA_; ++i) {
    for (int j = 0; j < colsB_; ++j) {
      C[(i * colsB_) + j] = Cexp[(i * M) + j];
    }
  }

  output_[0] = static_cast<double>(rowsA_);
  output_[1] = static_cast<double>(colsB_);
  for (int i = 0; i < rowsA_ * colsB_; ++i) {
    output_[2 + i] = C[i];
  }

  return true;
}

bool SequentialStrassenSeq::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}

}  // namespace borisov_s_strassen_seq
