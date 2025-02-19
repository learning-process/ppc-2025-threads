#include "seq/gnitienko_k_strassen_alg/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool gnitienko_k_strassen_algorithm::StrassenAlgSeq::PreProcessingImpl() {
  int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_1 = std::vector<double>(in_ptr, in_ptr + input_size);

  in_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  input_2 = std::vector<double>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0.0);
  
  size_ = static_cast<int>(std::sqrt(input_size));

  if (!((input_size > 0) && ((input_size & (input_size - 1)) == 0))) {
    int new_size = std::pow(2, std::ceil(std::log2(size_)));
    std::vector<double> extended_input_1(new_size * new_size, 0.0);
    std::vector<double> extended_input_2(new_size * new_size, 0.0);
    output_.resize(new_size * new_size);
    extend = new_size - size_;

    for (int i = 0; i < size_; ++i) {
      for (int j = 0; j < size_; ++j) {
        extended_input_1[i * new_size + j] = input_1[i * size_ + j];
        extended_input_2[i * new_size + j] = input_2[i * size_ + j];
      }
    }

    input_1 = std::move(extended_input_1);
    input_2 = std::move(extended_input_2);
    
    size_ = new_size;
  }
  return true;
}

bool gnitienko_k_strassen_algorithm::StrassenAlgSeq::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

void gnitienko_k_strassen_algorithm::StrassenAlgSeq::AddMatrix(const std::vector<double>& A,
                                                               const std::vector<double>& B, std::vector<double>& C,
                                                               int size) {
  for (int i = 0; i < size * size; ++i) {
    C[i] = A[i] + B[i];
  }
}

void gnitienko_k_strassen_algorithm::StrassenAlgSeq::SubMatrix(const std::vector<double>& A,
                                                                   const std::vector<double>& B, std::vector<double>& C,
                                                                   int size) {
  for (int i = 0; i < size * size; ++i) {
    C[i] = A[i] - B[i];
  }
}

void gnitienko_k_strassen_algorithm::StrassenAlgSeq::TrivialMultiply(const std::vector<double>& A,
                                                                     const std::vector<double>& B,
                                                                     std::vector<double>& C, int size) {
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      C[i * size + j] = 0;
      for (int k = 0; k < size; ++k) {
        C[i * size + j] += A[i * size + k] * B[k * size + j];
      }
    }
  }
}

void gnitienko_k_strassen_algorithm::StrassenAlgSeq::StrassenMultiply(const std::vector<double>& A,
                                                                      const std::vector<double>& B,
                                                                      std::vector<double>& C, int size) {
  if (size <= TRIVIAL_MULTIPLICATION_BOUND) {
    TrivialMultiply(A, B, C, size);
    return;
  }

  int half_size = size / 2;

  std::vector<double> A11(half_size * half_size), A12(half_size * half_size), A21(half_size * half_size),
      A22(half_size * half_size);
  std::vector<double> B11(half_size * half_size), B12(half_size * half_size), B21(half_size * half_size),
      B22(half_size * half_size);

  for (int i = 0; i < half_size; ++i) {
    for (int j = 0; j < half_size; ++j) {
      A11[i * half_size + j] = A[i * size + j];
      A12[i * half_size + j] = A[i * size + j + half_size];
      A21[i * half_size + j] = A[(i + half_size) * size + j];
      A22[i * half_size + j] = A[(i + half_size) * size + j + half_size];
    }
  }

  for (int i = 0; i < half_size; ++i) {
    for (int j = 0; j < half_size; ++j) {
      B11[i * half_size + j] = B[i * size + j];
      B12[i * half_size + j] = B[i * size + j + half_size];
      B21[i * half_size + j] = B[(i + half_size) * size + j];
      B22[i * half_size + j] = B[(i + half_size) * size + j + half_size];
    }
  }

  std::vector<double> D(half_size * half_size), D1(half_size * half_size), D2(half_size * half_size),
      H1(half_size * half_size), H2(half_size * half_size), V1(half_size * half_size), V2(half_size * half_size);

  // D = (A11 + A22) * (B11 + B22)
  std::vector<double> tempA(half_size * half_size), tempB(half_size * half_size);
  AddMatrix(A11, A22, tempA, half_size);
  AddMatrix(B11, B22, tempB, half_size);
  StrassenMultiply(tempA, tempB, D, half_size);

  // D1 = (A12 - A22) * (B21 + B22)
  SubMatrix(A12, A22, tempA, half_size);
  AddMatrix(B21, B22, tempB, half_size);
  StrassenMultiply(tempA, tempB, D1, half_size);

  // D2 = (A21 - A11) * (B11 + B12)
  SubMatrix(A21, A11, tempA, half_size);
  AddMatrix(B11, B12, tempB, half_size);
  StrassenMultiply(tempA, tempB, D2, half_size);

  // H1 = (A11 + A12) * B22
  AddMatrix(A11, A12, tempA, half_size);
  StrassenMultiply(tempA, B22, H1, half_size);

  // H2 = (A21 + A22) * B11
  AddMatrix(A21, A22, tempA, half_size);
  StrassenMultiply(tempA, B11, H2, half_size);

  // V1 = A22 * (B21 - B11)
  SubMatrix(B21, B11, tempB, half_size);
  StrassenMultiply(A22, tempB, V1, half_size);

  // V2 = A11 * (B12 - B22)
  SubMatrix(B12, B22, tempB, half_size);
  StrassenMultiply(A11, tempB, V2, half_size);

  std::vector<double> C11(half_size * half_size), C12(half_size * half_size), C21(half_size * half_size),
      C22(half_size * half_size);

  AddMatrix(D, D1, C11, half_size);
  AddMatrix(C11, V1, C11, half_size);
  SubMatrix(C11, H1, C11, half_size);
  AddMatrix(V2, H1, C12, half_size);
  AddMatrix(V1, H2, C21, half_size);
  AddMatrix(D, D2, C22, half_size);
  AddMatrix(C22, V2, C22, half_size);
  SubMatrix(C22, H2, C22, half_size);

  for (int i = 0; i < half_size; ++i) {
    for (int j = 0; j < half_size; ++j) {
      C[i * size + j] = C11[i * half_size + j];
      C[i * size + j + half_size] = C12[i * half_size + j];
      C[(i + half_size) * size + j] = C21[i * half_size + j];
      C[(i + half_size) * size + j + half_size] = C22[i * half_size + j];
    }
  }
}

bool gnitienko_k_strassen_algorithm::StrassenAlgSeq::RunImpl() {
  StrassenMultiply(input_1, input_2, output_, size_);
  if (extend != 0) {
    int original_size = size_ - extend;
    std::vector<double> res_(original_size * original_size);

    for (int i = 0; i < original_size; ++i) {
      for (int j = 0; j < original_size; ++j) {
        res_[i * original_size + j] = output_[i * size_ + j];
      }
    }

    output_ = std::move(res_);
  }
  return true;
}

bool gnitienko_k_strassen_algorithm::StrassenAlgSeq::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = round(output_[i] * 10000) / 10000;
  }
  return true;
}
