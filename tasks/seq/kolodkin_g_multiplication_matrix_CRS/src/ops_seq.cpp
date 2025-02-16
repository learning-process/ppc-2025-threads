#include "seq/kolodkin_g_multiplication_matrix_CRS/include/ops_seq.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <vector>

std::vector<Complex> kolodkin_g_multiplication_matrix_seq::parse_matrix_into_vec(const SparseMatrixCRS& mat) {
  std::vector<Complex> res;
  res.push_back(Complex(mat.numRows, 0));
  res.push_back(Complex(mat.numCols, 0));
  res.push_back(Complex(mat.values.size(), 0));
  res.push_back(Complex(mat.colIndices.size(), 0));
  res.push_back(Complex(mat.rowPtr.size(), 0));
  for (unsigned int i = 0; i < (unsigned int)mat.values.size(); i++) {
    res.push_back(mat.values[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)mat.colIndices.size(); i++) {
    res.push_back(Complex(mat.colIndices[i], 0));
  }
  for (unsigned int i = 0; i < (unsigned int)mat.rowPtr.size(); i++) {
    res.push_back(Complex(mat.rowPtr[i], 0));
  }
  return res;
}
bool kolodkin_g_multiplication_matrix_seq::check_matrixes_equality(
    const kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS& A,
    const kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS& B) {
  if (A.numCols != B.numCols || A.numRows != B.numRows) {
    return false;
  }
  for (unsigned int i = 0; i < (unsigned int)A.numRows; ++i) {
    unsigned int thisRowStart = A.rowPtr[i];
    unsigned int thisRowEnd = A.rowPtr[i + 1];
    unsigned int otherRowStart = B.rowPtr[i];
    unsigned int otherRowEnd = B.rowPtr[i + 1];

    if ((thisRowEnd - thisRowStart) != (otherRowEnd - otherRowStart)) {
      return false;
    }

    for (unsigned int j = thisRowStart; j < thisRowEnd; ++j) {
      bool found = false;
      for (unsigned int k = otherRowStart; k < otherRowEnd; ++k) {
        if (A.colIndices[j] == B.colIndices[k] && A.values[j] == B.values[k]) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
  }
  return true;
}
kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS kolodkin_g_multiplication_matrix_seq::parse_vector_into_matrix(
    std::vector<Complex>& vec) {
  SparseMatrixCRS res;
  res.numRows = vec[0].real();
  res.numCols = vec[1].real();
  unsigned int values_size = vec[2].real();
  unsigned int colIndices_size = vec[3].real();
  unsigned int rowPtr_size = vec[4].real();
  for (unsigned int i = 0; i < (unsigned int)values_size; i++) {
    res.values.push_back(vec[5 + i]);
  }
  for (unsigned int i = 0; i < (unsigned int)colIndices_size; i++) {
    res.colIndices.push_back(vec[5 + values_size + i].real());
  }
  for (unsigned int i = 0; i < (unsigned int)rowPtr_size; i++) {
    res.rowPtr.push_back(vec[5 + values_size + colIndices_size + i].real());
  }
  return res;
}

bool kolodkin_g_multiplication_matrix_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<Complex*>(task_data->inputs[0]);
  input_ = std::vector<Complex>(in_ptr, in_ptr + input_size);
  std::vector<Complex> matrix_a, matrix_b;
  for (unsigned int i = 0; i < (unsigned int)(5 + input_[2].real() + input_[3].real() + input_[4].real()); i++) {
    matrix_a.push_back(input_[i]);
  }
  for (unsigned int i = 5 + input_[2].real() + input_[3].real() + input_[4].real(); i < (unsigned int)input_.size();
       i++) {
    matrix_b.push_back(input_[i]);
  }
  A = parse_vector_into_matrix(matrix_a);
  B = parse_vector_into_matrix(matrix_b);
  return true;
}

bool kolodkin_g_multiplication_matrix_seq::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<Complex*>(task_data->inputs[0]);
  std::vector<Complex> vec = std::vector<Complex>(in_ptr, in_ptr + input_size);
  if (vec[1] != vec[5 + vec[2].real() + vec[3].real() + vec[4].real()]) {
    return false;
  }
  return true;
}

bool kolodkin_g_multiplication_matrix_seq::TestTaskSequential::RunImpl() {
  SparseMatrixCRS C(A.numRows, B.numCols);
  for (unsigned int i = 0; i < (unsigned int)A.numRows; ++i) {
    for (unsigned int j = A.rowPtr[i]; j < (unsigned int)A.rowPtr[i + 1]; ++j) {
      unsigned int colA = A.colIndices[j];
      Complex valueA = A.values[j];
      for (unsigned int k = B.rowPtr[colA]; k < (unsigned int)B.rowPtr[colA + 1]; ++k) {
        unsigned int colB = B.colIndices[k];
        Complex valueB = B.values[k];

        C.addValue(i, colB, valueA * valueB);
      }
    }
  }
  output_ = parse_matrix_into_vec(C);
  return true;
}

bool kolodkin_g_multiplication_matrix_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<Complex*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
