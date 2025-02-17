#include "seq/kolodkin_g_multiplication_matrix_CRS/include/ops_seq.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

std::vector<Complex> kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(const SparseMatrixCRS& mat) {
  std::vector<Complex> res(5 + mat.values.size() + mat.colIndices.size() + mat.rowPtr.size());
  res[0] = (double)mat.numRows;
  res[1] = (double)mat.numCols;
  res[2] = (double)mat.values.size();
  res[3] = (double)mat.colIndices.size();
  res[4] = (double)mat.rowPtr.size();
  for (unsigned int i = 0; i < (unsigned int)mat.values.size(); i++) {
    res[5 + i] = mat.values[i];
  }
  for (unsigned int i = 0; i < (unsigned int)mat.colIndices.size(); i++) {
    res[5 + mat.values.size() + i] = mat.colIndices[i];
  }
  for (unsigned int i = 0; i < (unsigned int)mat.rowPtr.size(); i++) {
    res[5 + mat.values.size() + mat.colIndices.size() + i] = mat.rowPtr[i];
  }
  return res;
}
bool kolodkin_g_multiplication_matrix_seq::CheckMatrixesEquality(
    const kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS& a,
    const kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS& b) {
  if (a.numCols != b.numCols || a.numRows != b.numRows) {
    return false;
  }
  for (unsigned int i = 0; i < (unsigned int)a.numRows; ++i) {
    unsigned int this_row_start = a.rowPtr[i];
    unsigned int this_row_end = a.rowPtr[i + 1];
    unsigned int other_row_start = b.rowPtr[i];
    unsigned int other_row_end = b.rowPtr[i + 1];

    if ((this_row_end - this_row_start) != (other_row_end - other_row_start)) {
      return false;
    }

    for (unsigned int j = this_row_start; j < this_row_end; ++j) {
      bool found = false;
      for (unsigned int k = other_row_start; k < other_row_end; ++k) {
        if (a.colIndices[j] == b.colIndices[k] && a.values[j] == b.values[k]) {
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
kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS kolodkin_g_multiplication_matrix_seq::ParseVectorIntoMatrix(
    std::vector<Complex>& vec) {
  SparseMatrixCRS res;
  res.numRows = (int)vec[0].real();
  res.numCols = (int)vec[1].real();
  auto values_size = (unsigned int)vec[2].real();
  auto col_indices_size = (unsigned int)vec[3].real();
  auto row_ptr_size = (unsigned int)vec[4].real();
  res.values.resize(values_size);
  res.colIndices.resize(col_indices_size);
  res.rowPtr.resize(row_ptr_size);
  for (unsigned int i = 0; i < values_size; i++) {
    res.values[i] = vec[5 + i];
  }
  for (unsigned int i = 0; i < col_indices_size; i++) {
    res.colIndices[i] = (int)vec[5 + values_size + i].real();
  }
  for (unsigned int i = 0; i < row_ptr_size; i++) {
    res.rowPtr[i] = (int)vec[5 + values_size + col_indices_size + i].real();
  }
  return res;
}

bool kolodkin_g_multiplication_matrix_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<Complex*>(task_data->inputs[0]);
  input_ = std::vector<Complex>(in_ptr, in_ptr + input_size);
  std::vector<Complex> matrix_a((unsigned int)(5 + input_[2].real() + input_[3].real() + input_[4].real()));
  std::vector<Complex> matrix_b((unsigned int)input_.size() -
                                (unsigned int)(5 + input_[2].real() + input_[3].real() + input_[4].real()));
  for (unsigned int i = 0; i < matrix_a.size(); i++) {
    matrix_a[i] = input_[i];
  }
  for (unsigned int i = 0; i < matrix_b.size(); i++) {
    matrix_b[i] = input_[(unsigned int)(5 + input_[2].real() + input_[3].real() + input_[4].real()) + i];
  }
  A_ = ParseVectorIntoMatrix(matrix_a);
  B_ = ParseVectorIntoMatrix(matrix_b);
  return true;
}

bool kolodkin_g_multiplication_matrix_seq::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<Complex*>(task_data->inputs[0]);
  std::vector<Complex> vec = std::vector<Complex>(in_ptr, in_ptr + input_size);
  return !(vec[1] != vec[5 + (int)(vec[2].real() + vec[3].real() + vec[4].real())].real());
}

bool kolodkin_g_multiplication_matrix_seq::TestTaskSequential::RunImpl() {
  SparseMatrixCRS c((NumRow)A_.numRows, (NumCol)B_.numCols);
  for (unsigned int i = 0; i < (unsigned int)A_.numRows; ++i) {
    for (unsigned int j = A_.rowPtr[i]; j < (unsigned int)A_.rowPtr[i + 1]; ++j) {
      unsigned int col_a = A_.colIndices[j];
      Complex value_a = A_.values[j];
      for (unsigned int k = B_.rowPtr[col_a]; k < (unsigned int)B_.rowPtr[col_a + 1]; ++k) {
        unsigned int col_b = B_.colIndices[k];
        Complex value_b = B_.values[k];

        c.AddValue((int)i, value_a * value_b, (int)col_b);
      }
    }
  }
  output_ = ParseMatrixIntoVec(c);
  return true;
}

bool kolodkin_g_multiplication_matrix_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<Complex*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}