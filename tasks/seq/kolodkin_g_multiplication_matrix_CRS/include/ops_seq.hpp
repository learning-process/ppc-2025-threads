#pragma once

#include <complex>
#include <iostream>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

typedef std::complex<double> Complex;

namespace kolodkin_g_multiplication_matrix_seq {

struct SparseMatrixCRS {
  std::vector<Complex> values;
  std::vector<int> colIndices;
  std::vector<int> rowPtr;
  int numRows;
  int numCols;
  SparseMatrixCRS() : numRows(0), numCols(0) {};
  SparseMatrixCRS(int rows, int cols) : numRows(rows), numCols(cols) { rowPtr.resize(rows + 1, 0); }

  void addValue(int row, int col, Complex value) {
    for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j) {
      if (colIndices[j] == col) {
        values[j] += value;
        return;
      }
    }
    colIndices.push_back(col);
    values.push_back(value);
    for (int i = row + 1; i <= numRows; ++i) {
      rowPtr[i]++;
    }
  }
  SparseMatrixCRS(const SparseMatrixCRS& other)
      : values(other.values),
        colIndices(other.colIndices),
        rowPtr(other.rowPtr),
        numRows(other.numRows),
        numCols(other.numCols) {};
  SparseMatrixCRS& operator=(const SparseMatrixCRS& A) {
    numCols = A.numCols;
    numRows = A.numRows;
    values = A.values;
    colIndices = A.colIndices;
    rowPtr = A.rowPtr;
    return *this;
  }
  void printSparseMatrix(const SparseMatrixCRS& matrix) {
    for (int i = 0; i < matrix.numRows; ++i) {
      for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; ++j) {
        std::cout << "Element at (" << i << ", " << matrix.colIndices[j] << ") = " << matrix.values[j] << std::endl;
      }
    }
  }
};
std::vector<Complex> parse_matrix_into_vec(const SparseMatrixCRS& mat);
SparseMatrixCRS parse_vector_into_matrix(std::vector<Complex>& vec);
bool check_matrixes_equality(const SparseMatrixCRS& A, const SparseMatrixCRS& B);
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Complex> input_, output_;
  SparseMatrixCRS A, B;
};

}  // namespace kolodkin_g_multiplication_matrix_seq