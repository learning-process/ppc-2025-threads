#pragma once

#include <complex>
#include <iostream>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using Complex = std::complex<double>;

namespace kolodkin_g_multiplication_matrix_seq {

struct SparseMatrixCRS {
  std::vector<Complex> values;
  std::vector<int> colIndices;
  std::vector<int> rowPtr;
  int numRows;
  int numCols;
  SparseMatrixCRS() : numRows(0), numCols(0) {};
  SparseMatrixCRS(double rows, int cols) : numRows((int)rows), numCols(cols) { rowPtr.resize(rows + 1, 0); }

  void AddValue(int row, Complex value, int col) {
    for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j) {
      if (colIndices[j] == col) {
        values[j] += value;
        return;
      }
    }
    colIndices.emplace_back(col);
    values.emplace_back(value);
    for (int i = row + 1; i <= numRows; ++i) {
      rowPtr[i]++;
    }
  }
  SparseMatrixCRS(const SparseMatrixCRS& other) = default;
  SparseMatrixCRS& operator=(const SparseMatrixCRS& other) = default;
  static void PrintSparseMatrix(const SparseMatrixCRS& matrix) {
    for (int i = 0; i < matrix.numRows; ++i) {
      for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; ++j) {
        std::cout << "Element at (" << i << ", " << matrix.colIndices[j] << ") = " << matrix.values[j] << '\n';
      }
    }
  }
};
std::vector<Complex> ParseMatrixIntoVec(const SparseMatrixCRS& mat);
SparseMatrixCRS ParseVectorIntoMatrix(std::vector<Complex>& vec);
bool CheckMatrixesEquality(const SparseMatrixCRS& a, const SparseMatrixCRS& b);
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Complex> input_, output_;
  SparseMatrixCRS A_, B_;
};

}  // namespace kolodkin_g_multiplication_matrix_seq