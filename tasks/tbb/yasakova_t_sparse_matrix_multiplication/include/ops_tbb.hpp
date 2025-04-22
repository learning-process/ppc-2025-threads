#include <complex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using ComplexNumber = std::complex<double>;

namespace yasakova_t_sparse_matrix_multiplication {

struct CompressedRowStorageMatrix {
  std::vector<ComplexNumber> nonZeroValues;
  std::vector<int> columnIndices;
  std::vector<int> rowPointers;
  int rowCount;
  int columnCount;
  CompressedRowStorageMatrix() : nonZeroValues({}), columnIndices({}), rowPointers({}), rowCount(0), columnCount(0) {};
  CompressedRowStorageMatrix(int rows, int cols) : rowCount(rows), columnCount(cols) {
    rowPointers.resize(rows + 1, 0);
  }

  void InsertElement(int row, ComplexNumber value, int col);
  CompressedRowStorageMatrix(const CompressedRowStorageMatrix& other) = default;
  CompressedRowStorageMatrix& operator=(const CompressedRowStorageMatrix& other) = default;
  static void DisplayMatrix(const CompressedRowStorageMatrix& matrix);
};
std::vector<ComplexNumber> ConvertMatrixToVector(const CompressedRowStorageMatrix& mat);
CompressedRowStorageMatrix ConvertVectorToMatrix(std::vector<ComplexNumber>& vec);
bool CompareMatrices(const CompressedRowStorageMatrix& first_matrix, const CompressedRowStorageMatrix& second_matrix);
bool AreEqualElems(const ComplexNumber& first_matrix, const ComplexNumber& second_matrix, double tolerance);
class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<ComplexNumber> inputData_, resultData_;
  CompressedRowStorageMatrix firstMatrix_, secondMatrix_;
};

}  // namespace yasakova_t_sparse_matrix_multiplication

yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix GenMatrix(
  unsigned int num_rows, unsigned int num_cols, unsigned int left_border_row, unsigned int right_border_row,
  unsigned int left_border_col, unsigned int right_border_col, int min_value, int max_value) {
if (left_border_row > right_border_row || left_border_col > right_border_col || right_border_row > num_rows ||
    right_border_col > num_cols || min_value > max_value) {
  throw("ERROR!");
}
yasakova_t_sparse_matrix_multiplication::CompressedRowStorageMatrix first_matrix((int)num_rows, (int)num_cols);
for (unsigned int i = left_border_row; i < right_border_row; i++) {
  for (unsigned int j = left_border_col; j < right_border_col; j++) {
    first_matrix.InsertElement(
        (int)i, ComplexNumber(min_value + (rand() % max_value), min_value + (rand() % max_value)), (int)j);
  }
}
return first_matrix;
}