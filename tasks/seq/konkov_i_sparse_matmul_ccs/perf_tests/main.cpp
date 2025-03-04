#include <gtest/gtest.h>

#include "seq/konkov_i_sparse_matmul_ccs/include/ops_seq.hpp"

TEST(konkov_i_SparseMatmulPerfTest, test_pipeline_run) {
  // ���� �� ������������������ � ����������� ����������
  // ������ � ������� ������ ��������� ��� ��������� ������� ����������
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  // ������ ����������� ������
  std::vector<double> A_values = {1.0, 2.0};  // �������� A
  std::vector<int> A_columns = {0, 1};        // ������� ��� A

  std::vector<double> B_values = {3.0, 4.0};  // �������� B
  std::vector<int> B_columns = {0, 1};        // ������� ��� B

  task.A_values = A_values;
  task.A_columns = A_columns;
  task.B_values = B_values;
  task.B_columns = B_columns;
  task.rowsA = 3;
  task.colsA = 2;
  task.rowsB = 2;
  task.colsB = 2;

  // ����� ������� ����������
  auto start = std::chrono::high_resolution_clock::now();
  task.ValidationImpl();
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "Pipeline run duration: " << duration.count() << " seconds\n";
}

TEST(konkov_i_SparseMatmulPerfTest, test_task_run) {
  // ���� �� ������������������ � �������������� ������
  // ������ � �������� ��������� ��� ��������� ������� ����������
  ppc::core::TaskDataPtr task_data = std::make_shared<ppc::core::TaskData>();
  konkov_i_sparse_matmul_ccs::SparseMatmulTask task(task_data);

  // ������ ����������� ������
  std::vector<double> A_values = {1.0, 2.0, 3.0};  // �������� A
  std::vector<int> A_columns = {0, 1, 2};          // ������� ��� A

  std::vector<double> B_values = {4.0, 5.0, 6.0};  // �������� B
  std::vector<int> B_columns = {0, 1, 2};          // ������� ��� B

  task.A_values = A_values;
  task.A_columns = A_columns;
  task.B_values = B_values;
  task.B_columns = B_columns;
  task.rowsA = 3;
  task.colsA = 3;
  task.rowsB = 3;
  task.colsB = 3;

  // ����� ������� ����������
  auto start = std::chrono::high_resolution_clock::now();
  task.ValidationImpl();
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "Task run duration: " << duration.count() << " seconds\n";
}
