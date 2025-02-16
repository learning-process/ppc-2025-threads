#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/kolodkin_g_multiplication_matrix_CRS/include/ops_seq.hpp"

TEST(kolodkin_g_multiplication_seq, test_matmul_only_real) {
  // Create data
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS A(3, 3);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS B(3, 3);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS C(3, 3);
  std::vector<Complex> in, in_a, in_b;
  std::vector<Complex> out(A.numCols * B.numRows * 100, 0);

  A.addValue(0, 0, Complex(1, 0));
  A.addValue(0, 2, Complex(2, 0));
  A.addValue(1, 1, Complex(3, 0));
  A.addValue(2, 0, Complex(4, 0));
  A.addValue(2, 1, Complex(5, 0));

  B.addValue(0, 1, Complex(6, 0));
  B.addValue(1, 0, Complex(7, 0));
  B.addValue(2, 2, Complex(8, 0));
  in_a = kolodkin_g_multiplication_matrix_seq::parse_matrix_into_vec(A);
  in_b = kolodkin_g_multiplication_matrix_seq::parse_matrix_into_vec(B);
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.push_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.push_back(in_b[i]);
  }
  C.addValue(0, 1, Complex(6, 0));
  C.addValue(0, 2, Complex(16, 0));
  C.addValue(1, 0, Complex(21, 0));
  C.addValue(2, 1, Complex(24, 0));
  C.addValue(2, 0, Complex(35, 0));
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kolodkin_g_multiplication_matrix_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS res =
      kolodkin_g_multiplication_matrix_seq::parse_vector_into_matrix(out);
  ASSERT_TRUE(kolodkin_g_multiplication_matrix_seq::check_matrixes_equality(res, C));
}

TEST(kolodkin_g_multiplication_seq, test_matmul_not_equal_rows_and_cols) {
  // Create data
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS A(3, 3);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS B(5, 3);
  std::vector<Complex> in, in_a, in_b;
  std::vector<Complex> out(A.numCols * B.numRows * 100, 0);

  A.addValue(0, 0, Complex(1, 0));
  A.addValue(0, 2, Complex(2, 0));
  A.addValue(1, 1, Complex(3, 0));
  A.addValue(2, 0, Complex(4, 0));
  A.addValue(2, 1, Complex(5, 0));

  B.addValue(0, 1, Complex(6, 0));
  B.addValue(1, 0, Complex(7, 0));
  B.addValue(2, 2, Complex(8, 0));
  in_a = kolodkin_g_multiplication_matrix_seq::parse_matrix_into_vec(A);
  in_b = kolodkin_g_multiplication_matrix_seq::parse_matrix_into_vec(B);
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.push_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.push_back(in_b[i]);
  }
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kolodkin_g_multiplication_matrix_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(kolodkin_g_multiplication_seq, test_matmul_with_imag) {
  // Create data
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS A(3, 3);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS B(3, 3);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS C(3, 3);
  std::vector<Complex> in, in_a, in_b;
  std::vector<Complex> out(A.numCols * B.numRows * 100, 0);

  A.addValue(0, 0, Complex(1, 1));
  A.addValue(0, 2, Complex(2, 2));
  A.addValue(1, 1, Complex(3, 3));
  A.addValue(2, 0, Complex(4, 4));
  A.addValue(2, 1, Complex(5, 5));

  B.addValue(0, 1, Complex(6, 6));
  B.addValue(1, 0, Complex(7, 7));
  B.addValue(2, 2, Complex(8, 8));
  in_a = kolodkin_g_multiplication_matrix_seq::parse_matrix_into_vec(A);
  in_b = kolodkin_g_multiplication_matrix_seq::parse_matrix_into_vec(B);
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.push_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.push_back(in_b[i]);
  }
  C.addValue(0, 1, Complex(0, 12));
  C.addValue(0, 2, Complex(0, 32));
  C.addValue(1, 0, Complex(0, 42));
  C.addValue(2, 1, Complex(0, 48));
  C.addValue(2, 0, Complex(0, 70));
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kolodkin_g_multiplication_matrix_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS res =
      kolodkin_g_multiplication_matrix_seq::parse_vector_into_matrix(out);
  ASSERT_TRUE(kolodkin_g_multiplication_matrix_seq::check_matrixes_equality(res, C));
}

TEST(kolodkin_g_multiplication_seq, test_matmul_rectangular_matrix) {
  // Create data
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS A(2, 3);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS B(3, 4);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS C(2, 4);
  std::vector<Complex> in, in_a, in_b;
  std::vector<Complex> out(A.numCols * B.numRows * 100, 0);

  A.addValue(0, 1, Complex(1, 0));
  A.addValue(0, 2, Complex(2, 0));
  A.addValue(1, 1, Complex(3, 0));

  B.addValue(0, 2, Complex(3, 0));
  B.addValue(1, 0, Complex(5, 0));
  B.addValue(1, 3, Complex(4, 0));
  B.addValue(2, 0, Complex(7, 0));
  B.addValue(2, 1, Complex(8, 0));
  in_a = kolodkin_g_multiplication_matrix_seq::parse_matrix_into_vec(A);
  in_b = kolodkin_g_multiplication_matrix_seq::parse_matrix_into_vec(B);
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.push_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.push_back(in_b[i]);
  }
  C.addValue(0, 0, Complex(19, 0));
  C.addValue(0, 3, Complex(4, 0));
  C.addValue(0, 1, Complex(16, 0));
  C.addValue(1, 0, Complex(15, 0));
  C.addValue(1, 3, Complex(12, 0));
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kolodkin_g_multiplication_matrix_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS res =
      kolodkin_g_multiplication_matrix_seq::parse_vector_into_matrix(out);
  ASSERT_TRUE(kolodkin_g_multiplication_matrix_seq::check_matrixes_equality(res, C));
}

TEST(kolodkin_g_multiplication_seq, test_matmul_with_negative_elems) {
  // Create data
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS A(2, 2);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS B(2, 2);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS C(2, 2);
  std::vector<Complex> in, in_a, in_b;
  std::vector<Complex> out(A.numCols * B.numRows * 100, 0);

  A.addValue(0, 0, Complex(-1, -1));
  A.addValue(1, 1, Complex(3, 3));

  B.addValue(0, 1, Complex(6, 6));
  B.addValue(1, 0, Complex(-7, -7));
  in_a = kolodkin_g_multiplication_matrix_seq::parse_matrix_into_vec(A);
  in_b = kolodkin_g_multiplication_matrix_seq::parse_matrix_into_vec(B);
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.push_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.push_back(in_b[i]);
  }
  C.addValue(0, 1, Complex(0, -12));
  C.addValue(1, 0, Complex(0, -42));
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  kolodkin_g_multiplication_matrix_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS res =
      kolodkin_g_multiplication_matrix_seq::parse_vector_into_matrix(out);
  ASSERT_TRUE(kolodkin_g_multiplication_matrix_seq::check_matrixes_equality(res, C));
}
