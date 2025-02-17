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
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS a(3, 3);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS b(3, 3);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS c(3, 3);
  std::vector<Complex> in;
  std::vector<Complex> in_a;
  std::vector<Complex> in_b;
  std::vector<Complex> out(a.numCols * b.numRows * 100, 0);

  a.AddValue(0, 0, Complex(1, 0));
  a.AddValue(0, 2, Complex(2, 0));
  a.AddValue(1, 1, Complex(3, 0));
  a.AddValue(2, 0, Complex(4, 0));
  a.AddValue(2, 1, Complex(5, 0));

  b.AddValue(0, 1, Complex(6, 0));
  b.AddValue(1, 0, Complex(7, 0));
  b.AddValue(2, 2, Complex(8, 0));
  in_a = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(a);
  in_b = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(b);
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.emplace_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.emplace_back(in_b[i]);
  }
  c.AddValue(0, 1, Complex(6, 0));
  c.AddValue(0, 2, Complex(16, 0));
  c.AddValue(1, 0, Complex(21, 0));
  c.AddValue(2, 1, Complex(24, 0));
  c.AddValue(2, 0, Complex(35, 0));
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
      kolodkin_g_multiplication_matrix_seq::ParseVectorIntoMatrix(out);
  ASSERT_TRUE(kolodkin_g_multiplication_matrix_seq::CheckMatrixesEquality(res, c));
}

TEST(kolodkin_g_multiplication_seq, test_matmul_not_equal_rows_and_cols) {
  // Create data
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS a(3, 3);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS b(5, 3);
  std::vector<Complex> in;
  std::vector<Complex> in_a;
  std::vector<Complex> in_b;
  std::vector<Complex> out(a.numCols * b.numRows * 100, 0);

  a.AddValue(0, 0, Complex(1, 0));
  a.AddValue(0, 2, Complex(2, 0));
  a.AddValue(1, 1, Complex(3, 0));
  a.AddValue(2, 0, Complex(4, 0));
  a.AddValue(2, 1, Complex(5, 0));

  b.AddValue(0, 1, Complex(6, 0));
  b.AddValue(1, 0, Complex(7, 0));
  b.AddValue(2, 2, Complex(8, 0));
  in_a = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(a);
  in_b = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(b);
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.emplace_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.emplace_back(in_b[i]);
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
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS a(3, 3);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS b(3, 3);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS c(3, 3);
  std::vector<Complex> in;
  std::vector<Complex> in_a;
  std::vector<Complex> in_b;
  std::vector<Complex> out(a.numCols * b.numRows * 100, 0);

  a.AddValue(0, 0, Complex(1, 1));
  a.AddValue(0, 2, Complex(2, 2));
  a.AddValue(1, 1, Complex(3, 3));
  a.AddValue(2, 0, Complex(4, 4));
  a.AddValue(2, 1, Complex(5, 5));

  b.AddValue(0, 1, Complex(6, 6));
  b.AddValue(1, 0, Complex(7, 7));
  b.AddValue(2, 2, Complex(8, 8));
  in_a = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(a);
  in_b = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(b);
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.emplace_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.emplace_back(in_b[i]);
  }
  c.AddValue(0, 1, Complex(0, 12));
  c.AddValue(0, 2, Complex(0, 32));
  c.AddValue(1, 0, Complex(0, 42));
  c.AddValue(2, 1, Complex(0, 48));
  c.AddValue(2, 0, Complex(0, 70));
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
      kolodkin_g_multiplication_matrix_seq::ParseVectorIntoMatrix(out);
  ASSERT_TRUE(kolodkin_g_multiplication_matrix_seq::CheckMatrixesEquality(res, c));
}

TEST(kolodkin_g_multiplication_seq, test_matmul_rectangular_matrix) {
  // Create data
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS a(2, 3);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS b(3, 4);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS c(2, 4);
  std::vector<Complex> in;
  std::vector<Complex> in_a;
  std::vector<Complex> in_b;
  std::vector<Complex> out(a.numCols * b.numRows * 100, 0);

  a.AddValue(0, 1, Complex(1, 0));
  a.AddValue(0, 2, Complex(2, 0));
  a.AddValue(1, 1, Complex(3, 0));

  b.AddValue(0, 2, Complex(3, 0));
  b.AddValue(1, 0, Complex(5, 0));
  b.AddValue(1, 3, Complex(4, 0));
  b.AddValue(2, 0, Complex(7, 0));
  b.AddValue(2, 1, Complex(8, 0));
  in_a = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(a);
  in_b = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(b);
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.emplace_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.emplace_back(in_b[i]);
  }
  c.AddValue(0, 0, Complex(19, 0));
  c.AddValue(0, 3, Complex(4, 0));
  c.AddValue(0, 1, Complex(16, 0));
  c.AddValue(1, 0, Complex(15, 0));
  c.AddValue(1, 3, Complex(12, 0));
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
      kolodkin_g_multiplication_matrix_seq::ParseVectorIntoMatrix(out);
  ASSERT_TRUE(kolodkin_g_multiplication_matrix_seq::CheckMatrixesEquality(res, c));
}

TEST(kolodkin_g_multiplication_seq, test_matmul_with_negative_elems) {
  // Create data
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS a(2, 2);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS b(2, 2);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS c(2, 2);
  std::vector<Complex> in;
  std::vector<Complex> in_a;
  std::vector<Complex> in_b;
  std::vector<Complex> out(a.numCols * b.numRows * 100, 0);

  a.AddValue(0, 0, Complex(-1, -1));
  a.AddValue(1, 1, Complex(3, 3));

  b.AddValue(0, 1, Complex(6, 6));
  b.AddValue(1, 0, Complex(-7, -7));
  in_a = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(a);
  in_b = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(b);
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.emplace_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.emplace_back(in_b[i]);
  }
  c.AddValue(0, 1, Complex(0, -12));
  c.AddValue(1, 0, Complex(0, -42));
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
      kolodkin_g_multiplication_matrix_seq::ParseVectorIntoMatrix(out);
  ASSERT_TRUE(kolodkin_g_multiplication_matrix_seq::CheckMatrixesEquality(res, c));
}
