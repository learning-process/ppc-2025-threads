#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "seq/solovev_a_ccs_mmult_sparse/include/ccs_mmult_sparse.hpp"

TEST(solovev_a_ccs_mmult_sparse, test_I) {
  solovev_a_matrix::MatrixInCCS_Sparse M1(1, 1, 1);
  solovev_a_matrix::MatrixInCCS_Sparse M2(1, 1, 1);
  solovev_a_matrix::MatrixInCCS_Sparse M3;

  M1.col_p = {0, 1};
  M1.row = {0};
  M1.val = {std::complex<double>(0.0, 1.0)};

  M2.col_p = {0, 1};
  M2.row = {0};
  M2.val = {std::complex<double>(0.0, -1.0)};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&M3));

  solovev_a_matrix::Seq_MatMultCCS test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  std::complex<double> correct_reply(1.0, 0.0);
  ASSERT_NEAR(std::abs(M3.val[0] - correct_reply), 0.0, 1e-6);
}

TEST(solovev_a_ccs_mmult_sparse, test_II) {
  std::complex<double> vvector(1.0, 1.0);
  solovev_a_matrix::MatrixInCCS_Sparse M1(50, 1, 50);
  solovev_a_matrix::MatrixInCCS_Sparse M2(1, 50, 50);
  solovev_a_matrix::MatrixInCCS_Sparse M3(50, 50, 2500);

  M1.col_p = {0, 50};

  for (int i = 0; i <= 50; i++) {
    M2.col_p.push_back(i);
  }
  for (int i = 0; i < 50; i++) {
    M1.row.push_back((double)i);
    M1.val.emplace_back(vvector);
    M2.row.push_back(0.0);
    M2.val.emplace_back(vvector);
  };

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&M3));

  solovev_a_matrix::Seq_MatMultCCS test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  std::complex<double> correct_reply(0.0, 0.0);
  for (int i = 0; i < 50 * 50; i++) {
    ASSERT_EQ(M3.val[i], correct_reply);
  }
}

TEST(solovev_a_ccs_mmult_sparse, test_III) {
  std::complex<double> vvector(2.0, 1.0);
  solovev_a_matrix::MatrixInCCS_Sparse M1(50, 50);
  solovev_a_matrix::MatrixInCCS_Sparse M2(50, 1);
  solovev_a_matrix::MatrixInCCS_Sparse M3(50, 1);

  int l = 1;
  int m = 0;

  for (int i = 0; i <= 50; i++) {
    M1.col_p.push_back(m);
    m += l;
    l++;
  }

  l = 1;
  m = 0;
  for (int i = 0; i < M1.col_p[50]; i++) {
    M1.val.emplace_back(vvector);
    if (m >= l) {
      m = 0;
      l++;
    }
    M1.row.push_back(m);
    m++;
  }

  M2.col_p = {0, 50};
  for (int i = 0; i < 50; i++) {
    M2.val.emplace_back(vvector);
    M2.row.push_back(i);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&M3));

  solovev_a_matrix::Seq_MatMultCCS test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  l = 50;
  for (size_t i = 0; i < M3.val.size(); i++) {
    ASSERT_EQ(M3.val[i], std::complex<double>(3.0 * l, 4.0 * l));
    l--;
  }
}

TEST(solovev_a_ccs_mmult_sparse, test_IV) {
  std::complex<double> vvector_one(2.0, 1.0);
  std::complex<double> vvector_two(3.0, 4.0);
  solovev_a_matrix::MatrixInCCS_Sparse M1(5, 5);
  solovev_a_matrix::MatrixInCCS_Sparse M2(5, 5);
  solovev_a_matrix::MatrixInCCS_Sparse M3(5, 5);

  M1.col_p = {0, 0, 1, 1, 1, 1};
  M2.col_p = {0, 0, 1, 1, 1, 1};

  M1.val = {vvector_one};
  M1.row = {1};
  M2.val = {vvector_two};
  M2.row = {1};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&M3));

  solovev_a_matrix::Seq_MatMultCCS test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  std::complex<double> correct_reply(2.0, 11.0);

  ASSERT_EQ(M3.val[0], correct_reply);
}

TEST(solovev_a_ccs_mmult_sparse, test_V) {
  std::complex<double> vvector(2.0, 1.0);
  solovev_a_matrix::MatrixInCCS_Sparse M1(50, 50);
  solovev_a_matrix::MatrixInCCS_Sparse M2(50, 50);
  solovev_a_matrix::MatrixInCCS_Sparse M3(50, 50);

  for (int i = 0; i <= 50; i++) {
    M1.col_p.push_back(i);
    M2.col_p.push_back(i);
  }

  for (int i = 0; i < 50; i++) {
    M1.row.push_back(i);
    M1.val.emplace_back(vvector);
    M2.row.push_back(i);
    M2.val.emplace_back(vvector);
  };

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&M2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&M3));

  solovev_a_matrix::Seq_MatMultCCS test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  std::complex<double> correct_reply(3.0, 4.0);
  for (size_t i = 0; i < M3.val.size(); i++) {
    ASSERT_EQ(M3.val[i], correct_reply);
  }
}
