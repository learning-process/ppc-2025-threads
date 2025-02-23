#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/vavilov_v_cannon/include/ops_seq.hpp"

TEST(vavilov_v_cannon_seq, test_fixed_4x4) {
  constexpr unsigned int N = 4;
  std::vector<double> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> B = {1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0};
  std::vector<double> expected_output = {4, 6, 6, 4, 12, 14, 14, 12, 20, 22, 22, 20, 28, 30, 30, 28};
  std::vector<double> C(N * N, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (unsigned int i = 0; i < N * N; i++) {
    EXPECT_EQ(expected_output[i], C[i]);
  }
}

TEST(vavilov_v_cannon_seq, test_225) {
  constexpr unsigned int N = 225;
  std::vector<double> A(N * N, 1.0);
  std::vector<double> B(N * N, 1.0);
  std::vector<double> C(N * N, 0.0);
  std::vector<double> expected_output(N * N, N);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (unsigned int i = 0; i < N * N; i++) {
    EXPECT_EQ(expected_output[i], C[i]);
  }
}

TEST(vavilov_v_cannon_seq, test_225_from_file) {
  std::string line;
  std::ifstream test_file(ppc::util::GetAbsolutePath("seq/vavilov_v_cannon/data/test.txt"));
  unsigned int N = 0;
  if (test_file.is_open()) {
    getline(test_file, line);
  }
  test_file.close();

  N = std::stoi(line);

  std::vector<double> A(N * N, 1.0);
  std::vector<double> B(N * N, 1.0);
  std::vector<double> C(N * N, 0.0);
  std::vector<double> expected_output(N * N, N);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (unsigned int i = 0; i < N * N; i++) {
    EXPECT_EQ(expected_output[i], C[i]);
  }
}

TEST(vavilov_v_cannon_seq, test_identity_matrix) {
  constexpr unsigned int N = 225;
  std::vector<double> A(N * N, 1.0);
  std::vector<double> I(N * N, 0.0);
  std::vector<double> C(N * N, 0.0);

  for (unsigned int i = 0; i < N; i++) {
    I[i * N + i] = 1.0;
  }

  auto expected_output = A;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(I.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(I.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (unsigned int i = 0; i < N * N; i++) {
    EXPECT_EQ(expected_output[i], C[i]);
  }
}

TEST(vavilov_v_cannon_seq, test_zero_matrix) {
  constexpr unsigned int N = 225;
  std::vector<double> A(N * N, 1.0);
  std::vector<double> B(N * N, 0.0);
  std::vector<double> C(N * N, 0.0);
  std::vector<double> expected_output(N * N, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data_seq->inputs_count.emplace_back(A.size());
  task_data_seq->inputs_count.emplace_back(B.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  task_data_seq->outputs_count.emplace_back(C.size());

  vavilov_v_cannon_seq::CannonSequential task_seq(task_data_seq);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  for (unsigned int i = 0; i < N * N; i++) {
    EXPECT_EQ(expected_output[i], C[i]);
  }
}
