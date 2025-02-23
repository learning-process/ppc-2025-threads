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
#include "seq/korotin_e_crs_multiplication/include/ops_seq.hpp"

namespace korotin_e_crs_multiplication_seq {

std::vector<double> GetRandomMatrix(unsigned int M, unsigned int N) {
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(-100.0, 100.0);
  std::vector<double> res(M * N);
  for (unsigned int i = 0; i < M; i++)
    for (unsigned int j = 0; j < N; j++) {
      res[i * N + j] = distrib(gen);
    }
  return res;
}

void MakeCRS(std::vector<unsigned int> &rI, std::vector<unsigned int> &col, std::vector<double> &val,
             std::vector<double> src, unsigned int M, unsigned int N) {
  rI = std::vector<unsigned int>(M + 1, 0);
  col.clear();
  val.clear();
  for (unsigned int i = 0; i < M; i++)
    for (unsigned int j = 0; j < N; j++) {
      if (src[i * N + j] != 0) {
        val.push_back(src[i * N + j]);
        col.push_back(j);
        rI[i + 1]++;
      }
    }
  for (unsigned int i = 0; i <= M; i++) {
    rI[i] += rI[i - 1];
  }
}

}  // namespace korotin_e_crs_multiplication_seq

TEST(korotin_e_crs_multiplication_seq, test_rnd_50_50_50) {
  const unsigned int M = 50, N = 50, P = 50;

  std::vector<double> A, B, A_val, B_val;
  std::vector<unsigned int> A_rI, A_col, B_rI, B_col;

  A = korotin_e_crs_multiplication_seq::GetRandimMatrix(M, N);
  B = korotin_e_crs_multiplication_seq::GetRandomMatrix(N, P);
  korotin_e_crs_multiplication_seq::MakeCRS(A_rI, A_col, A_val, A, M, N);
  korotin_e_crs_multiplication_seq::MakeCRS(B_rI, B_col, B_val, B, N, P);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_rI.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_val.data()));
  task_data_seq->inputs_count.emplace_back(A_rI.size());
  task_data_seq->inputs_count.emplace_back(A_col.size());
  task_data_seq->inputs_count.emplace_back(A_val.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_rI.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_val.data()));
  task_data_seq->inputs_count.emplace_back(B_rI.size());
  task_data_seq->inputs_count.emplace_back(B_col.size());
  task_data_seq->inputs_count.emplace_back(B_val.size());

  std::vector<unsigned int> out_rI(A_rI.size(), 0), out_col(M * P);
  std::vector<double> out_val(M * P);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_rI.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_col.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_val.data()));
  task_data_seq->outputs_count.emplace_back(out_rI.size());

  korotin_e_crs_multiplication_seq::CrsMultiplicationSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<double> C(M * P, 0), C_val;
  std::vector<unsigned int> C_rI, C_col;
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < P; j++) {
        C[i * P + j] += A[i * N + k] * B[k * P + j];
      }
    }
  }
  korotin_e_crs_multiplication_seq::MakeCRS(C_rI, C_col, C_val, C, M, P);

  EXPECT_EQ(C_rI, out_rI);
  EXPECT_EQ(C_col, out_col);
  EXPECT_EQ(C_val, out_val);
}
