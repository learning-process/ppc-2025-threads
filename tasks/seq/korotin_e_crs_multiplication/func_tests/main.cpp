#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/korotin_e_crs_multiplication/include/ops_seq.hpp"

namespace korotin_e_crs_multiplication_seq {

std::vector<double> GetRandomMatrix(unsigned int M, unsigned int N) {
  std::random_device rd;
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
             const std::vector<double> &src, unsigned int M, unsigned int N) {
  std::cout << "Entered\n";
  rI = std::vector<unsigned int>(M + 1, 0);
  col.clear();
  val.clear();
  std::cout << "Initialized\n";
  for (unsigned int i = 0; i < M; i++)
    for (unsigned int j = 0; j < N; j++) {
      if (src[i * N + j] != 0) {
        val.push_back(src[i * N + j]);
        col.push_back(j);
        rI[i + 1]++;
      }
    }
  std::cout << "hmm\n";
  for (unsigned int i = 1; i <= M; i++) {
    rI[i] += rI[i - 1];
  }
}

}  // namespace korotin_e_crs_multiplication_seq

TEST(korotin_e_crs_multiplication_seq, test_rnd_50_50_50) {
  std::cout << "Hello" << std::endl;
  const unsigned int M = 50, N = 50, P = 50;
  std::cout << "I'm here" << std::endl;
  std::vector<double> A, B, A_val, B_val;
  std::vector<unsigned int> A_rI, A_col, B_rI, B_col;
  std::cout << "Pupupu" << std::endl;
  A = korotin_e_crs_multiplication_seq::GetRandomMatrix(M, N);
  B = korotin_e_crs_multiplication_seq::GetRandomMatrix(N, P);
  std::cout << "A, B created" << std::endl;
  A_rI = std::vector<unsigned int>(M + 1, 0);
  std::cout << "Initialized" << std::endl;
  for (unsigned int i = 0; i < M; i++)
    for (unsigned int j = 0; j < N; j++) {
      if (A[i * N + j] != 0) {
        A_val.push_back(A[i * N + j]);
        A_col.push_back(j);
        A_rI[i + 1]++;
      }
    }
  std::cout << "hmm" << std::endl;
  for (unsigned int i = 1; i <= M; i++) {
    A_rI[i] += A_rI[i - 1];
  }
  B_rI = std::vector<unsigned int>(N + 1, 0);
  std::cout << "Initialized" << std::endl;
  for (unsigned int i = 0; i < N; i++)
    for (unsigned int j = 0; j < P; j++) {
      if (B[i * P + j] != 0) {
        B_val.push_back(B[i * P + j]);
        B_col.push_back(j);
        B_rI[i + 1]++;
      }
    }
  std::cout << "hmm" << std::endl;
  for (unsigned int i = 1; i <= N; i++) {
    B_rI[i] += B_rI[i - 1];
  }
  std::cout << "What?" << std::endl;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_rI.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_col.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A_val.data()));
  task_data_seq->inputs_count.emplace_back(A_rI.size());
  task_data_seq->inputs_count.emplace_back(A_col.size());
  task_data_seq->inputs_count.emplace_back(A_val.size());

  std::cout << "Inputed A" << std::endl;

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_rI.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_col.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B_val.data()));
  task_data_seq->inputs_count.emplace_back(B_rI.size());
  task_data_seq->inputs_count.emplace_back(B_col.size());
  task_data_seq->inputs_count.emplace_back(B_val.size());

  std::cout << "Inputed B" << std::endl;

  std::vector<unsigned int> out_rI(A_rI.size(), 0), out_col(M * P);
  std::vector<double> out_val(M * P);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_rI.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_col.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_val.data()));
  task_data_seq->outputs_count.emplace_back(out_rI.size());

  std::cout << "Outed" << std::endl;

  korotin_e_crs_multiplication_seq::CrsMultiplicationSequential test_task_sequential(task_data_seq);
  std::cout << "Start" << std::endl;
  ASSERT_EQ(test_task_sequential.Validation(), true);
  std::cout << "Valid" << std::endl;
  test_task_sequential.PreProcessing();
  std::cout << "PreProc" << std::endl;
  test_task_sequential.Run();
  std::cout << "Run" << std::endl;
  test_task_sequential.PostProcessing();
  std::cout << "PostProc" << std::endl;

  std::vector<double> C(M * P, 0), C_val;
  std::vector<unsigned int> C_rI, C_col;
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int k = 0; k < N; k++) {
      for (unsigned int j = 0; j < P; j++) {
        C[i * P + j] += A[i * N + k] * B[k * P + j];
      }
    }
  }
  std::cout << "Done C" << std::endl;
  korotin_e_crs_multiplication_seq::MakeCRS(C_rI, C_col, C_val, C, M, P);
  std::cout << "Done CRS C" << std::endl;
  EXPECT_EQ(C_rI, out_rI);
  EXPECT_EQ(C_col, out_col);
  EXPECT_EQ(C_val, out_val);
}
