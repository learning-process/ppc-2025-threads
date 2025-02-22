#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "seq/borisov_s_strassen_seq/include/ops_seq.hpp"

namespace {

std::vector<double> MultiplyNaiveDouble(const std::vector<double>& A, const std::vector<double>& B, int rowsA,
                                        int colsA, int colsB) {
  std::vector<double> C(rowsA * colsB, 0.0);
  for (int i = 0; i < rowsA; ++i) {
    for (int j = 0; j < colsB; ++j) {
      double sum = 0.0;
      for (int k = 0; k < colsA; ++k) {
        sum += A[(i * colsA) + k] * B[(k * colsB) + j];
      }
      C[(i * colsB) + j] = sum;
    }
  }
  return C;
}

}  // namespace

TEST(borisov_s_strassen_seq, OneByOne) {
  std::vector<double> in_data = {1.0, 1.0, 1.0, 1.0, 7.5, 2.5};

  size_t output_count = 3;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());  // Число элементов (double)

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], 1.0);
  EXPECT_DOUBLE_EQ(out_ptr[1], 1.0);
  EXPECT_DOUBLE_EQ(out_ptr[2], 18.75);

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, TwoByTwo) {
  std::vector<double> A = {1.0, 2.5, 3.0, 4.0};
  std::vector<double> B = {1.5, 2.0, 0.5, 3.5};
  std::vector<double> C_expected = {2.75, 10.75, 6.5, 20.0};

  std::vector<double> in_data = {2.0, 2.0, 2.0, 2.0};
  in_data.insert(in_data.end(), A.begin(), A.end());
  in_data.insert(in_data.end(), B.begin(), B.end());

  size_t output_count = 2 + 4;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], 2.0);
  EXPECT_DOUBLE_EQ(out_ptr[1], 2.0);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(out_ptr[2 + i], C_expected[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Rectangular2x3_3x4) {
  std::vector<double> A = {1.0, 2.5, 3.0, 4.0, 5.5, 6.0};
  std::vector<double> B = {0.5, 1.0, 2.0, 1.5, 2.0, 0.5, 1.0, 3.0, 4.0, 2.5, 0.5, 1.0};

  auto C_expected = MultiplyNaiveDouble(A, B, 2, 3, 4);

  std::vector<double> in_data = {2.0, 3.0, 3.0, 4.0};
  in_data.insert(in_data.end(), A.begin(), A.end());
  in_data.insert(in_data.end(), B.begin(), B.end());

  size_t output_count = 2 + (2 * 4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], 2.0);
  EXPECT_DOUBLE_EQ(out_ptr[1], 4.0);

  std::vector<double> C_result(2 * 4);
  for (int i = 0; i < 2 * 4; ++i) {
    C_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(C_expected.size(), C_result.size());
  for (size_t i = 0; i < C_expected.size(); ++i) {
    EXPECT_NEAR(C_expected[i], C_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Square5x5_Random) {
  const int n = 5;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> dist(0.0, 10.0);

  std::vector<double> A(n * n);
  std::vector<double> B(n * n);
  for (int i = 0; i < n * n; ++i) {
    A[i] = dist(rng);
    B[i] = dist(rng);
  }

  auto C_expected = MultiplyNaiveDouble(A, B, n, n, n);

  std::vector<double> in_data = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                                 static_cast<double>(n)};
  in_data.insert(in_data.end(), A.begin(), A.end());
  in_data.insert(in_data.end(), B.begin(), B.end());

  size_t output_count = 2 + (n * n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], 5.0);
  EXPECT_DOUBLE_EQ(out_ptr[1], 5.0);

  std::vector<double> C_result(n * n);
  for (int i = 0; i < n * n; ++i) {
    C_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(C_expected.size(), C_result.size());
  for (size_t i = 0; i < C_expected.size(); ++i) {
    EXPECT_NEAR(C_expected[i], C_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Square20x20_Random) {
  const int n = 20;
  std::mt19937 rng(7777);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<double> A(n * n);
  std::vector<double> B(n * n);
  for (int i = 0; i < n * n; ++i) {
    A[i] = dist(rng);
    B[i] = dist(rng);
  }

  auto C_expected = MultiplyNaiveDouble(A, B, n, n, n);

  std::vector<double> in_data = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                                 static_cast<double>(n)};
  in_data.insert(in_data.end(), A.begin(), A.end());
  in_data.insert(in_data.end(), B.begin(), B.end());

  size_t output_count = 2 + (n * n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(n));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(n));

  std::vector<double> C_result(n * n);
  for (int i = 0; i < n * n; ++i) {
    C_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(C_expected.size(), C_result.size());
  for (size_t i = 0; i < C_expected.size(); ++i) {
    EXPECT_NEAR(C_expected[i], C_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Square32x32_Random) {
  const int n = 32;
  std::mt19937 rng(7777);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<double> A(n * n);
  std::vector<double> B(n * n);
  for (int i = 0; i < n * n; ++i) {
    A[i] = dist(rng);
    B[i] = dist(rng);
  }

  auto C_expected = MultiplyNaiveDouble(A, B, n, n, n);

  std::vector<double> in_data = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                                 static_cast<double>(n)};
  in_data.insert(in_data.end(), A.begin(), A.end());
  in_data.insert(in_data.end(), B.begin(), B.end());

  size_t output_count = 2 + (n * n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(n));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(n));

  std::vector<double> C_result(n * n);
  for (int i = 0; i < n * n; ++i) {
    C_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(C_expected.size(), C_result.size());
  for (size_t i = 0; i < C_expected.size(); ++i) {
    EXPECT_NEAR(C_expected[i], C_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Square128x128_Random) {
  const int n = 128;
  std::mt19937 rng(7777);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<double> A(n * n);
  std::vector<double> B(n * n);
  for (int i = 0; i < n * n; ++i) {
    A[i] = dist(rng);
    B[i] = dist(rng);
  }

  auto C_expected = MultiplyNaiveDouble(A, B, n, n, n);

  std::vector<double> in_data = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                                 static_cast<double>(n)};
  in_data.insert(in_data.end(), A.begin(), A.end());
  in_data.insert(in_data.end(), B.begin(), B.end());

  size_t output_count = 2 + (n * n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(n));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(n));

  std::vector<double> C_result(n * n);
  for (int i = 0; i < n * n; ++i) {
    C_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(C_expected.size(), C_result.size());
  for (size_t i = 0; i < C_expected.size(); ++i) {
    EXPECT_NEAR(C_expected[i], C_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Square129x129_Random) {
  const int n = 129;
  std::mt19937 rng(7777);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<double> A(n * n);
  std::vector<double> B(n * n);
  for (int i = 0; i < n * n; ++i) {
    A[i] = dist(rng);
    B[i] = dist(rng);
  }

  auto C_expected = MultiplyNaiveDouble(A, B, n, n, n);

  std::vector<double> in_data = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                                 static_cast<double>(n)};
  in_data.insert(in_data.end(), A.begin(), A.end());
  in_data.insert(in_data.end(), B.begin(), B.end());

  size_t output_count = 2 + (n * n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(n));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(n));

  std::vector<double> C_result(n * n);
  for (int i = 0; i < n * n; ++i) {
    C_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(C_expected.size(), C_result.size());
  for (size_t i = 0; i < C_expected.size(); ++i) {
    EXPECT_NEAR(C_expected[i], C_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Square240x240_Random) {
  const int n = 240;
  std::mt19937 rng(7777);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<double> A(n * n);
  std::vector<double> B(n * n);
  for (int i = 0; i < n * n; ++i) {
    A[i] = dist(rng);
    B[i] = dist(rng);
  }

  auto C_expected = MultiplyNaiveDouble(A, B, n, n, n);

  std::vector<double> in_data = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                                 static_cast<double>(n)};
  in_data.insert(in_data.end(), A.begin(), A.end());
  in_data.insert(in_data.end(), B.begin(), B.end());

  size_t output_count = 2 + (n * n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(n));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(n));

  std::vector<double> C_result(n * n);
  for (int i = 0; i < n * n; ++i) {
    C_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(C_expected.size(), C_result.size());
  for (size_t i = 0; i < C_expected.size(); ++i) {
    EXPECT_NEAR(C_expected[i], C_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, ValidCase) {
  std::vector<int> input_data = {2, 3, 3, 2};
  input_data.insert(input_data.end(), {1, 2, 3, 4, 5, 6});
  input_data.insert(input_data.end(), {7, 8, 9, 10, 11, 12});

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size());

  task_data->outputs.push_back(nullptr);
  task_data->outputs_count.push_back(0);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();

  EXPECT_TRUE(task.ValidationImpl());
}

TEST(borisov_s_strassen_seq, MismatchCase) {
  std::vector<double> input_data = {
      2.0,
      2.0,
      3.0,
      3.0,
  };
  input_data.insert(input_data.end(), {1.0, 2.0, 3.0, 4.0});
  input_data.insert(input_data.end(), {5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0});

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size());

  task_data->outputs.push_back(nullptr);
  task_data->outputs_count.push_back(0);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(borisov_s_strassen_seq, NotEnoughDataCase) {
  std::vector<double> input_data = {2.0, 2.0, 2.0, 2.0};
  input_data.insert(input_data.end(), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size());

  task_data->outputs.push_back(nullptr);
  task_data->outputs_count.push_back(0);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();

  EXPECT_FALSE(task.ValidationImpl());
}

TEST(borisov_s_strassen_seq, Rectangular16x17_Random) {
  const int rowsA = 32;
  const int colsA = 34;
  const int colsB = 35;

  std::mt19937 rng(7777);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<double> A(rowsA * colsA);
  std::vector<double> B(colsA * colsB);
  for (double& x : A) {
    x = dist(rng);
  }
  for (double& x : B) {
    x = dist(rng);
  }

  auto C_expected = MultiplyNaiveDouble(A, B, rowsA, colsA, colsB);

  std::vector<double> in_data = {static_cast<double>(rowsA), static_cast<double>(colsA), static_cast<double>(colsA),
                                 static_cast<double>(colsB)};
  in_data.insert(in_data.end(), A.begin(), A.end());
  in_data.insert(in_data.end(), B.begin(), B.end());

  size_t output_count = 2 + (rowsA * colsB);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(rowsA));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(colsB));

  std::vector<double> C_result(rowsA * colsB);
  for (int i = 0; i < rowsA * colsB; ++i) {
    C_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(C_expected.size(), C_result.size());
  for (size_t i = 0; i < C_expected.size(); ++i) {
    EXPECT_NEAR(C_expected[i], C_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Rectangular19x23_Random) {
  const int rowsA = 19;
  const int colsA = 23;
  const int colsB = 21;

  std::mt19937 rng(777);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<double> A(rowsA * colsA);
  std::vector<double> B(colsA * colsB);
  for (double& x : A) {
    x = dist(rng);
  }
  for (double& x : B) {
    x = dist(rng);
  }

  auto C_expected = MultiplyNaiveDouble(A, B, rowsA, colsA, colsB);

  std::vector<double> in_data = {static_cast<double>(rowsA), static_cast<double>(colsA), static_cast<double>(colsA),
                                 static_cast<double>(colsB)};
  in_data.insert(in_data.end(), A.begin(), A.end());
  in_data.insert(in_data.end(), B.begin(), B.end());

  size_t output_count = 2 + (rowsA * colsB);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(rowsA));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(colsB));

  std::vector<double> C_result(rowsA * colsB);
  for (int i = 0; i < rowsA * colsB; ++i) {
    C_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(C_expected.size(), C_result.size());
  for (size_t i = 0; i < C_expected.size(); ++i) {
    EXPECT_NEAR(C_expected[i], C_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Rectangular32x64_Random) {
  const int rowsA = 32;
  const int colsA = 64;
  const int colsB = 32;

  std::mt19937 rng(7777);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<double> A(rowsA * colsA);
  std::vector<double> B(colsA * colsB);
  for (double& x : A) {
    x = dist(rng);
  }
  for (double& x : B) {
    x = dist(rng);
  }

  auto C_expected = MultiplyNaiveDouble(A, B, rowsA, colsA, colsB);

  std::vector<double> in_data = {static_cast<double>(rowsA), static_cast<double>(colsA), static_cast<double>(colsA),
                                 static_cast<double>(colsB)};
  in_data.insert(in_data.end(), A.begin(), A.end());
  in_data.insert(in_data.end(), B.begin(), B.end());

  size_t output_count = 2 + (rowsA * colsB);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(rowsA));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(colsB));

  std::vector<double> C_result(rowsA * colsB);
  for (int i = 0; i < rowsA * colsB; ++i) {
    C_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(C_expected.size(), C_result.size());
  for (size_t i = 0; i < C_expected.size(); ++i) {
    EXPECT_NEAR(C_expected[i], C_result[i], 1e-9);
  }

  delete[] out_ptr;
}
