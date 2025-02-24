#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<double> GenerateRandomVector(size_t len, double min_val = -1000.0, double max_val = 1000.0) {
  std::vector<double> vect(len);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(min_val, max_val);

  for (size_t i = 0; i < len; ++i) {
    vect[i] = dis(gen);
  }

  return vect;
}

void CreateTest(size_t len) {
  std::vector<double> in = GenerateRandomVector(len);
  std::vector<double> out(len, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential hoare_sort_simple_merge_sequential(
      task_data_seq);
  ASSERT_TRUE(hoare_sort_simple_merge_sequential.Validation());
  ASSERT_TRUE(hoare_sort_simple_merge_sequential.PreProcessing());
  ASSERT_TRUE(hoare_sort_simple_merge_sequential.Run());
  ASSERT_TRUE(hoare_sort_simple_merge_sequential.PostProcessing());

  std::vector<double> ref(len);
  std::ranges::copy(in, ref.begin());
  std::ranges::sort(ref);

  for (size_t i = 0; i < len; i++) {
    EXPECT_EQ(out[i], ref[i]);
  }
}

}  // namespace

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_empty_vect) {
  std::vector<double> in = {};
  std::vector<double> out = {};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs_count.emplace_back(out.size());

  nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential hoare_sort_simple_merge_sequential(
      task_data_seq);
  ASSERT_FALSE(hoare_sort_simple_merge_sequential.Validation());
}

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_vect_len_1) { CreateTest(1); }

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_vect_len_2) { CreateTest(2); }

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_vect_len_10) { CreateTest(10); }

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_vect_len_100) { CreateTest(100); }

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_vect_len_150) { CreateTest(150); }

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_vect_len_200) { CreateTest(200); }

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_vect_len_1000) { CreateTest(1000); }

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_vect_len_2000) { CreateTest(2000); }

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_vect_len_5000) { CreateTest(5000); }

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_invalid_output) {
  std::vector<double> in = {1.0, 2.0, 3.0};
  std::vector<double> out = {};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential hoare_sort_simple_merge_sequential(
      task_data_seq);
  ASSERT_FALSE(hoare_sort_simple_merge_sequential.Validation());
}

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_input_and_output_sizes_not_equal) {
  std::vector<double> in = {1.0, 2.0, 3.0};
  std::vector<double> out = {0.0, 0.0};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential hoare_sort_simple_merge_sequential(
      task_data_seq);
  ASSERT_FALSE(hoare_sort_simple_merge_sequential.Validation());
}