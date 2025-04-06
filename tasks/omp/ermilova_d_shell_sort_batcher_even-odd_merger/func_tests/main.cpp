#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/ermilova_d_shell_sort_batcher_even-odd_merger/include/ops_omp.hpp"

namespace {
std::vector<int> GenerateRandomVector(size_t size, int lower_bound = -1000, int upper_bound = 1000) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = static_cast<int>(lower_bound + (gen() % (upper_bound - lower_bound + 1)));
  }
  return vec;
}
}  // namespace

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_create_empty_input) {
  // Create data
  std::vector<int> in = {};
  std::vector<int> out(in);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  ASSERT_FALSE(sut.Validation());
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_create_input_and_output_with_different_size) {
  // Create data
  std::vector<int> in = GenerateRandomVector(5);
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  ASSERT_FALSE(sut.Validation());
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_single_element) {
  // Create data
  std::vector<int> in = GenerateRandomVector(1);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_small_even_size) {
  // Create data
  std::vector<int> in = {3, 1, 4, 2};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_small_odd_size) {
  // Create data
  std::vector<int> in = {5, 2, 3};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_descending_sorted) {
  // Create data
  std::vector<int> in = {1, 2, 3, 4, 5};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_ascending_sorted) {
  // Create data
  std::vector<int> in = {1, 2, 3, 4, 5};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_all_equal_elements) {
  // Create data
  std::vector<int> in = {7, 7, 7, 7, 7};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_duplicates_elements) {
  // Create data
  std::vector<int> in = {7, 7, 7, 7, 7};
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_100_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(100);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_1000_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(1000);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_347_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(347);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_with_boundary_sedgwick_gap_109) {
  // Create data
  std::vector<int> in = GenerateRandomVector(109);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_128_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(128);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_27_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(27);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_809_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(809);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_500_random_elements) {
  // Create data
  std::vector<int> in = GenerateRandomVector(500);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_with_boundary_sedgwick_gap_729) {
  // Create data
  std::vector<int> in = GenerateRandomVector(729);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_omp, test_sort_with_boundary_sedgwick_gap_457) {
  // Create data
  std::vector<int> in = GenerateRandomVector(457);
  std::vector<int> out(in.size(), 0);

  std::vector<int> expected = in;
  std::ranges::sort(expected);

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  // Create Task
  ermilova_d_shell_sort_batcher_even_odd_merger_omp::OmpTask sut(task_data);
  sut.Validation();
  sut.PreProcessing();
  sut.Run();
  sut.PostProcessing();
  ASSERT_EQ(expected, out);
}
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, vec_negative_values) {
//   // Create data
//
//   bool is_resersed = false;
//
//   std::vector<int> in = {-578, -23546, -1231, -6, -18247, -789, -2348, -3, -213980, -123345};
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, vec_repeating_value) {
//   // Create data
//
//   bool is_resersed = false;
//
//   std::vector<int> in = {10, 10, 8, 9399, 10, 10, 546, 2387, 3728};
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, sorted_vec) {
//   // Create data
//
//   bool is_resersed = false;
//
//   std::vector<int> in = {13, 56, 90, 123, 345, 567, 1000, 1230, 12340};
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, reverse_sorted_vec) {
//   // Create data
//
//   bool is_resersed = true;
//
//   std::vector<int> in = {12340, 9999, 5678, 800, 567, 340, 230, 10, 8, 3, 1};
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref, std::greater<>());
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, vec_1) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 1;
//
//   bool is_resersed = false;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, vec_10) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 10;
//
//   bool is_resersed = false;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, vec_100) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 100;
//
//   bool is_resersed = false;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, vec_1000) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 1000;
//
//   bool is_resersed = false;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, vec_10000) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 10000;
//
//   bool is_resersed = false;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, vec_8) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 8;
//
//   bool is_resersed = false;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, vec_128) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 128;
//
//   bool is_resersed = false;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, vec_27) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 27;
//
//   bool is_resersed = false;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, vec_729) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 729;
//
//   bool is_resersed = false;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, vec_457) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 457;
//
//   bool is_resersed = false;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, vec_809) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 809;
//
//   bool is_resersed = false;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, reverse_sort_vec_500) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 500;
//
//   bool is_resersed = true;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref, std::greater<>());
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, reverse_sort_vec_347) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 809;
//
//   bool is_resersed = true;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref, std::greater<>());
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, reverse_sort_vec_1000) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 1000;
//
//   bool is_resersed = true;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref, std::greater<>());
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, reverse_sort_vec_10000) {
//   // Create data
//   const int upper_border_test = 1000;
//   const int lower_border_test = -1000;
//   const int size = 10000;
//
//   bool is_resersed = true;
//
//   std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref, std::greater<>());
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, reverse_duplicates_vec) {
//   // Create data
//
//   bool is_resersed = false;
//
//   std::vector<int> in = {5, 4, 3, 2, 1, 5, 4, 3, 2, 1};
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, reverse_duplicates_vec_reverse_sort) {
//   // Create data
//
//   bool is_resersed = true;
//
//   std::vector<int> in = {5, 4, 3, 2, 1, 5, 4, 3, 2, 1};
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref, std::greater<>());
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, duplicates_vec) {
//   // Create data
//
//   bool is_resersed = false;
//
//   std::vector<int> in = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref);
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }
//
// TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, duplicates_vec_reverse_sort) {
//   // Create data
//
//   bool is_resersed = true;
//
//   std::vector<int> in = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
//   std::vector<int> out(in.size(), 0);
//
//   std::vector<int> ref = in;
//   std::ranges::sort(ref, std::greater<>());
//
//   // Create task_data
//   auto task_data_seq = std::make_shared<ppc::core::TaskData>();
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
//   task_data_seq->inputs_count.emplace_back(in.size());
//   task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   task_data_seq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential test_task_sequential(task_data_seq);
//   ASSERT_EQ(test_task_sequential.Validation(), true);
//   test_task_sequential.PreProcessing();
//   test_task_sequential.Run();
//   test_task_sequential.PostProcessing();
//   EXPECT_EQ(ref, out);
// }