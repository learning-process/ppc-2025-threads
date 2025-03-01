#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/petrov_a_radix_double_batcher/include/ops_seq.hpp"

namespace {
std::vector<double> RandomVector(size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dist(-5000, 8000);
  std::vector<double> vec(size);
  std::ranges::generate(vec, [&dist, &gen] { return dist(gen); });
  return vec;
}
}  // namespace

TEST(petrov_a_radix_double_batcher_seq, test_0) {
  auto in = RandomVector(0);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(petrov_a_radix_double_batcher_seq, test_1) {
  auto in = RandomVector(1);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(petrov_a_radix_double_batcher_seq, test_2) {
  auto in = RandomVector(2);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(petrov_a_radix_double_batcher_seq, test_3) {
  auto in = RandomVector(3);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(petrov_a_radix_double_batcher_seq, test_4) {
  auto in = RandomVector(4);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(petrov_a_radix_double_batcher_seq, test_5) {
  auto in = RandomVector(5);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(petrov_a_radix_double_batcher_seq, test_6) {
  auto in = RandomVector(6);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(petrov_a_radix_double_batcher_seq, test_7) {
  auto in = RandomVector(7);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(petrov_a_radix_double_batcher_seq, test_8) {
  auto in = RandomVector(8);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(petrov_a_radix_double_batcher_seq, test_9) {
  auto in = RandomVector(9);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(petrov_a_radix_double_batcher_seq, test_10) {
  auto in = RandomVector(10);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(petrov_a_radix_double_batcher_seq, test_11) {
  auto in = RandomVector(11);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(petrov_a_radix_double_batcher_seq, test_111) {
  auto in = RandomVector(111);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

TEST(petrov_a_radix_double_batcher_seq, test_213) {
  auto in = RandomVector(213);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}