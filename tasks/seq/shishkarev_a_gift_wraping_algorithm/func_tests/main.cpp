#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/shishkarev_a_gift_wraping_algorithm/include/ops_seq.hpp"

TEST(shishkarev_a_gift_wraping_algorithm_seq, test_convex_hull_simple) {
  constexpr size_t kCount = 10;

  std::vector<shishkarev_a_gift_wraping_algorithm_seq::Vertex> in(kCount);
  std::vector<shishkarev_a_gift_wraping_algorithm_seq::Vertex> out(kCount);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = {.x = rand() % 100, .y = rand() % 100};
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shishkarev_a_gift_wraping_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_FALSE(out.empty());
  EXPECT_LE(out.size(), in.size());
}

TEST(shishkarev_a_gift_wraping_algorithm_seq, test_convex_hull_medium) {
  constexpr size_t kCount = 1000;

  std::vector<shishkarev_a_gift_wraping_algorithm_seq::Vertex> in(kCount);
  std::vector<shishkarev_a_gift_wraping_algorithm_seq::Vertex> out(kCount);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = {.x = rand() % 100, .y = rand() % 100};
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shishkarev_a_gift_wraping_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_FALSE(out.empty());
  EXPECT_LE(out.size(), in.size());
}

TEST(shishkarev_a_gift_wraping_algorithm_seq, test_convex_hull_large) {
  constexpr size_t kCount = 100000;

  std::vector<shishkarev_a_gift_wraping_algorithm_seq::Vertex> in(kCount);
  std::vector<shishkarev_a_gift_wraping_algorithm_seq::Vertex> out(kCount);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = {.x = rand() % 100, .y = rand() % 100};
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shishkarev_a_gift_wraping_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_FALSE(out.empty());
  EXPECT_LE(out.size(), in.size());
}

TEST(shishkarev_a_gift_wraping_algorithm_seq, test_convex_hull_from_file) {
  std::ifstream test_file(ppc::util::GetAbsolutePath("seq/shishkarev_a_gift_wraping_algorithm/data/test.txt"));

  size_t count;
  test_file >> count;

  std::vector<shishkarev_a_gift_wraping_algorithm_seq::Vertex> in(count);
  std::vector<shishkarev_a_gift_wraping_algorithm_seq::Vertex> out(count);

  for (size_t i = 0; i < count; i++) {
    test_file >> in[i].x >> in[i].y;
  }
  test_file.close();

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shishkarev_a_gift_wraping_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_FALSE(out.empty());
  EXPECT_LE(out.size(), in.size());
}
