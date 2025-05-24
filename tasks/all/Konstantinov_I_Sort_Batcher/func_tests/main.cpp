#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/Konstantinov_I_Sort_Batcher/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace mpi = boost::mpi;

TEST(Konstantinov_I_Sort_Batcher_all, invalid_input) {
  mpi::communicator world;
  std::vector<double> in{1.0};
  std::vector<double> out(1);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  if (world.rank() == 0) {
    EXPECT_EQ(test_task.ValidationImpl(), false);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, negative_values) {
  mpi::communicator world;
  std::vector<double> in{-3.14, -1.0, -104.5, -0.1, -990.90};
  std::vector<double> exp_out{-990.90, -104.5, -3.14, -1.0, -0.1};
  std::vector<double> out(5);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, positive_values) {
  std::vector<double> in{3.14, 1.0, 104.5, 0.1, 990.90};
  std::vector<double> exp_out{0.1, 1.0, 3.14, 104.5, 990.90};
  std::vector<double> out(5);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  mpi::communicator world;
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, mixed_values) {
  std::vector<double> in{0.0, -2.4, 3.4, -1.1, 2.2};
  std::vector<double> exp_out{-2.4, -1.1, 0.0, 2.2, 3.4};
  std::vector<double> out(5);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  mpi::communicator world;
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, duplicate_values) {
  std::vector<double> in{6.5, 1.2, 6.5, 3.3, 1.2};
  std::vector<double> exp_out{1.2, 1.2, 3.3, 6.5, 6.5};
  std::vector<double> out(5);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  mpi::communicator world;
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, sorted_input) {
  std::vector<double> in{-6.6, -3.3, 0.0, 4.4, 6.6};
  std::vector<double> exp_out{-6.6, -3.3, 0.0, 4.4, 6.6};
  std::vector<double> out(5);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  mpi::communicator world;
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, large_array) {
  constexpr size_t kSize = 100000;
  std::vector<double> in(kSize);
  std::vector<double> exp_out(kSize);

  for (size_t i = 0; i < kSize; ++i) {
    in[i] = static_cast<double>(kSize - i);
    exp_out[i] = static_cast<double>(i + 1);
  }

  std::vector<double> out(kSize);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  mpi::communicator world;
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
  ASSERT_EQ(test_task.ValidationImpl(), true);
  test_task.PreProcessingImpl();
  test_task.RunImpl();
  test_task.PostProcessingImpl();

  if (world.rank() == 0) {
    EXPECT_EQ(exp_out, out);
  }
}