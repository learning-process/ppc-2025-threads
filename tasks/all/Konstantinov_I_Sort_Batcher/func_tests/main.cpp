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

TEST(Konstantinov_I_Sort_Batcher_all, test_empty_array) {
  mpi::environment env;
  mpi::communicator world;

  std::vector<double> in, out;

  if (world.rank() == 0) {
    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());

    konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
    ASSERT_TRUE(test_task.ValidationImpl());
    test_task.PreProcessingImpl();
    test_task.RunImpl();
    test_task.PostProcessingImpl();

    EXPECT_TRUE(out.empty());
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, test_wrong_size) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<double> in(2, 0.0);
    std::vector<double> out(1);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());

    konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
    ASSERT_FALSE(test_task.ValidationImpl());
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, test_scalar) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<double> in{3.14};
    std::vector<double> exp_out{3.14};
    std::vector<double> out(1);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());

    konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
    ASSERT_TRUE(test_task.ValidationImpl());
    test_task.PreProcessingImpl();
    test_task.RunImpl();
    test_task.PostProcessingImpl();

    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, test_negative_values) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<double> in{-3.14, -1.0, -100.5, -0.1, -999.99};
    std::vector<double> exp_out{-999.99, -100.5, -3.14, -1.0, -0.1};
    std::vector<double> out(5);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());

    konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
    ASSERT_TRUE(test_task.ValidationImpl());
    test_task.PreProcessingImpl();
    test_task.RunImpl();
    test_task.PostProcessingImpl();

    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, test_already_sorted) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<double> in = {-100.0, -50.0, -1.0, 0.0, 1.0, 50.0, 100.0};
    std::vector<double> exp_out = in;
    std::vector<double> out(in.size());

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());

    konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
    ASSERT_TRUE(test_task.ValidationImpl());
    test_task.PreProcessingImpl();
    test_task.RunImpl();
    test_task.PostProcessingImpl();

    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, test_reverse_sorted) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<double> in = {100.0, 50.0, 1.0, 0.0, -1.0, -50.0, -100.0};
    std::vector<double> exp_out = in;
    std::ranges::sort(exp_out);
    std::vector<double> out(in.size());

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());

    konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
    ASSERT_TRUE(test_task.ValidationImpl());
    test_task.PreProcessingImpl();
    test_task.RunImpl();
    test_task.PostProcessingImpl();

    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, test_duplicate_values) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<double> in = {3.14, -1.0, 3.14, 0.0, -1.0, 42.0, 0.0};
    std::vector<double> exp_out = in;
    std::ranges::sort(exp_out);
    std::vector<double> out(in.size());

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());

    konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
    ASSERT_TRUE(test_task.ValidationImpl());
    test_task.PreProcessingImpl();
    test_task.RunImpl();
    test_task.PostProcessingImpl();

    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, test_random_100_values) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    constexpr size_t kCount = 100;
    std::vector<double> in(kCount);
    std::vector<double> out(kCount);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    for (auto& num : in) {
      num = dist(gen);
    }
    std::vector<double> exp_out = in;
    std::ranges::sort(exp_out);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());

    konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
    ASSERT_TRUE(test_task.ValidationImpl());
    test_task.PreProcessingImpl();
    test_task.RunImpl();
    test_task.PostProcessingImpl();

    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, test_alternating_sign_values) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    std::vector<double> in = {10.5, -9.3, 8.1, -7.7, 6.6, -5.5, 4.4, -3.3, 2.2, -1.1};
    std::vector<double> exp_out = in;
    std::ranges::sort(exp_out);
    std::vector<double> out(in.size());

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());

    konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
    ASSERT_TRUE(test_task.ValidationImpl());
    test_task.PreProcessingImpl();
    test_task.RunImpl();
    test_task.PostProcessingImpl();

    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, test_random_10000_values) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    constexpr size_t kCount = 11887;
    std::vector<double> in(kCount);
    std::vector<double> out(kCount);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10000.0, 10000.0);

    for (auto& num : in) {
      num = dist(gen);
    }
    std::vector<double> exp_out = in;
    std::ranges::sort(exp_out);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());

    konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
    ASSERT_TRUE(test_task.ValidationImpl());
    test_task.PreProcessingImpl();
    test_task.RunImpl();
    test_task.PostProcessingImpl();

    EXPECT_EQ(exp_out, out);
  }
}

TEST(Konstantinov_I_Sort_Batcher_all, test_random_huge_size) {
  mpi::environment env;
  mpi::communicator world;

  if (world.rank() == 0) {
    constexpr size_t kCount = 1000000;
    std::vector<double> in(kCount);
    std::vector<double> out(kCount);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10000.0, 10000.0);

    for (auto& num : in) {
      num = dist(gen);
    }
    std::vector<double> exp_out = in;
    std::ranges::sort(exp_out);

    auto task_data = std::make_shared<ppc::core::TaskData>();
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data->outputs_count.emplace_back(out.size());

    konstantinov_i_sort_batcher_all::RadixSortBatcherall test_task(task_data);
    ASSERT_TRUE(test_task.ValidationImpl());
    test_task.PreProcessingImpl();
    test_task.RunImpl();
    test_task.PostProcessingImpl();

    EXPECT_EQ(exp_out, out);
  }
}