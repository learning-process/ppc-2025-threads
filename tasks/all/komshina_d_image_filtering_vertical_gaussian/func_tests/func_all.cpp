#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/komshina_d_image_filtering_vertical_gaussian/include/ops_all.hpp"
#include "core/task/include/task.hpp"

TEST(komshina_d_image_filtering_vertical_gaussian_all, EmptyImage) {
  std::size_t width = 0;
  std::size_t height = 0;
  std::vector<unsigned char> in = {};
  std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<unsigned char> expected = {};
  std::vector<unsigned char> out(expected.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, ZeroWidthImage) {
  std::size_t width = 0;
  std::size_t height = 3;
  std::vector<unsigned char> in = {255, 255, 255};
  std::vector<float> kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<unsigned char> expected = {};
  std::vector<unsigned char> out(expected.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, ValidationInvalidKernelSize) {
  std::size_t width = 3;
  std::size_t height = 3;
  std::vector<unsigned char> in = {255, 255, 255};
  std::vector<float> kernel = {1, 1};
  std::vector<unsigned char> out(9);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, ValidationInvalidOutputSize) {
  std::size_t width = 3;
  std::size_t height = 3;
  std::vector<unsigned char> in = {255, 255, 255};
  std::vector<float> kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<unsigned char> out(5);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, RandomImage) {
  std::size_t width = 5;
  std::size_t height = 5;
  std::vector<unsigned char> in(width * height * 3);
  std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<unsigned char> out(in.size());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = dis(gen);
  }

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);

  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_GE(out[i], 0);
    EXPECT_LE(out[i], 255);
  }
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, RandomImage2) {
  std::size_t width = 17;
  std::size_t height = 23;
  std::vector<unsigned char> in(width * height * 3);
  std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<unsigned char> out(in.size());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = dis(gen);
  }

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);

  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_GE(out[i], 0);
    EXPECT_LE(out[i], 255);
  }
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, EdgeHandling) {
  std::size_t width = 1;
  std::size_t height = 5;
  std::vector<unsigned char> in = {10, 10, 10, 50, 50, 50, 100, 100, 100, 50, 50, 50, 10, 10, 10};
  std::vector<float> kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<unsigned char> out(in.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (auto pixel : out) {
    EXPECT_GE(pixel, 0);
    EXPECT_LE(pixel, 255);
  }
}

TEST(komshina_d_image_filtering_vertical_gaussian_all, MpiReceiveChunksIntoOutput) {
  int rank_int = 0;
  int size_int = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_int);
  MPI_Comm_size(MPI_COMM_WORLD, &size_int);

  std::size_t rank = static_cast<std::size_t>(rank_int);
  std::size_t size = static_cast<std::size_t>(size_int);

  if (size < 2) {
    return;
  }

  std::size_t width = 2;
  std::size_t height = 4 * size;
  std::size_t chunk_rows = height / size;
  std::size_t chunk_size = chunk_rows * width * 3;

  std::vector<unsigned char> in(chunk_size, static_cast<unsigned char>(rank * 50));
  std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<unsigned char> out(width * height * 3, 0);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_all::TestTaskALL test_task(task_data);

  if (test_task.Validation()) {
    test_task.PreProcessing();
    test_task.Run();
    test_task.PostProcessing();

    if (rank == 0) {
      for (std::size_t proc = 0; proc < size; ++proc) {
        std::size_t start = proc * chunk_size;
        std::size_t end = start + chunk_size;
        for (std::size_t i = start; i < end; ++i) {
          EXPECT_EQ(out[i], static_cast<unsigned char>(proc * 50));
        }
      }
    }
  }
}