#include "all/titov_s_ImageFilter_HorizGaussian3x3/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  width_ = height_ = static_cast<int>(std::sqrt(input_size));
  input_.assign(in_ptr, in_ptr + input_size);

  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  output_ = std::vector<double>(input_size, 0.0);

  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::ValidationImpl() {
  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  size_t size = input_.size();
  auto sqrt_size = static_cast<size_t>(std::sqrt(size));
  if (kernel_.size() != 3 || sqrt_size * sqrt_size != size) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::DistributeData(int world_rank, int world_size,
                                                                                   int height, int width, int start_row,
                                                                                   int end_row,
                                                                                   std::vector<double> &local_input) {
  if (world_rank == 0) {
    auto src_range = std::ranges::subrange(input_.begin() + start_row * width, input_.begin() + end_row * width);
    std::ranges::copy(src_range, local_input.begin());

    for (int p = 1; p < world_size; ++p) {
      const int p_start = (p * (height / world_size)) + std::min(p, height % world_size);
      const int p_end = p_start + (height / world_size) + ((p < (height % world_size)) ? 1 : 0);
      const int p_size = (p_end - p_start) * width;

      world_.send(p, 0, input_.data() + (p_start * width), p_size);
    }
  } else {
    world_.recv(0, 0, local_input.data(), static_cast<int>(local_input.size()));
  }
  return true;
}

void titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::ProcessRows(const std::vector<double> &local_input,
                                                                                std::vector<double> &local_output,
                                                                                int width, int local_rows,
                                                                                int num_threads) {
  const double sum = kernel_[0] + kernel_[1] + kernel_[2];
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  const int rows_per_thread = (local_rows + num_threads - 1) / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    const int thread_start = t * rows_per_thread;
    const int thread_end = std::min(thread_start + rows_per_thread, local_rows);

    threads.emplace_back([=, this, &local_input, &local_output] {
      for (int i = thread_start; i < thread_end; ++i) {
        const int row_offset = i * width;
        for (int j = 0; j < width; ++j) {
          const double left = (j > 0) ? local_input[row_offset + j - 1] : 0.0;
          const double center = local_input[row_offset + j];
          const double right = (j < (width - 1)) ? local_input[row_offset + j + 1] : 0.0;

          local_output[row_offset + j] = (left * kernel_[0] + center * kernel_[1] + right * kernel_[2]) / sum;
        }
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }
}

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::CollectResults(
    int world_rank, int world_size, int height, int width, int start_row, const std::vector<double> &local_output) {
  if (world_rank == 0) {
    std::ranges::copy(local_output, output_.begin() + start_row * width);

    for (int p = 1; p < world_size; ++p) {
      const int p_start = p * (height / world_size) + std::min(p, height % world_size);
      const int p_end = p_start + (height / world_size) + (p < (height % world_size) ? 1 : 0);
      const int p_size = (p_end - p_start) * width;

      world_.recv(p, 0, output_.data() + p_start * width, p_size);
    }
  } else {
    world_.send(0, 0, local_output.data(), static_cast<int>(local_output.size()));
  }
  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::RunImpl() {
  const int width = width_;
  const int height = height_;
  const int world_size = world_.size();
  const int world_rank = world_.rank();
  const int num_threads = ppc::util::GetPPCNumThreads();

  const int rows_per_process = height / world_size;
  const int remainder = height % world_size;

  const int extra_row = (world_rank < remainder) ? 1 : 0;
  const int start_row = world_rank * rows_per_process + std::min(world_rank, remainder);
  const int end_row = start_row + rows_per_process + extra_row;

  const int local_rows = end_row - start_row;

  std::vector<double> local_input(local_rows * width);
  std::vector<double> local_output(local_rows * width, 0.0);

  if (!DistributeData(world_rank, world_size, height, width, start_row, end_row, local_input)) {
    return false;
  }

  ProcessRows(local_input, local_output, width, local_rows, num_threads);

  if (!CollectResults(world_rank, world_size, height, width, start_row, local_output)) {
    return false;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(output_, out_ptr);
  return true;
}