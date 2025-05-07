#include "all/titov_s_ImageFilter_HorizGaussian3x3/include/ops_all.hpp"

#include <cmath>
#include <cstddef>
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

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::RunImpl() {
  const double sum = kernel_[0] + kernel_[1] + kernel_[2];
  const int width = width_;
  const int height = height_;
  const int world_size = world_.size();
  const int world_rank = world_.rank();

  const int rows_per_process = height / world_size;
  const int remainder = height % world_size;
  int start_row = world_rank * rows_per_process + std::min(world_rank, remainder);
  int end_row = start_row + rows_per_process + (world_rank < remainder ? 1 : 0);
  const int local_rows = end_row - start_row;

  std::vector<double> local_input(local_rows * width);
  std::vector<double> local_output(local_rows * width, 0.0);

  if (world_rank == 0) {
    std::copy(input_.begin() + start_row * width, input_.begin() + end_row * width, local_input.begin());
    for (int p = 1; p < world_size; p++) {
      int p_start = p * rows_per_process + std::min(p, remainder);
      int p_end = p_start + rows_per_process + (p < remainder ? 1 : 0);
      int p_size = (p_end - p_start) * width;
      world_.send(p, 0, input_.data() + p_start * width, p_size);
    }
  } else {
    world_.recv(0, 0, local_input.data(), local_input.size());
  }

  const int num_threads = ppc::util::GetPPCNumThreads();
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
          double left = (j > 0) ? local_input[row_offset + j - 1] : 0.0;
          double center = local_input[row_offset + j];
          double right = (j < width - 1) ? local_input[row_offset + j + 1] : 0.0;
          local_output[row_offset + j] = (left * kernel_[0] + center * kernel_[1] + right * kernel_[2]) / sum;
        }
      }
    });
  }

  for (auto &t : threads) t.join();

  if (world_rank == 0) {
    std::copy(local_output.begin(), local_output.end(), output_.begin() + start_row * width);
    for (int p = 1; p < world_size; p++) {
      int p_start = p * rows_per_process + std::min(p, remainder);
      int p_size = (rows_per_process + (p < remainder ? 1 : 0)) * width;
      world_.recv(p, 0, output_.data() + p_start * width, p_size);
    }
  } else {
    world_.send(0, 0, local_output.data(), local_output.size());
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
  std::copy(output_.begin(), output_.end(), out_ptr);
  return true;
}