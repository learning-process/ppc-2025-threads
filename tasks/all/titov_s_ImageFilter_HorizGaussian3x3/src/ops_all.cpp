#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"
#include "all/titov_s_ImageFilter_HorizGaussian3x3/include/ops_all.hpp"

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

  // Получаем информацию о ранге и размере коммуникатора
  const int world_size = world_.size();
  const int world_rank = world_.rank();

  // Calculate rows per process
  const int rows_per_process = height / world_size;
  int start_row = world_rank * rows_per_process;
  int end_row = (world_rank == world_size - 1) ? height : (start_row + rows_per_process);

  // Add overlap rows for boundary conditions
  if (world_rank > 0) start_row--;
  if (world_rank < world_size - 1) end_row++;

  // Allocate local buffers
  std::vector<double> local_input((end_row - start_row) * width);
  std::vector<double> local_output((end_row - start_row) * width);

  // Root process distributes data
  if (world_rank == 0) {
    // Copy own data first
    std::copy(input_.begin() + start_row * width, input_.begin() + end_row * width, local_input.begin());

    // Send to other processes
    for (int p = 1; p < world_size; p++) {
      int p_start = p * rows_per_process;
      int p_end = (p == world_size - 1) ? height : (p_start + rows_per_process);
      if (p > 0) p_start--;
      if (p < world_size - 1) p_end++;

      world_.send(p, 0, &input_[p_start * width], (p_end - p_start) * width);
    }
  } else {
    world_.recv(0, 0, local_input.data(), local_input.size());
  }

  // STL threads for intra-process parallelism
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  const int local_height = end_row - start_row;
  const int rows_per_thread = local_height / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    const int thread_start = t * rows_per_thread;
    const int thread_end = (t == num_threads - 1) ? local_height : (thread_start + rows_per_thread);

    threads.emplace_back([=, &local_input, &local_output] {
      for (int i = thread_start; i < thread_end; ++i) {
        const int row_offset = i * width;
        for (int j = 0; j < width; ++j) {
          double val = local_input[row_offset + j] * kernel_[1];
          if (j > 0) {
            val += local_input[row_offset + j - 1] * kernel_[0];
          }
          if (j < width - 1) {
            val += local_input[row_offset + j + 1] * kernel_[2];
          }
          local_output[row_offset + j] = val / sum;
        }
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  // Gather results
  if (world_rank == 0) {
    // Copy root's results (excluding overlaps)
    std::copy(local_output.begin() + (world_rank > 0 ? width : 0),
              local_output.end() - (world_rank < world_size - 1 ? width : 0),
              output_.begin() + (world_rank * rows_per_process) * width);

    // Receive from other processes
    for (int p = 1; p < world_size; p++) {
      int p_start = p * rows_per_process;
      int p_end = (p == world_size - 1) ? height : (p_start + rows_per_process);
      world_.recv(p, 0, &output_[p_start * width], (p_end - p_start) * width);
    }
  } else {
    // Send results to root (excluding overlaps)
    int send_start = (world_rank > 0) ? width : 0;
    int send_end = local_output.size() - ((world_rank < world_size - 1) ? width : 0);
    world_.send(0, 0, local_output.data() + send_start, send_end - send_start);
  }

  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);

  for (size_t i = 0; i < output_.size(); i++) {
    out_ptr[i] = output_[i];
  }

  return true;
}