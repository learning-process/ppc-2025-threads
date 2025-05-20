#include "all/petrov_o_vertical_image_filtration/include/ops_all.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

#include "core/util/include/util.hpp"

namespace petrov_o_vertical_image_filtration_all {

TaskAll::TaskAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)), env_(), world_() {}

bool TaskAll::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];
  size_t input_size = width_ * height_;

  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  if (world_.rank() == 0 && width_ >= 3 && height_ >= 3) {
    output_ = std::vector<int>((width_ - 2) * (height_ - 2), 0);
  } else if (world_.rank() == 0) {
    output_.clear();
  }
  return true;
}

bool TaskAll::ValidationImpl() {
  size_t width = task_data->inputs_count[0];
  size_t height = task_data->inputs_count[1];

  if (width < 3 || height < 3) {
    return false;
  }
  return task_data->outputs_count[0] == (width - 2) * (height - 2);
}

bool TaskAll::RunImpl() {
  int rank = world_.rank();
  int comm_size = world_.size();

  if (height_ < 3 || width_ < 3) {
    if (rank == 0) {
      output_.clear();
    }
    world_.barrier();
    return true;
  }

  size_t total_output_rows = height_ - 2;
  size_t output_cols = width_ - 2;

  size_t rows_per_rank_base = total_output_rows / comm_size;
  size_t remainder_rows = total_output_rows % comm_size;

  size_t my_num_rows = rows_per_rank_base + (rank < remainder_rows ? 1 : 0);
  size_t my_start_output_row = 0;
  for (int r = 0; r < rank; ++r) {
    my_start_output_row += (rows_per_rank_base + (r < remainder_rows ? 1 : 0));
  }

  std::vector<int> local_output;
  if (my_num_rows > 0) {
    local_output.resize(my_num_rows * output_cols);
  }

  if (my_num_rows > 0) {
    int num_tbb_threads = ppc::util::GetPPCNumThreads();
    if (num_tbb_threads <= 0) {
      num_tbb_threads = oneapi::tbb::info::default_concurrency();
    }
    if (num_tbb_threads <= 0) {
      num_tbb_threads = 1;
    }

    oneapi::tbb::task_arena arena(num_tbb_threads);
    arena.execute([&] {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, my_num_rows), [&](const tbb::blocked_range<size_t> &r_range) {
        for (size_t local_i_out = r_range.begin(); local_i_out != r_range.end(); ++local_i_out) {
          size_t global_i_out = my_start_output_row + local_i_out;
          size_t i_in = global_i_out + 1;

          for (size_t j_out = 0; j_out < output_cols; ++j_out) {
            size_t j_in = j_out + 1;
            float sum = 0.0F;
            for (int ki = -1; ki <= 1; ++ki) {
              for (int kj = -1; kj <= 1; ++kj) {
                sum += static_cast<float>(input_[((i_in + ki) * width_) + (j_in + kj)]) *
                       gaussian_kernel_[((ki + 1) * 3) + (kj + 1)];
              }
            }
            local_output[(local_i_out * output_cols) + j_out] = static_cast<int>(sum);
          }
        }
      });
    });
  }

  if (total_output_rows > 0) {
    if (rank == 0) {
      output_.assign(total_output_rows * output_cols, 0);

      std::vector<int> recv_counts(comm_size);
      std::vector<int> displs(comm_size);
      int current_displ = 0;
      for (int r = 0; r < comm_size; ++r) {
        size_t rows_for_r = rows_per_rank_base + (r < remainder_rows ? 1 : 0);
        recv_counts[r] = static_cast<int>(rows_for_r * output_cols);
        displs[r] = current_displ;
        current_displ += recv_counts[r];
      }
      boost::mpi::gatherv(world_, local_output, output_.data(), recv_counts, displs, 0);
    } else {
      boost::mpi::gatherv(world_, local_output, 0);
    }
  } else if (rank == 0) {
    output_.clear();
  }

  world_.barrier();
  return true;
}

bool TaskAll::PostProcessingImpl() {
  if (world_.rank() == 0) {
    if (width_ >= 3 && height_ >= 3 && !output_.empty()) {
      for (size_t i = 0; i < output_.size(); i++) {
        reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
      }
    }
  }
  return true;
}

}  // namespace petrov_o_vertical_image_filtration_all
