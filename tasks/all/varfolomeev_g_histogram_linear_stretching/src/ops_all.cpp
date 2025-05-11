#include "all/varfolomeev_g_histogram_linear_stretching/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/operations.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

bool varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] == task_data->outputs_count[0] && task_data->inputs_count[0] > 0;
  }
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::PreProcessingImpl() {
  input_image_.clear();
  result_image_.clear();
  if (world_.rank() == 0) {
    input_image_.resize(task_data->inputs_count[0]);
    auto* input_ptr = reinterpret_cast<uint8_t*>(task_data->inputs[0]);
    std::ranges::copy(input_ptr, input_ptr + task_data->inputs_count[0], input_image_.begin());
  }

  return true;
}

bool varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::RunImpl() {
  std::vector<uint8_t> local_data;
  ScatterData(local_data);

  int global_min = 0;
  int global_max = 255;
  FindMinMax(local_data, global_min, global_max);

  StretchHistogram(local_data, global_min, global_max);

  GatherResults(local_data);

  return true;
}

bool varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<uint8_t*>(task_data->outputs[0]);
    std::ranges::copy(result_image_.begin(), result_image_.end(), output_ptr);
  }
  return true;
}

void varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::ScatterData(std::vector<uint8_t>& local_data) {
  if (world_.rank() == 0) {
    for (int proc = 1; proc < world_.size(); ++proc) {
      std::vector<uint8_t> proc_data;
      for (size_t i = proc; i < input_image_.size(); i += world_.size()) {
        proc_data.push_back(input_image_[i]);
      }
      world_.send(proc, 0, proc_data);  // NOLINT
    }

    for (size_t i = 0; i < input_image_.size(); i += world_.size()) {
      local_data.push_back(input_image_[i]);
    }
  } else {
    world_.recv(0, 0, local_data);  // NOLINT
  }
}

void varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::FindMinMax(const std::vector<uint8_t>& local_data,
                                                                            int& global_min, int& global_max) {
  int local_min = 255;
  int local_max = 0;

  if (!local_data.empty()) {
    local_min = *std::ranges::min_element(local_data);
    local_max = *std::ranges::max_element(local_data);
  }

  boost::mpi::all_reduce(world_, local_min, global_min, boost::mpi::minimum<int>());
  boost::mpi::all_reduce(world_, local_max, global_max, boost::mpi::maximum<int>());

  if (global_min == global_max) {
    global_min = 0;
    global_max = 255;
  }
}

void varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::StretchHistogram(std::vector<uint8_t>& local_data,
                                                                                  int global_min, int global_max) {
  if (global_min != global_max) {
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(local_data.size()); ++i) {
      local_data[i] =
          static_cast<uint8_t>(std::round((local_data[i] - global_min) * 255.0 / (global_max - global_min)));
    }
  }
}

void varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::GatherResults(const std::vector<uint8_t>& local_data) {
  if (world_.rank() == 0) {
    result_image_.resize(input_image_.size());

    for (size_t i = 0; i < local_data.size(); ++i) {
      const size_t pos = i * world_.size();
      if (pos < result_image_.size()) {
        result_image_[pos] = local_data[i];
      }
    }

    for (int proc = 1; proc < world_.size(); ++proc) {
      std::vector<uint8_t> proc_data;
      world_.recv(proc, 0, proc_data);  // NOLINT

      for (size_t i = 0; i < proc_data.size(); ++i) {
        const size_t pos = (i * world_.size()) + proc;
        if (pos < result_image_.size()) {
          result_image_[pos] = proc_data[i];
        }
      }
    }
  } else {
    world_.send(0, 0, local_data);  // NOLINT
  }
}
