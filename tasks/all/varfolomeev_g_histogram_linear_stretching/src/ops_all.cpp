#include "all/varfolomeev_g_histogram_linear_stretching/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <vector>

bool varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] == task_data->outputs_count[0] && task_data->inputs_count[0] > 0;
  }
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    input_image_.resize(task_data->inputs_count[0]);
    auto* input_ptr = reinterpret_cast<uint8_t*>(task_data->inputs[0]);
    std::copy(input_ptr, input_ptr + task_data->inputs_count[0], input_image_.begin());
  }
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::RunImpl() {
  std::vector<uint8_t> local_data;
  ScatterData(local_data);

  int min_val = 0;
  int max_val = 255;
  FindMinMax(local_data, min_val, max_val);

  StretchHistogram(local_data, min_val, max_val);

  GatherResults(local_data);

  return true;
}

bool varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<uint8_t*>(task_data->outputs[0]);
    std::copy(result_image_.begin(), result_image_.end(), output_ptr);
  }
  return true;
}

void varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::ScatterData(std::vector<uint8_t>& local_data) {
  if (world_.rank() == 0) {
    // Distribute data among processes
    for (int proc = 1; proc < world_.size(); ++proc) {
      std::vector<uint8_t> proc_data;
      for (size_t i = proc; i < input_image_.size(); i += world_.size()) {
        proc_data.push_back(input_image_[i]);
      }
      world_.send(proc, 0, proc_data);
    }

    // Process local portion
    for (size_t i = 0; i < input_image_.size(); i += world_.size()) {
      local_data.push_back(input_image_[i]);
    }
  } else {
    world_.recv(0, 0, local_data);
  }
}

void varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::FindMinMax(const std::vector<uint8_t>& local_data,
                                                                            int& min_val, int& max_val) {
  // Find local min and max
  int local_min = 255, local_max = 0;
  if (!local_data.empty()) {
    local_min = *std::min_element(local_data.begin(), local_data.end());
    local_max = *std::max_element(local_data.begin(), local_data.end());
  }

  // Reduce to get global min and max
  boost::mpi::all_reduce(world_, local_min, min_val, boost::mpi::minimum<int>());
  boost::mpi::all_reduce(world_, local_max, max_val, boost::mpi::maximum<int>());

  // Handle case when all pixels are equal
  if (min_val == max_val) {
    min_val = 0;
    max_val = 255;
  }
}

void varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::StretchHistogram(std::vector<uint8_t>& local_data,
                                                                                  int min_val, int max_val) {
  if (min_val != max_val) {
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(local_data.size()); ++i) {
      local_data[i] =
          static_cast<uint8_t>(std::round(static_cast<double>(local_data[i] - min_val) * 255 / (max_val - min_val)));
    }
  }
}

void varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::GatherResults(const std::vector<uint8_t>& local_data) {
  if (world_.rank() == 0) {
    result_image_.resize(input_image_.size());

    // Process local results
    for (size_t i = 0; i < local_data.size(); ++i) {
      size_t pos = i * world_.size();
      if (pos < result_image_.size()) {
        result_image_[pos] = local_data[i];
      }
    }

    // Receive results from other processes
    for (int proc = 1; proc < world_.size(); ++proc) {
      std::vector<uint8_t> proc_data;
      world_.recv(proc, 0, proc_data);

      for (size_t i = 0; i < proc_data.size(); ++i) {
        size_t pos = i * world_.size() + proc;
        if (pos < result_image_.size()) {
          result_image_[pos] = proc_data[i];
        }
      }
    }
  } else {
    world_.send(0, 0, local_data);
  }
}
