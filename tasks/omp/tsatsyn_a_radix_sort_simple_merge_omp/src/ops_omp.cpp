#include "omp/tsatsyn_a_radix_sort_simple_merge_omp/include/ops_omp.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

bool tsatsyn_a_radix_sort_simple_merge_omp::TestTaskOpenMP::ValidationImpl() { return task_data->inputs_count[0] != 0; }

bool tsatsyn_a_radix_sort_simple_merge_omp::TestTaskOpenMP::PreProcessingImpl() {
  auto *temp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_data_ = std::vector<double>(temp_ptr, temp_ptr + task_data->inputs_count[0]);
  output_.resize(task_data->inputs_count[0]);
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_omp::TestTaskOpenMP::RunImpl() {
  std::vector<uint64_t> pozitive_copy;
  std::vector<uint64_t> negative_copy;

#pragma omp parallel
  {
    std::vector<uint64_t> local_positive;
    std::vector<uint64_t> local_negative;

#pragma omp for nowait
    for (int i = 0; i < static_cast<int>(input_data_.size()); ++i) {
      if (input_data_[i] > 0.0) {
        local_positive.push_back(*reinterpret_cast<const uint64_t *>(&input_data_[i]));
      } else {
        local_negative.push_back(*reinterpret_cast<const uint64_t *>(&input_data_[i]));
      }
    }

#pragma omp critical
    {
      pozitive_copy.insert(pozitive_copy.end(), local_positive.begin(), local_positive.end());
      negative_copy.insert(negative_copy.end(), local_negative.begin(), local_negative.end());
    }
  }

  for (int bit = 0; bit < 64; bit++) {
#pragma omp parallel
    {
#pragma omp single
      {
        std::vector<uint64_t> group0;
        std::vector<uint64_t> group1;
        group0.reserve(pozitive_copy.size());
        group1.reserve(pozitive_copy.size());

        for (int i = 0; i < static_cast<int>(pozitive_copy.size()); i++) {
          if (((pozitive_copy[i] >> bit) & 1) != 0U) {
            group1.push_back(pozitive_copy[i]);
          } else {
            group0.push_back(pozitive_copy[i]);
          }
        }
        pozitive_copy = std::move(group0);
        pozitive_copy.insert(pozitive_copy.end(), group1.begin(), group1.end());
      }
    }
  }

  for (int bit = 0; bit < 64; bit++) {
#pragma omp parallel
    {
#pragma omp single
      {
        std::vector<uint64_t> group0;
        std::vector<uint64_t> group1;
        group0.reserve(negative_copy.size());
        group1.reserve(negative_copy.size());

        for (int i = 0; i < static_cast<int>(negative_copy.size()); i++) {
          if (((negative_copy[i] >> bit) & 1) != 0U) {
            group1.push_back(negative_copy[i]);
          } else {
            group0.push_back(negative_copy[i]);
          }
        }
        negative_copy = std::move(group0);
        negative_copy.insert(negative_copy.end(), group1.begin(), group1.end());
      }
    }
  }

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(negative_copy.size()); i++) {
    output_[static_cast<int>(negative_copy.size()) - 1 - i] = *reinterpret_cast<const double *>(&negative_copy[i]);
  }

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(pozitive_copy.size()); ++i) {
    output_[negative_copy.size() + i] = *reinterpret_cast<const double *>(&pozitive_copy[i]);
  }

  return true;
}
bool tsatsyn_a_radix_sort_simple_merge_omp::TestTaskOpenMP::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
