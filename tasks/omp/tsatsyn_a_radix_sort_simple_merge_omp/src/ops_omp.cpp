#include "omp/tsatsyn_a_radix_sort_simple_merge_omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

bool tsatsyn_a_radix_sort_simple_merge_omp::TestTaskOpenMP::PreProcessingImpl() {
  auto *temp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_data_ = std::vector<double>(temp_ptr, temp_ptr + task_data->inputs_count[0]);
  output_.resize(task_data->inputs_count[0]);
#pragma omp parallel
  {
    std::cout << "Hello from thread " << omp_get_thread_num() << std::endl;
    // Выполняем другие действия
  }
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_omp::TestTaskOpenMP::ValidationImpl() { return task_data->inputs_count[0] != 0; }

bool tsatsyn_a_radix_sort_simple_merge_omp::TestTaskOpenMP::RunImpl() {
  std::vector<uint64_t> pozitive_copy;
  std::vector<uint64_t> negative_copy;
  for (int i = 0; i < static_cast<int>(input_data_.size()); i++) {
    if (input_data_[i] > 0.0) {
      pozitive_copy.emplace_back(*reinterpret_cast<uint64_t *>(&input_data_[i]));
    } else {
      negative_copy.emplace_back(*reinterpret_cast<uint64_t *>(&input_data_[i]));
    }
  }
  for (int bit = 0; bit < 64; bit++) {
    std::vector<uint64_t> group0;
    std::vector<uint64_t> group1;
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(pozitive_copy.size()); ++i) {
      if (((pozitive_copy[i] >> bit) & 1) != 0U) {
#pragma omp critical
        group1.push_back(pozitive_copy[i]);
      } else {
#pragma omp critical
        group0.push_back(pozitive_copy[i]);
      }
    }
    pozitive_copy.clear();
    pozitive_copy.insert(pozitive_copy.end(), group0.begin(), group0.end());
    pozitive_copy.insert(pozitive_copy.end(), group1.begin(), group1.end());
  }

  for (int bit = 0; bit < 64; bit++) {
    std::vector<uint64_t> group0;
    std::vector<uint64_t> group1;
    for (uint64_t b : negative_copy) {
      if (((b >> bit) & 1) != 0U) {
        group1.push_back(b);
      } else {
        group0.push_back(b);
      }
    }
    negative_copy.clear();
    negative_copy.insert(negative_copy.end(), group1.begin(), group1.end());
    negative_copy.insert(negative_copy.end(), group0.begin(), group0.end());
  }
  for (int i = 0; i < static_cast<int>(negative_copy.size()); i++) {
    output_[i] = *reinterpret_cast<double *>(&negative_copy[i]);
  }
  for (int i = 0; i < static_cast<int>(pozitive_copy.size()); i++) {
    output_[negative_copy.size() + i] = *reinterpret_cast<double *>(&pozitive_copy[i]);
  }
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_omp::TestTaskOpenMP::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
