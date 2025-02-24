#include "seq/tsatsyn_a_radix_sort_simple_merge_seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential::PreProcessingImpl() {
  auto *tempPtr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_data = std::vector<double>(tempPtr, tempPtr + task_data->inputs_count[0]);
  // std::copy(tempPtr, tempPtr + task_data->inputs_count[0], input_data.begin());
  output.resize(task_data->inputs_count[0]);
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] != 0;
}

bool tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential::RunImpl() {
  std::vector<uint64_t> pozitive_copy, negative_copy;
  for (int i = 0; i < input_data.size(); i++) {
    if (input_data[i] > 0.0) {
      pozitive_copy.emplace_back(*reinterpret_cast<uint64_t *>(&input_data[i]));
    } else {
      negative_copy.emplace_back(*reinterpret_cast<uint64_t *>(&input_data[i]));
    }
  }
  for (int bit = 0; bit < 64; bit++) {
    std::vector<uint64_t> group0, group1;
    for (uint64_t b : pozitive_copy) {
      if ((b >> bit) & 1) {
        group1.push_back(b);
      } else {
        group0.push_back(b);
      }
    }

    pozitive_copy.clear();
    pozitive_copy.insert(pozitive_copy.end(), group0.begin(), group0.end());
    pozitive_copy.insert(pozitive_copy.end(), group1.begin(), group1.end());
  }
  for (int bit = 0; bit < 64; bit++) {
    std::vector<uint64_t> group0, group1;

    for (uint64_t b : negative_copy) {
      if ((b >> bit) & 1) {
        group1.push_back(b);
      } else {
        group0.push_back(b);
      }
    }
    negative_copy.clear();
    negative_copy.insert(negative_copy.end(), group1.begin(), group1.end());
    negative_copy.insert(negative_copy.end(), group0.begin(), group0.end());
  }
  for (int i = 0; i < negative_copy.size(); i++) {
    output[i] = *reinterpret_cast<double *>(&negative_copy[i]);
  }
  for (int i = 0; i < pozitive_copy.size(); i++) {
    output[negative_copy.size() + i] = *reinterpret_cast<double *>(&pozitive_copy[i]);
  }
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output[i];
  }
  return true;
}
