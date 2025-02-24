#include "seq/tsatsyn_a_radix_sort_simple_merge_seq/include/ops_seq.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

bool tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential::PreProcessingImpl() {
  auto *temp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_data_ = std::vector<double>(temp_ptr, temp_ptr + task_data->inputs_count[0]);
  output_.resize(task_data->inputs_count[0]);
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] != 0;
}

bool tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential::RunImpl() {
  std::vector<uint64_t> pozitive_copy;
  std::vector<uint64_t> negative_copy;
  for (int i = 0; i < static_cast<int>(input_data_.size()); i++) {
    if (input_data_[i] > 0.0) {
      pozitive_copy.emplace_back(*reinterpret_cast<uint64_t *>(&input_data_[i]));
    } else {
      negative_copy.emplace_back(*reinterpret_cast<uint64_t *>(&input_data_[i]));
    }
  }
  int positive_bits = 0;
  if (!pozitive_copy.empty()) {
    uint64_t max_positive = *std::ranges::max_element(pozitive_copy);
    positive_bits = std::bit_width(max_positive);
  }
  int negative_bits = 0;
  if (!negative_copy.empty()) {
    uint64_t min_negative = *std::ranges::min_element(negative_copy);
    negative_bits = min_negative == 0 ? 0 : std::bit_width(min_negative);
  }
  for (int bit = 0; bit < positive_bits; bit++) {
    std::vector<uint64_t> group0;
    std::vector<uint64_t> group1;
    for (uint64_t b : pozitive_copy) {
      if (((b >> bit) & 1) != 0U) {
        group1.push_back(b);
      } else {
        group0.push_back(b);
      }
    }

    pozitive_copy.clear();
    pozitive_copy.insert(pozitive_copy.end(), group0.begin(), group0.end());
    pozitive_copy.insert(pozitive_copy.end(), group1.begin(), group1.end());
  }
  if (!negative_copy.empty()) {
    for (int bit = 0; bit < negative_bits; bit++) {
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
  }
  for (int i = 0; i < static_cast<int>(pozitive_copy.size()); i++) {
    output_[negative_copy.size() + i] = *reinterpret_cast<double *>(&pozitive_copy[i]);
  }
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
