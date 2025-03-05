#include "stl/tsatsyn_a_radix_sort_simple_merge_stl/include/ops_stl.hpp"

#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
void MatMul(const std::vector<int> &in_vec, int rc_size, std::vector<int> &out_vec) {
  for (int i = 0; i < rc_size; ++i) {
    for (int j = 0; j < rc_size; ++j) {
      out_vec[(i * rc_size) + j] = 0;
      for (int k = 0; k < rc_size; ++k) {
        out_vec[(i * rc_size) + j] += in_vec[(i * rc_size) + k] * in_vec[(k * rc_size) + j];
      }
    }
  }
}
}  // namespace

bool tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL::PreProcessingImpl() {
  // Init value for input and output
  auto *temp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_data_ = std::vector<double>(temp_ptr, temp_ptr + task_data->inputs_count[0]);
  output_.resize(task_data->inputs_count[0]);
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] != 0;
}

bool tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL::RunImpl() {
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

bool tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
