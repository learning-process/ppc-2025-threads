#include "stl/tsatsyn_a_radix_sort_simple_merge_stl/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

bool tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL::PreProcessingImpl() {
  // Init value for input and output
  auto* temp_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
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
  {
    std::mutex pos_mtx;
    std::mutex neg_mtx;
    std::for_each(std::execution::par_unseq, input_data_.begin(), input_data_.end(), [&](double num) {
      uint64_t bits;
     std::memcpy(&bits,&num,sizeof(double));
      if (num > 0.0) {
        std::lock_guard lock(pos_mtx);
        pozitive_copy.push_back(bits);
      } else {
        std::lock_guard lock(neg_mtx);
        negative_copy.push_back(bits);
      }
    });
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
  {
    std::for_each(std::execution::par_unseq, negative_copy.begin(), negative_copy.end(),
                  [&, size = negative_copy.size()](uint64_t &b) {
                    double value;
                    std::memcpy(&value, &b, sizeof(double));
                    size_t i = &b - negative_copy.data();
                    output_[i] = value;
                  });
  }
  {
    std::for_each(std::execution::par_unseq, pozitive_copy.begin(), pozitive_copy.end(),
                  [&, offset = negative_copy.size()](uint64_t &b) {
                    double value;
                    std::memcpy(&value, &b, sizeof(double));
                    size_t i = &b - pozitive_copy.data() + offset;
                    output_[i] = value;
                  });
  }
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
