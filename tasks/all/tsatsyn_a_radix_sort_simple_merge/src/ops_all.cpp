#include "all/tsatsyn_a_radix_sort_simple_merge/include/ops_all.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace constants {
constexpr int kChunk = 100;
}  // namespace constants

bool tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL::ValidationImpl() {
  // Check equality of counts elements
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] != 0;
  }
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL::PreProcessingImpl() {
  // Init value for input and output
  if (world_.rank() == 0) {
    auto *temp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    input_data_ = std::vector<double>(temp_ptr, temp_ptr + task_data->inputs_count[0]);
    output_.resize(task_data->inputs_count[0]);
  }
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL::RunImpl() {
  bool is_pozitive = false;
  bool is_negative = false;
  if (world_.rank() == 0) {
    for (int proc = 1; proc < static_cast<int>(world_.size()); proc++) {
      for (int j = proc; j < static_cast<int>(input_data_.size()); j += world_.size()) {
        input_data_[j] <= 0.0 ? is_negative = true : is_pozitive = true;
        local_data_.push_back(input_data_[j]);
      }
      world_.send(proc, 0, local_data_);
      local_data_.clear();
    }
    for (int j = 0; j < static_cast<int>(input_data_.size()); j += world_.size()) {
      local_data_.push_back(input_data_[j]);
    }
  } else {
    world_.recv(0, 0, local_data_);
  }
  std::vector<uint64_t> pozitive_copy;
  std::vector<uint64_t> negative_copy;
#pragma omp parallel
  {
    std::vector<uint64_t> local_positive;
    std::vector<uint64_t> local_negative;
#pragma omp for nowait
    for (int i = 0; i < static_cast<int>(local_data_.size()); ++i) {
      if (local_data_[i] > 0.0) {
        local_positive.push_back(*reinterpret_cast<const uint64_t *>(&local_data_[i]));
      } else {
        local_negative.push_back(*reinterpret_cast<const uint64_t *>(&local_data_[i]));
      }
    }
#pragma omp critical
    {
      pozitive_copy.insert(pozitive_copy.end(), local_positive.begin(), local_positive.end());
      negative_copy.insert(negative_copy.end(), local_negative.begin(), local_negative.end());
    }
  }

  if (!pozitive_copy.empty()) {
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
  }
  if (!negative_copy.empty()) {
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
  }
  if (world_.rank() == 0) {
    std::vector<uint64_t> local_copy_for_recv;
    if (is_pozitive) {
      for (int proc = 1; proc < static_cast<int>(world_.size()); proc++) {
        world_.recv(proc, 1, local_copy_for_recv);
        pozitive_copy.insert(pozitive_copy.end(), local_copy_for_recv.begin(), local_copy_for_recv.end());
        local_copy_for_recv.clear();
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
    }

    if (is_negative) {
      for (int proc = 1; proc < static_cast<int>(world_.size()); proc++) {
        world_.recv(proc, 2, local_copy_for_recv);
        negative_copy.insert(negative_copy.end(), local_copy_for_recv.begin(), local_copy_for_recv.end());
        local_copy_for_recv.clear();
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
    }
#pragma omp parallel for schedule(guided, constants::kChunk)
    for (int i = 0; i < static_cast<int>(negative_copy.size()); i++) {
      output_[static_cast<int>(negative_copy.size()) - 1 - i] = *reinterpret_cast<const double *>(&negative_copy[i]);
    }
#pragma omp parallel for schedule(guided, constants::kChunk)
    for (int i = 0; i < static_cast<int>(pozitive_copy.size()); i++) {
      output_[negative_copy.size() + i] = *reinterpret_cast<const double *>(&pozitive_copy[i]);
    }
  } else {
    if (!pozitive_copy.empty()) {
      world_.send(0, 1, pozitive_copy);
    }
    if (!negative_copy.empty()) {
      world_.send(0, 2, negative_copy);
    }
  }
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_.size(); i++) {
      reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
    }
  }
  return true;
}
