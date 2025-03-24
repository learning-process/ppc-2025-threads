#include "all/tsatsyn_a_radix_sort_simple_merge/include/ops_all.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

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
inline void SendData(boost::mpi::communicator &world, bool &is_pozitive, bool &is_negative,
                     std::vector<double> &local_data, std::vector<double> &input_data) {
  for (int proc = 1; proc < world.size(); proc++) {
    for (int j = proc; j < static_cast<int>(input_data.size()); j += world_.size()) {
      input_data[j] <= 0.0 ? is_negative = true : is_pozitive = true;
      local_data.push_back(input_data[j]);
    }
    world.send(proc, 0, local_data);
    local_data.clear();
  }
  for (int j = 0; j < static_cast<int>(input_data.size()); j += world.size()) {
    local_data.push_back(input_data[j]);
  }
}
inline void ParallelParse(std::vector<uint64_t> &pozitive_copy, std::vector<uint64_t> &negative_copy,
                          std::vector<double> &local_data) {
#pragma omp parallel
  {
    std::vector<uint64_t> local_positive;
    std::vector<uint64_t> local_negative;
#pragma omp for nowait
    for (int i = 0; i < static_cast<int>(local_data.size()); ++i) {
      if (local_data[i] > 0.0) {
        local_positive.push_back(*reinterpret_cast<const uint64_t *>(&local_data[i]));
      } else {
        local_negative.push_back(*reinterpret_cast<const uint64_t *>(&local_data[i]));
      }
    }
#pragma omp critical
    {
      pozitive_copy.insert(pozitive_copy.end(), local_positive.begin(), local_positive.end());
      negative_copy.insert(negative_copy.end(), local_negative.begin(), local_negative.end());
    }
  }
}
inline void RadixSort(std::vector<uint64_t> &data) {
  for (int bit = 0; bit < 64; bit++) {
#pragma omp parallel
    {
#pragma omp single
      {
        std::vector<uint64_t> group0;
        std::vector<uint64_t> group1;
        group0.reserve(data.size());
        group1.reserve(data.size());
        for (int i = 0; i < static_cast<int>(data.size()); i++) {
          if (((data[i] >> bit) & 1) != 0U) {
            group1.push_back(data[i]);
          } else {
            group0.push_back(data[i]);
          }
        }
        data = std::move(group0);
        data.insert(data.end(), group1.begin(), group1.end());
      }
    }
  }
}
inline void FinalParse(std::vector<uint64_t> &data, int code, boost::mpi::communicator &world) {
  std::vector<uint64_t> local_copy_for_recv;
  for (int proc = 1; proc < world.size(); proc++) {
    world.recv(proc, code, local_copy_for_recv);
    data.insert(data.end(), local_copy_for_recv.begin(), local_copy_for_recv.end());
    local_copy_for_recv.clear();
  }
  RadixSort(data);
}
bool tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL::RunImpl() {
  bool is_pozitive = false;
  bool is_negative = false;
  if (world_.rank() == 0) {
    SendData(world_, is_pozitive, is_negative, local_data_, input_data_);
  } else {
    world_.recv(0, 0, local_data_);
  }
  std::vector<uint64_t> pozitive_copy;
  std::vector<uint64_t> negative_copy;
  ParallelParse(pozitive_copy, negative_copy, local_data_);
  if (!pozitive_copy.empty()) {
    RadixSort(pozitive_copy);
  }
  if (!negative_copy.empty()) {
    RadixSort(negative_copy);
  }
  if (world_.rank() == 0) {
    if (is_pozitive) {
      FinalParse(pozitive_copy, 1, world_);
    }
    if (is_negative) {
      FinalParse(negative_copy, 2, world_);
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
