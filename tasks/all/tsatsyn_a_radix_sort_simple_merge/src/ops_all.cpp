#include "all/tsatsyn_a_radix_sort_simple_merge/include/ops_all.hpp"

#include <boost/mpi/communicator.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace {
inline void SendData(boost::mpi::communicator &world, bool &is_pozitive, bool &is_negative,
                     std::vector<double> &local_data, std::vector<double> &input_data) {
  // std::cout << "DATASZ " << input_data.size()<<std::endl;
  for (int proc = 1; proc < world.size(); proc++) {
    for (size_t j = proc; j < input_data.size(); j += world.size()) {
      input_data[j] < 0.0 ? is_negative = true : is_pozitive = true;
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
      if (local_data[i] >= 0.0) {
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
inline double Uint64ToDouble(uint64_t value) {
  double result = NAN;
  static_assert(sizeof(double) == sizeof(uint64_t), "Size mismatch");
  std::memcpy(&result, &value, sizeof(double));
  return result;
}

inline void WriteNegativePart(const std::vector<uint64_t> &negative_copy, std::vector<double> &output) {
  const size_t size = negative_copy.size();
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(size); ++i) {
    const size_t output_idx = size - 1 - i;
    output[output_idx] = Uint64ToDouble(negative_copy[i]);
  }
}

inline void WritePositivePart(const std::vector<uint64_t> &positive_copy, const size_t offset,
                              std::vector<double> &output) {
  const size_t size = positive_copy.size();
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(size); ++i) {
    const size_t output_idx = offset + i;
    output[output_idx] = Uint64ToDouble(positive_copy[i]);
  }
}
inline void FinalParse(std::vector<uint64_t> &data, int code, boost::mpi::communicator &world) {
  std::vector<uint64_t> local_copy_for_recv;
  for (int proc = 1; proc < world.size(); proc++) {
    world.recv(proc, code, local_copy_for_recv);
    data.insert(data.end(), local_copy_for_recv.begin(), local_copy_for_recv.end());
    local_copy_for_recv.clear();
  }

  if (!data.empty()) {
    RadixSort(data);
  }
}
inline void SafeDataWrite(const std::vector<uint64_t> &negative_copy, const std::vector<uint64_t> &positive_copy,
                          std::vector<double> &output, bool is_negative, bool is_positive) {
  if (is_negative) {
    WriteNegativePart(negative_copy, output);
  }
  if (is_positive) {
    WritePositivePart(positive_copy, negative_copy.size(), output);
  }
}
}  // namespace
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
    SendData(world_, is_pozitive, is_negative, local_data_, input_data_);
  } else {
    world_.recv(0, 0, local_data_);
  }
  std::vector<uint64_t> pozitive_copy;
  std::vector<uint64_t> negative_copy;
  ParallelParse(pozitive_copy, negative_copy, local_data_);
  // std::cout <<world_.rank() << " POZ " << pozitive_copy.size() << " NEG " << negative_copy.size() << std::endl;
  if (!pozitive_copy.empty()) {
    RadixSort(pozitive_copy);
  }
  if (!negative_copy.empty()) {
    RadixSort(negative_copy);
  }
  /*std::cout << " POZ PROC " << world_.rank() << " " << pozitive_copy.size()<< std::endl;
  for (auto &val : pozitive_copy) {
    std::cout << val << " ";
  }
  std::cout << std::endl << " NEG PROC " << world_.rank() << " " << negative_copy.size() << std::endl;
  for (auto &val : negative_copy) {
    std::cout << val << " ";
  }
  std::cout << std::endl;*/
  if (world_.rank() == 0) {
    if (is_pozitive) {
      FinalParse(pozitive_copy, 1, world_);
    }
    if (is_negative) {
      FinalParse(negative_copy, 2, world_);
    }
    // std::cout << " POZitive na PROC NULL " << world_.rank() << " s" << pozitive_copy.size() << "s" << std::endl;
    // std::cout << " NEGative na PROC NULL " << world_.rank() << " s" << negative_copy.size() << "s" << std::endl;

    SafeDataWrite(negative_copy, pozitive_copy, output_, is_negative, is_pozitive);
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
    assert(output_.size() == task_data->outputs_count[0]);
    for (size_t i = 0; i < output_.size(); i++) {
      double value = output_[i];
      std::memcpy(reinterpret_cast<double *>(task_data->outputs[0]) + i, &value, sizeof(double));
    }
  }
  return true;
}
