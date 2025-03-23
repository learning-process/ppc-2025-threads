#include "stl/tsatsyn_a_radix_sort_simple_merge_stl/include/ops_stl.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <mutex>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
inline void ToSplitData(std::vector<uint64_t> &poz, std::vector<uint64_t> &neg, std::vector<double> input) {
  std::mutex pos_mtx;
  std::mutex neg_mtx;

  size_t num_threads = ppc::util::GetPPCNumThreads();
  size_t chunk_size = (input.size() + num_threads - 1) / num_threads;

  auto split_worker = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      double num = input[i];
      uint64_t bits = 0;
      bits = std::bit_cast<uint64_t>(num);
      if (num >= 0) {
        std::lock_guard<std::mutex> lock(pos_mtx);
        poz.push_back(bits);
      } else {
        std::lock_guard<std::mutex> lock(neg_mtx);
        neg.push_back(bits);
      }
    }
  };
  std::vector<std::future<void>> split_futures;
  for (size_t t = 0; t < num_threads; ++t) {
    size_t start = t * chunk_size;
    size_t end = std::min(start + chunk_size, input.size());
    split_futures.emplace_back(std::async(std::launch::async, split_worker, start, end));
  }
  for (auto &f : split_futures) {
    f.wait();
  }
}
inline void SortRadix(std::vector<uint64_t> &data, bool is_negative) {
  for (int bit = 0; bit < 64; bit++) {
    std::vector<uint64_t> group0;
    std::vector<uint64_t> group1;
    for (uint64_t b : data) {
      if (((b >> bit) & 1) != 0U) {
        group1.push_back(b);
      } else {
        group0.push_back(b);
      }
    }
    if (is_negative) {
      data = std::move(group1);
      data.insert(data.end(), group0.begin(), group0.end());

    } else {
      data = std::move(group0);
      data.insert(data.end(), group1.begin(), group1.end());
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
  ToSplitData(pozitive_copy, negative_copy, input_data_);

  size_t num_threads = ppc::util::GetPPCNumThreads();
  if (num_threads >= 2) {
    auto future_neg = std::async(std::launch::async, [&negative_copy]() { SortRadix(negative_copy, true); });
    auto future_pos = std::async(std::launch::async, [&pozitive_copy]() { SortRadix(pozitive_copy, false); });
    future_neg.wait();
    future_pos.wait();
  } else {
    SortRadix(negative_copy, true);
    SortRadix(pozitive_copy, false);
  }

  auto future_neg = std::async(std::launch::async, [&]() {
    for (size_t i = 0; i < negative_copy.size(); ++i) {
      output_[i] = std::bit_cast<double>(negative_copy[i]);
    }
  });

  auto future_pos = std::async(std::launch::async, [&]() {
    size_t offset = negative_copy.size();
    for (size_t i = 0; i < pozitive_copy.size(); ++i) {
      output_[offset + i] = std::bit_cast<double>(pozitive_copy[i]);
    }
  });

  future_neg.wait();
  future_pos.wait();
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_stl::TestTaskSTL::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
