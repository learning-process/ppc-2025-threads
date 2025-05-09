#include "stl/mezhuev_m_bitwise_integer_sort_with_simple_merge/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

namespace mezhuev_m_bitwise_integer_sort_stl {

namespace {

void SeparateNumbers(const std::vector<int>& input, std::vector<int>& negative, std::vector<int>& positive) {
  for (int num : input) {
    if (num < 0) {
      negative.push_back(-num);
    } else {
      positive.push_back(num);
    }
  }
}

void RadixSort(std::vector<int>& data, int exp) {
  std::vector<int> count(10, 0);
  std::vector<int> output(data.size());

  for (size_t i = 0; i < data.size(); ++i) {
    int digit = (data[i] / exp) % 10;
    count[digit]++;
  }

  for (int i = 1; i < 10; ++i) {
    count[i] += count[i - 1];
  }

  for (int i = data.size() - 1; i >= 0; --i) {
    int digit = (data[i] / exp) % 10;
    output[--count[digit]] = data[i];
  }

  data = std::move(output);
}

void ProcessNumbers(std::vector<int>& numbers, int max_value) {
  int exp = 1;
  while (max_value / exp > 0) {
    RadixSort(numbers, exp);
    exp *= 10;
  }
}

}  // namespace

bool SortSTL::PreProcessingImpl() {
  size_t input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);

  input_ = (input_size == 0) ? std::vector<int>() : std::vector<int>(in_ptr, in_ptr + input_size);
  output_ = std::vector<int>(input_size, 0);

  if (input_.empty()) return true;

  max_value_ = *std::max_element(input_.begin(), input_.end(), [](int a, int b) { return std::abs(a) < std::abs(b); });
  max_value_ = std::abs(max_value_);

  return true;
}

bool SortSTL::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool SortSTL::RunImpl() {
  if (input_.empty()) {
    output_ = input_;
    return true;
  }

  std::vector<int> negative, positive;
  SeparateNumbers(input_, negative, positive);

  auto sort_in_threads = [&](std::vector<int>& numbers) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    size_t chunk_size = numbers.size() / num_threads;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
      size_t start = i * chunk_size;
      size_t end = (i == num_threads - 1) ? numbers.size() : start + chunk_size;
      threads.emplace_back([&, start, end]() {
        std::vector<int> chunk(numbers.begin() + start, numbers.begin() + end);
        ProcessNumbers(chunk, max_value_);
        std::copy(chunk.begin(), chunk.end(), numbers.begin() + start);
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }

    std::sort(numbers.begin(), numbers.end());
  };

  sort_in_threads(positive);
  sort_in_threads(negative);

  std::reverse(negative.begin(), negative.end());
  for (int& num : negative) num = -num;

  output_.clear();
  output_.insert(output_.end(), negative.begin(), negative.end());
  output_.insert(output_.end(), positive.begin(), positive.end());

  return true;
}

bool SortSTL::PostProcessingImpl() {
  if (input_.empty()) return true;

  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(output_.begin(), output_.end(), out_ptr);

  return true;
}

}  // namespace mezhuev_m_bitwise_integer_sort_stl