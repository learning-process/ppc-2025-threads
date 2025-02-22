#include "seq/malyshev_v_radix_sort/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace malyshev_v_radix_sort {
namespace {
union DoubleWrapper {
  double d;
  uint64_t u;
};

void CountingSort(std::vector<double>& arr, int exp) {
  const int RADIX = 256;
  size_t n = arr.size();
  std::vector<double> output(n);
  std::vector<int> count(RADIX, 0);

  for (size_t i = 0; i < n; i++) {
    DoubleWrapper dw;
    dw.d = arr[i];
    uint64_t value = dw.u;
    int index = (value >> (8 * exp)) & 0xFF;
    count[index]++;
  }

  for (int i = 1; i < RADIX; i++) {
    count[i] += count[i - 1];
  }

  for (int i = n - 1; i >= 0; i--) {
    DoubleWrapper dw;
    dw.d = arr[i];
    uint64_t value = dw.u;
    int index = (value >> (8 * exp)) & 0xFF;
    output[count[index] - 1] = arr[i];
    count[index]--;
  }

  arr = output;
}

void RadixSort(std::vector<double>& arr) {
  if (arr.empty()) return;

  const int BYTES = sizeof(uint64_t);
  for (int exp = 0; exp < BYTES; exp++) {
    CountingSort(arr, exp);
  }

  // Handle negative numbers
  std::partition(arr.begin(), arr.end(), [](double x) { return std::signbit(x); });
}
}  // namespace

bool RadixSortSequential::PreProcessingImpl() {
  res_ = std::vector<double>(reinterpret_cast<double*>(task_data->inputs[0]),
                             reinterpret_cast<double*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  return true;
}

bool RadixSortSequential::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0;
}

bool RadixSortSequential::RunImpl() {
  RadixSort(res_);
  return true;
}

bool RadixSortSequential::PostProcessingImpl() {
  double* output = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < res_.size(); i++) {
    output[i] = res_[i];
  }
  return true;
}
}  // namespace malyshev_v_radix_sort