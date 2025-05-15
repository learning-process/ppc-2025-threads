#include "tbb/malyshev_v_radix_sort/include/ops_tbb.hpp"

#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace malyshev_v_radix_sort_tbb {

namespace {
union DoubleConverter {
  double d;
  uint64_t u;
};

uint64_t ConvertDoubleToUInt64(double d) {
  DoubleConverter converter;
  converter.d = d;
  return (converter.u & (1ULL << 63)) != 0U ? ~converter.u : (converter.u ^ (1ULL << 63));
}

void RadixSort(std::vector<double>& data, int exp) {
  std::vector<int> count(256, 0);
  std::vector<double> output(data.size());

  for (size_t i = 0; i < data.size(); ++i) {
    uint64_t bits = ConvertDoubleToUInt64(data[i]);
    int digit = static_cast<int>((bits >> (exp * 8)) & 0xFF);
    count[digit]++;
  }

  for (int i = 1; i < 256; ++i) {
    count[i] += count[i - 1];
  }

  for (size_t i = data.size(); i-- > 0;) {
    uint64_t bits = ConvertDoubleToUInt64(data[i]);
    int digit = static_cast<int>((bits >> (exp * 8)) & 0xFF);
    output[--count[digit]] = data[i];
  }

  data = output;
}

void ProcessNumbers(std::vector<double>& numbers) {
  for (int exp = 0; exp < 8; ++exp) {
    RadixSort(numbers, exp);
  }
}

}  // namespace

bool SortTBB::PreProcessingImpl() {
  input_ = std::vector<double>(reinterpret_cast<double*>(task_data->inputs[0]),
                               reinterpret_cast<double*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  output_.resize(task_data->outputs_count[0]);
  return true;
}

bool SortTBB::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool SortTBB::RunImpl() {
  if (input_.empty()) {
    output_ = input_;
    return true;
  }

  std::vector<double> negative;
  std::vector<double> positive;

  for (double num : input_) {
    if (num < 0) {
      negative.push_back(num);
    } else {
      positive.push_back(num);
    }
  }

  oneapi::tbb::task_arena arena;
  arena.execute([&] {
    tbb::task_group tg;
    tg.run([&] { ProcessNumbers(negative); });
    tg.run([&] { ProcessNumbers(positive); });
    tg.wait();
  });

  std::ranges::reverse(negative);
  output_.clear();
  output_.insert(output_.end(), negative.begin(), negative.end());
  output_.insert(output_.end(), positive.begin(), positive.end());

  return true;
}

bool SortTBB::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}

}  // namespace malyshev_v_radix_sort_tbb