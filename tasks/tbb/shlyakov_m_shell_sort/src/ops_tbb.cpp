#include "tbb/shlyakov_m_shell_sort/include/ops_tbb.hpp"

#include <tbb/task_arena.h>
#include <tbb/task_group.h>

#include <algorithm>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <utility>
#include <vector>

namespace shlyakov_m_shell_sort_tbb {

bool TestTaskTBB::PreProcessingImpl() {
  std::size_t input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);
  output_ = input_;
  return true;
}

bool TestTaskTBB::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool TestTaskTBB::RunImpl() {
  int n = static_cast<int>(input_.size());
  if (n < 2) {
    output_ = input_;
    return true;
  }

  int num_threads = ppc::util::GetPPCNumThreads();
  num_threads = std::min(num_threads, n);

  int seg_size = (n + num_threads - 1) / num_threads;

  std::vector<std::pair<int, int>> segments;
  segments.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    int left = i * seg_size;
    int right = std::min(n - 1, left + seg_size - 1);
    segments.emplace_back(left, right);
  }

  tbb::task_arena arena(num_threads);
  arena.execute([&] {
    tbb::task_group tg;
    for (auto [l, r] : segments) {
      tg.run([this, l, r] { ShellSort(l, r, input_); });
    }
    tg.wait();
  });

  std::vector<int> buffer;
  int merged_end = segments[0].second;
  for (std::size_t i = 1; i < segments.size(); ++i) {
    int seg_end = segments[i].second;
    Merge(0, merged_end, seg_end, input_, buffer);
    merged_end = seg_end;
  }

  output_ = input_;
  return true;
}

void ShellSort(int left, int right, std::vector<int>& arr) {
  int size = right - left + 1;
  int gap = 1;
  while (gap <= size / 3) {
    gap = gap * 3 + 1;
  }
  for (; gap > 0; gap /= 3) {
    for (int k = left + gap; k <= right; ++k) {
      int tmp = arr[k];
      int j = k;
      while (j >= left + gap && arr[j - gap] > tmp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = tmp;
    }
  }
}

void Merge(int left, int mid, int right, std::vector<int>& arr, std::vector<int>& buffer) {
  int i = left;
  int j = mid + 1;
  int k = 0;

  int merge_size = right - left + 1;
  if (buffer.size() < static_cast<std::size_t>(merge_size)) {
    buffer.resize(static_cast<std::size_t>(merge_size));
  }

  while (i <= mid || j <= right) {
    if (i > mid) {
      buffer[k] = arr[j];
      ++k;
      ++j;
    } else if (j > right) {
      buffer[k] = arr[i];
      ++k;
      ++i;
    } else if (arr[i] <= arr[j]) {
      buffer[k] = arr[i];
      ++k;
      ++i;
    } else {
      buffer[k] = arr[j];
      ++k;
      ++j;
    }
  }

  for (int idx = 0; idx < k; ++idx) {
    arr[left + idx] = buffer[idx];
  }
}

bool TestTaskTBB::PostProcessingImpl() {
  for (std::size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

}  // namespace shlyakov_m_shell_sort_tbb