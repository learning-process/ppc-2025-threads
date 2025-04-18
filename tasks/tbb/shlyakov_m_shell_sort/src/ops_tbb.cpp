#include "tbb/shlyakov_m_shell_sort/include/ops_tbb.hpp"

#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>

#include <algorithm>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <utility>
#include <vector>

namespace shlyakov_m_shell_sort_tbb {

bool TestTaskTBB::PreProcessingImpl() {
  std::size_t sz = task_data->inputs_count[0];
  auto* ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(ptr, ptr + sz);
  output_ = input_;
  return true;
}

bool TestTaskTBB::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool TestTaskTBB::RunImpl() {
  int n = static_cast<int>(input_.size());
  if (n < 2) {
    return true;
  }

  int threads = ppc::util::GetPPCNumThreads();
  threads = std::min(threads, n);
  int seg = (n + threads - 1) / threads;

  std::vector<std::pair<int, int>> segs;
  segs.reserve(threads);
  for (int i = 0; i < threads; ++i) {
    int l = i * seg;
    int r = std::min(n - 1, l + seg - 1);
    segs.emplace_back(l, r);
  }

  tbb::task_arena arena(threads);
  arena.execute([&] {
    tbb::task_group tg;
    for (auto [l, r] : segs) {
      tg.run([this, l, r] { ShellSort(l, r, input_); });
    }
    tg.wait();
  });

  std::vector<int> buf;
  int end = segs[0].second;
  for (std::size_t i = 1; i < segs.size(); ++i) {
    int r = segs[i].second;
    Merge(0, end, r, input_, buf);
    end = r;
  }

  output_ = input_;
  return true;
}

void ShellSort(int left, int right, std::vector<int>& arr) {
  int gap = 1;
  int size = right - left + 1;
  while (gap <= size / 3) {
    gap = gap * 3 + 1;
  }
  for (; gap > 0; gap /= 3) {
    for (int k = left + gap; k <= right; ++k) {
      int val = arr[k];
      int j = k;
      while (j >= left + gap && arr[j - gap] > val) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = val;
    }
  }
}

void Merge(int left, int mid, int right, std::vector<int>& arr, std::vector<int>& buffer) {
  int merge_size = right - left + 1;
  if (buffer.size() < static_cast<std::size_t>(merge_size)) {
    buffer.resize(static_cast<std::size_t>(merge_size));
  }

  int i = left;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) {
      buffer[k++] = arr[i++];
    } else {
      buffer[k++] = arr[j++];
    }
  }
  while (i <= mid) {
    buffer[k++] = arr[i++];
  }
  while (j <= right) {
    buffer[k++] = arr[j++];
  }
  for (int idx = 0; idx < merge_size; ++idx) {
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