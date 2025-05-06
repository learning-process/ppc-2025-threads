#include "stl/shlyakov_m_shell_sort/include/ops_stl.hpp"

#include <algorithm>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace shlyakov_m_shell_sort_stl {

bool TestTaskSTL::PreProcessingImpl() {
  const std::size_t sz = task_data->inputs_count[0];
  auto* ptr = reinterpret_cast<int*>(task_data->inputs[0]);

  input_.assign(ptr, ptr + sz);
  output_ = input_;
  return true;
}

bool TestTaskSTL::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool TestTaskSTL::RunImpl() {
  const int n = static_cast<int>(input_.size());
  if (n < 2) {
    return true;
  }

  const int max_threads = ppc::util::GetPPCNumThreads();
  int threads = std::min(max_threads, n);
  const int seg_size = (n + threads - 1) / threads;

  std::vector<std::pair<int, int>> segs;
  segs.reserve(threads);
  for (int idx = 0; idx < threads; ++idx) {
    const int l = idx * seg_size;
    const int r = std::min(n - 1, l + seg_size - 1);
    segs.emplace_back(l, r);
  }

  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (const auto& seg : segs) {
    const int l = seg.first;
    const int r = seg.second;
    workers.emplace_back([this, l, r] { ShellSort(l, r, input_); });
  }

  for (auto& t : workers) {
    t.join();
  }

  std::vector<int> buf(input_.size());

  int end = segs.front().second;
  for (std::size_t i = 1; i < segs.size(); ++i) {
    const int r = segs[i].second;
    Merge(0, end, r, input_, buf);
    end = r;
  }

  output_ = input_;
  return true;
}

void ShellSort(int left, int right, std::vector<int>& arr) {
  const int size = right - left + 1;

  std::vector<int> gaps;
  for (int k = 0;; ++k) {
    int gap = (k % 2 == 0) ? (9 * (1 << k) - (1 << (k / 2)) + 1) : (8 * (1 << k) - (6 * (1 << ((k + 1) / 2))) + 1);
    if (gap > size) break;
    gaps.push_back(gap);
  }
  std::reverse(gaps.begin(), gaps.end());

  for (int gap : gaps) {
    for (int k = left + gap; k <= right; ++k) {
      const int val = arr[k];
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
  const int merge_size = right - left + 1;
  if (buffer.size() < static_cast<std::size_t>(merge_size)) {
    buffer.resize(static_cast<std::size_t>(merge_size));
  }

  auto it_left = arr.begin() + left;
  auto it_mid = arr.begin() + mid + 1;
  auto it_right = arr.begin() + right + 1;

  auto buf_it = buffer.begin();

  auto left_it = it_left, right_it = it_mid;
  while (left_it != arr.begin() + mid + 1 && right_it != it_right) {
    *buf_it++ = (*left_it <= *right_it) ? *left_it++ : *right_it++;
  }
  std::copy(left_it, arr.begin() + mid + 1, buf_it);
  std::copy(right_it, it_right, buf_it);

  std::copy(buffer.begin(), buffer.begin() + merge_size, it_left);
}

void ParallelMerge(std::vector<std::pair<int, int>>& segs, std::vector<int>& arr, std::vector<int>& buffer) {
  while (segs.size() > 1) {
    std::vector<std::pair<int, int>> new_segs;
    std::vector<std::thread> workers;

    for (size_t i = 0; i < segs.size(); i += 2) {
      if (i + 1 < segs.size()) {
        const int l = segs[i].first;
        const int mid = segs[i].second;
        const int r = segs[i + 1].second;

        workers.emplace_back([&arr, &buffer, l, mid, r] { Merge(l, mid, r, arr, buffer); });
        new_segs.emplace_back(l, r);
      } else {
        new_segs.push_back(segs[i]);
      }
    }

    for (auto& t : workers) {
      t.join();
    }

    segs = std::move(new_segs);
  }
}

bool TestTaskSTL::PostProcessingImpl() {
  for (std::size_t idx = 0; idx < output_.size(); ++idx) {
    reinterpret_cast<int*>(task_data->outputs[0])[idx] = output_[idx];
  }
  return true;
}

}  // namespace shlyakov_m_shell_sort_stl