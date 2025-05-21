#include "all/shlyakov_m_shell_sort/include/ops_all.hpp"

#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "all/shlyakov_m_shell_sort/include/ops_all.hpp"

namespace shlyakov_m_shell_sort_all {

bool TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    const std::size_t sz = task_data->inputs_count[0];
    auto* ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_.assign(ptr, ptr + sz);
  }
  return true;
}

bool TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] == task_data->outputs_count[0];
  } else {
    return 1;
  }
}

bool TestTaskALL::RunImpl() {
  int size = input_.size();
  boost::mpi::broadcast(world_, size, 0);

  int delta = size / world_.size();
  int extra = size % world_.size();
  std::vector<int> local_sizes(world_.size(), delta);

  if (world_.rank() == 0) {
    for (int i = 0; i < extra; ++i) local_sizes[i]++;
  }

  std::vector<int> local_data;
  if (world_.rank() == 0) {
    std::vector<std::vector<int>> chunks;
    int offset = 0;
    for (int i = 0; i < world_.size(); ++i) {
      chunks.emplace_back(input_.begin() + offset, input_.begin() + offset + local_sizes[i]);
      offset += local_sizes[i];
    }
    boost::mpi::scatter(world_, chunks, local_data, 0);
  } else {
    boost::mpi::scatter(world_, local_data, 0);
  }

  const int n = static_cast<int>(local_data.size());
  if (n > 1) {
    const int max_threads = ppc::util::GetPPCNumThreads();
    int threads = std::min(max_threads, n);
    const int seg_size = (n + threads - 1) / threads;

    tbb::task_arena arena(threads);
    arena.execute([&] {
      tbb::task_group tg;
      for (int i = 0; i < threads; ++i) {
        int l = i * seg_size;
        int r = std::min(n - 1, l + seg_size - 1);
        tg.run([&, l, r] { ShellSort(l, r, local_data); });
      }
      tg.wait();
    });

    std::vector<int> buf;
    int end = seg_size - 1;
    for (int i = 1; i < threads; ++i) {
      int r = std::min(n - 1, i * seg_size + seg_size - 1);
      Merge(0, end, r, local_data, buf);
      end = r;
    }
  }

  std::vector<std::vector<int>> gathered_data;
  boost::mpi::gather(world_, local_data, gathered_data, 0);

  if (world_.rank() == 0) {
    output_.clear();
    for (auto& chunk : gathered_data) {
      Merge(0, output_.size() - 1, chunk.size() - 1, output_, chunk);
    }
  }

  return true;
}

void ShellSort(int left, int right, std::vector<int>& arr) {
  int gap = 1;
  const int size = right - left + 1;
  while (gap <= size / 3) {
    gap = gap * 3 + 1;
  }

  for (; gap > 0; gap /= 3) {
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

  int i = left;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= right) {
    buffer[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
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

bool TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (std::size_t idx = 0; idx < output_.size(); ++idx) {
      reinterpret_cast<int*>(task_data->outputs[0])[idx] = output_[idx];
    }
  }
  return true;
}

}  // namespace shlyakov_m_shell_sort_all