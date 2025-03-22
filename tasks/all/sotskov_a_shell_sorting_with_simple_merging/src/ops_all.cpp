#include "all/sotskov_a_shell_sorting_with_simple_merging/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

class ThreadPool {
 public:
  explicit ThreadPool(size_t num_threads) {
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) {
              return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    for (auto& worker : workers_) {
      worker.join();
    }
  }

  template <typename F>
  void Enqueue(F&& task) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      tasks_.emplace(std::forward<F>(task));
    }
    condition_.notify_one();
  }

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_{false};
};

void sotskov_a_shell_sorting_with_simple_merging_all::ShellSort(std::vector<int>& arr, size_t left, size_t right) {
  size_t array_size = right - left + 1;
  size_t gap = 1;
  while (gap < array_size / 3) {
    gap = gap * 3 + 1;
  }

  while (gap > 0) {
    for (size_t i = left + gap; i <= right; ++i) {
      int current_element = arr[i];
      size_t j = i;
      while (j >= left + gap && arr[j - gap] > current_element) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = current_element;
    }
    gap /= 3;
  }
}

void sotskov_a_shell_sorting_with_simple_merging_all::ParallelMerge(std::vector<int>& arr, size_t left, size_t mid,
                                                                    size_t right) {
  std::vector<int> temp(right - left + 1);
  size_t i = left;
  size_t j = mid + 1;
  size_t k = 0;

  while (i <= mid && j <= right) {
    temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];
  }
  while (i <= mid) {
    temp[k++] = arr[i++];
  }
  while (j <= right) {
    temp[k++] = arr[j++];
  }
  std::ranges::copy(temp, arr.begin() + static_cast<std::ptrdiff_t>(left));
}

void sotskov_a_shell_sorting_with_simple_merging_all::ShellSortWithSimpleMerging(std::vector<int>& arr) {
  int array_size = static_cast<int>(arr.size());
  int num_threads = std::min(ppc::util::GetPPCNumThreads(), static_cast<int>(std::thread::hardware_concurrency()));
  int chunk_size = std::max(1, (array_size + num_threads - 1) / num_threads);

  ThreadPool pool(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    int left = i * chunk_size;
    int right = std::min(left + chunk_size - 1, array_size - 1);
    if (left < right) {
      pool.Enqueue([&arr, left, right]() { ShellSort(arr, left, right); });
    }
  }

  pool.~ThreadPool();

  for (int size = chunk_size; size < array_size; size *= 2) {
    for (int i = 0; i < array_size; i += 2 * size) {
      int left = i;
      int mid = std::min(i + size - 1, array_size - 1);
      int right = std::min(i + (2 * size) - 1, array_size - 1);
      if (mid < right) {
        ParallelMerge(arr, left, mid, right);
      }
    }
  }
}

bool sotskov_a_shell_sorting_with_simple_merging_all::TestTaskALL::PreProcessingImpl() {
  if (rank_ == 0) {
    input_.resize(task_data->inputs_count[0]);
    auto* src = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(src, src + task_data->inputs_count[0], input_.begin());
  }

  const int total = static_cast<int>(task_data->inputs_count[0]);
  std::vector<int> counts(size_);
  std::vector<int> displs(size_);

  int base_size = total / size_;
  int remainder = total % size_;
  for (int i = 0; i < size_; ++i) {
    counts[i] = (i < remainder) ? base_size + 1 : base_size;
    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
  }

  int local_size = counts[rank_];
  std::vector<int> local(local_size);

  std::vector<int> dummy_input;
  if (rank_ != 0) {
    dummy_input.resize(local_size);
  }

  boost::mpi::scatterv(world_, (rank_ == 0) ? input_.data() : dummy_input.data(), counts, displs, local.data(),
                       counts[rank_], 0);

  input_ = std::move(local);
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_all::TestTaskALL::PostProcessingImpl() {
  const int total = static_cast<int>(task_data->inputs_count[0]);
  std::vector<int> counts(size_);
  std::vector<int> displs(size_);

  int base_size = total / size_;
  int remainder = total % size_;
  for (int i = 0; i < size_; ++i) {
    counts[i] = (i < remainder) ? base_size + 1 : base_size;
    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
  }

  std::vector<int> result;
  if (rank_ == 0) {
    if (total <= 0) {
      throw std::runtime_error("Invalid total size for vector resize");
    }
    result.assign(total, 0);
  }

  std::vector<int> dummy_output;
  if (rank_ != 0) {
    if (input_.size() <= 0) {
      throw std::runtime_error("Invalid input_ size");
    }
    dummy_output.resize(input_.size());
  }

  boost::mpi::gatherv(world_, input_.data(), static_cast<int>(input_.size()),
                      (rank_ == 0) ? result.data() : dummy_output.data(), counts, displs, 0);

  if (rank_ == 0) {
    auto* dst = reinterpret_cast<int*>(task_data->outputs[0]);
    std::ranges::move(result, dst);
  }
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_all::TestTaskALL::ValidationImpl() {
  if (rank_ != 0) {
    return true;
  }

  std::size_t input_size = task_data->inputs_count[0];
  std::size_t output_size = task_data->outputs_count[0];
  if (input_size != output_size) {
    return false;
  }

  for (std::size_t i = 1; i < output_size; ++i) {
    if (task_data->outputs[0][i] < task_data->outputs[0][i - 1]) {
      return false;
    }
  }
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_all::TestTaskALL::RunImpl() {
  ShellSortWithSimpleMerging(input_);
  return true;
}