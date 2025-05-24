#include "all/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <future>
#include <limits>
#include <ranges>
#include <thread>
#include <utility>
#include <vector>

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == task_data->outputs_count[0];
  }

  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::PreProcessingImpl() {
  rank_ = world_.rank();

  if (rank_ == 0) {
    sizes.resize(world_.size(), 0);
    auto* input_pointer = reinterpret_cast<int*>(task_data->inputs[0]);
    size_ = static_cast<int>(task_data->inputs_count[0]);
    sizes[0] = size_;
    array_mpi_ = std::vector<int>(input_pointer, input_pointer + size_);
  }

  return true;
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::ShellSort(int start) {
  int n = mini_batch_stl_;

  int gap = 1;
  while (gap < n / 3) {
    gap = 3 * gap + 1;
  }

  while (gap >= 1) {
    for (int i = start + gap; i < start + mini_batch_stl_; ++i) {
      int temp = mass_stl_[i];
      int j = i;
      while (j >= start + gap && mass_stl_[j - gap] > temp) {
        mass_stl_[j] = mass_stl_[j - gap];
        j -= gap;
      }
      mass_stl_[j] = temp;
    }
    gap /= 3;
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::MergeBlocksSTL(int id_l, int id_r,
                                                                                           int len) {
  int left_id = 0;
  int right_id = 0;
  int merged_id = 0;

  while (left_id < len || right_id < len) {
    if (left_id < len && right_id < len) {
      if (mass_stl_[id_l + left_id] < mass_stl_[id_r + right_id]) {
        array_stl_[id_l + merged_id] = mass_stl_[id_l + left_id];
        left_id += 2;
      } else {
        array_stl_[id_l + merged_id] = mass_stl_[id_r + right_id];
        right_id += 2;
      }
    } else if (left_id < len) {
      array_stl_[id_l + merged_id] = mass_stl_[id_l + left_id];
      left_id += 2;
    } else {
      array_stl_[id_l + merged_id] = mass_stl_[id_r + right_id];
      right_id += 2;
    }
    merged_id += 2;
  }

  for (int i = 0; i < merged_id; i += 2) {
    mass_stl_[id_l + i] = array_stl_[id_l + i];
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::LastMergeSTL() {
  int even_index = 0;
  int odd_index = 1;
  int result_index = 0;

  while (even_index < n_stl_ || odd_index < n_) {
    if (even_index < n_ && odd_index < n_) {
      if (mass_stl_[even_index] < mass_stl_[odd_index]) {
        array_stl_[result_index++] = mass_stl_[even_index];
        even_index += 2;
      } else {
        array_stl_[result_index++] = mass_stl_[odd_index];
        odd_index += 2;
      }
    } else if (even_index < n_) {
      array_stl_[result_index++] = mass_stl_[even_index];
      even_index += 2;
    } else {
      array_stl_[result_index++] = mass_stl_[odd_index];
      odd_index += 2;
    }
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::MergeSTL() {
  int current_threads = c_threads_stl_;

  while (current_threads > 1) {
    std::vector<std::future<void>> futures;
    int l = mini_batch_stl_ * (c_threads_stl_ / current_threads);

    for (int i = 0; i < current_threads / 2; ++i) {
      futures.emplace_back(
          std::async(std::launch::async, [this, i, l]() { MergeBlocksSTL((i * 2 * l), (i * 2 * l) + l, l); }));

      futures.emplace_back(std::async(std::launch::async,
                                      [this, i, l]() { MergeBlocksSTL((i * 2 * l) + 1, (i * 2 * l) + l + 1, l - 1); }));
    }

    for (auto& future : futures) {
      future.get();
    }

    current_threads /= 2;
  }

  LastMergeSTL();
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::ParallelShellSort() {
  FindThreadVariablesSTL();

  std::vector<std::future<void>> futures;
  futures.reserve(c_threads_stl_);

  for (int i = 0; i < c_threads_stl_; ++i) {
    futures.emplace_back(std::async(std::launch::async, [this, i]() { ShellSort(i * mini_batch_stl_); }));
  }

  for (auto& future : futures) {
    future.get();
  }

  MergeSTL();
}

std::vector<int> volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::Merge(std::vector<int>& v1,
                                                                                              std::vector<int>& v2) {
  size_t id1 = 0;
  size_t id2 = 0;
  std::vector<int> ans;
  while (id1 < v1.size() && id2 < v2.size()) {
    if (v1[id1] < v2[id2]) {
      ans.push_back(v1[id1]);
      ++id1;
    } else {
      ans.push_back(v2[id2]);
      ++id2;
    }
  }

  while (id1 < v1.size()) {
    ans.push_back(v1[id1]);
    ++id1;
  }

  while (id2 < v2.size()) {
    ans.push_back(v2[id2]);
    ++id2;
  }

  return ans;
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::FindThreadVariablesSTL() {
  auto max_threads =
      std::min(static_cast<unsigned int>(ppc::util::GetPPCNumThreads()), std::thread::hardware_concurrency());
  c_threads_stl_ = static_cast<int>(std::pow(2, std::floor(std::log2(max_threads))));
  n_stl_ = mini_batch_mpi_ + (((2 * c_threads_stl_) - mini_batch_mpi_ % (2 * c_threads_stl_))) % (2 * c_threads_stl_);
  mass_stl_.resize(n_stl_, std::numeric_limits<int>::max());
  mini_batch_stl_ = n_stl_ / c_threads_stl_;
  std::ranges::copy(array_stl_ | std::views::take(mini_batch_stl_), mass_stl_.begin());
  array_stl_.resize(n_stl_);
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::DistributeData() {
  if (rank_ == 0) {
    c_threads_mpi_ = static_cast<int>(std::pow(2, std::floor(std::log2(world_size_))));
    n_mpi_ = size_ + (((2 * c_threads_mpi_) - size_ % (2 * c_threads_mpi_))) % (2 * c_threads_mpi_);
    mass_mpi_.resize(n_mpi_, std::numeric_limits<int>::max());
    mini_batch_mpi_ = n_mpi_ / c_threads_mpi_;
    std::ranges::copy(array_mpi_ | std::views::take(mini_batch_mpi_), mass_mpi_.begin());
    array_mpi_.resize(n_mpi_);
  }

  broadcast(world_, sizes, 0);

  array_stl_.resize(sizes[world_.rank()]);
  scatterv(world_, mass_mpi_.data(), sizes, array_stl_.data(), 0);
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::GatherAndMerge() {
  if (world_.rank() == 0) {
    array_mpi_ = array_stl_;
    std::vector<int> data;

    for (int i = 1; i < world_size_; ++i) {
      world_.recv(i, 0, data);
      array_mpi_ = Merge(array_mpi_, data);
    }
  } else {
    world_.send(0, 0, array_stl_);
  }
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::RunImpl() {
  DistributeData();
  ParallelShellSort();
  GatherAndMerge();
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::PostProcessingImpl() {
  int* ptr_ans = reinterpret_cast<int*>(task_data->outputs[0]);

  std::ranges::copy(array_mpi_ | std::views::take(size_), ptr_ans);
  return true;
}