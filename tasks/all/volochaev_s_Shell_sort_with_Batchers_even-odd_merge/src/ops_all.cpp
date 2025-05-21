#include "all/volochaev_s_shell_sort_with_batchers_even-odd_merge_all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <future>
#include <ranges>
#include <thread>
#include <vector>

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::PreProcessingImpl() {
  MPI_Init(nullptr, nullptr);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  if (rank_ == 0) {
    size_ = static_cast<int>(task_data->inputs_count[0]);
    auto* input_pointer = reinterpret_cast<int*>(task_data->inputs[0]);
    array_ = std::vector<int>(input_pointer, input_pointer + size_);
  }

  MPI_Bcast(&size_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::ValidationImpl() {
  if (rank_ == 0) {
    return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::DistributeData() {
  int local_size = size_ / world_size_;
  int remainder = size_ % world_size_;

  std::vector<int> counts(world_size_, local_size);
  std::vector<int> displacements(world_size_, 0);

  for (int i = 0; i < remainder; ++i) {
    counts[i]++;
  }

  for (int i = 1; i < world_size_; ++i) {
    displacements[i] = displacements[i - 1] + counts[i - 1];
  }

  local_data_.resize(counts[rank_]);
  MPI_Scatterv(array_.data(), counts.data(), displacements.data(), MPI_INT, local_data_.data(), local_data_.size(),
               MPI_INT, 0, MPI_COMM_WORLD);
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::FindThreadVariables() {
  unsigned int max_threads = std::thread::hardware_concurrency();
  c_threads_ = static_cast<int>(std::pow(2, std::floor(std::log2(max_threads))));
  n_local_ = local_data_.size();
  mini_batch_ = n_local_ / c_threads_;

  if (mini_batch_ == 0) {
    mini_batch_ = n_local_;
    c_threads_ = 1;
  }

  mass_.resize(n_local_);
  std::ranges::copy(local_data_, mass_.begin());
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::ShellSort(int start) {
  int n = mini_batch_;
  int gap = 1;

  while (gap < n / 3) {
    gap = 3 * gap + 1;
  }

  while (gap >= 1) {
    for (int i = start + gap; i < start + mini_batch_; ++i) {
      int temp = mass_[i];
      int j = i;
      while (j >= start + gap && mass_[j - gap] > temp) {
        mass_[j] = mass_[j - gap];
        j -= gap;
      }
      mass_[j] = temp;
    }
    gap /= 3;
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::MergeBlocks(int id_l, int id_r, int len) {
  std::vector<int> temp(len * 2);
  int i = 0, j = 0, k = 0;

  while (i < len && j < len) {
    if (mass_[id_l + i] < mass_[id_r + j]) {
      temp[k++] = mass_[id_l + i++];
    } else {
      temp[k++] = mass_[id_r + j++];
    }
  }

  while (i < len) temp[k++] = mass_[id_l + i++];
  while (j < len) temp[k++] = mass_[id_r + j++];

  std::copy(temp.begin(), temp.end(), mass_.begin() + id_l);
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::MergeLocal() {
  int current_threads = c_threads_;

  while (current_threads > 1) {
    std::vector<std::future<void>> futures;
    int l = mini_batch_ * (c_threads_ / current_threads);

    for (int i = 0; i < current_threads / 2; ++i) {
      futures.emplace_back(
          std::async(std::launch::async, [this, i, l]() { MergeBlocks((i * 2 * l), (i * 2 * l) + l, l); }));

      futures.emplace_back(
          std::async(std::launch::async, [this, i, l]() { MergeBlocks((i * 2 * l) + 1, (i * 2 * l) + l + 1, l - 1); }));
    }

    for (auto& future : futures) {
      future.get();
    }

    current_threads /= 2;
  }

  std::ranges::copy(mass_, local_data_.begin());
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::ParallelShellSortLocal() {
  FindThreadVariables();

  std::vector<std::future<void>> futures;
  futures.reserve(c_threads_);

  for (int i = 0; i < c_threads_; ++i) {
    futures.emplace_back(std::async(std::launch::async, [this, i]() { ShellSort(i * mini_batch_); }));
  }

  for (auto& future : futures) {
    future.get();
  }

  MergeLocal();
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::LastMerge() {
  std::vector<int> temp(mass_.size());
  size_t i = 0, j = 1, k = 0;
  size_t n = mass_.size();

  while (i < n && j < n) {
    if (mass_[i] < mass_[j]) {
      temp[k++] = mass_[i];
      i += 2;
    } else {
      temp[k++] = mass_[j];
      j += 2;
    }
  }

  while (i < n) {
    temp[k++] = mass_[i];
    i += 2;
  }
  while (j < n) {
    temp[k++] = mass_[j];
    j += 2;
  }

  mass_ = std::move(temp);
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::GatherAndMerge() {
  std::vector<int> all_sizes(world_size_);
  int local_size = local_data_.size();
  MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> displacements(world_size_, 0);
  for (int i = 1; i < world_size_; ++i) {
    displacements[i] = displacements[i - 1] + all_sizes[i - 1];
  }

  if (rank_ == 0) {
    array_.resize(size_);
  }

  MPI_Gatherv(local_data_.data(), local_size, MPI_INT, array_.data(), all_sizes.data(), displacements.data(), MPI_INT,
              0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    n_ = size_;
    mass_ = array_;
    LastMerge();
    array_ = mass_;
  }
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::RunImpl() {
  DistributeData();
  ParallelShellSortLocal();
  GatherAndMerge();
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_all::ShellSortALL::PostProcessingImpl() {
  if (rank_ == 0) {
    int* ptr_ans = reinterpret_cast<int*>(task_data->outputs[0]);
    std::ranges::copy(array_ | std::views::take(size_), ptr_ans);
  }

  MPI_Finalize();
  return true;
}
