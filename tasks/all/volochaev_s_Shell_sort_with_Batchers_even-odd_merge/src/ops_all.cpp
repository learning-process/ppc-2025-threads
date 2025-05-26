#include <algorithm>
#include <all/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_all.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_all {

bool ShellSortALL::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs.empty() || task_data->outputs.empty() || task_data->inputs_count[0] <= 0 ||
        task_data->outputs_count[0] != n_input_ || task_data->inputs_count[0] != task_data->outputs_count[0]) {
      return false;
    }
  }
  return true;
}

bool ShellSortALL::PreProcessingImpl() {
  boost::mpi::broadcast(world_, n_input_, 0);

  effective_num_procs_ = static_cast<int>(std::pow(2, std::floor(std::log2(world_.size()))));
  auto e_n_f = static_cast<unsigned int>(effective_num_procs_);
  n_ = n_input_ + (((2 * e_n_f) - n_input_ % (2 * e_n_f))) % (2 * e_n_f);
  loc_proc_lenght_ = n_ / effective_num_procs_;

  if (world_.rank() == 0) {
    mas_.resize(n_);
    memcpy(mas_.data(), task_data->inputs[0], sizeof(long long int) * n_input_);

    for (unsigned int i = n_input_; i < n_; ++i) {
      mas_[i] = LLONG_MAX;
    }
  }

  loc_.resize(loc_proc_lenght_);
  loc_tmp_.resize(loc_proc_lenght_);

  return true;
}

bool ShellSortALL::RunImpl() {
  boost::mpi::scatter(world_, mas_.data(), static_cast<int>(loc_proc_lenght_), loc_.data(), 0);

  BatcherSort();

  for (unsigned int i = effective_num_procs_; i > 1; i /= 2) {
    if (world_.rank() < static_cast<int>(i)) {
      unsigned int len = loc_proc_lenght_ * (effective_num_procs_ / i);

      OddEvenMergeMPI(len);

      if (world_.rank() > 0 && world_.rank() % 2 == 0) {
        world_.send(world_.rank() / 2, 0, loc_tmp_.data(), 2 * static_cast<int>(len));
      }
      if (world_.rank() > 0 && world_.rank() < static_cast<int>(i) / 2) {
        loc_.resize(2 * len);
        world_.recv(world_.rank() * 2, 0, loc_.data(), 2 * static_cast<int>(len));
      } else if (world_.rank() == 0 && i != 2) {
        std::copy(loc_tmp_.begin(), loc_tmp_.end(), loc_.begin());
      }
    }
  }

  return true;
}

bool ShellSortALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    memcpy(task_data->outputs[0], loc_tmp_.data(), sizeof(long long int) * n_input_);
  }
  return true;
}

// === Shell Sort реализация ===
void ShellSortALL::ShellSort(std::vector<long long>& arr, size_t start, size_t size) {
  std::vector<size_t> gaps = generate_gaps(size);

  for (auto h : gaps) {
    for (size_t i = h; i < size; ++i) {
      long long temp = arr[start + i];
      size_t j = i;
      while (j >= h && arr[start + j - h] > temp) {
        arr[start + j] = arr[start + j - h];
        j -= h;
      }
      arr[start + j] = temp;
    }
  }
}

// Генерация шагов по Седжуику
std::vector<size_t> ShellSortALL::generate_gaps(size_t size) {
  std::vector<size_t> gaps;
  size_t h = 1;
  while (h < size) {
    gaps.push_back(h);
    h = 3 * h + 1;
  }
  std::reverse(gaps.begin(), gaps.end());
  return gaps;
}

bool ShellSortALL::BatcherSort() {
  if (ppc::util::GetPPCNumThreads() > 2 * loc_proc_lenght_) {
    ShellSort(loc_, 0, loc_proc_lenght_);
    loc_tmp_ = loc_;
    return true;
  }

  unsigned int effective_num_threads =
      static_cast<int>(std::pow(2, std::floor(std::log2(ppc::util::GetPPCNumThreads()))));
  unsigned int n_by_proc =
      loc_proc_lenght_ +
      (((2 * effective_num_threads) - (loc_proc_lenght_ % (2 * effective_num_threads))) % (2 * effective_num_threads));
  unsigned int loc_length = n_by_proc / effective_num_threads;

  loc_.resize(n_by_proc, LLONG_MAX);
  loc_tmp_.resize(n_by_proc);

  // Параллельно сортируем каждую часть
  std::vector<std::thread> threads;
  for (unsigned int tid = 0; tid < effective_num_threads; ++tid) {
    threads.emplace_back([this, tid, loc_length]() { this->ShellSort(loc_, tid * loc_length, loc_length); });
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  // Merge step
  for (unsigned int i = effective_num_procs_; i > 1; i /= 2) {
    std::vector<std::thread> merge_threads;
    for (int tid = 0; tid < static_cast<int>(i); ++tid) {
      merge_threads.emplace_back([this, tid, i, loc_length]() {
        auto stride = static_cast<unsigned int>(tid / 2);
        unsigned int bias = tid % 2;
        unsigned int len = loc_length * (effective_num_procs_ / i);

        std::vector<long long> left_part(loc_.begin() + (stride * 2 * len) + bias,
                                         loc_.begin() + (stride * 2 * len) + len + bias);
        std::vector<long long> right_part(loc_.begin() + (stride * 2 * len) + len + bias,
                                          loc_.begin() + (stride * 2 * len) + 2 * len + bias);
        std::vector<long long> merged(len);

        std::merge(left_part.begin(), left_part.end(), right_part.begin(), right_part.end(), merged.begin());

        std::copy(merged.begin(), merged.end(), loc_tmp_.begin() + (stride * 2 * len) + bias);
      });
    }

    for (auto& t : merge_threads) {
      if (t.joinable()) t.join();
    }

    std::swap(loc_, loc_tmp_);
  }

  FinalMergeSTL(loc_, loc_tmp_);
  std::copy(loc_tmp_.begin(), loc_tmp_.end(), loc_.begin());

  return true;
}

bool ShellSortALL::OddEvenMergeSTL(std::vector<long long>& tmp, const std::vector<long long>& left,
                                   const std::vector<long long>& right) {
  size_t iter_tmp = 0, iter_l = 0, iter_r = 0;
  size_t len = std::min(left.size(), right.size());

  while (iter_l < len && iter_r < len) {
    if (left[iter_l] < right[iter_r]) {
      tmp[iter_tmp] = left[iter_l];
      iter_l += 2;
    } else {
      tmp[iter_tmp] = right[iter_r];
      iter_r += 2;
    }
    iter_tmp += 2;
  }

  while (iter_l < len) {
    tmp[iter_tmp] = left[iter_l];
    iter_l += 2;
    iter_tmp += 2;
  }

  while (iter_r < len) {
    tmp[iter_tmp] = right[iter_r];
    iter_r += 2;
    iter_tmp += 2;
  }

  for (size_t i = 0; i < tmp.size(); i += 2) {
    const_cast<long long*>(left.data())[i] = tmp[i];
  }

  return true;
}

// === FinalMergeOMP ===
bool ShellSortALL::FinalMergeSTL(std::vector<long long>& loc, std::vector<long long>& loc_tmp) {
  size_t n = loc.size();
  size_t iter_even = 0, iter_odd = 1, iter_tmp = 0;

  while (iter_even < n && iter_odd < n) {
    if (loc[iter_even] < loc[iter_odd]) {
      loc_tmp[iter_tmp++] = loc[iter_even];
      iter_even += 2;
    } else {
      loc_tmp[iter_tmp++] = loc[iter_odd];
      iter_odd += 2;
    }
  }

  while (iter_even < n) {
    loc_tmp[iter_tmp++] = loc[iter_even];
    iter_even += 2;
  }

  while (iter_odd < n) {
    loc_tmp[iter_tmp++] = loc[iter_odd];
    iter_odd += 2;
  }

  std::copy(loc_tmp.begin(), loc_tmp.end(), loc.begin());

  return true;
}

bool ShellSortALL::OddEvenMergeMPI(unsigned int len) {
  if (world_.rank() % 2 == 0) {
    loc_.resize(2 * len);
    loc_tmp_.resize(2 * len);

    world_.recv(world_.rank() + 1, 0, loc_.data() + len, static_cast<int>(len));

    size_t iter_l = 0, iter_r = 0, iter_tmp = 0;

    while (iter_l < len && iter_r < len) {
      if (loc_[iter_l] < loc_[len + iter_r]) {
        loc_tmp_[iter_tmp++] = loc_[iter_l++];
      } else {
        loc_tmp_[iter_tmp++] = loc_[len + iter_r++];
      }
    }

    while (iter_l < len) loc_tmp_[iter_tmp++] = loc_[iter_l++];
    while (iter_r < len) loc_tmp_[iter_tmp++] = loc_[len + iter_r++];

    std::copy(loc_tmp_.begin(), loc_tmp_.end(), loc_.begin());
  } else {
    world_.send(world_.rank() - 1, 0, loc_.data(), static_cast<int>(len));
  }

  return true;
}

}  // namespace volochaev_s_shell_sort_with_batchers_even_odd_merge_all