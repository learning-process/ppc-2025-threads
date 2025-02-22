#include "seq/deryabin_m_hoare_sort_simple_merge/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<double**>(task_data->outputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  remainder_ = dimension_ % chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

void deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::HoaraSort(std::vector<double>& a, size_t first, size_t last) {
  size_t i = first;
  size_t j = last;
  double tmp = 0;
  double x =
      std::max(std::min(a[first], a[(first + last) / 2]),
               std::min(std::max(a[first], a[(first + last) / 2]),
                        a[last]));  // выбор опорного элемента как медианы первого, среднего и последнего элементов
  do {
    while (a[i] < x) {
      i++;
    }
    while (a[j] > x) {
      j--;
    }
    if (i <= j) {
      if (i < j) {
        tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
      }
      i++;
      j--;
    }
  } while (i <= j);
  if (i < last) {
    HoaraSort(a, i, last);
  }
  if (first < j) {
    HoaraSort(a, first, j);
  }
}

void deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::MergeTwoParts(std::vector<double>& a, size_t left,
                                                                                    size_t right) {
  size_t middle = left + ((right - left) / 2);
  size_t l_cur = left;
  size_t r_cur = middle + 1;
  double* l_buff{new double[right - left + 1]{}};
  double* r_buff{new double[right - left + 1]{}};
  std::copy(a.begin() + left, a.begin() + r_cur, l_buff);
  std::copy(a.begin() + r_cur, a.begin() + right + 1, r_buff + r_cur);
  for (size_t i = left; i <= right; i++) {
    if (l_cur <= middle && r_cur <= right) {
      if (l_buff[l_cur] < r_buff[r_cur]) {
        a[i] = l_buff[l_cur];
        l_cur++;
      } else {
        a[i] = r_buff[r_cur];
        r_cur++;
      }
    } else if (l_cur <= middle) {
      a[i] = l_buff[l_cur];
      l_cur++;
    } else {
      a[i] = r_buff[r_cur];
      r_cur++;
    }
  }
  chunk_count_--;
}

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::RunImpl() {
  size_t count = 0;
  while (count != chunk_count_) {
    if (count < chunk_count_ - 1) {
      HoaraSort(input_array_A_, count * min_chunk_size_, ((count + 1) * min_chunk_size_) - 1);
    } else {
      HoaraSort(input_array_A_, count * min_chunk_size_, ((count + 1) * min_chunk_size_) - 1 + remainder_);
    }
    count++;
  }
  for (size_t i = 0; i < (size_t)(log((double)chunk_count_) / std::numbers::ln2); i++) {
    for (size_t j = 0; j < chunk_count_; j++) {
      if (j == 0) {
        if (chunk_count_ % 2 != 0) {
          MergeTwoParts(input_array_A_, dimension_ - 1 - (2 * min_chunk_size_ * (i + 1)) - remainder_, dimension_ - 1);
          j--;
        }
        if (i == (size_t)(log((double)chunk_count_) / std::numbers::ln2) - 1) {
          MergeTwoParts(input_array_A_, 0, dimension_ - 1);
        } else {
          MergeTwoParts(input_array_A_, 0, (2 * min_chunk_size_ * (i + 1)) - 1);
        }
      } else {
        if (chunk_count_ - j == 2) {
          MergeTwoParts(input_array_A_, min_chunk_size_ * (i + 1) * (j + 1), dimension_ - 1);
        } else {
          MergeTwoParts(input_array_A_, min_chunk_size_ * (i + 1) * (j + 1), (min_chunk_size_ * (i + 1) * (j + 3)) - 1);
        }
      }
    }
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double**>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}
