#include "seq/deryabin_m_cannons_algorithm/include/ops_seq.hpp"

#include <algorithm>

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<double*>(task_data->inputs[0]);
  dimension_ = sizeof(input_array_A_) / sizeof(input_array_A_[0]);
  chunk_count_ = reinterpret_cast<size_type>(taskData->inputs[1]);
  min_chunk_size_ = dimension_ / chunk_count_;
  remainder_ = dimension_ % chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] > 2 &&
         task_data->inputs[1] >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

void deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::HoaraSort(double* a, size_t first, size_t last) {
    size_t i = first;
    size_t j = last;
    double tmp;
    double x = std::max(std::min(a[first], a[(first + last) / 2]), std::min(std::max(a[first], a[(first + last) / 2]), a[last])); // выбор опорного элемента как медианы первого, среднего и последнего элементов
    do {
      while (a[i] < x) i++;
      while (a[j] > x) j--;
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
    if (i < last) HoaraSort(a, i, last);
    if (first < j) HoaraSort(a, first, j);
}

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::RunImpl() {
  size_t count_ = 0;
  while (count_ != chunk_count_) {
    if (count_ < chunk_count_ - 1) {
      HoaraSort(input_array_A_, count_ * min_chunk_size_, ((count_ + 1) * min_chunk_size_) - 1);
    } else {
      HoaraSort(input_array_A_, count_ * min_chunk_size_, ((count_ + 1) * min_chunk_size_) - 1 + remainder_);
    }
    count_++;
  }

  
  auto dimension = (unsigned short)sqrt(static_cast<unsigned short>(input_matrix_A_.size()));
  while (i != dimension) {
    j = 0;
    while (j != dimension) {
      count = 0;
      while (count != dimension) {
        output_matrix_C_[(i * dimension) + j] +=
            input_matrix_A_[(i * dimension) + count] * input_matrix_B_[(count * dimension) + j];
        count++;
      }
      j++;
    }
    i++;
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = output_matrix_C_;
  return true;
}
