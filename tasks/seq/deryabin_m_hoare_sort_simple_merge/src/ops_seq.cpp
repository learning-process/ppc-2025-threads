#include "seq/deryabin_m_cannons_algorithm/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::PreProcessingImpl() {
  input_array_A_ = std::vector<double>(task_data->inputs_count[0]);
  auto* tmp_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  std::copy(tmp_ptr_a, tmp_ptr_a + task_data->inputs_count[0], input_matrix_A_.begin());
  dimension_ = input_array_A_.size();
  chunk_count_ = reinterpret_cast<size_type>(taskData->inputs[1]);
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] > 2 &&
         task_data->inputs[1] >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::RunImpl() {
  unsigned short i = 0;
  unsigned short j = 0;
  unsigned short count = 0;
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
