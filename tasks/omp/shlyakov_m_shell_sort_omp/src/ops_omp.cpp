#include "omp/shlyakov_m_shell_sort_omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool shlyakov_m_shell_sort_omp::TestTaskOpenMP::PreProcessingImpl() {
  std::size_t input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  output_ = input_;

  return true;
}

bool shlyakov_m_shell_sort_omp::TestTaskOpenMP::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool shlyakov_m_shell_sort_omp::TestTaskOpenMP::RunImpl() {
  int array_size = static_cast<int>(input_.size());
  int num_threads = omp_get_max_threads();
  int chunk_size = (array_size + num_threads - 1) / num_threads;

#pragma omp parallel
  {
    std::vector<int> temp_buffer;

#pragma omp for schedule(dynamic)
    for (int i = 0; i < num_threads; ++i) {
      int left = i * chunk_size;
      int right = std::min(left + chunk_size - 1, array_size - 1);

      if (left < right) {
        int sub_array_size = right - left + 1;
        int gap = 1;

        while (gap <= sub_array_size / 3) {
          gap = gap * 3 + 1;
        }

        while (gap > 0) {
          for (int k = left + gap; k <= right; ++k) {
            int current_element = input_[k];
            int j = k;

            while (j >= left + gap && input_[j - gap] > current_element) {
              input_[j] = input_[j - gap];
              j -= gap;
            }
            input_[j] = current_element;
          }
          gap /= 3;
        }
      }
    }

    for (int size = chunk_size; size < array_size; size *= 2) {
#pragma omp for schedule(dynamic)
      for (int left = 0; left < array_size; left += 2 * size) {
        int mid = std::min(left + size - 1, array_size - 1);
        int right = std::min(left + 2 * size - 1, array_size - 1);

        if (mid < right) {
          int i = left;
          int j = mid + 1;
          int k = 0;
          const int merge_size = right - left + 1;

          if (temp_buffer.size() < static_cast<std::size_t>(merge_size)) {
            temp_buffer.resize(static_cast<std::size_t>(merge_size));
          }

          while (i <= mid && j <= right) {
            temp_buffer[k++] = (input_[i] <= input_[j]) ? input_[i++] : input_[j++];
          }

          while (i <= mid) {
            temp_buffer[k++] = input_[i++];
          }

          while (j <= right) {
            temp_buffer[k++] = input_[j++];
          }

          std::ranges::copy(temp_buffer.begin(), temp_buffer.begin() + k, input_.begin() + left);
        }
      }
    }
  }

  output_ = input_;
  return true;
}
bool shlyakov_m_shell_sort_omp::TestTaskOpenMP::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
// namespace shlyakov_m_shell_sort_omp