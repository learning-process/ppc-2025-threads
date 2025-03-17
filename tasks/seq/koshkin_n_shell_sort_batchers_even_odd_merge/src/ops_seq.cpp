#include "seq/koshkin_n_shell_sort_batchers_even_odd_merge/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

std::vector<int> koshkin_n_shell_sort_batchers_even_odd_merge_seq::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-100000, 100000);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

void koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::swap(std::vector<int> &a, int i, int j,
                                                                                bool order) {
  if ((order && a[i] > a[j]) || (!order && a[i] < a[j])) {
    std::swap(a[i], a[j]);
  }
}

void koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::BatcherMerge(std::vector<int> &a, int start,
                                                                                        int end, bool order) {
  int distance = end - start;
  if (distance <= 1) return;

  int mid = start + distance / 2;

  BatcherMerge(a, start, mid, order);
  BatcherMerge(a, mid, end, order);

  // четно-нечетное слияние
  for (int i = start; i + 1 < end; i += 2) {
    swap(a, i, i + 1, order);
  }
  for (int i = start + 1; i + 1 < end; i += 2) {
    swap(a, i, i + 1, order);
  }
}

void koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::shellBatcherSort(std::vector<int> &a,
                                                                                            bool order) {
  int n = a.size();
  int gap = 1;

  // генерация шагов Кнута
  while (gap < n / 3) gap = 3 * gap + 1;

  // шаги Шелла
  while (gap > 0) {
    for (int i = gap; i < n; i++) {
      int temp = a[i];
      int j;
      for (j = i; j >= gap && ((order && a[j - gap] > temp) || (!order && a[j - gap] < temp)); j -= gap) {
        a[j] = a[j - gap];
      }
      a[j] = temp;
    }
    gap /= 3;
  }

  // четно-нечетное слияние Бэтчера
  BatcherMerge(a, 0, n, order);
}

bool koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  input_order = *reinterpret_cast<bool *>(task_data->inputs[1]);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0] && !task_data->inputs.empty() &&
         !task_data->outputs.empty();
}

bool koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::RunImpl() {
  shellBatcherSort(input_, input_order);
  output_ = input_;
  return true;
}

bool koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}