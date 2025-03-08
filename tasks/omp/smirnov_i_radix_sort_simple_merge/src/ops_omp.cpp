#include "omp/smirnov_i_radix_sort_simple_merge/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

std::vector<int> smirnov_i_radix_sort_simple_merge_omp::TestTaskOpenMP::merge(std::vector<int> mas1,
                                                                              std::vector<int> mas2) {
  std::vector<int> res;
  int p1 = 0, p2 = 0;
  while (static_cast<int>(mas1.size()) != p1 && static_cast<int>(mas2.size()) != p2) {
    if (mas1[p1] < mas2[p2]) {
      res.push_back(mas1[p1]);
      p1++;
    } else if (mas2[p2] < mas1[p1]) {
      res.push_back(mas2[p2]);
      p2++;
    } else {
      res.push_back(mas1[p1]);
      res.push_back(mas2[p2]);
      p1++;
      p2++;
    }
  }
  while (static_cast<int>(mas1.size()) != p1) {
    res.push_back(mas1[p1]);
    p1++;
  }
  while (static_cast<int>(mas2.size()) != p2) {
    res.push_back(mas2[p2]);
    p2++;
  }
  return res;
}
void smirnov_i_radix_sort_simple_merge_omp::TestTaskOpenMP::radix_sort(std::vector<int>& mas) {
  int longest = *std::ranges::max_element(mas.begin(), mas.end());
  int len = std::ceil(std::log10(longest + 1));
  std::vector<int> sorting(mas.size());
  int base = 1;
  for (int j = 0; j < len; j++, base *= 10) {
    std::vector<int> counting(10, 0);
    for (size_t i = 0; i < mas.size(); i++) {
      counting[mas[i] / base % 10]++;
    }
    std::partial_sum(counting.begin(), counting.end(), counting.begin());
    for (int i = static_cast<int>(mas.size() - 1); i >= 0; i--) {
      int pos = counting[mas[i] / base % 10] - 1;
      sorting[pos] = mas[i];
      counting[mas[i] / base % 10]--;
    }
    std::swap(mas, sorting);
  }
}
bool smirnov_i_radix_sort_simple_merge_omp::TestTaskOpenMP::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  mas_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);
  printf("222\n");
  fflush(stdout);
  return true;
}

bool smirnov_i_radix_sort_simple_merge_omp::TestTaskOpenMP::ValidationImpl() {
  printf("111\n");
  fflush(stdout);
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}
bool smirnov_i_radix_sort_simple_merge_omp::TestTaskOpenMP::RunImpl() {
  std::vector<std::vector<int>> A, B;
  std::vector<int> sort_res;
#pragma omp parallel
  {
    printf("fff\n");
    fflush(stdout);
    int num = omp_get_thread_num();
    int all = omp_get_num_threads();
    std::vector<int> local_mas;
    int start = num * mas_.size() / all;
    int end = std::min((num + 1) * mas_.size() / all, mas_.size());
    for (int i = start; i < end; i++) {
      local_mas.push_back(mas_[i]);
    }
    radix_sort(local_mas);
    if (!local_mas.empty()) {
      printf("first %d %ld %d \n", num, local_mas.size(), local_mas[0]);
      fflush(stdout);
    }
#pragma omp critical
    {
      if (!local_mas.empty()) A.push_back(local_mas);
    }

#pragma omp barrier
    while (static_cast<int>(A.size()) != 1) {
      std::vector<int> mas1{}, mas2{}, merge_mas{};
      printf("ssss %d\n", num);
      fflush(stdout);
#pragma omp critical
      {
        if (static_cast<int>(A.size()) >= 2) {
          mas1 = A[0];
          A.erase(A.begin());
          mas2 = A[0];
          A.erase(A.begin());
        }
      }
      if (!mas1.empty() && !mas2.empty()) {
        merge_mas = merge(mas1, mas2);
      }

      if (!merge_mas.empty()) {
#pragma omp critical
        B.push_back(merge_mas);
      }
#pragma omp critical
      {
        if (static_cast<int>(A.size()) == 1) {
          B.push_back(A[0]);
          A.erase(A.begin());
        }
      }
#pragma omp barrier
#pragma omp single
      { std::swap(A, B); }
#pragma omp barrier
    }
#pragma omp single
    {
      printf("ttt3 %d %ld\n", num, A.size());
      fflush(stdout);
      if (static_cast<int>(A.size()) == 1) {
        sort_res.resize(A[0].size());
        std::copy(A[0].begin(), A[0].end(), sort_res.begin());
      }
    }
    printf("second %d %ld\n", num, sort_res.size());
    fflush(stdout);
#pragma omp single
    {
      output_ = sort_res;
      printf("third %d %ld\n", num, sort_res.size());
      fflush(stdout);
      printf("fourth %d %ld\n", num, output_.size());
      fflush(stdout);
    }
  }
  printf("fifth %ld\n", output_.size());
  fflush(stdout);
  return true;
}
bool smirnov_i_radix_sort_simple_merge_omp::TestTaskOpenMP::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}