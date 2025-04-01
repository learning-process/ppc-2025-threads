#include "omp/example/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "omp/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_omp.hpp"

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::PreProcessingImpl() {
  // Init value for input and output
  unsigned int size = task_data->inputs_count[0];
  auto* input_pointer = reinterpret_cast<int*>(task_data->inputs[0]);
  array_ = std::vector<int>(input_pointer, input_pointer + size);

  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::InitializeParallelSections() {
#pragma omp parallel
  {
    thread_id_ = omp_get_thread_num();
#pragma omp single
    thread_num_ = omp_get_num_threads();
  }
  dim_size_ = int(log10(double(threadnum_)) / log10(2.0)) + 1;
}

int volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::GrayCode(int ring_id, int dim_size) {
  if ((ring_id == 0) && (dim_size == 1)) {
    return 0;
  }
  if ((ring_id == 1) && (dim_size == 1)) {
    return 1;
  }
  int res = 0;
  if (ring_id < (1 << (dim_size - 1))) {
    res = GrayCode(ring_id, dim_size - 1);
  } else {
    res = (1 << (dim_size - 1)) + GrayCode((1 << dimSize) - 1 - ring_id, dim_size - 1);
  }
  return res;
}

int volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::ReverseGrayCode(int cube_id, int dim_size) {
  int ans = 0;
  for (int i = 0; i < (1 << dim_size); i++) {
    if (cube_id == GrayCode(i, dim_size)) {
      ans = i;
      break;
    }
  }
  return ans;
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::SetBlockPairs(int* p_block_pairs,
                                                                                          int iter) {
  int pair_num = 0, first_value, second_value;
  bool exist = true;
  for (int i = 0; i < 2 * thread_num_; i++) {
    first_value = GrayCode(i, dim_size_);
    exist = false;
    for (int j = 0; (j < pair_num) && (!exist); j++)
      if (p_block_pairs[2 * j + 1] == first_value) exist = true;
    if (!exist) {
      second_value = first_value ^ (1 << (dim_size_ - iter - 1));
      p_block_pairs[2 * pair_num] = first_value;
      p_block_pairs[2 * pair_num + 1] = second_value;
      ++pair_num;
    }
  }
}

int volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::FindMyPair(int* p_block_pairs, int thread_id,
                                                                                      int iter) {
  int block_id = 0, id, res = 0;
  for (int i = 0; i < threadnum_; i++) {
    block_id = p_block_pairs[2 * i];
    if (iter == 0) id = block_id % (1 << (dim_size_ - iter - 1));
    if ((iter > 0) && (iter < dim_size_ - 1))
      id = ((block_id >> (dim_size_ - iter)) << (dim_size_ - iter - 1)) | (block_id % (1 << (dim_size_ - iter - 1)));
    if (iter == dim_size_ - 1) id = block_id >> 1;
    if (id == thread_id) {
      res = i;
      break;
    }
  }
  return res;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::IsSorted(std::vector<int>& p_data,
                                                                                     int size) {
  for (int i = 1; i < size; i++) {
    if (p_data[i] < p_data[i - 1]) return false;
  }
  return true;
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::ShellSort(std::vector<int>& p_arr,
                                                                                      int start, int finish) {
  int n = finish - start;
  int gap = n / 2;

  while (gap > 0) {
    for (int i = start + gap; i < finish; ++i) {
      int temp = p_arr[i];
      int j = i;
      while (j >= gap && p_arr[j - gap] > temp) {
        p_arr[j] = p_arr[j - gap];
        j -= gap;
      }
      p_arr[j] = temp;
    }
    gap /= 2;
    gap /= 2;
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::MergeBlocks(std::vector<int>& p_data,
                                                                                        int index_1, int block_size_1,
                                                                                        int index_2, int block_size_2) {
  int* p_temp_array = new int[block_size_1 + block_size_2];
  int i1 = index_1, i2 = index_2, curr = 0;
  while ((i1 < index_1 + block_size_1) && (i2 < index_2 + block_size_2)) {
    if (p_data[i1] < p_data[i2])
      p_temp_array[curr++] = p_data[i1++];
    else {
      p_temp_array[curr++] = p_data[i2++];
    }
    while (i1 < index_1 + block_size_1) p_temp_array[curr++] = p_data[i1++];
    while (i2 < index_2 + block_size_2) p_temp_array[curr++] = p_data[i2++];
    for (int i = 0; i < block_size_1 + block_size_2; i++) p_data[index_1 + i] = p_temp_array[i];
  }
  delete[] p_temp_array;
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::ParallelShellSort(std::vector<int>& p_data,
                                                                                              int size) {
  InitializeParallelSections();
  int* index = new int[2 * thread_num_];
  int* block_size = new int[2 * thread_num_];
  int* block_pairs = new int[2 * thread_num_];
  for (int i = 0; i < 2 * thread_num_; i++) {
    index[i] = int((i * size) / double(2 * thread_num_));
    if (i < 2 * thread_num_ - 1)
      block_size[i] = int(((i + 1) * size) / double(2 * thread_num_)) - index[i];
    else
      block_size[i] = size - index[i];
  }
#pragma omp parallel
  {
    int block_id = ReverseGrayCode(thread_num_ + thread_id_, dim_size_);
    ShellSort(p_data, index[block_id], index[block_id] + block_size[block_id] - 1);
    block_id = ReverseGrayCode(thread_id_, dim_size_);
    ShellSort(p_data, index[block_id], index[block_id] + block_size[block_id] - 1);
  }
  for (int iter = 0; (iter < dim_size_) && (!IsSorted(p_data, size)); iter++) {
    SetBlockPairs(block_pairs, iter);
#pragma omp parallel
    {
      int my_pair_num = FindMyPair(block_pairs, thread_id_, iter);
      int first_block = ReverseGrayCode(block_pairs[2 * my_pair_num], dim_size_);
      int second_block = ReverseGrayCode(block_pairs[2 * my_pair_num + 1], dim_size_);
      MergeBlocks(p_data, index[first_block], block_size[first_block], index[second_block], block_size[second_block]);
    }
  }
  int iter = 1;
  while (!IsSorted(p_data, size)) {
#pragma omp parallel
    {
      if (iter % 2 == 0)
        MergeBlocks(p_data, index[2 * thread_id_], block_size[2 * thread_id_], index[2 * thread_id_ + 1],
                    block_size[2 * thread_id_ + 1]);
      else {
        if (thread_id_ < thread_num_ - 1)
          MergeBlocks(p_data, index[2 * thread_id_ + 1], block_size[2 * thread_id_ + 1], index[2 * thread_id_ + 2],
                      block_size[2 * thread_id_ + 2]);
      }
    }
    Iter++;
  }
  delete[] index;
  delete[] block_size;
  delete[] block_pairs;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::RunImpl() {
  ParallelShellSort(array_, array_.size());
  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::PostProcessingImpl() {
  for (size_t i = 0; i < array_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = array_[i];
  }
  return true;
}