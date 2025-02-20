#include "seq/kovalev_k_radix_sort_batcher_merge/include/header.hpp"

bool kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge::radix_unsigned(unsigned long long* inp_arr,
                                                                                   unsigned long long* mas_tmp) {
  unsigned char* masc = reinterpret_cast<unsigned char*>(inp_arr);
  int count[256];
  size_t sizetype = sizeof(unsigned long long), j, i;
  for (i = 0; i < sizetype; i++) {
    countbyte(inp_arr, count, i);
    for (j = 0; j < n_; j++) mas_tmp[count[masc[j * sizetype + i]]++] = inp_arr[j];
    memcpy(inp_arr, mas_tmp, sizeof(unsigned long long) * n_);
  }
  return true;
}

bool kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge::countbyte(unsigned long long* inp_arr, int* count,
                                                                              unsigned int byte) {
  unsigned char* masc = reinterpret_cast<unsigned char*>(inp_arr);
  unsigned int i, bias = sizeof(unsigned long long);
  int tmp1, tmp2;
  for (i = 0; i < 256; i++) count[i] = 0;
  for (i = 0; i < n_; i++) count[masc[i * bias + byte]]++;
  tmp1 = count[0];
  count[0] = 0;
  for (i = 1; i < 256; i++) {
    tmp2 = count[i];
    count[i] = count[i - 1] + tmp1;
    tmp1 = tmp2;
  }
  return true;
}

bool kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge::ValidationImpl() {
  return (task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == task_data->inputs_count[0]);
}

bool kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge::PreProcessingImpl() {
  n_ = task_data->inputs_count[0];
  mas_ = std::vector<long long int>(n_);
  tmp_ = std::vector<long long int>(n_);
  void* ptr_input = task_data->inputs[0];
  void* ptr_vec = mas_.data();
  memcpy(ptr_vec, ptr_input, sizeof(long long int) * n_);
  return true;
}

bool kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge::RunImpl() {
  unsigned int count = 0, i = 0;
  bool ret = radix_unsigned(reinterpret_cast<unsigned long long*>(mas_.data()),
                            reinterpret_cast<unsigned long long*>(tmp_.data()));
  while (count < n_ && mas_[count++] >= 0);
  if (count == n_) return ret;
  count--;
  for (; count < n_; count++) tmp_[i++] = mas_[count];
  count = 0;
  for (; i < n_; i++) tmp_[i] = mas_[count++];
  memcpy(mas_.data(), tmp_.data(), sizeof(long long int) * n_);
  return ret;
}

bool kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge::PostProcessingImpl() {
  memcpy(reinterpret_cast<long long int*>(task_data->outputs[0]), mas_.data(), sizeof(long long int) * n_);
  return true;
}