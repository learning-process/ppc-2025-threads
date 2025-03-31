#include "omp/example/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "omp/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_omp.hpp"

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::PreProcessingImpl() {
  // Init value for input and output
  int size = task_data->inputs_count[0];
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
    threadid_ = omp_get_thread_num();
#pragma omp single
    threadnum_ = omp_get_num_threads();
  }
  dimsize_ = int(log10(double(threadnum_)) / log10(2.0)) + 1;
}

int volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::GrayCode(int ringId, int dimSize) {
  if ((ringId == 0) && (dimSize == 1)) return 0;
  if ((ringId == 1) && (dimSize == 1)) return 1;
  int res;
  if (ringId < (1 << (dimSize - 1)))
    res = GrayCode(ringId, dimSize - 1);
  else
    res = (1 << (dimSize - 1)) + GrayCode((1 << dimSize) - 1 - ringId, dimSize - 1);
  return res;
}

int volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::ReverseGrayCode(int CubeID, int DimSize) {
  for (int i = 0; i < (1 << DimSize); i++) {
    if (CubeID == GrayCode(i, DimSize)) return i;
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::SetBlockPairs(int* BlockPairs, int Iter) {
  int pairNum = 0, firstValue, secondValue;
  bool exist;
  for (int i = 0; i < 2 * threadnum_; i++) {
    firstValue = GrayCode(i, dimsize_);
    exist = false;
    for (int j = 0; (j < pairNum) && (!exist); j++)
      if (BlockPairs[2 * j + 1] == firstValue) exist = true;
    if (!exist) {
      secondValue = firstValue ^ (1 << (dimsize_ - Iter - 1));
      BlockPairs[2 * pairNum] = firstValue;
      BlockPairs[2 * pairNum + 1] = secondValue;
      ++pairNum;
    }
  }
}

int volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::FindMyPair(int* BlockPairs, int ThreadID,
                                                                                      int Iter) {
  int BlockID = 0, id, res;
  for (int i = 0; i < threadnum_; i++) {
    BlockID = BlockPairs[2 * i];
    if (Iter == 0) id = BlockID % (1 << dimsize_ - Iter - 1);
    if ((Iter > 0) && (Iter < dimsize_ - 1))
      id = ((BlockID >> (dimsize_ - Iter)) << (dimsize_ - Iter - 1)) | (BlockID % (1 << (dimsize_ - Iter - 1)));
    if (Iter == dimsize_ - 1) id = BlockID >> 1;
    if (id == ThreadID) {
      res = i;
      break;
    }
  }
  return res;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::IsSorted(std::vector<int>& pData,
                                                                                     int size) {
  for (int i = 1; i < size; i++) {
    if (pData[i] < pData[i - 1]) return false;
  }
  return true;
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::ShellSort(std::vector<int>& arr, int start,
                                                                                      int finish) {
  int n = finish - start;
  int gap = n / 2;

  while (gap > 0) {
    for (int i = start + gap; i < finish; ++i) {
      int temp = arr[i];
      int j = i;
      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = temp;
    }
    gap /= 2;
    gap /= 2;
  }
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::MergeBlocks(std::vector<int>& pData,
                                                                                        int Index1, int BlockSize1,
                                                                                        int Index2, int BlockSize2) {
  int* pTempArray = new int[BlockSize1 + BlockSize2];
  int i1 = Index1, i2 = Index2, curr = 0;
  while ((i1 < Index1 + BlockSize1) && (i2 < Index2 + BlockSize2)) {
    if (pData[i1] < pData[i2])
      pTempArray[curr++] = pData[i1++];
    else {
      pTempArray[curr++] = pData[i2++];
    }
    while (i1 < Index1 + BlockSize1) pTempArray[curr++] = pData[i1++];
    while (i2 < Index2 + BlockSize2) pTempArray[curr++] = pData[i2++];
    for (int i = 0; i < BlockSize1 + BlockSize2; i++) pData[Index1 + i] = pTempArray[i];
  }
  delete[] pTempArray;
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_omp::ShellSortOMP::ParallelShellSort(std::vector<int>& pData,
                                                                                              int Size) {
  InitializeParallelSections();
  int* Index = new int[2 * threadnum_];
  int* BlockSize = new int[2 * threadnum_];
  int* BlockPairs = new int[2 * threadnum_];
  for (int i = 0; i < 2 * threadnum_; i++) {
    Index[i] = int((i * Size) / double(2 * threadnum_));
    if (i < 2 * threadnum_ - 1)
      BlockSize[i] = int(((i + 1) * Size) / double(2 * threadnum_)) - Index[i];
    else
      BlockSize[i] = Size - Index[i];
  }
#pragma omp parallel
  {
    int BlockID = ReverseGrayCode(threadnum_ + threadid_, dimsize_);
    ShellSort(pData, Index[BlockID], Index[BlockID] + BlockSize[BlockID] - 1);
    BlockID = ReverseGrayCode(threadid_, dimsize_);
    ShellSort(pData, Index[BlockID], Index[BlockID] + BlockSize[BlockID] - 1);
  }
  for (int Iter = 0; (Iter < dimsize_) && (!IsSorted(pData, Size)); Iter++) {
    SetBlockPairs(BlockPairs, Iter);
#pragma omp parallel
    {
      int MyPairNum = FindMyPair(BlockPairs, threadid_, Iter);
      int FirstBlock = ReverseGrayCode(BlockPairs[2 * MyPairNum], dimsize_);
      int SecondBlock = ReverseGrayCode(BlockPairs[2 * MyPairNum + 1], dimsize_);
      MergeBlocks(pData, Index[FirstBlock], BlockSize[FirstBlock], Index[SecondBlock], BlockSize[SecondBlock]);
    }
  }
  int Iter = 1;
  while (!IsSorted(pData, Size)) {
#pragma omp parallel
    {
      if (Iter % 2 == 0)
        MergeBlocks(pData, Index[2 * threadid_], BlockSize[2 * threadid_], Index[2 * threadid_ + 1],
                    BlockSize[2 * threadid_ + 1]);
      else {
        if (threadid_ < threadnum_ - 1)
          MergeBlocks(pData, Index[2 * threadid_ + 1], BlockSize[2 * threadid_ + 1], Index[2 * threadid_ + 2],
                      BlockSize[2 * threadid_ + 2]);
      }
    }
    Iter++;
  }
  delete[] Index;
  delete[] BlockSize;
  delete[] BlockPairs;
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