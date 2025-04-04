#include "tbb/lavrentiev_A_CCS_TBB/include/ops_tbb.hpp"

#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/parallel_for.h"

lavrentiev_a_ccs_tbb::Sparse lavrentiev_a_ccs_tbb::CCSTBB::ConvertToSparse(std::pair<int, int> size,
                                                                           const std::vector<double> &values) {
  auto [nsize, elements, rows, columns_sum] = Sparse();
  columns_sum.resize(size.second);
  for (int i = 0; i < size.second; ++i) {
    for (int j = 0; j < size.first; ++j) {
      if (values[i + (size.second * j)] != 0) {
        elements.emplace_back(values[i + (size.second * j)]);
        rows.emplace_back(j);
        columns_sum[i] += 1;
      }
    }
    if (i != size.second - 1) {
      columns_sum[i + 1] = columns_sum[i];
    }
  }
  return {.size = size, .elements = elements, .rows = rows, .columnsSum = columns_sum};
}

lavrentiev_a_ccs_tbb::Sparse lavrentiev_a_ccs_tbb::CCSTBB::Transpose(const Sparse &sparse) {
  auto [size, elements, rows, columns_sum] = Sparse();
  size.first = sparse.size.second;
  size.second = sparse.size.first;
  int need_size = std::max(sparse.size.first, sparse.size.second);
  std::vector<std::vector<double>> new_elements(need_size);
  std::vector<std::vector<int>> new_indexes(need_size);
  int counter = 0;
  for (int i = 0; i < static_cast<int>(sparse.columnsSum.size()); ++i) {
    for (int j = 0; j < GetElementsCount(i, sparse.columnsSum); ++j) {
      new_elements[sparse.rows[counter]].emplace_back(sparse.elements[counter]);
      new_indexes[sparse.rows[counter]].emplace_back(i);
      counter++;
    }
  }
  for (int i = 0; i < static_cast<int>(new_elements.size()); ++i) {
    for (int j = 0; j < static_cast<int>(new_elements[i].size()); ++j) {
      elements.emplace_back(new_elements[i][j]);
      rows.emplace_back(new_indexes[i][j]);
    }
    if (i > 0) {
      columns_sum.emplace_back(new_elements[i].size() + columns_sum[i - 1]);
    } else {
      columns_sum.emplace_back(new_elements[i].size());
    }
  }
  return {.size = size, .elements = elements, .rows = rows, .columnsSum = columns_sum};
}

int lavrentiev_a_ccs_tbb::CCSTBB::CalculateStartIndex(int index, const std::vector<int> &columns_sum) {
  if (index != 0) {
    return columns_sum[index] - GetElementsCount(index, columns_sum);
  }
  return 0;
}

lavrentiev_a_ccs_tbb::Sparse lavrentiev_a_ccs_tbb::CCSTBB::MatMul(const Sparse &matrix1, const Sparse &matrix2) {
  oneapi::tbb::task_arena worker(ppc::util::GetPPCNumThreads());
  auto [size, elements, rows, columns_sum] = Sparse();
  columns_sum.resize(matrix2.size.second);
  rows.resize(matrix2.columnsSum.size() * matrix1.columnsSum.size() +
              std::max(matrix1.columnsSum.size(), matrix2.columnsSum.size()));
  elements.resize(matrix2.columnsSum.size() * matrix1.columnsSum.size() +
                  std::max(matrix1.columnsSum.size(), matrix2.columnsSum.size()));
  auto new_matrix1 = Transpose(matrix1);
  std::map<int, std::pair<double, int>> thread_data;
  worker.execute([&] {
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, matrix2.columnsSum.size()),
                              [&](const oneapi::tbb::blocked_range<int> &blocked_range) {
                                for (int i = blocked_range.begin(); i != blocked_range.end(); ++i) {
                                  for (int j = 0; j < static_cast<int>(new_matrix1.columnsSum.size()); ++j) {
                                    double sum = 0.0;
                                    for (int x = 0; x < GetElementsCount(j, new_matrix1.columnsSum); x++) {
                                      for (int y = 0; y < GetElementsCount(i, matrix2.columnsSum); y++) {
                                        if (new_matrix1.rows[CalculateStartIndex(j, new_matrix1.columnsSum) + x] ==
                                            matrix2.rows[CalculateStartIndex(i, matrix2.columnsSum) + y]) {
                                          sum +=
                                              new_matrix1.elements[x + CalculateStartIndex(j, new_matrix1.columnsSum)] *
                                              matrix2.elements[y + CalculateStartIndex(i, matrix2.columnsSum)];
                                        }
                                      }
                                    }
                                    if (sum != 0) {
                                      elements[i * matrix2.size.second + j] = sum;
                                      rows[i * matrix2.size.second + j] = j;
                                      columns_sum[i]++;
                                    }
                                  }
                                }
                              }),
        tbb::auto_partitioner();
  });
  for (size_t i = 1; i < columns_sum.size(); ++i) {
    columns_sum[i] = columns_sum[i] + columns_sum[i - 1];
  }
  size.first = matrix2.size.second;
  size.second = matrix2.size.second;
  std::vector<double> elems;
  std::vector<int> nrows;
  for (size_t i = 0; i < elements.size(); ++i) {
    if (elements[i] != 0.0) {
      elems.emplace_back(elements[i]);
      nrows.emplace_back(rows[i]);
    }
  }
  return {.size = size, .elements = elems, .rows = nrows, .columnsSum = columns_sum};
}

int lavrentiev_a_ccs_tbb::CCSTBB::GetElementsCount(int index, const std::vector<int> &columns_sum) {
  if (index == 0) {
    return columns_sum[index];
  }
  return columns_sum[index] - columns_sum[index - 1];
}

std::vector<double> lavrentiev_a_ccs_tbb::CCSTBB::ConvertFromSparse(const Sparse &matrix) {
  std::vector<double> nmatrix(matrix.size.first * matrix.size.second);
  int counter = 0;
  for (size_t i = 0; i < matrix.columnsSum.size(); ++i) {
    for (int j = 0; j < GetElementsCount(static_cast<int>(i), matrix.columnsSum); ++j) {
      nmatrix[i + (matrix.size.second * matrix.rows[counter])] = matrix.elements[counter];
      counter++;
    }
  }
  return nmatrix;
}

bool lavrentiev_a_ccs_tbb::CCSTBB::PreProcessingImpl() {
  A_.size = {static_cast<int>(task_data->inputs_count[0]), static_cast<int>(task_data->inputs_count[1])};
  B_.size = {static_cast<int>(task_data->inputs_count[2]), static_cast<int>(task_data->inputs_count[3])};
  if (IsEmpty()) {
    return true;
  }
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  auto am = std::vector<double>(in_ptr, in_ptr + (A_.size.first * A_.size.second));
  A_ = ConvertToSparse(A_.size, am);
  auto *in_ptr2 = reinterpret_cast<double *>(task_data->inputs[1]);
  auto bm = std::vector<double>(in_ptr2, in_ptr2 + (B_.size.first * B_.size.second));
  B_ = ConvertToSparse(B_.size, bm);
  return true;
}

bool lavrentiev_a_ccs_tbb::CCSTBB::IsEmpty() const {
  return A_.size.first * A_.size.second == 0 || B_.size.first * B_.size.second == 0;
}

bool lavrentiev_a_ccs_tbb::CCSTBB::ValidationImpl() {
  return task_data->inputs_count[0] * task_data->inputs_count[3] == task_data->outputs_count[0] &&
         task_data->inputs_count[0] == task_data->inputs_count[3] &&
         task_data->inputs_count[1] == task_data->inputs_count[2];
}

bool lavrentiev_a_ccs_tbb::CCSTBB::RunImpl() {
  Answer_ = MatMul(A_, B_);
  return true;
}

bool lavrentiev_a_ccs_tbb::CCSTBB::PostProcessingImpl() {
  std::ranges::copy(ConvertFromSparse(Answer_), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}