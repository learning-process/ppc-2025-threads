#include "seq/Lavrentiev_A_CCS_SEQ/include/ops_seq.hpp"

#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>


bool lavrentiev_a_ccs_seq::CCSSequential::PreProcessingImpl() {
  A_.size = {static_cast<int>(task_data->inputs_count[0]), static_cast<int>(task_data->inputs_count[1])};
  B_.size = {static_cast<int>(task_data->inputs_count[2]), static_cast<int>(task_data->inputs_count[3])};
  if (IsEmpty()) {
    return true;
  }
  std::vector<double> Am;
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  for (int i = 0; i < A_.size.first * A_.size.second; ++i) {
    Am.push_back(in_ptr[i]);
  }
  A_ = ConvertToSparse(A_.size, Am);
  std::vector<double> Bm;
  auto *in_ptr2 = reinterpret_cast<double *>(task_data->inputs[1]);
  for (int i = 0; i < B_.size.first * B_.size.second; ++i) {
    Bm.push_back(in_ptr2[i]);
  }
  B_ = ConvertToSparse(B_.size, Bm);
  return true;
}

bool lavrentiev_a_ccs_seq::CCSSequential::IsEmpty() const {
  return A_.size.first * A_.size.second == 0 || B_.size.first * B_.size.second == 0;
}

lavrentiev_a_ccs_seq::Sparse lavrentiev_a_ccs_seq::CCSSequential::ConvertToSparse(std::pair<int, int> size,
                                                                                  const std::vector<double> &values) {
  std::vector<int> columnsSum(size.second);
  std::vector<double> elements;
  std::vector<int> rows;
  for (int i = 0; i < size.second; ++i) {
    for (int j = 0; j < size.first; ++j) {
      if (values[i + (size.second * j)] != 0) {
        elements.emplace_back(values[i + (size.second * j)]);
        rows.emplace_back(j);
        columnsSum[i] += 1;
      }
    }
    if (i != size.second - 1) {
      columnsSum[i + 1] = columnsSum[i];
    }
  }
  return {.size = size, .elements = elements, .rows = rows, .columnsSum = columnsSum};
}

lavrentiev_a_ccs_seq::Sparse lavrentiev_a_ccs_seq::CCSSequential::MatMul(const Sparse &matrix1, const Sparse &matrix2) {
  auto [size, elements, rows, columns_sum] = Sparse();
  columns_sum.resize(matrix2.size.second);
  Sparse new_matrix1 = Transpose(matrix1);
  for (int i = 0; i < static_cast<int>(matrix2.columnsSum.size()); ++i) {
    for (int j = 0; j < static_cast<int>(new_matrix1.columnsSum.size()); ++j) {
      double sum = 0.0;
      int start_index1 = j != 0 ? new_matrix1.columnsSum[j] - GetElementsCount(j, new_matrix1.columnsSum) : 0;
      int start_index2 = i != 0 ? matrix2.columnsSum[i] - GetElementsCount(i, matrix2.columnsSum) : 0;
      for (int n = 0; n < GetElementsCount(j, new_matrix1.columnsSum); n++) {
        for (int n2 = 0; n2 < GetElementsCount(i, matrix2.columnsSum); n2++) {
          if (new_matrix1.rows[start_index1 + n] == matrix2.rows[start_index2 + n2]) {
            sum += new_matrix1.elements[n + start_index1] * matrix2.elements[n2 + start_index2];
          }
        }
      }
      if (sum > kMEpsilon) {
        elements.emplace_back(sum);
        rows.emplace_back(j);
        columns_sum[i]++;
      }
    }
  }
  for (auto i = 1; i < static_cast<int>(columns_sum.size()); ++i) {
    columns_sum[i] = columns_sum[i] + columns_sum[i - 1];
  }
  size.first = matrix2.size.second;
  size.second = matrix2.size.second;

  return {.size = size, .elements = elements, .rows = rows, .columnsSum = columns_sum};
}

int lavrentiev_a_ccs_seq::CCSSequential::GetElementsCount(int index, const std::vector<int> &columns_sum) {
  if (index == 0) {
    return columns_sum[index];
  }
  return columns_sum[index] - columns_sum[index - 1];
}

std::vector<double> lavrentiev_a_ccs_seq::CCSSequential::ConvertFromSparse(const Sparse &matrix) {
  std::vector<double> nmatrix(matrix.size.first * matrix.size.second);
  int counter = 0;
  for (size_t i = 0; i < matrix.columnsSum.size(); ++i) {
    auto barier = i == 0 ? matrix.columnsSum[0] : matrix.columnsSum[i] - matrix.columnsSum[i - 1];
    for (int j = 0; j < barier; ++j) {
      nmatrix[i + (matrix.size.second * matrix.rows[counter])] = matrix.elements[counter];
      counter++;
    }
  }
  return nmatrix;
}

bool lavrentiev_a_ccs_seq::CCSSequential::ValidationImpl() {
  return task_data->inputs_count[0] * task_data->inputs_count[3] == task_data->outputs_count[0] &&
         task_data->inputs_count[0] == task_data->inputs_count[3] &&
         task_data->inputs_count[1] == task_data->inputs_count[2];
}

lavrentiev_a_ccs_seq::Sparse lavrentiev_a_ccs_seq::CCSSequential::Transpose(const Sparse &sparse) {
  auto [size, elements, rows, columns_sum] = Sparse();
  size.first = sparse.size.second;
  size.second = sparse.size.first;
  int need_size = std::max(sparse.size.first, sparse.size.second);
  std::vector<std::vector<double>> new_elements(need_size);
  std::vector<std::vector<int>> new_indexes(need_size);
  int counter = 0;
  for (int i = 0; i < static_cast<int>(sparse.columnsSum.size()); ++i) {
    auto limit = i == 0 ? sparse.columnsSum[0] : sparse.columnsSum[i] - sparse.columnsSum[i - 1];
    for (int j = 0; j < limit; ++j) {
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

bool lavrentiev_a_ccs_seq::CCSSequential::RunImpl() {
  Answer_ = MatMul(A_, B_);
  return true;
}

bool lavrentiev_a_ccs_seq::CCSSequential::PostProcessingImpl() {
  std::vector<double> result = ConvertFromSparse(Answer_);
  for (auto i = 0; i < static_cast<int>(result.size()); ++i) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = result[i];
  }
  return true;
}

std::vector<double> lavrentiev_a_ccs_seq::GenerateRandomMatrix(int size) {
  std::vector<double> data(size);
  std::random_device device;
  std::mt19937 generator(device());
  for (int i = 0; i < size; ++i) {
    if (i % 6) {
      data[i] = static_cast<double>(generator() % 1000);
    }
  }
  std::ranges::shuffle(data, generator);
  return data;
}

std::vector<double> lavrentiev_a_ccs_seq::GenerateSingleMatrix(int size) {
  std::vector<double> test_data(size, 0.0);
  int sqrt = static_cast<int>(std::sqrt(size));
  for (int i = 0; i < sqrt; ++i) {
    for (int j = 0; j < sqrt; ++j) {
      if (i == j) {
        test_data[(sqrt * i) + j] = 1.0;
      }
    }
  }
  return test_data;
}