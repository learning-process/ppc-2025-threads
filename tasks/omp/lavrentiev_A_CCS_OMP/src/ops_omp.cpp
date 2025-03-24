#include "omp/lavrentiev_A_CCS_OMP/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

lavrentiev_a_ccs_omp::Sparse lavrentiev_a_ccs_omp::CCSOMP::ConvertToSparse(std::pair<int, int> size,
                                                                           const std::vector<double> &values) {
  std::vector<int> columns_sum(size.second);
  std::vector<double> elements;
  std::vector<int> rows;
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

lavrentiev_a_ccs_omp::Sparse lavrentiev_a_ccs_omp::CCSOMP::Transpose(const Sparse &sparse) {
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

lavrentiev_a_ccs_omp::Sparse lavrentiev_a_ccs_omp::CCSOMP::MatMul(const Sparse &matrix1, const Sparse &matrix2) {
  Sparse result_matrix;
  result_matrix.columnsSum.resize(matrix2.size.second);
  auto new_matrix1 = Transpose(matrix1);
  std::vector<std::pair<std::vector<double>, std::vector<int>>> threads_data(ppc::util::GetPPCNumThreads());
#pragma omp parallel
  {
    std::pair<std::vector<double>, std::vector<int>> current_thread_data;
#pragma omp for
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
        if (sum != 0) {
          current_thread_data.first.push_back(sum);
          current_thread_data.second.push_back(j);
          result_matrix.columnsSum[i]++;
        }
      }
    }
    threads_data[omp_get_thread_num()] = std::move(current_thread_data);
  }
  for (size_t i = 1; i < result_matrix.columnsSum.size(); ++i) {
    result_matrix.columnsSum[i] = result_matrix.columnsSum[i] + result_matrix.columnsSum[i - 1];
  }
  for (const auto &data : threads_data) {
    for (size_t i = 0; i < data.first.size(); ++i) {
      result_matrix.elements.push_back(data.first[i]);
      result_matrix.rows.push_back(data.second[i]);
    }
  }
  result_matrix.size.first = matrix2.size.second;
  result_matrix.size.second = matrix2.size.second;
  return {.size = result_matrix.size,
          .elements = result_matrix.elements,
          .rows = result_matrix.rows,
          .columnsSum = result_matrix.columnsSum};
}

int lavrentiev_a_ccs_omp::CCSOMP::GetElementsCount(int index, const std::vector<int> &columns_sum) {
  if (index == 0) {
    return columns_sum[index];
  }
  return columns_sum[index] - columns_sum[index - 1];
}

std::vector<double> lavrentiev_a_ccs_omp::CCSOMP::ConvertFromSparse(const Sparse &matrix) {
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

bool lavrentiev_a_ccs_omp::CCSOMP::PreProcessingImpl() {
  A_.size = {static_cast<int>(task_data->inputs_count[0]), static_cast<int>(task_data->inputs_count[1])};
  B_.size = {static_cast<int>(task_data->inputs_count[2]), static_cast<int>(task_data->inputs_count[3])};
  if (IsEmpty()) {
    return true;
  }
  std::vector<double> am(A_.size.first * A_.size.second);
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  for (int i = 0; i < A_.size.first * A_.size.second; ++i) {
    am[i] = in_ptr[i];
  }
  A_ = ConvertToSparse(A_.size, am);
  std::vector<double> bm(B_.size.first * B_.size.second);
  auto *in_ptr2 = reinterpret_cast<double *>(task_data->inputs[1]);
  for (int i = 0; i < B_.size.first * B_.size.second; ++i) {
    bm[i] = in_ptr2[i];
  }
  B_ = ConvertToSparse(B_.size, bm);
  return true;
}

bool lavrentiev_a_ccs_omp::CCSOMP::IsEmpty() const {
  return A_.size.first * A_.size.second == 0 || B_.size.first * B_.size.second == 0;
}

bool lavrentiev_a_ccs_omp::CCSOMP::ValidationImpl() {
  return task_data->inputs_count[0] * task_data->inputs_count[3] == task_data->outputs_count[0] &&
         task_data->inputs_count[0] == task_data->inputs_count[3] &&
         task_data->inputs_count[1] == task_data->inputs_count[2];
}

bool lavrentiev_a_ccs_omp::CCSOMP::RunImpl() {
  Answer_ = MatMul(A_, B_);
  return true;
}

bool lavrentiev_a_ccs_omp::CCSOMP::PostProcessingImpl() {
  std::vector<double> result = ConvertFromSparse(Answer_);
  for (auto i = 0; i < static_cast<int>(result.size()); ++i) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = result[i];
  }
  return true;
}