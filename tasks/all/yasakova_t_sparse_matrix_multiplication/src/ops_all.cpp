#include "all/yasakova_t_sparse_matrix_multiplication/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

void yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS::AddValue(int row, Complex value, int col) {
  for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j) {
    if (colIndices[j] == col) {
      values[j] += value;
      return;
    }
  }
  colIndices.emplace_back(col);
  values.emplace_back(value);
  for (int i = row + 1; i <= numRows; ++i) {
    rowPtr[i]++;
  }
}

void yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS::PrintSparseMatrix(
    const yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS& matrix) {
  for (int i = 0; i < matrix.numRows; ++i) {
    for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; ++j) {
      std::cout << "Element at (" << i << ", " << matrix.colIndices[j] << ") = " << matrix.values[j] << '\n';
    }
  }
}

bool yasakova_t_sparse_matrix_mult_all::AreEqualElems(const Complex& a, const Complex& b, double epsilon) {
  return std::abs(a.real() - b.real()) < epsilon && std::abs(a.imag() - b.imag()) < epsilon;
}

void yasakova_t_sparse_matrix_mult_all::AddResult(std::vector<CoordVal>& results, int row, int col, Complex val) {
  results.push_back({row, col, val});
}

std::vector<Complex> yasakova_t_sparse_matrix_mult_all::ParseMatrixIntoVec(const SparseMatrixCRS& mat) {
  std::vector<Complex> res = {};
  res.reserve(5 + mat.values.size() + mat.colIndices.size() + mat.rowPtr.size());
  res.emplace_back((double)mat.numRows);
  res.emplace_back((double)mat.numCols);
  res.emplace_back((double)mat.values.size());
  res.emplace_back((double)mat.colIndices.size());
  res.emplace_back((double)mat.rowPtr.size());
  for (unsigned int i = 0; i < (unsigned int)mat.values.size(); i++) {
    res.emplace_back(mat.values[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)mat.colIndices.size(); i++) {
    res.emplace_back(mat.colIndices[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)mat.rowPtr.size(); i++) {
    res.emplace_back(mat.rowPtr[i]);
  }
  return res;
}

bool yasakova_t_sparse_matrix_mult_all::CheckMatrixesEquality(
    const yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS& a,
    const yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS& b) {
  if (a.numCols != b.numCols || a.numRows != b.numRows) {
    return false;
  }
  for (unsigned int i = 0; i < (unsigned int)a.numRows; ++i) {
    unsigned int this_row_start = a.rowPtr[i];
    unsigned int this_row_end = a.rowPtr[i + 1];
    unsigned int other_row_start = b.rowPtr[i];
    unsigned int other_row_end = b.rowPtr[i + 1];
    if ((this_row_end - this_row_start) != (other_row_end - other_row_start)) {
      return false;
    }
    for (unsigned int j = this_row_start; j < this_row_end; ++j) {
      bool found = false;
      for (unsigned int k = other_row_start; k < other_row_end; ++k) {
        if (a.colIndices[j] == b.colIndices[k] && AreEqualElems(a.values[j], b.values[k], 0.000001)) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
  }
  return true;
}

yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS yasakova_t_sparse_matrix_mult_all::ParseVectorIntoMatrix(
    std::vector<Complex>& vec) {
  SparseMatrixCRS res;
  res.numRows = (int)vec[0].real();
  res.numCols = (int)vec[1].real();
  auto values_size = (unsigned int)vec[2].real();
  auto col_indices_size = (unsigned int)vec[3].real();
  auto row_ptr_size = (unsigned int)vec[4].real();
  res.values.reserve(values_size);
  res.colIndices.reserve(col_indices_size);
  res.rowPtr.reserve(row_ptr_size);
  for (unsigned int i = 0; i < values_size; i++) {
    res.values.emplace_back(vec[5 + i]);
  }
  for (unsigned int i = 0; i < col_indices_size; i++) {
    res.colIndices.emplace_back((int)vec[5 + values_size + i].real());
  }
  for (unsigned int i = 0; i < row_ptr_size; i++) {
    res.rowPtr.emplace_back((int)vec[5 + values_size + col_indices_size + i].real());
  }
  return res;
}

yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS yasakova_t_sparse_matrix_mult_all::BuildResultMatrix(
    const std::vector<yasakova_t_sparse_matrix_mult_all::CoordVal>& all_results, int a_num_rows, int b_num_cols) {
  yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS c(a_num_rows, b_num_cols);
  std::map<std::pair<int, int>, Complex> result_map;

  for (const auto& rv : all_results) {
    result_map[{rv.row, rv.col}] += rv.value;
  }

  c.rowPtr.resize(a_num_rows + 1);
  c.values.reserve(result_map.size());
  c.colIndices.reserve(result_map.size());

  c.rowPtr[0] = 0;

  for (int i = 0; i < a_num_rows; ++i) {
    for (auto& kv : result_map) {
      if (kv.first.first == i && kv.second != Complex(0)) {
        c.colIndices.push_back(kv.first.second);
        c.values.push_back(kv.second);
      }
    }
    c.rowPtr[i + 1] = static_cast<int>(c.colIndices.size());

    for (auto it = result_map.begin(); it != result_map.end();) {
      if (it->first.first == i) {
        it = result_map.erase(it);
      } else {
        ++it;
      }
    }
  }
  return c;
}

void yasakova_t_sparse_matrix_mult_all::TestTaskALL::ProcessRowsRange(int start_row, int end_row, 
                                                                    std::vector<CoordVal>& local_results) {
  for (int i = start_row; i < end_row; ++i) {
    for (int j_idx = A_.rowPtr[i]; j_idx < A_.rowPtr[i + 1]; ++j_idx) {
      int col_a = A_.colIndices[j_idx];
      Complex value_a = A_.values[j_idx];

      for (int k_idx = B_.rowPtr[col_a]; k_idx < B_.rowPtr[col_a + 1]; ++k_idx) {
        int col_b = B_.colIndices[k_idx];
        Complex value_b = B_.values[k_idx];

        #pragma omp critical
        AddResult(local_results, i, col_b, value_a * value_b);
      }
    }
  }
}

bool yasakova_t_sparse_matrix_mult_all::TestTaskALL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<Complex*>(task_data->inputs[0]);
  input_ = std::vector<Complex>(in_ptr, in_ptr + input_size);
  std::vector<Complex> matrix_a = {};
  std::vector<Complex> matrix_b = {};
  matrix_a.reserve(5 + (unsigned int)(input_[2].real() + input_[3].real() + input_[4].real()));
  matrix_b.reserve(input_.size() - (unsigned int)(5 + input_[2].real() + input_[3].real() + input_[4].real()));
  for (unsigned int i = 0; i < (unsigned int)(5 + input_[2].real() + input_[3].real() + input_[4].real()); i++) {
    matrix_a.emplace_back(input_[i]);
  }
  for (auto i = (unsigned int)(5 + input_[2].real() + input_[3].real() + input_[4].real());
       i < (unsigned int)input_.size(); i++) {
    matrix_b.emplace_back(input_[i]);
  }
  A_ = ParseVectorIntoMatrix(matrix_a);
  B_ = ParseVectorIntoMatrix(matrix_b);
  return true;
}

bool yasakova_t_sparse_matrix_mult_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    unsigned int input_size = task_data->inputs_count[0];
    auto* in_ptr = reinterpret_cast<Complex*>(task_data->inputs[0]);
    std::vector<Complex> vec = std::vector<Complex>(in_ptr, in_ptr + input_size);
    return !(vec[1] != vec[5 + (int)(vec[2].real() + vec[3].real() + vec[4].real())].real());
  }
  return true;
}

bool yasakova_t_sparse_matrix_mult_all::TestTaskALL::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  int a_num_rows = A_.numRows;
  int a_num_cols = A_.numCols;
  int b_num_rows = B_.numRows;
  int b_num_cols = B_.numCols;

  MPI_Bcast(&a_num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int row_per_proc = a_num_rows / size;
  int remainder = a_num_rows % size;

  int start_row = (rank * row_per_proc) + std::min(rank, remainder);
  int end_row = start_row + row_per_proc + (rank < remainder ? 1 : 0);

  std::vector<CoordVal> local_results;
  local_results.reserve((end_row - start_row) * b_num_cols);

  // Установка числа потоков OpenMP
  int num_threads = ppc::util::GetPPCNumThreads();
  omp_set_num_threads(num_threads);

  // Параллельная обработка с использованием OpenMP
  #pragma omp parallel
  {
    std::vector<CoordVal> thread_local_results;
    thread_local_results.reserve((end_row - start_row) * b_num_cols / num_threads);
    
    #pragma omp for schedule(static)
    for (int i = start_row; i < end_row; ++i) {
      for (int j_idx = A_.rowPtr[i]; j_idx < A_.rowPtr[i + 1]; ++j_idx) {
        int col_a = A_.colIndices[j_idx];
        Complex value_a = A_.values[j_idx];

        for (int k_idx = B_.rowPtr[col_a]; k_idx < B_.rowPtr[col_a + 1]; ++k_idx) {
          int col_b = B_.colIndices[k_idx];
          Complex value_b = B_.values[k_idx];

          AddResult(thread_local_results, i, col_b, value_a * value_b);
        }
      }
    }
    
    #pragma omp critical
    local_results.insert(local_results.end(), thread_local_results.begin(), thread_local_results.end());
  }

  // Сбор результатов с помощью MPI
  int local_size_bytes = static_cast<int>(local_results.size() * sizeof(CoordVal));
  std::vector<int> recv_counts(size);
  MPI_Gather(&local_size_bytes, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<int> displs(size);
    if (!displs.empty()) {
      displs[0] = 0;
      for (int i = 1; i < size; ++i) {
        displs[i] = displs[i - 1] + recv_counts[i - 1];
      }
    }
    int total_bytes = displs[size - 1] + recv_counts[size - 1];

    std::vector<char> recv_buffer(total_bytes);
    std::vector<CoordVal> send_buffer(local_results.begin(), local_results.end());

    MPI_Gatherv(send_buffer.data(), static_cast<int>(send_buffer.size() * sizeof(CoordVal)), MPI_BYTE,
                recv_buffer.data(), recv_counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    std::vector<CoordVal> all_results;
    all_results.reserve(total_bytes / sizeof(CoordVal));
    auto* ptr = reinterpret_cast<CoordVal*>(recv_buffer.data());
    size_t count_coords = total_bytes / sizeof(CoordVal);
    for (size_t i = 0; i < count_coords; ++i) {
      all_results.push_back(ptr[i]);
    }
    
    yasakova_t_sparse_matrix_mult_all::SparseMatrixCRS c(a_num_rows, b_num_cols);
    c = BuildResultMatrix(all_results, a_num_rows, b_num_cols);
    output_ = ParseMatrixIntoVec(c);
  } else {
    MPI_Gatherv(local_results.data(), static_cast<int>(local_results.size() * sizeof(CoordVal)), MPI_BYTE, 
                nullptr, nullptr, nullptr, MPI_BYTE, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool yasakova_t_sparse_matrix_mult_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_.size(); i++) {
      reinterpret_cast<Complex*>(task_data->outputs[0])[i] = output_[i];
    }
  }
  return true;
}