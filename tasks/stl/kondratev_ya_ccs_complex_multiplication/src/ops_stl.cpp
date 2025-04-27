#include "stl/kondratev_ya_ccs_complex_multiplication/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool kondratev_ya_ccs_complex_multiplication_stl::IsZero(const std::complex<double> &value) {
  return std::norm(value) < kEpsilonForZero;
}

bool kondratev_ya_ccs_complex_multiplication_stl::IsEqual(const std::complex<double> &a,
                                                          const std::complex<double> &b) {
  return std::norm(a - b) <= kEpsilonForZero;
}

bool kondratev_ya_ccs_complex_multiplication_stl::TestTaskSTL::PreProcessingImpl() {
  a_ = *reinterpret_cast<CCSMatrix *>(task_data->inputs[0]);
  b_ = *reinterpret_cast<CCSMatrix *>(task_data->inputs[1]);

  if (a_.rows == 0 || a_.cols == 0 || b_.rows == 0 || b_.cols == 0) {
    return false;
  }

  if (a_.cols != b_.rows) {
    return false;
  }

  return true;
}

bool kondratev_ya_ccs_complex_multiplication_stl::TestTaskSTL::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 1 && task_data->inputs[0] != nullptr &&
         task_data->inputs[1] != nullptr && task_data->outputs[0] != nullptr;
}

bool kondratev_ya_ccs_complex_multiplication_stl::TestTaskSTL::RunImpl() {
  c_ = a_ * b_;
  return true;
}

bool kondratev_ya_ccs_complex_multiplication_stl::TestTaskSTL::PostProcessingImpl() {
  *reinterpret_cast<CCSMatrix *>(task_data->outputs[0]) = c_;
  return true;
}

kondratev_ya_ccs_complex_multiplication_stl::CCSMatrix
kondratev_ya_ccs_complex_multiplication_stl::CCSMatrix::operator*(const CCSMatrix &other) const {
  CCSMatrix result({rows, other.cols});
  result.values.reserve(std::min(rows * other.cols, static_cast<int>(values.size() * other.values.size())));
  result.row_index.reserve(result.values.capacity());
  result.col_ptrs.resize(other.cols + 1, 0);

  auto temp_cols = ComputeColumns(other);
  FillResultFromTempCols(temp_cols, other.cols, result);

  return result;
}

std::vector<std::vector<std::pair<int, std::complex<double>>>>
kondratev_ya_ccs_complex_multiplication_stl::CCSMatrix::ComputeColumns(const CCSMatrix &other) const {
  std::vector<std::vector<std::pair<int, std::complex<double>>>> temp_cols(other.cols);

  auto worker = [&](int start_col, int end_col) {
    for (int result_col = start_col; result_col < end_col; ++result_col) {
      std::vector<std::complex<double>> local_temp_col(rows, std::complex<double>(0.0, 0.0));

      for (int k = other.col_ptrs[result_col]; k < other.col_ptrs[result_col + 1]; k++) {
        int row_other = other.row_index[k];
        std::complex<double> val_other = other.values[k];

        for (int i = col_ptrs[row_other]; i < col_ptrs[row_other + 1]; i++) {
          int row_this = row_index[i];
          local_temp_col[row_this] += values[i] * val_other;
        }
      }

      for (int i = 0; i < rows; i++) {
        if (!IsZero(local_temp_col[i])) {
          temp_cols[result_col].emplace_back(i, local_temp_col[i]);
        }
      }
    }
  };

  int num_threads = std::min(ppc::util::GetPPCNumThreads(), static_cast<int>(std::thread::hardware_concurrency()));
  std::vector<std::thread> threads;
  int cols_per_thread = other.cols / num_threads;

  for (int i = 0; i < num_threads; ++i) {
    int start_col = i * cols_per_thread;
    int end_col = (i == num_threads - 1) ? other.cols : start_col + cols_per_thread;
    threads.emplace_back(worker, start_col, end_col);
  }

  for (auto &t : threads) {
    t.join();
  }

  return temp_cols;
}

void kondratev_ya_ccs_complex_multiplication_stl::CCSMatrix::FillResultFromTempCols(
    const std::vector<std::vector<std::pair<int, std::complex<double>>>> &temp_cols, int cols, CCSMatrix &result) {
  std::vector<int> col_sizes(cols);
  std::ranges::transform(temp_cols, col_sizes.begin(), [](const auto &col) { return static_cast<int>(col.size()); });

  std::vector<int> col_offsets(cols + 1, 0);
  std::partial_sum(col_sizes.begin(), col_sizes.end(), col_offsets.begin() + 1);

  int total_nonzero = col_offsets[cols];
  result.values.resize(total_nonzero);
  result.row_index.resize(total_nonzero);

  for (int i = 0; i <= cols; i++) {
    result.col_ptrs[i] = col_offsets[i];
  }

  auto worker_fill = [&](int start_col, int end_col) {
    for (int col = start_col; col < end_col; ++col) {
      int offset = col_offsets[col];
      for (const auto &[row, value] : temp_cols[col]) {
        result.row_index[offset] = row;
        result.values[offset] = value;
        offset++;
      }
    }
  };

  int num_threads = std::min(ppc::util::GetPPCNumThreads(), static_cast<int>(std::thread::hardware_concurrency()));
  std::vector<std::thread> threads;
  int cols_per_thread = cols / num_threads;

  threads.clear();
  for (int i = 0; i < num_threads; ++i) {
    int start_col = i * cols_per_thread;
    int end_col = (i == num_threads - 1) ? cols : start_col + cols_per_thread;
    threads.emplace_back(worker_fill, start_col, end_col);
  }

  for (auto &t : threads) {
    t.join();
  }
}