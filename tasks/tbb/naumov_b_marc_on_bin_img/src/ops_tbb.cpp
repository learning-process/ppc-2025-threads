#include "tbb/naumov_b_marc_on_bin_img/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

std::vector<int> naumov_b_marc_on_bin_img_tbb::GenerateRandomBinaryMatrix(int rows, int cols, double probability) {
  const int total_elements = rows * cols;
  const int target_ones = static_cast<int>(total_elements * probability);

  std::vector<int> matrix(total_elements, 1);

  const int zeros_needed = total_elements - target_ones;

  for (int i = 0; i < zeros_needed; ++i) {
    matrix[i] = 0;
  }

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(matrix.begin(), matrix.end(), g);

  return matrix;
}

std::vector<int> naumov_b_marc_on_bin_img_tbb::GenerateSparseBinaryMatrix(int rows, int cols, double probability) {
  return GenerateRandomBinaryMatrix(rows, cols, probability);
}

std::vector<int> naumov_b_marc_on_bin_img_tbb::GenerateDenseBinaryMatrix(int rows, int cols, double probability) {
  return GenerateRandomBinaryMatrix(rows, cols, probability);
}

std::vector<int> naumov_b_marc_on_bin_img_tbb::TestTaskTBB::FindAdjacentLabels(int row, int col) {
  std::vector<int> neighbors;

  if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
    return neighbors;
  }

  if (col > 0 && output_image_[(row * cols_) + (col - 1)] != 0) {
    neighbors.push_back(output_image_[(row * cols_) + (col - 1)]);
  }
  if (row > 0 && output_image_[((row - 1) * cols_) + col] != 0) {
    neighbors.push_back(output_image_[((row - 1) * cols_) + col]);
  }

  return neighbors;
}

void naumov_b_marc_on_bin_img_tbb::TestTaskTBB::UnionLabels(int a, int b) {
  int ra = FindRoot(a);
  int rb = FindRoot(b);
  if (ra != rb) {
    if (ra < rb) {
      label_parent_[rb] = ra;
    } else {
      label_parent_[ra] = rb;
    }
  }
}

int naumov_b_marc_on_bin_img_tbb::TestTaskTBB::FindRoot(int x) {
  while (label_parent_[x] != x) {
    x = label_parent_[x];
  }
  return x;
}

bool naumov_b_marc_on_bin_img_tbb::TestTaskTBB::PreProcessingImpl() {
  rows_ = static_cast<int>(task_data->inputs_count[0]);
  cols_ = static_cast<int>(task_data->inputs_count[1]);
  size_t total = size_t(rows_) * size_t(cols_);

  input_image_.resize(total);
  output_image_.resize(total);

  label_parent_.clear();
  label_parent_.resize(total + 1);

  int* in = reinterpret_cast<int*>(task_data->inputs[0]);
  for (size_t i = 0; i < total; ++i) {
    input_image_[i] = in[i];
    output_image_[i] = 0;
  }
  return true;
}

bool naumov_b_marc_on_bin_img_tbb::TestTaskTBB::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) return false;
  int m = static_cast<int>(task_data->inputs_count[0]);
  int n = static_cast<int>(task_data->inputs_count[1]);
  if (m <= 0 || n <= 0) return false;

  size_t total = size_t(m) * size_t(n);
  if (!task_data->inputs[0]) return false;
  int* in = reinterpret_cast<int*>(task_data->inputs[0]);
  for (size_t i = 0; i < total; ++i) {
    if (in[i] != 0 && in[i] != 1) return false;
  }
  return true;
}

bool naumov_b_marc_on_bin_img_tbb::TestTaskTBB::RunImpl() {
  const int R = rows_, C = cols_;
  const size_t total = size_t(R) * size_t(C);

  int next_label = 1;

  for (size_t i = 0; i < total + 1; ++i) {
    label_parent_[i] = int(i);
  }

  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      size_t idx = size_t(i) * C + j;
      if (input_image_[idx] == 0) {
        output_image_[idx] = 0;
        continue;
      }

      int left_label = (j > 0 ? output_image_[idx - 1] : 0);
      int top_label = (i > 0 ? output_image_[idx - C] : 0);

      if (left_label == 0 && top_label == 0) {
        output_image_[idx] = next_label;
        label_parent_[next_label] = next_label;
        ++next_label;
      } else if (left_label > 0 && top_label > 0) {
        int min_l = std::min(left_label, top_label);
        int max_l = std::max(left_label, top_label);
        output_image_[idx] = min_l;
        UnionLabels(min_l, max_l);
      } else {
        output_image_[idx] = (left_label > 0 ? left_label : top_label);
      }
    }
  }

  for (int lbl = 1; lbl < next_label; ++lbl) {
    label_parent_[lbl] = FindRoot(lbl);
  }

  tbb::parallel_for(size_t(0), total, [this](size_t idx) {
    if (output_image_[idx] > 0) {
      output_image_[idx] = label_parent_[output_image_[idx]];
    }
  });

  return true;
}

bool naumov_b_marc_on_bin_img_tbb::TestTaskTBB::PostProcessingImpl() {
  if (task_data->outputs.empty()) return false;
  int* out = reinterpret_cast<int*>(task_data->outputs[0]);
  size_t total = output_image_.size();
  for (size_t i = 0; i < total; ++i) {
    out[i] = output_image_[i];
  }
  return true;
}
