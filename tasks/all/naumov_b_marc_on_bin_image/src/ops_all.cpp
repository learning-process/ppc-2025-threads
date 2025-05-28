#include "all/naumov_b_marc_on_bin_image/include/ops_all.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <execution>
#include <functional>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/util/include/util.hpp"

std::vector<int> naumov_b_marc_on_bin_image_all::GenerateRandomBinaryMatrix(int rows, int cols, double probability) {
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

std::vector<int> naumov_b_marc_on_bin_image_all::GenerateSparseBinaryMatrix(int rows, int cols, double probability) {
  return GenerateRandomBinaryMatrix(rows, cols, probability);
}

std::vector<int> naumov_b_marc_on_bin_image_all::GenerateDenseBinaryMatrix(int rows, int cols, double probability) {
  return GenerateRandomBinaryMatrix(rows, cols, probability);
}

bool naumov_b_marc_on_bin_image_all::TestTaskALL::PreProcessingImpl() {
  const int rank = world_.rank();
  const int num_procs = world_.size();

  if (rank == 0) {
    global_rows_ = task_data->inputs_count[0];
    global_cols_ = task_data->inputs_count[1];
    input_image_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
                                    reinterpret_cast<int*>(task_data->inputs[0]) + global_rows_ * global_cols_);
  }

  boost::mpi::broadcast(world_, global_rows_, 0);
  boost::mpi::broadcast(world_, global_cols_, 0);

  local_rows_ = global_rows_ / num_procs;
  const int remainder = global_rows_ % num_procs;
  if (rank < remainder) local_rows_++;

  input_image_.resize((local_rows_ + 2 * halo_) * global_cols_);

  if (rank == 0) {
    for (int p = 1; p < num_procs; ++p) {
      const int p_rows = global_rows_ / num_procs + (p < remainder ? 1 : 0);
      const int start = std::accumulate(&local_rows_, &local_rows_ + p, 0);
      world_.send(p, 0, &input_image_[start * global_cols_], (p_rows + 2 * halo_) * global_cols_);
    }
  } else {
    world_.recv(0, 0, input_image_);
  }

  output_image_.assign((local_rows_ + 2 * halo_) * global_cols_, 0);
  label_parent_.clear();
  current_label_ = 0;

  return true;
}

bool naumov_b_marc_on_bin_image_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs.empty() || task_data->outputs.empty()) return false;
    const size_t expected = global_rows_ * global_cols_;
    const int* data = reinterpret_cast<int*>(task_data->inputs[0]);
    return std::all_of(data, data + expected, [](int v) { return v == 0 || v == 1; });
  }
  return true;
}

bool naumov_b_marc_on_bin_image_all::TestTaskALL::RunImpl() {
  for (int i = halo_; i < local_rows_ + halo_; ++i) {
    for (int j = 0; j < global_cols_; ++j) {
      if (input_image_[i * global_cols_ + j] == 1) {
        ProcessPixel(i, j);
      }
    }
  }

  CreateAndJoinThreads();
  ExchangeBoundaries();
  ResolveGlobalLabels();

  return true;
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::ProcessPixel(int row, int col) {
  const auto neighbors = FindAdjacentLabels(row, col);
  if (neighbors.empty()) {
    AssignNewLabel(row, col);
  } else {
    AssignMinLabel(row, col, neighbors);
  }
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::AssignNewLabel(int row, int col) {
  output_image_[row * global_cols_ + col] = ++current_label_;
  if (label_parent_.size() <= current_label_) {
    label_parent_.resize(current_label_ + 1);
  }
  label_parent_[current_label_] = current_label_;
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::AssignMinLabel(int row, int col, const std::vector<int>& neighbors) {
  const int min_label = *std::min_element(neighbors.begin(), neighbors.end());
  output_image_[row * global_cols_ + col] = min_label;
  for (const int lbl : neighbors) {
    if (lbl != min_label) UnionLabels(min_label, lbl);
  }
}

std::vector<int> naumov_b_marc_on_bin_image_all::TestTaskALL::FindAdjacentLabels(int row, int col) {
  std::vector<int> labels;
  if (col > 0 && output_image_[row * global_cols_ + col - 1] != 0)
    labels.push_back(output_image_[row * global_cols_ + col - 1]);
  if (row > 0 && output_image_[(row - 1) * global_cols_ + col] != 0)
    labels.push_back(output_image_[(row - 1) * global_cols_ + col]);
  return labels;
}

int naumov_b_marc_on_bin_image_all::TestTaskALL::FindRoot(int label) {
  while (label_parent_[label] != label) {
    label_parent_[label] = label_parent_[label_parent_[label]];
    label = label_parent_[label];
  }
  return label;
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::UnionLabels(int label1, int label2) {
  int root1 = FindRoot(label1);
  int root2 = FindRoot(label2);
  if (root1 != root2) {
    if (root1 < root2)
      label_parent_[root2] = root1;
    else
      label_parent_[root1] = root2;
  }
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::ExchangeBoundaries() {
  const int rank = world_.rank();
  const int num_procs = world_.size();

  if (rank > 0) {
    world_.send(rank - 1, 0, &output_image_[halo_ * global_cols_], global_cols_);
    world_.recv(rank - 1, 0, &output_image_[0], global_cols_);
  }

  if (rank < num_procs - 1) {
    world_.send(rank + 1, 0, &output_image_[(local_rows_ + halo_ - 1) * global_cols_], global_cols_);
    world_.recv(rank + 1, 0, &output_image_[(local_rows_ + halo_) * global_cols_], global_cols_);
  }
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::ResolveGlobalLabels() {
  const int rank = world_.rank();
  const int num_procs = world_.size();

  std::vector<int> send_buf, recv_buf;
  if (rank < num_procs - 1) {
    send_buf.assign(output_image_.begin() + (local_rows_ + halo_ - 1) * global_cols_,
                    output_image_.begin() + (local_rows_ + halo_) * global_cols_);
  }

  std::vector<std::vector<int>> boundaries(num_procs);
  boost::mpi::all_gather(world_, send_buf, boundaries);

  int label_offset = 0;
  std::vector<int> offsets(num_procs, 0);
  for (int p = 0; p < num_procs; ++p) {
    offsets[p] = label_offset;
    if (p == rank) {
      label_offset += *std::max_element(label_parent_.begin(), label_parent_.end());
    }
    boost::mpi::broadcast(world_, label_offset, p);
  }

  std::vector<int> global_parent(label_offset + 1);
  std::iota(global_parent.begin(), global_parent.end(), 0);

  for (int p = 0; p < num_procs - 1; ++p) {
    const auto& bottom = boundaries[p];
    const auto& top = boundaries[p + 1];

    for (int i = 0; i < global_cols_; ++i) {
      if (bottom[i] == 0 || top[i] == 0) continue;

      const int global_bottom = offsets[p] + bottom[i];
      const int global_top = offsets[p + 1] + top[i];

      int root_bottom = global_bottom;
      while (global_parent[root_bottom] != root_bottom) {
        global_parent[root_bottom] = global_parent[global_parent[root_bottom]];
        root_bottom = global_parent[root_bottom];
      }

      int root_top = global_top;
      while (global_parent[root_top] != root_top) {
        global_parent[root_top] = global_parent[global_parent[root_top]];
        root_top = global_parent[root_top];
      }

      if (root_bottom != root_top) {
        global_parent[std::max(root_bottom, root_top)] = std::min(root_bottom, root_top);
      }
    }
  }

  if (world_.rank() == 0) {
    boost::mpi::broadcast(world_, global_parent.data(), global_parent.size(), 0);
  } else {
    boost::mpi::broadcast(world_, global_parent.data(), global_parent.size(), 0);
  }

  for (size_t i = 0; i < output_image_.size(); ++i) {
    if (output_image_[i] != 0) {
      const int global_label = offsets[rank] + output_image_[i];
      int root = global_label;
      while (global_parent[root] != root) {
        root = global_parent[root];
      }
      output_image_[i] = root;
    }
  }

  std::unordered_map<int, int> label_map;
  int new_label = 1;
  for (auto& label : output_image_) {
    if (label != 0 && label_map.find(label) == label_map.end()) {
      label_map[label] = new_label++;
    }
    if (label != 0) label = label_map[label];
  }
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::CreateAndJoinThreads() {
  for (int label = 1; label <= current_label_; label++) {
    FindRoot(label);
  }

  const size_t total = (local_rows_ + 2 * halo_) * global_cols_;
  unsigned num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 1;
  }
  std::vector<std::thread> threads(num_threads);

  const size_t chunk = total / num_threads;
  for (unsigned t = 0; t < num_threads; ++t) {
    const size_t start = t * chunk;
    const size_t end = (t == num_threads - 1) ? total : start + chunk;
    threads[t] = std::thread([this, start, end] {
      for (size_t idx = start; idx < end; ++idx) {
        if (output_image_[idx] != 0) {
          output_image_[idx] = label_parent_[output_image_[idx]];
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

bool naumov_b_marc_on_bin_image_all::TestTaskALL::PostProcessingImpl() {
  std::vector<int> local_result(output_image_.begin() + halo_ * global_cols_,
                                output_image_.begin() + (local_rows_ + halo_) * global_cols_);

  if (world_.rank() == 0) {
    output_image_.resize(global_rows_ * global_cols_);
  }

  boost::mpi::gather(world_, local_result.data(), local_rows_ * global_cols_, output_image_.data(), 0);

  if (world_.rank() == 0) {
    std::copy(output_image_.begin(), output_image_.end(), reinterpret_cast<int*>(task_data->outputs[0]));
  }

  return true;
}