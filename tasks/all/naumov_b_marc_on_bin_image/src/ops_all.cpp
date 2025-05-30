#include "all/naumov_b_marc_on_bin_image/include/ops_all.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <random>
#include <thread>
#include <utility>
#include <vector>


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
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);

  if (rank_ == 0) {
    rows_ = static_cast<int>(task_data->inputs_count[0]);
    cols_ = static_cast<int>(task_data->inputs_count[1]);
    input_image_.resize(rows_ * cols_);
    int* input_data = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(input_data, input_data +(rows_ * cols_), input_image_.begin());
  }

  MPI_Bcast(&rows_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  base_rows_ = rows_ / num_procs_;
  remainder_ = rows_ % num_procs_;
  local_start_row_ = rank_ * base_rows_ + std::min(rank_, remainder_);
  local_rows_ = base_rows_ + (rank_ < remainder_ ? 1 : 0);
  base_label_ = (rank_ + 1) * (rows_ * cols_) + 1;

  local_image_.resize(local_rows_ * cols_);
  std::vector<int> counts(num_procs_);
  std::vector<int> displs(num_procs_);

  for (int i = 0; i < num_procs_; ++i) {
    int lrows = base_rows_ + (i < remainder_ ? 1 : 0);
    counts[i] = lrows * cols_;
    displs[i] = (i * base_rows_ + std::min(i, remainder_)) * cols_;
  }

  MPI_Scatterv(input_image_.data(), counts.data(), displs.data(), MPI_INT, local_image_.data(), counts[rank_], MPI_INT,
               0, MPI_COMM_WORLD);

  return true;
}

bool naumov_b_marc_on_bin_image_all::TestTaskALL::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  if (task_data->inputs_count[0] <= 0 || task_data->inputs_count[1] <= 0) {
    return false;
  }

  size_t expected_size = task_data->inputs_count[0] * task_data->inputs_count[1];
  if (task_data->inputs[0] == nullptr) {
    return false;
  }

  int* input_data = reinterpret_cast<int*>(task_data->inputs[0]);
  for (size_t i = 0; i < expected_size; ++i) {
    if (input_data[i] != 0 && input_data[i] != 1) {
      return false;
    }
  }

  return true;
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::ProcessPixel(int row, int col) {
  std::vector<int> neighbors = FindAdjacentLabels(row, col);
  if (neighbors.empty()) {
    AssignNewLabel(row, col);
  } else {
    AssignMinLabel(row, col, neighbors);
  }
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::AssignNewLabel(int row, int col) {
  int new_label = base_label_ + (++current_label_);
  local_output_[row * cols_ + col] = new_label;
  if (static_cast<size_t>(current_label_) >= local_label_parent_.size()) {
    local_label_parent_.resize(current_label_ + 1, 0);
  }
  local_label_parent_[current_label_] = current_label_;
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::AssignMinLabel(int row, int col, const std::vector<int>& neighbors) {
  int min_label = *std::ranges::min_element(neighbors.begin(), neighbors.end());
  local_output_[(row * cols_) + col] = min_label;

  for (int neighbor_label : neighbors) {
    if (neighbor_label != min_label) {
      UnionLabels(min_label, neighbor_label);
    }
  }
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::UnionLabels(int label1, int label2) {
  int local_label1 = label1 - base_label_;
  int local_label2 = label2 - base_label_;

  if (local_label1 < 0 || local_label1 >= static_cast<int>(local_label_parent_.size()) || local_label2 < 0 ||
      local_label2 >= static_cast<int>(local_label_parent_.size())) {
    return;
  }

  int root1 = FindRoot(local_label1);
  int root2 = FindRoot(local_label2);

  if (root1 != root2) {
    if (root1 < root2) {
      local_label_parent_[root2] = root1;
    } else {
      local_label_parent_[root1] = root2;
    }
  }
}

std::vector<int> naumov_b_marc_on_bin_image_all::TestTaskALL::FindAdjacentLabels(int row, int col) {
  std::vector<int> neighbors;
  if (col > 0 && local_output_[(row * cols_) + col - 1] != 0) {
    neighbors.push_back(local_output_[(row * cols_) + col - 1]);
  }
  if (row > 0 && local_output_[((row - 1) * cols_) + col] != 0) {
    neighbors.push_back(local_output_[((row - 1) * cols_) + col]);
  }
  return neighbors;
}

int naumov_b_marc_on_bin_image_all::TestTaskALL::FindRoot(int label) {
  while (local_label_parent_[label] != label) {
    label = local_label_parent_[label];
  }
  return label;
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::LocalLabeling() {
  local_output_.assign(local_rows_ * cols_, 0);
  current_label_ = 0;
  local_label_parent_.clear();

  for (int i = 0; i < local_rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (local_image_[(i * cols_) + j] == 1) {
        ProcessPixel(i, j);
      }
    }
  }

  const int num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  int rows_per_thread = (local_rows_ + num_threads - 1) / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    int start_row = t * rows_per_thread;
    int end_row = std::min(start_row + rows_per_thread, local_rows_);

    if (start_row < local_rows_) {
      threads.emplace_back([&, start_row, end_row]() {
        for (int i = start_row; i < end_row; ++i) {
          for (int j = 0; j < cols_; ++j) {
            if (local_output_[(i * cols_) + j] != 0) {
              int label = local_output_[(i * cols_) + j];
              int root = FindRoot(label - base_label_);
              local_output_[(i * cols_) + j] = root + base_label_;
            }
          }
        }
      });
    }
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::MergeLabelsBetweenProcesses() {
  if (num_procs_ == 1) {return;}

  std::vector<int> last_row(cols_, 0);
  std::vector<int> next_first_row(cols_, 0);

  if (rank_ < num_procs_ - 1 && local_rows_ > 0) {
    std::copy(local_output_.end() - cols_, local_output_.end(), last_row.begin());
    MPI_Send(last_row.data(), cols_, MPI_INT, rank_ + 1, 0, MPI_COMM_WORLD);
  }

  if (rank_ > 0) {
    MPI_Status status;
    MPI_Recv(next_first_row.data(), cols_, MPI_INT, rank_ - 1, 0, MPI_COMM_WORLD, &status);
  }

  std::vector<std::pair<int, int>> equivalences;
  if (rank_ > 0 && local_rows_ > 0) {
    for (int j = 0; j < cols_; ++j) {
      int global_label1 = next_first_row[j];
      int global_label2 = local_output_[j];
      if (global_label1 != 0 && global_label2 != 0) {
        equivalences.emplace_back(global_label1, global_label2);
      }
    }
  }

  int local_count = static_cast<int>(equivalences.size()) * 2;
  std::vector<int> counts(num_procs_, 0);
  std::vector<int> displs(num_procs_, 0);

  MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    int total = 0;
    for (int i = 0; i < num_procs_; ++i) {
      displs[i] = total;
      total += counts[i];
    }
    all_equivalences_.resize(total);
  }

  std::vector<int> local_flat;
  local_flat.reserve(equivalences.size() * 2);
  for (auto& p : equivalences) {
    local_flat.push_back(p.first);
    local_flat.push_back(p.second);
  }

  MPI_Gatherv(local_flat.data(), local_flat.size(), MPI_INT, all_equivalences_.data(), counts.data(), displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);
}

void naumov_b_marc_on_bin_image_all::TestTaskALL::UpdateGlobalLabels() {
  if (global_output_.empty()) return;

  std::map<int, int> parent;

  for (int label : global_output_) {
    if (label != 0 && parent.find(label) == parent.end()) {
      parent[label] = label;
    }
  }
  for (size_t i = 0; i < all_equivalences_.size(); i += 2) {
    int l1 = all_equivalences_[i];
    int l2 = all_equivalences_[i + 1];
    if (parent.find(l1) == parent.end()) parent[l1] = l1;
    if (parent.find(l2) == parent.end()) parent[l2] = l2;
  }

  std::function<int(int)> find = [&](int x) {
    if (parent[x] != x) {
      parent[x] = find(parent[x]);
    }
    return parent[x];
  };

  for (size_t i = 0; i < all_equivalences_.size(); i += 2) {
    int root1 = find(all_equivalences_[i]);
    int root2 = find(all_equivalences_[i + 1]);
    if (root1 != root2) {
      if (root1 < root2){
        parent[root2] = root1;
      }else{
        parent[root1] = root2;
      }
    }
  }

  for (int& label : global_output_) {
    if (label != 0) {label = find(label);}
  }

  std::map<int, int> renumber;
  int new_label = 1;
  for (int& label : global_output_) {
    if (label != 0) {
      if (renumber.find(label) == renumber.end()) {
        renumber[label] = new_label++;
      }
      label = renumber[label];
    }
  }
}

bool naumov_b_marc_on_bin_image_all::TestTaskALL::RunImpl() {
  LocalLabeling();
  MergeLabelsBetweenProcesses();

  std::vector<int> counts(num_procs_);
  std::vector<int> displs(num_procs_);
  for (int i = 0; i < num_procs_; ++i) {
    int lrows = base_rows_ + (i < remainder_ ? 1 : 0);
    counts[i] = lrows * cols_;
    displs[i] = (i * base_rows_ + std::min(i, remainder_)) * cols_;
  }

  if (rank_ == 0) {
    global_output_.resize(rows_ * cols_);
  } else {
    global_output_.resize(0);
  }

  MPI_Gatherv(local_output_.data(), counts[rank_], MPI_INT, global_output_.data(), counts.data(), displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    UpdateGlobalLabels();
  }

  return true;
}

bool naumov_b_marc_on_bin_image_all::TestTaskALL::PostProcessingImpl() {
  int* output_data = reinterpret_cast<int*>(task_data->outputs[0]);
  const int data_size = rows_ * cols_;

  if (rank_ == 0) {
    std::ranges::copy(global_output_.begin(), global_output_.end(), output_data);
  }

  MPI_Bcast(output_data, data_size, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}