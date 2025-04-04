#include "omp/shkurinskaya_e_bin_labeling/include/ops_omp.hpp"

#include <algorithm>
#include <climits>
#include <iostream>
#include <vector>

int shkurinskaya_e_bin_labeling_omp::TaskOMP::FindRoot(int index) {
  while (parent_[index] != index) {
    int grandparent = parent_[parent_[index]];
    parent_[index] = grandparent;
    index = grandparent;
  }
  return index;
}

void shkurinskaya_e_bin_labeling_omp::TaskOMP::UnionSets(int index_a, int index_b) {
  int root_a = findRoot(index_a);
  int root_b = findRoot(index_b);

  if (root_a != root_b) {
    {
      if (rank_[root_a] < rank_[root_b]) {
        parent_[root_a] = root_b;
      } else if (rank_[root_a] > rank_[root_b]) {
        parent_[root_b] = root_a;
      } else {
        parent_[root_b] = root_a;
        rank_[root_a]++;
      }
    }
  }
}

bool shkurinskaya_e_bin_labeling_omp::TaskOMP::PreProcessingImpl() {
  // Init value for input and output
  std::cout << "PreProcessingImpl: Initializing inputs and outputs...\n";
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  width_ = reinterpret_cast<int*>(task_data->inputs[1])[0];
  height_ = reinterpret_cast<int*>(task_data->inputs[2])[0];
  std::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], input_.begin());
  // Init value for output
  res_.assign(task_data->inputs_count[0], 0);
  parent_.assign(task_data->inputs_count[0], 0);
  rank_.assign(task_data->inputs_count[0], 0);
  label_.assign(task_data->inputs_count[0], 0);
  return true;
}

bool shkurinskaya_e_bin_labeling_omp::TaskOMP::ValidationImpl() {
  std::cout << "ValidationImpl: Validating input data...\n";
  // Check count elements of output
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == task_data->inputs_count[0] &&
         task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1;
}

bool shkurinskaya_e_bin_labeling_omp::TaskOMP::RunImpl() {
  std::cout << "[DEBUG] RunImpl: Starting processing...\n";

  // Первый этап
#pragma omp parallel for
  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      int index = (i * width_) + j;
      if (input_[index] == 1) {
        parent_[index] = index;
        rank_[index] = 0;
      } else {
        parent_[index] = -1;
      }
    }
  }

  // Второй этап
#pragma omp parallel for
  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      int index = (i * width_) + j;
      if (input_[index] != 1) { continue; }

      const int directions[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

      for (int d = 0; d < 8; ++d) {
        int ni = i + directions[d][0];
        int nj = j + directions[d][1];

        if (ni >= 0 && ni < height_ && nj >= 0 && nj < width_) {
          int neighbor_index = (ni * width_) + nj;

          if (input_[neighbor_index] == 1) {
            int root_a = index;
            int root_b = neighbor_index;

            // Находим корни
            while (parent_[root_a] != root_a) {
              root_a = parent_[root_a];
            }
            while (parent_[root_b] != root_b) {
              root_b = parent_[root_b];
            }

            // Объединение
            if (root_a != root_b) {
              if (rank_[root_a] < rank_[root_b]) {
                parent_[root_a] = root_b;
              } else if (rank_[root_a] > rank_[root_b]) {
                parent_[root_b] = root_a;
              } else {
                parent_[root_b] = root_a;
                rank_[root_a]++;
              }
            }
          }
        }
      }
    }
  }

  // Третий этап
#pragma omp parallel for
  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      int index = (i * width_) + j;
      if (input_[index] == 1) {
        while (parent_[index] != parent_[parent_[index]]) {
          parent_[index] = parent_[parent_[index]];
        }
      }
    }
  }
  std::cout << "[DEBUG] RunImpl: Processing completed\n";
  return true;
}

bool shkurinskaya_e_bin_labeling_omp::TaskOMP::PostProcessingImpl() {
  std::cout << "PostProcessingImpl: Starting post-processing...\n";
  // mark the parent_ with smallest label
  int comp = 1;
  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      int index = (i * width_) + j;
      int root = index;
      if (parent_[root] == -1) {
        continue;
      }
      while (parent_[root] != root) {
        root = parent_[root];
      }
      if (label_[parent_[root]] == 0) {
        label_[parent_[root]] = comp++;
      }
      res_[index] = label_[parent_[root]];
    }
  }
  std::ranges::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
