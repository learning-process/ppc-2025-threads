#include "omp/laganina_e_component_labeling/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace laganina_e_component_labeling_omp {

TestTaskOpenMP::TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool TestTaskOpenMP::ValidationImpl() {
  if (task_data == nullptr || task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }

  const auto size = static_cast<int>(task_data->inputs_count[0] * task_data->inputs_count[1]);
  const int* input = reinterpret_cast<const int*>(task_data->inputs[0]);

  for (int i = 0; i < size; ++i) {
    if (input[i] != 0 && input[i] != 1) {
      return false;
    }
  }
  return true;
}

bool TestTaskOpenMP::PreProcessingImpl() {
  m_ = static_cast<int>(task_data->inputs_count[0]);
  n_ = static_cast<int>(task_data->inputs_count[1]);

  binary_.resize(m_ * n_);
  const int* input = reinterpret_cast<const int*>(task_data->inputs[0]);
  std::copy_n(input, m_ * n_, binary_.begin());

  return true;
}

bool TestTaskOpenMP::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(binary_.cbegin(), binary_.cend(), output);
  return true;
}

bool TestTaskOpenMP::RunImpl() {
  LabelConnectedComponents();
  return true;
}

namespace {
void CompressPath(std::vector<int>& parent, int node, int& root) {
  while (parent[node] != node) {
    parent[node] = parent[parent[node]];
    node = parent[node];
  }
  root = node;
}

void ProcessNeighbor(int idx, int neighbor_idx, std::vector<int>& parent, std::vector<int>& binary, bool& changed) {
  if (binary[neighbor_idx] != 1) return;

  int root;
  CompressPath(parent, idx, root);

  int neighbor_root;
  CompressPath(parent, neighbor_idx, neighbor_root);

  if (root != neighbor_root) {
    const int new_root = std::min(root, neighbor_root);
#pragma omp atomic write
    parent[root] = new_root;
#pragma omp atomic write
    parent[neighbor_root] = new_root;
    changed = true;
  }
}
}  // namespace

void TestTaskOpenMP::LabelConnectedComponents() {
  const int size = m_ * n_;
  std::vector<int> parent(size);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < size; ++i) {
    parent[i] = (binary_[i] == 1) ? i : -1;
  }

  bool changed = false;
  int iterations = 0;
  constexpr int kMaxIterations = 100;

  do {
    changed = false;
    iterations++;

// Left-Right Top-Bottom pass
#pragma omp parallel for reduction(|| : changed) schedule(dynamic)
    for (int i = 0; i < m_; ++i) {
      for (int j = 0; j < n_; ++j) {
        const int idx = (i * n_) + j;
        if (binary_[idx] != 1) continue;

        if (j > 0) ProcessNeighbor(idx, idx - 1, parent, binary_, changed);
        if (i > 0) ProcessNeighbor(idx, idx - n_, parent, binary_, changed);
      }
    }

// Right-Left Bottom-Top pass
#pragma omp parallel for reduction(|| : changed) schedule(dynamic)
    for (int i = m_ - 1; i >= 0; --i) {
      for (int j = n_ - 1; j >= 0; --j) {
        const int idx = (i * n_) + j;
        if (binary_[idx] != 1) continue;

        if (j < n_ - 1) ProcessNeighbor(idx, idx + 1, parent, binary_, changed);
        if (i < m_ - 1) ProcessNeighbor(idx, idx + n_, parent, binary_, changed);
      }
    }
  } while (changed && iterations < kMaxIterations);

// Final path compression and labeling
#pragma omp parallel for schedule(static)
  for (int i = 0; i < size; ++i) {
    if (binary_[i] == 1) {
      int root;
      CompressPath(parent, i, root);
      binary_[i] = root + 2;
    }
  }

  // Label normalization
  std::vector<int> label_map(size + 2, 0);
  std::atomic<int> next_label{2};

#pragma omp parallel
  {
    std::unordered_map<int, int> local_map;

#pragma omp for nowait
    for (int i = 0; i < size; ++i) {
      if (binary_[i] >= 2) {
        local_map.try_emplace(binary_[i], 0);
      }
    }

#pragma omp critical
    {
      for (const auto& [key, _] : local_map) {
        if (label_map[key] == 0) {
          label_map[key] = next_label++;
        }
      }
    }
  }

// Apply normalized labels
#pragma omp parallel for schedule(static)
  for (int i = 0; i < size; ++i) {
    if (binary_[i] >= 2) {
      binary_[i] = label_map[binary_[i]];
    }
  }
}

}  // namespace laganina_e_component_labeling_omp