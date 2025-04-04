#include "omp/laganina_e_component_labeling/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <vector>

#include "core/task/include/task.hpp"

namespace laganina_e_component_labeling_omp {

// Helper function for path compression
void CompressPath(std::vector<int>& parent, int node, int& root) {
  while (parent[node] != node) {
    parent[node] = parent[parent[node]];  // Path compression
    node = parent[node];
  }
  root = node;
}

// NormalizeLabels implementation
void NormalizeLabels(std::vector<int>& vec) {
  std::unordered_map<int, int> label_map;
  int current_label = 1;

#pragma omp parallel
  {
    std::unordered_map<int, int> local_map;
#pragma omp for nowait
    for (size_t i = 0; i < vec.size(); ++i) {
      if (vec[i] != 0) {
        local_map.try_emplace(vec[i], 0);
      }
    }

#pragma omp critical
    {
      for (const auto& [key, _] : local_map) {
        if (label_map.find(key) == label_map.end()) {
          label_map[key] = current_label++;
        }
      }
    }
  }

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] != 0) {
      vec[i] = label_map[vec[i]];
    }
  }
}

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
  std::ranges::copy(binary_.cbegin(), binary_.cend(), output);
  return true;
}

bool TestTaskOpenMP::RunImpl() {
  LabelConnectedComponents();
  return true;
}

void TestTaskOpenMP::InitializeParents(std::vector<int>& parent) {
  const int size = m_ * n_;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < size; ++i) {
    parent[i] = binary_[i] ? i : -1;
  }
}

void TestTaskOpenMP::ProcessSweep(bool reverse, std::vector<int>& parent, bool& changed) {
  bool local_changed = false;

#pragma omp parallel for reduction(|| : local_changed) schedule(static)
  for (int row_idx = 0; row_idx < m_; ++row_idx) {
    const int row = reverse ? m_ - 1 - row_idx : row_idx;

    for (int col_idx = 0; col_idx < n_; ++col_idx) {
      const int col = reverse ? n_ - 1 - col_idx : col_idx;
      const int current = row * n_ + col;

      if (parent[current] == -1) continue;

      const std::pair<int, int> neighbors[] = {{row - (reverse ? -1 : 1), col}, {row, col - (reverse ? -1 : 1)}};

      for (const auto& [nr, nc] : neighbors) {
        if (nr >= 0 && nr < m_ && nc >= 0 && nc < n_) {
          const int neighbor = nr * n_ + nc;
          if (parent[neighbor] != -1) {
            UnionNodes(current, neighbor, parent, local_changed);
          }
        }
      }
    }
  }
  changed = local_changed;
}

int TestTaskOpenMP::FindRoot(std::vector<int>& parent, int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

void TestTaskOpenMP::UnionNodes(int a, int b, std::vector<int>& parent, bool& changed) {
  int root_a = FindRoot(parent, a);
  int root_b = FindRoot(parent, b);

  if (root_a != root_b) {
    if (root_b < root_a) std::swap(root_a, root_b);

#pragma omp atomic write
    parent[root_b] = root_a;

    changed = true;
  }
}

void TestTaskOpenMP::FinalizeRoots(std::vector<int>& parent) {
  const int size = m_ * n_;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < size; ++i) {
    if (parent[i] != -1) {
      parent[i] = FindRoot(parent, i);
    }
  }
}

void TestTaskOpenMP::AssignLabels(std::vector<int>& parent) {
  std::vector<int> labels(m_ * n_ + 1, 0);
  int current_label = 1;

#pragma omp parallel
  {
    std::vector<int> local_roots;
#pragma omp for nowait
    for (int i = 0; i < m_ * n_; ++i) {
      if (parent[i] != -1 && parent[i] == i) {
        local_roots.push_back(i);
      }
    }

#pragma omp critical
    {
      for (int root : local_roots) {
        if (labels[root] == 0) {
          labels[root] = current_label++;
        }
      }
    }
  }

#pragma omp parallel for schedule(static)
  for (int i = 0; i < m_ * n_; ++i) {
    binary_[i] = parent[i] != -1 ? labels[parent[i]] : 0;
  }
}

void TestTaskOpenMP::LabelConnectedComponents() {
  std::vector<int> parent(m_ * n_);
  InitializeParents(parent);

  constexpr int kMaxIterations = 100;
  bool changed;
  int iterations = 0;

  do {
    changed = false;
    ProcessSweep(false, parent, changed);
    ProcessSweep(true, parent, changed);
  } while (changed && ++iterations < kMaxIterations);

  FinalizeRoots(parent);
  AssignLabels(parent);
}

}  // namespace laganina_e_component_labeling_omp