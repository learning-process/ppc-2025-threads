#include "stl/laganina_e_component_labeling/include/ops_stl.hpp"

#include <algorithm>
#include <atomic>
#include <execution>
#include <functional>
#include <numeric>
#include <ranges>
#include <vector>

bool laganina_e_component_labeling_stl::TestTaskSTL::ValidationImpl() {
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

bool laganina_e_component_labeling_stl::TestTaskSTL::PreProcessingImpl() {
  m_ = static_cast<int>(task_data->inputs_count[0]);
  n_ = static_cast<int>(task_data->inputs_count[1]);
  binary_.resize(m_ * n_);
  const int* input = reinterpret_cast<const int*>(task_data->inputs[0]);
  std::copy_n(input, m_ * n_, binary_.begin());
  return true;
}

bool laganina_e_component_labeling_stl::TestTaskSTL::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(binary_.cbegin(), binary_.cend(), output);
  return true;
}

void laganina_e_component_labeling_stl::TestTaskSTL::InitializeParents(std::vector<int>& parent) {
  const int size = m_ * n_;

  std::for_each(std::execution::par_unseq, std::views::iota(0, size).begin(), std::views::iota(0, size).end(),
                [&](int i) { parent[i] = binary_[i] ? i : -1; });
}

void laganina_e_component_labeling_stl::TestTaskSTL::ProcessSweep(bool reverse, std::vector<int>& parent,
                                                                  bool& changed) const {
  changed = std::transform_reduce(std::execution::par_unseq, std::views::iota(0, m_).begin(),
                                  std::views::iota(0, m_).end(), false, std::logical_or<>(),
                                  [&](int row_idx) { return ProcessRow(row_idx, reverse, parent); });
}

bool laganina_e_component_labeling_stl::TestTaskSTL::ProcessRow(int row_idx, bool reverse,
                                                                std::vector<int>& parent) const {
  const int row = reverse ? m_ - 1 - row_idx : row_idx;
  bool row_changed = false;

  for (int col_idx = 0; col_idx < n_; ++col_idx) {
    const int col = reverse ? n_ - 1 - col_idx : col_idx;
    const int current = (row * n_) + col;

    if (parent[current] == -1) {
      continue;
    }

    const int vert_neighbor_row = row - (reverse ? -1 : 1);
    const int vert_neighbor = (vert_neighbor_row * n_) + col;
    if (vert_neighbor_row >= 0 && vert_neighbor_row < m_ && parent[vert_neighbor] != -1) {
      row_changed |= UnionNodes(current, vert_neighbor, parent);
    }

    const int horz_neighbor_col = col - (reverse ? -1 : 1);
    const int horz_neighbor = (row * n_) + horz_neighbor_col;
    if (horz_neighbor_col >= 0 && horz_neighbor_col < n_ && parent[horz_neighbor] != -1) {
      row_changed |= UnionNodes(current, horz_neighbor, parent);
    }
  }

  return row_changed;
}

bool laganina_e_component_labeling_stl::TestTaskSTL::CheckNeighbor(int nr, int nc, int current,
                                                                   std::vector<int>& parent) const {
  if (nr >= 0 && nr < m_ && nc >= 0 && nc < n_) {
    const int neighbor = (nr * n_) + nc;
    if (parent[neighbor] != -1) {
      return UnionNodes(current, neighbor, parent);
    }
  }
  return false;
}

int laganina_e_component_labeling_stl::TestTaskSTL::FindRoot(std::vector<int>& parent, int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

bool laganina_e_component_labeling_stl::TestTaskSTL::UnionNodes(int a, int b, std::vector<int>& parent) {
  int root_a = FindRoot(parent, a);
  int root_b = FindRoot(parent, b);

  if (root_a != root_b) {
    if (root_b < root_a) {
      std::swap(root_a, root_b);
    }

    parent[root_b] = root_a;

    return true;
  }
  return false;
}

void laganina_e_component_labeling_stl::TestTaskSTL::FinalizeRoots(std::vector<int>& parent) const {
  const int size = m_ * n_;

  std::for_each(std::execution::par_unseq, std::views::iota(0, size).begin(), std::views::iota(0, size).end(),
                [&](int i) {
                  if (parent[i] != -1) {
                    parent[i] = FindRoot(parent, i);
                  }
                });
}

void laganina_e_component_labeling_stl::TestTaskSTL::AssignLabels(std::vector<int>& parent) {
  const int size = m_ * n_;
  std::vector<std::atomic<int>> labels(size + 1);

  std::for_each(std::execution::par_unseq, std::views::iota(0, size + 1).begin(), std::views::iota(0, size + 1).end(),
                [&](int i) { labels[i].store(0, std::memory_order_relaxed); });

  std::atomic<int> current_label = 1;

  std::for_each(std::execution::par_unseq, std::views::iota(0, size).begin(), std::views::iota(0, size).end(),
                [&](int i) {
                  if (parent[i] == i) {
                    int expected = 0;
                    labels[i].compare_exchange_strong(expected, current_label++, std::memory_order_release,
                                                      std::memory_order_relaxed);
                    // labels[i].compare_exchange_strong(expected, current_label++, std::memory_order_relaxed,
                    //                                   std::memory_order_relaxed);
                  }
                });

  std::for_each(std::execution::par_unseq, std::views::iota(0, size).begin(), std::views::iota(0, size).end(),
                [&](int i) { binary_[i] = (parent[i] != -1) ? labels[parent[i]].load(std::memory_order_acquire) : 0; });
  /*std::for_each(std::execution::par_unseq, std::views::iota(0, size).begin(), std::views::iota(0, size).end(),
                [&](int i) { binary_[i] = (parent[i] != -1) ? labels[parent[i]].load(std::memory_order_relaxed) : 0;
     });*/
}

void laganina_e_component_labeling_stl::TestTaskSTL::LabelConnectedComponents() {
  std::vector<int> parent(m_ * n_);
  InitializeParents(parent);

  constexpr int kMaxIterations = 100;
  bool changed = false;
  int iterations = 0;

  do {
    changed = false;
    ProcessSweep(false, parent, changed);
    ProcessSweep(true, parent, changed);
  } while (changed && ++iterations < kMaxIterations);

  FinalizeRoots(parent);
  AssignLabels(parent);
}

bool laganina_e_component_labeling_stl::TestTaskSTL::RunImpl() {
  LabelConnectedComponents();
  return true;
}
