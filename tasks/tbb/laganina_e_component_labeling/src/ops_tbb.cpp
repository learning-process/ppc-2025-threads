#include "tbb/laganina_e_component_labeling/include/ops_tbb.hpp"

#include <oneapi/tbb/mutex.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "core/util/include/util.hpp"

using namespace oneapi::tbb;

bool laganina_e_component_labeling_tbb::TestTaskTBB::ValidationImpl() {
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
// 1234
bool laganina_e_component_labeling_tbb::TestTaskTBB::PreProcessingImpl() {
  m_ = static_cast<int>(task_data->inputs_count[0]);
  n_ = static_cast<int>(task_data->inputs_count[1]);
  binary_.resize(m_ * n_);
  const int* input = reinterpret_cast<const int*>(task_data->inputs[0]);
  std::copy_n(input, m_ * n_, binary_.begin());
  return true;
}

bool laganina_e_component_labeling_tbb::TestTaskTBB::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(binary_.cbegin(), binary_.cend(), output);
  return true;
}

void laganina_e_component_labeling_tbb::TestTaskTBB::InitializeParents(std::vector<int>& parent) {
  int size = m_ * n_;
  parallel_for(blocked_range<int>(0, size), [&](const blocked_range<int>& r) {
    for (int i = r.begin(); i != r.end(); ++i) {
      parent[i] = (binary_[i] != 0) ? i : -1;
    }
  });
}

void laganina_e_component_labeling_tbb::TestTaskTBB::ProcessSweep(bool reverse, std::vector<int>& parent,
                                                                  bool changed) const {
  changed = parallel_reduce(
      blocked_range<int>(0, m_), false,
      [&](const blocked_range<int>& r, bool local_changed) {
        for (int row_idx = r.begin(); row_idx != r.end(); ++row_idx) {
          local_changed |= ProcessRow(row_idx, reverse, parent);
        }
        return local_changed;
      },
      [](bool a, bool b) { return a || b; });
}

bool laganina_e_component_labeling_tbb::TestTaskTBB::ProcessRow(int row_idx, bool reverse,
                                                                std::vector<int>& parent) const {
  const int row = reverse ? m_ - 1 - row_idx : row_idx;
  bool row_changed = false;

  for (int col_idx = 0; col_idx < n_; ++col_idx) {
    const int col = reverse ? n_ - 1 - col_idx : col_idx;
    const int current = (row * n_) + col;

    if (parent[current] == -1) continue;

    // Check vertical neighbor
    const int vert_neighbor_row = row - (reverse ? -1 : 1);
    if (vert_neighbor_row >= 0 && vert_neighbor_row < m_) {
      const int vert_neighbor = (vert_neighbor_row * n_) + col;
      if (parent[vert_neighbor] != -1) {
        row_changed |= UnionNodes(current, vert_neighbor, parent);
      }
    }

    // Check horizontal neighbor
    const int horz_neighbor_col = col - (reverse ? -1 : 1);
    if (horz_neighbor_col >= 0 && horz_neighbor_col < n_) {
      const int horz_neighbor = (row * n_) + horz_neighbor_col;
      if (parent[horz_neighbor] != -1) {
        row_changed |= UnionNodes(current, horz_neighbor, parent);
      }
    }
  }

  return row_changed;
}

int laganina_e_component_labeling_tbb::TestTaskTBB::FindRoot(std::vector<int>& parent, int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];  // Path compression
    x = parent[x];
  }
  return x;
}

bool laganina_e_component_labeling_tbb::TestTaskTBB::UnionNodes(int a, int b, std::vector<int>& parent) {
  int root_a = FindRoot(parent, a);
  int root_b = FindRoot(parent, b);

  if (root_a != root_b) {
    if (root_b < root_a) std::swap(root_a, root_b);
    parent[root_b] = root_a;
    return true;
  }
  return false;
}

void laganina_e_component_labeling_tbb::TestTaskTBB::FinalizeRoots(std::vector<int>& parent) const {
  int size = m_ * n_;
  parallel_for(blocked_range<int>(0, size), [&](const blocked_range<int>& r) {
    for (int i = r.begin(); i != r.end(); ++i) {
      if (parent[i] != -1) {
        parent[i] = FindRoot(parent, i);
      }
    }
  });
}

void laganina_e_component_labeling_tbb::TestTaskTBB::AssignLabels(std::vector<int>& parent) {
  std::vector<int> labels(m_ * n_ + 1, 0);
  int current_label = 1;
  tbb::mutex label_mutex;

  // First pass: collect all unique roots
  combinable<std::unordered_set<int>> unique_roots;
  parallel_for(blocked_range<int>(0, m_ * n_), [&](const blocked_range<int>& r) {
    for (int i = r.begin(); i != r.end(); ++i) {
      if (parent[i] != -1 && parent[i] == i) {
        unique_roots.local().insert(i);
      }
    }
  });

  // Assign labels to root
  unique_roots.combine_each([&](const std::unordered_set<int>& local_roots) {
    tbb::mutex::scoped_lock lock(label_mutex);
    for (int root : local_roots) {
      if (labels[root] == 0) {
        labels[root] = current_label++;
      }
    }
  });

  // Second pass: assign labels to all pixels
  parallel_for(blocked_range<int>(0, m_ * n_), [&](const blocked_range<int>& r) {
    for (int i = r.begin(); i != r.end(); ++i) {
      binary_[i] = (parent[i] != -1) ? labels[parent[i]] : 0;
    }
  });
}

void laganina_e_component_labeling_tbb::TestTaskTBB::LabelConnectedComponents() {
  std::vector<int> parent(m_ * n_);
  InitializeParents(parent);

  constexpr int kMaxIterations = 100;
  int iterations = 0;
  bool changed;

  do {
    bool forward_changed = false;
    bool backward_changed = false;

    ProcessSweep(false, parent, forward_changed);
    ProcessSweep(true, parent, backward_changed);

    changed = forward_changed || backward_changed;
  } while (changed && ++iterations < kMaxIterations);

  FinalizeRoots(parent);
  AssignLabels(parent);
}

bool laganina_e_component_labeling_tbb::TestTaskTBB::RunImpl() {
  task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&] { LabelConnectedComponents(); });
  return true;
}