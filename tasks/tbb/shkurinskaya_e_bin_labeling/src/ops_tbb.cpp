#include "tbb/shkurinskaya_e_bin_labeling/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <climits>
#include <vector>
namespace shkurinskaya_e_bin_labeling_tbb {

bool TaskTBB::ValidationImpl() {
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == task_data->inputs_count[0] &&
         task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1;
}

bool TaskTBB::PreProcessingImpl() {
  const int total_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + total_size);

  width_ = reinterpret_cast<int *>(task_data->inputs[1])[0];
  height_ = reinterpret_cast<int *>(task_data->inputs[2])[0];

  res_.resize(total_size);
  parent_.resize(total_size);
  rank_.resize(total_size);
  label_.resize(total_size);
  return true;
}

bool TaskTBB::RunImpl() {
  const int W = width_;
  const int H = height_;

  tbb::parallel_for(tbb::blocked_range<int>(0, H), [&](const tbb::blocked_range<int> &rows) {
    for (int i = rows.begin(); i < rows.end(); ++i) {
      int base = i * W;
      for (int j = 0; j < W; ++j) {
        int idx = base + j;
        if (input_[idx] == 1) {
          parent_[idx] = idx;
          rank_[idx] = 0;
        } else {
          parent_[idx] = -1;
        }
      }
    }
  });

  ProcessUnion();

  tbb::parallel_for(tbb::blocked_range<int>(0, H), [&](const tbb::blocked_range<int> &rows) {
    for (int i = rows.begin(); i < rows.end(); ++i) {
      int base = i * W;
      for (int j = 0; j < W; ++j) {
        int idx = base + j;
        if (input_[idx] == 1) {
          while (true) {
            int p = parent_[idx];
            if (p < 0) break;
            int gp = parent_[p];
            if (gp < 0) break;
            if (p == gp) break;
            parent_[idx] = gp;
          }
        }
      }
    }
  });

  return true;
}

void TaskTBB::ProcessUnion() {
  const int W = width_;
  const int H = height_;
  // 1) Собираем все пары соседних 1–1 пикселей (последовательно):
  std::vector<std::pair<int, int>> allPairs;
  allPairs.reserve(H * W / 2);  // грубая оценка; можно чуть больше или меньше

  for (int i = 0; i < H; ++i) {
    int base = i * W;
    for (int j = 0; j < W; ++j) {
      int idx = base + j;
      if (input_[idx] != 1) continue;
      // 1. Вправо
      if (j + 1 < W && input_[idx + 1] == 1) {
        allPairs.emplace_back(idx, idx + 1);
      }
      // 2. Вниз
      if (i + 1 < H && input_[(i + 1) * W + j] == 1) {
        allPairs.emplace_back(idx, (i + 1) * W + j);
      }
      // 3. Вниз-вправо
      if (i + 1 < H && j + 1 < W && input_[(i + 1) * W + (j + 1)] == 1) {
        allPairs.emplace_back(idx, (i + 1) * W + (j + 1));
      }
      // 4. Вниз-влево
      if (i + 1 < H && j > 0 && input_[(i + 1) * W + (j - 1)] == 1) {
        allPairs.emplace_back(idx, (i + 1) * W + (j - 1));
      }
    }
  }

  // 2) Параллельно обрабатываем каждую пару, вызывая UnionSets:
  tbb::parallel_for(tbb::blocked_range<size_t>(0, allPairs.size()), [&](const tbb::blocked_range<size_t> &range) {
    for (size_t z = range.begin(); z < range.end(); ++z) {
      const auto &pr = allPairs[z];
      UnionSets(pr.first, pr.second);
    }
  });
}

void TaskTBB::UnionSets(int idx_a, int idx_b) {
  int rootA = FindRoot(idx_a);
  int rootB = FindRoot(idx_b);
  if (rootA == rootB || rootA < 0 || rootB < 0) return;

  tbb::spin_mutex::scoped_lock lock(uf_mutex_);

  rootA = FindRoot(rootA);
  rootB = FindRoot(rootB);
  if (rootA == rootB) return;

  if (rank_[rootA] < rank_[rootB]) {
    std::swap(rootA, rootB);
  }
  parent_[rootB] = rootA;
  if (rank_[rootA] == rank_[rootB]) {
    rank_[rootA]++;
  }
}

int TaskTBB::FindRoot(int v) {
  int u = v;
  while (true) {
    int p = parent_[u];
    if (p < 0 || p == u) break;
    u = p;
  }
  if (u < 0) return -1;
  int root = u;
  u = v;
  while (true) {
    int p = parent_[u];
    if (p < 0 || p == root) break;
    parent_[u] = root;
    u = p;
  }
  return root;
}

bool TaskTBB::IsValidIndex(int i, int j) const { return (i >= 0 && i < height_ && j >= 0 && j < width_); }

bool TaskTBB::PostProcessingImpl() {
  const int W = width_;
  const int H = height_;

  std::fill(label_.begin(), label_.end(), 0);

  int comp = 1;

  for (int i = 0; i < H; ++i) {
    int base = i * W;
    for (int j = 0; j < W; ++j) {
      int idx = base + j;
      if (parent_[idx] < 0) {
        res_[idx] = 0;
        continue;
      }

      int root = idx;
      while (parent_[root] != root) {
        root = parent_[root];
      }

      if (label_[root] == 0) {
        label_[root] = comp++;
      }
      res_[idx] = label_[root];
    }
  }
  int *out_ptr = reinterpret_cast<int *>(task_data->outputs[0]);
  std::ranges::copy(res_.begin(), res_.end(), out_ptr);
  return true;
}

}  // namespace shkurinskaya_e_bin_labeling_tbb
