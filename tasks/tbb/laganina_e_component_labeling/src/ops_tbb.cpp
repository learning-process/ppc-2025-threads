#include "tbb/laganina_e_component_labeling/include/ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "core/task/include/task.hpp"

bool laganina_e_component_labeling_tbb::TestTaskTBB::ValidationImpl() {
  if (!task_data || !task_data->inputs[0] || !task_data->outputs[0]) return false;

  const int* input = reinterpret_cast<int*>(task_data->inputs[0]);
  const int size = task_data->inputs_count[0] * task_data->inputs_count[1];

  return std::all_of(input, input + size, [](int val) { return val == 0 || val == 1; });
}

bool laganina_e_component_labeling_tbb::TestTaskTBB::PreProcessingImpl() {
  rows_ = task_data->inputs_count[0];
  cols_ = task_data->inputs_count[1];
  const int size = rows_ * cols_;

  data_.resize(size);
  const int* input = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(input, input + size, data_.begin());

  return true;
}

bool laganina_e_component_labeling_tbb::TestTaskTBB::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(data_.begin(), data_.end(), output);
  return true;
}

bool laganina_e_component_labeling_tbb::TestTaskTBB::RunImpl() {
  label_components();
  return true;
}

int laganina_e_component_labeling_tbb::TestTaskTBB::UnionFind::find(int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

void laganina_e_component_labeling_tbb::TestTaskTBB::UnionFind::unite(int x, int y) {
  int rx = find(x);
  int ry = find(y);
  if (rx != ry && rx != -1 && ry != -1) {
    if (rx < ry) {
      parent[ry] = rx;
    } else {
      parent[rx] = ry;
    }
  }
}

void laganina_e_component_labeling_tbb::TestTaskTBB::process_components(UnionFind& uf) {
  tbb::parallel_for(tbb::blocked_range2d<int>(0, rows_, 16, 0, cols_, 64), [&](const tbb::blocked_range2d<int>& r) {
    for (int i = r.rows().begin(); i < r.rows().end(); ++i) {
      for (int j = r.cols().begin(); j < r.cols().end(); ++j) {
        const int idx = i * cols_ + j;
        if (data_[idx] == 0) continue;

        if (j > 0 && data_[idx - 1]) uf.unite(idx, idx - 1);
        if (i > 0 && data_[idx - cols_]) uf.unite(idx, idx - cols_);
        if (j < cols_ - 1 && data_[idx + 1]) uf.unite(idx, idx + 1);
        if (i < rows_ - 1 && data_[idx + cols_]) uf.unite(idx, idx + cols_);
      }
    }
  });
}

void laganina_e_component_labeling_tbb::TestTaskTBB::assign_final_labels(int size, UnionFind uf) {
  tbb::concurrent_unordered_map<int, int> label_map;

  tbb::parallel_for(0, size, [&](int i) {
    if (data_[i]) data_[i] = uf.find(i) + 1;
  });

  tbb::parallel_for(0, size, [&](int i) {
    if (data_[i] > 0) label_map.insert({data_[i], 0});
  });

  std::vector<int> keys;
  for (auto& p : label_map) keys.push_back(p.first);
  std::sort(keys.begin(), keys.end());

  int next_label = 1;
  for (auto& k : keys) label_map[k] = next_label++;

  tbb::parallel_for(0, size, [&](int i) {
    if (data_[i] > 0) data_[i] = label_map[data_[i]];
  });
}

void laganina_e_component_labeling_tbb::TestTaskTBB::label_components() {
  const int size = rows_ * cols_;
  UnionFind uf(size, data_);

  process_components(uf);
  // hh99ib
  assign_final_labels(size, uf);
}
