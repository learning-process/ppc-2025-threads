#include "tbb/laganina_e_component_labeling/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/concurrent_unordered_set.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "core/task/include/task.hpp"

bool laganina_e_component_labeling_tbb::TestTaskTBB::ValidationImpl() {
  if ((task_data == nullptr) || (task_data->inputs[0] == nullptr) || (task_data->outputs[0] == nullptr)) {
    return false;
  }
  const unsigned int* input = reinterpret_cast<unsigned int*>(task_data->inputs[0]);
  const unsigned int size = task_data->inputs_count[0] * task_data->inputs_count[1];

  return std::all_of(input, input + size, [](int val) { return val == 0 || val == 1; });
}

bool laganina_e_component_labeling_tbb::TestTaskTBB::PreProcessingImpl() {
  rows = static_cast<int>(task_data->inputs_count[0]);
  cols = static_cast<int>(task_data->inputs_count[1]);
  const int size = rows * cols;

  data.resize(size);
  const int* input = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(input, input + size, data.begin());

  return true;
}

bool laganina_e_component_labeling_tbb::TestTaskTBB::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(data.begin(), data.end(), output);
  return true;
}

bool laganina_e_component_labeling_tbb::TestTaskTBB::RunImpl() {
  Label_Components();
  return true;
}

int laganina_e_component_labeling_tbb::TestTaskTBB::UnionFind::Find(int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

void laganina_e_component_labeling_tbb::TestTaskTBB::UnionFind::Unite(int x, int y) {
  int rx = Find(x);
  int ry = Find(y);
  if ((rx != ry) && (rx != -1) && (ry != -1)) {
    if (rx < ry) {
      parent[ry] = rx;
    } else {
      parent[rx] = ry;
    }
  }
}

void laganina_e_component_labeling_tbb::TestTaskTBB::Assign_Final_Labels(int size, UnionFind uf) {
  tbb::concurrent_unordered_map<int, int> label_map;

  tbb::parallel_for(0, size, [&](int i) {
    if (data[i]) data[i] = uf.Find(i) + 1;
  });

  tbb::parallel_for(0, size, [&](int i) {
    if (data[i] > 0) {
      label_map.insert({data[i], 0});
    }
  });

  std::vector<int> keys;
  for (auto& p : label_map) {
    keys.push_back(p.first);
  }
  std::sort(keys.begin(), keys.end());

  int next_label = 1;
  for (auto& k : keys) {
    label_map[k] = next_label++;
  }

  tbb::parallel_for(0, size, [&](int i) {
    if (data[i] > 0) {
      data[i] = label_map[data[i]];
    }
  });
}

void laganina_e_component_labeling_tbb::TestTaskTBB::Label_Components() {
  const int size = rows * cols;
  UnionFind uf(size, data);

  Process_Components(uf);
  // hh99ib
  Assign_Final_Labels(size, uf);
}
void laganina_e_component_labeling_tbb::TestTaskTBB::Process_Components(UnionFind& uf) {
  tbb::parallel_for(tbb::blocked_range2d<int>(0, rows, 16, 0, cols, 64), [&](const auto& r) { Process_Range(r, uf); });
}

void laganina_e_component_labeling_tbb::TestTaskTBB::Process_Range(const tbb::blocked_range2d<int>& range,
                                                                   UnionFind& uf) {
  for (int i = range.rows().begin(); i < range.rows().end(); ++i) {
    const tbb::blocked_range<int> col_range = range.cols();
    Process_Row(i, col_range, uf);
  }
}

void laganina_e_component_labeling_tbb::TestTaskTBB::Process_Row(int row, const tbb::blocked_range<int>& col_range,
                                                                 UnionFind& uf) {
  for (int j = col_range.begin(); j < col_range.end(); ++j) {
    const int idx = (row * cols) + j;
    if (data[idx] == 0) {
      continue;
    }

    Check_All_Neighbors(row, j, idx, uf);
  }
}
void laganina_e_component_labeling_tbb::TestTaskTBB::Check_All_Neighbors(int row, int col, int idx, UnionFind& uf) {
  if (col > 0 && data[idx - 1]) {
    uf.Unite(idx, idx - 1);
  }
  if (row > 0 && data[idx - cols]) {
    uf.Unite(idx, idx - cols);
  }

  if (col < cols - 1 && data[idx + 1]) {
    uf.Unite(idx, idx + 1);
  }

  if (row < rows - 1 && data[idx + cols]) {
    uf.Unite(idx, idx + cols);
  }
}