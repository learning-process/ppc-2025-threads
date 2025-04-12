#pragma once

#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>

#include <unordered_set>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace laganina_e_component_labeling_tbb {

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;

  bool PreProcessingImpl() override;

  bool PostProcessingImpl() override;
  bool RunImpl() override;

 private:
  int rows;
  int cols;
  std::vector<int> data;

  struct UnionFind {
    std::vector<int> parent;

    UnionFind(int size, std::vector<int> data) : parent(size) {
      tbb::parallel_for(0, size, [&](int i) { parent[i] = data[i] ? i : -1; });
    }

    int Find(int x);
    void Unite(int x, int y);
  };
  void Process_Components(UnionFind& uf);
  void Assign_Final_Labels(int size, UnionFind uf);
  void Process_Range(const tbb::blocked_range2d<int>& range, UnionFind& uf);
  void Process_Row(int row, const tbb::blocked_range<int>& col_range, UnionFind& uf);
  void Check_All_Neighbors(int row, int col, int idx, UnionFind& uf);
  void Label_Components();
};

inline void NormalizeLabels(std::vector<int> vec) {
  std::vector<int> unique_labels;
  std::unordered_set<int> seen;

  for (int val : vec) {
    if (val != 0 && seen.find(val) == seen.end()) {
      unique_labels.push_back(val);
      seen.insert(val);
    }
  }

  std::unordered_map<int, int> label_map;
  int current_label = 1;
  for (int val : unique_labels) {
    label_map[val] = current_label++;
  }
  for (int& val : vec) {
    if (val != 0) {
      val = label_map[val];
    }
  }
}

}  // namespace laganina_e_component_labeling_tbb
