#pragma once

#include <unordered_map>
#include <vector>

#include "core/task/include/task.hpp"

namespace laganina_e_component_labeling_omp {

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int m_;
  int n_;

  std::vector<int> binary_;

  void InitializeParents(std::vector<int>& parent);
  void ProcessSweep(bool reverse, std::vector<int>& parent, bool& changed);
  void UnionNodes(int a, int b, std::vector<int>& parent, bool& changed);
  bool CheckNeighbor(int nr, int nc, int current, std::vector<int>& parent);
  bool ProcessRow(int row_idx, bool reverse, std::vector<int>& parent);
  int FindRoot(std::vector<int>& parent, int x);
  void FinalizeRoots(std::vector<int>& parent);
  void AssignLabels(std::vector<int>& parent);
  void LabelConnectedComponents();
};

inline void NormalizeLabels(std::vector<int>& vec) {
  std::unordered_map<int, int> label_map;
  int current_label = 1;
  {
    std::unordered_map<int, int> local_map;
    for (int unsigned int i = 0; i < vec.size(); ++i) {
      if (vec[i] != 0) {
        local_map.try_emplace(vec[i], 0);
      }
    }
    {
      for (const auto& [key, _] : local_map) {
        if (label_map.find(key) == label_map.end()) {
          label_map[key] = current_label++;
        }
      }
    }
  }
  for (unsigned int i = 0; i < vec.size(); ++i) {
    if (vec[i] != 0) {
      vec[i] = label_map[vec[i]];
    }
  }
}

void CompressPath(std::vector<int>& parent, int node, int& root);

}  // namespace laganina_e_component_labeling_omp