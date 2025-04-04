#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/task/include/task.hpp"

namespace laganina_e_component_labeling_omp {

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data);

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
  int FindRoot(std::vector<int>& parent, int x);
  void FinalizeRoots(std::vector<int>& parent);
  void AssignLabels(std::vector<int>& parent);
  void LabelConnectedComponents();
};

}  // namespace laganina_e_component_labeling_omp