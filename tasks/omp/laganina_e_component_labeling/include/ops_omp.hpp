#pragma once

#include <unordered_map>
#include <unordered_set>
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
  bool UnionNodes(int a, int b, std::vector<int>& parent);
  bool CheckNeighbor(int nr, int nc, int current, std::vector<int>& parent);
  bool ProcessRow(int row_idx, bool reverse, std::vector<int>& parent);
  int FindRoot(std::vector<int>& parent, int x);
  void FinalizeRoots(std::vector<int>& parent);
  void AssignLabels(std::vector<int>& parent);
  void LabelConnectedComponents();
};

inline void NormalizeLabels(std::vector<int>& vec) {
  std::vector<int> unique_labels;  // Хранит уникальные значения в порядке их первого появления
  std::unordered_set<int> seen;  // Для быстрой проверки уникальности

  // Собираем уникальные ненулевые значения, сохраняя порядок первого появления
  for (int val : vec) {
    if (val != 0 && seen.find(val) == seen.end()) {
      unique_labels.push_back(val);
      seen.insert(val);
    }
  }

  // Создаем отображение: старое значение -> новая метка (начиная с 1)
  std::unordered_map<int, int> label_map;
  int current_label = 1;
  for (int val : unique_labels) {
    label_map[val] = current_label++;
  }

  // Заменяем значения в векторе, сохраняя нули
  for (int& val : vec) {
    if (val != 0) {
      val = label_map[val];
    }
  }
}
void CompressPath(std::vector<int>& parent, int node, int& root);

}  // namespace laganina_e_component_labeling_omp