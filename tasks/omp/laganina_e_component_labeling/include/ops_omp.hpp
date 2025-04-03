#pragma once
#include <atomic>
#include <unordered_map>
#include <utility>
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
  std::vector<int> step1_;

  void LabelConnectedComponents();  // Переименовано в CamelCase
};

inline void NormalizeLabels(std::vector<int>& vec) {  // Переименовано и оптимизировано
  std::unordered_map<int, int> label_map;
  int current_label = 2;

  // Collect unique labels
  for (const auto& val : vec) {                  // Добавлен const
    if (val != 0 && !label_map.contains(val)) {  // Использован contains
      label_map[val] = current_label++;
    }
  }

  // Replace labels
  for (auto& val : vec) {
    if (val != 0) {
      val = label_map[val];
    }
  }
}

}  // namespace laganina_e_component_labeling_omp