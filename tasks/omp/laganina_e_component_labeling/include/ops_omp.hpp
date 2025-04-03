#pragma once
#include <map>
#include <memory>
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

  // Методы 2
  void label_connected_components();
};

inline void normalize_labels(std::vector<int>& vec) {
  std::map<int, int> label_map;
  int current_label = 2;

  // Собираем уникальные метки
  for (auto& val : vec) {
    if (val != 0 && label_map.count(val) == 0) {
      label_map[val] = current_label++;
    }
  }

  // Заменяем метки
  for (auto& val : vec) {
    if (val != 0) {
      val = label_map[val];
    }
  }
}

}  // namespace laganina_e_component_labeling_omp