#include "seq/laganina_e_component_labeling/include/ops_seq.hpp"

#include <utility>
#include <vector>

int laganina_e_component_labeling_seq::TestTaskSequential::find(int x) {
  int j = x;
  while (parent[j] != 0) {
    j = parent[j];
  }
  return j;
}

bool laganina_e_component_labeling_seq::TestTaskSequential::Union_sets(int x, int y) {
  int rootX = find(x);  // Находим корень множества, содержащего X
  int rootY = find(y);  // Находим корень множества, содержащего Y

  if (rootX != rootY) {
    parent[rootY] = rootX;  // Объединяем множества
  }
  return true;
}

std::vector<int> laganina_e_component_labeling_seq::TestTaskSequential::neighbors_labels(int x, int y) {
  std::vector<int> labels(2);  // ћаксимум 2 соседа
  int count = 0;               // —чЄтчик добавленных элементов

  if (x == 0 && y == 0) {
    // (0, 0) Ч нет соседей
  } else if (x == 0) {
    // (0, y) Ч только левый сосед
    if (step1[x * n + (y - 1)] != 0) {
      labels[count++] = step1[x * n + (y - 1)];
    }
  } else if (y == 0) {
    // (x, 0) Ч только верхний сосед
    if (step1[(x - 1) * n + y] != 0) {
      labels[count++] = step1[(x - 1) * n + y];
    }
  } else {
    // (x, y) Ч левый и верхний сосед
    if (step1[x * n + (y - 1)] != 0) {
      labels[count++] = step1[x * n + (y - 1)];
    }
    if (step1[(x - 1) * n + y] != 0) {
      labels[count++] = step1[(x - 1) * n + y];
    }
  }

  // ”меньшаем размер вектора до количества добавленных элементов
  labels.resize(count);

  return labels;
}

bool laganina_e_component_labeling_seq::TestTaskSequential::ValidationImpl() {
  if (!task_data || !task_data->inputs[0] || !task_data->inputs_count[0] || !task_data->outputs[0] ||
      !task_data->inputs_count[1]) {
    return false;
  }
  if ((task_data->inputs_count[0] <= 0) || (task_data->inputs_count[1] <= 0)) {
    return false;
  }
  int size = task_data->inputs_count[0] * task_data->inputs_count[0];
  for (int i = 0; i < size; i++) {
    if ((task_data->inputs[0][i] != 0) && (task_data->inputs[0][i] != 1)) {
      return false;
    }
  }
  return true;
}

bool laganina_e_component_labeling_seq::TestTaskSequential::PreProcessingImpl() {
  m = static_cast<int>(task_data->inputs_count[0]);
  n = static_cast<int>(task_data->inputs_count[1]);
  step1.resize(m * n, 0);
  labeled_binary.resize(m * n, 0);
  parent.resize((m * n) + 1);
  for (int i = 0; i < (m * n) + 1; ++i) {
    parent[i] = 0;  // »значально каждый элемент ¤вл¤етс¤ своим родителем
  }
  binary.resize(m * n);
  for (int i = 0; i < m * n; ++i) {
    binary[i] = reinterpret_cast<int*>(task_data->inputs[0])[i];
  }
  return true;
}

bool laganina_e_component_labeling_seq::TestTaskSequential::RunImpl() {
  int label = 1;  // Ќачальна¤ метка

  // ѕервый проход: маркировка компонент
  for (int l = 0; l < m; ++l) {
    for (int p = 0; p < n; ++p) {
      if (binary[l * n + p]) {
        auto neighbors = neighbors_labels(l, p);
        if (neighbors.empty()) {
          // Ќова¤ метка
          step1[l * n + p] = label;
          label++;
        } else {
          // Ќазначаем минимальную метку из соседей
          int minLabel = *std::min_element(neighbors.begin(), neighbors.end());
          step1[l * n + p] = minLabel;

          // ќбъедин¤ем метки
          for (int neighborLabel : neighbors) {
            if (neighborLabel != minLabel) {
              Union_sets(minLabel, neighborLabel);
            }
          }
        }
      }
    }
  }

  // ¬торой проход: замена меток на корневые значени¤
  for (int l = 0; l < m; ++l) {
    for (int p = 0; p < n; ++p) {
      if (binary[l * n + p]) {
        labeled_binary[l * n + p] = find(step1[l * n + p]);
      }
    }
  }
  return true;
}

bool laganina_e_component_labeling_seq::TestTaskSequential::PostProcessingImpl() {
  for (int i = 0; i < m * n; ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = labeled_binary[i];
  }
  return true;
}
