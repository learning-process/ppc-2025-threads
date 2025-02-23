include "seq/laganina_e_component_labeling/include/ops_seq.hpp"
#include <utility>

    std::pair<int, int>
    laganina_e_component_labeling_seq::TestTaskSequential::find(int x) {
  int index;  // ”брали const
  while (x > 0) {
    bool found = false;  // ‘лаг дл¤ проверки, найдена ли вершина x
    for (index = 0; index < parent.size(); ++index) {
      if (parent[index].first == x) {
        x = parent[index].second;  // ѕереходим к родителю
        found = true;
        break;
      }
    }
    if (!found) {
      return std::make_pair(-1, -1);  // ¬ершина x не найдена
    }
  }
  return std::make_pair(x, index);  // x теперь корень
}

bool laganina_e_component_labeling_seq::TestTaskSequential::Union_sets(int x, int y) {
  int j = find(x).first;
  int k = find(y).first;
  if ((j == -1) || (k == -1)) {
    return false;
  }
  if (j != k) {
    parent[find(y).second].second = j;
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
  internal_order_test();
  if ((taskData->inputs_count[0] <= 0) || (taskData->inputs_count[1] <= 0)) {
    return false;
  }
  for (int i = 0; i < n * m; ++i) {
    if (!(taskData->inputs[0][i] == 0 || taskData->inputs[0][i] == 1)) {
      return false;
    }
  }
  return true;
}

bool laganina_e_component_labeling_seq::TestTaskSequential::PreprocessingImpl() {
  internal_order_test();
  m = static_cast<int>(taskData->inputs_count[0]);
  n = static_cast<int>(taskData->inputs_count[1]);
  step1.resize(m * n, 0);
  labeled_binary.resize(m * n, 0);
  parent.resize(m * n);
  for (int i = 0; i < m * n; ++i) {
    parent[i] = {i, i};  // »значально каждый элемент ¤вл¤етс¤ своим родителем
  }
  binary.resize(m * n);
  for (int i = 0; i < m * n; ++i) {
    binary[i] = reinterpret_cast<int*>(taskData->inputs[0])[i];
  }
}

bool laganina_e_component_labeling_seq::TestTaskSequential::RunImpl() {
  internal_order_test();
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
              Union(minLabel, neighborLabel);
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
        labeled_binary[l * n + p] = find(step1[l * n + p]).first;
      }
    }
  }
  return true;
}

bool laganina_e_component_labeling_seq::TestTaskSequential::Postprocessing() {
  internal_order_test();
  for (int i = 0; i < m * n; ++i) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = labeled_binary[i];
  }
  return true;
}
