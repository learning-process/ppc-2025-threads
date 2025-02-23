
#include "seq/laganina_e_component_labeling/include/ops_seq.hpp"

std::pair<int, int> laganina_e_component_labeling_seq::TestTaskSequential::find(int x) {
  int index;  // Убрали const
  while (x > 0) {
    bool found = false;  // Флаг для проверки, найдена ли вершина x
    for (index = 0; index < parent.size(); ++index) {
      if (parent[index].first == x) {
        x = parent[index].second;  // Переходим к родителю
        found = true;
        break;
      }
    }
    if (!found) {
      return std::make_pair(- 1,-1);  // Вершина x не найдена
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
  std::vector<int> labels(2);  // Максимум 2 соседа
  int count = 0;               // Счётчик добавленных элементов

  if (x == 0 && y == 0) {
    // (0, 0) — нет соседей
  } else if (x == 0) {
    // (0, y) — только левый сосед
    if (step1[x * n + (y - 1)] != 0) {
      labels[count++] = step1[x*n+( y - 1)];
    }
  } else if (y == 0) {
    // (x, 0) — только верхний сосед
    if (step1[(x - 1)*n+ y] != 0) {
      labels[count++] = step1[(x - 1)*n+ y];
    }
  }
    else {
    // (x, y) — левый и верхний сосед
    if (step1[x * n + (y - 1)] != 0) {
      labels[count++] = step1[x * n + (y - 1)];
    }
    if (step1[(x - 1) * n + y] != 0) {
      labels[count++] = step1[(x - 1) * n + y];
    }
  }

  // Уменьшаем размер вектора до количества добавленных элементов
  labels.resize(count);

  return labels;
}

bool laganina_e_component_labeling_seq::TestTaskSequential::ValidationImpl() {
  internal_order_test();
  if ((taskData->inputs_count[0] <= 0) || (taskData->inputs_count[1] <= 0)){
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
    parent[i] = {i, i};  // Изначально каждый элемент является своим родителем
  }
  binary.resize(m * n);
  for (int i = 0; i < m * n; ++i) {
    binary[i] = reinterpret_cast<int*>(taskData->inputs[0])[i];
  }
}

bool laganina_e_component_labeling_seq::TestTaskSequential::RunImpl() {
  internal_order_test();
  int label = 1;  // Начальная метка

  // Первый проход: маркировка компонент
  for (int l = 0; l < m; ++l) {
    for (int p = 0; p < n; ++p) {
      if (binary[l * n + p]) {
        auto neighbors = neighbors_labels(l, p);
        if (neighbors.empty()) {
          // Новая метка
          step1[l * n + p] = label;
          label++;
        } else {
          // Назначаем минимальную метку из соседей
          int minLabel = *std::min_element(neighbors.begin(), neighbors.end());
          step1[l * n + p] = minLabel;

          // Объединяем метки
          for (int neighborLabel : neighbors) {
            if (neighborLabel != minLabel) {
              Union(minLabel, neighborLabel);
            }
          }
        }
      }
    }
  }

  // Второй проход: замена меток на корневые значения
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
