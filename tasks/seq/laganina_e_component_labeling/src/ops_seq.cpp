#include "seq/laganina_e_component_labeling/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

int laganina_e_component_labeling_seq::TestTaskSequential::Find(int x) {
  int j = x;
  while (parent_[j] != 0) {
    j = parent_[j];
  }
  return j;
}

bool laganina_e_component_labeling_seq::TestTaskSequential::UnionSets(int x, int y) {
  int root_x = Find(x);  // Находим корень множества, содержащего X
  int root_y = Find(y);  // Находим корень множества, содержащего Y

  if (root_x != root_y) {
    parent_[root_y] = root_x;  // Объединяем множества
  }
  return true;
}

std::vector<int> laganina_e_component_labeling_seq::TestTaskSequential::NeighborsLabels(int x, int y) {
  std::vector<int> labels(2);  // ћаксимум 2 соседа
  int count = 0;               // —чЄтчик добавленных элементов

  if (x == 0 && y == 0) {
    // (0, 0) Ч нет соседей
  } else if (x == 0) {
    // (0, y) Ч только левый сосед
    if (step1_[(x * n_) + (y - 1)] != 0) {
      labels[count++] = step1_[(x * n_) + (y - 1)];
    }
  } else if (y == 0) {
    // (x, 0) Ч только верхний сосед
    if (step1_[((x - 1) * n_) + y] != 0) {
      labels[count++] = step1_[((x - 1) * n_) + y];
    }
  } else {
    // (x, y) Ч левый и верхний сосед
    if (step1_[(x * n_) + (y - 1)] != 0) {
      labels[count++] = step1_[(x * n_) + (y - 1)];
    }
    if (step1_[((x - 1) * n_) + y] != 0) {
      labels[count++] = step1_[((x - 1) * n_) + y];
    }
  }

  // ”меньшаем размер вектора до количества добавленных элементов
  labels.resize(count);

  return labels;
}

bool laganina_e_component_labeling_seq::TestTaskSequential::ValidationImpl() {
  if ((task_data == nullptr) || (task_data->inputs[0] == nullptr) || (task_data->outputs[0] == nullptr)) {
    return false;
  }
  if ((task_data->inputs_count[0] <= 0) || (task_data->inputs_count[1] <= 0)) {
    return false;
  }
  int size = static_cast<int>((task_data->inputs_count[0]) * (task_data->inputs_count[0]));
  for (int i = 0; i < size; i++) {
    if ((task_data->inputs[0][i] != 0) && (task_data->inputs[0][i] != 1)) {
      return false;
    }
  }
  return true;
}

bool laganina_e_component_labeling_seq::TestTaskSequential::PreProcessingImpl() {
  m_ = static_cast<int>(task_data->inputs_count[0]);
  n_ = static_cast<int>(task_data->inputs_count[1]);
  step1_.resize(m_ * n_, 0);
  labeled_binary_.resize(m_ * n_, 0);
  parent_.resize((m_ * n_) + 1);
  for (int i = 0; i < (m_ * n_) + 1; ++i) {
    parent_[i] = 0;  // »значально каждый элемент ¤вл¤етс¤ своим родителем
  }
  binary_.resize(m_ * n_);
  for (int i = 0; i < m_ * n_; ++i) {
    binary_[i] = reinterpret_cast<int*>(task_data->inputs[0])[i];
  }
  return true;
}

bool laganina_e_component_labeling_seq::TestTaskSequential::RunImpl() {
  FirstPass();
  SecondPass();
  return true;
}

void laganina_e_component_labeling_seq::TestTaskSequential::FirstPass() {
  int label = 1;  // Начальная метка

  for (int l = 0; l < m_; ++l) {
    for (int p = 0; p < n_; ++p) {
      if (binary_[(l * n_) + p] != 0) {
        ProcessPixel(l, p, label);
      }
    }
  }
}

void laganina_e_component_labeling_seq::TestTaskSequential::ProcessPixel(int l, int p, int& label) {
  auto neighbors = NeighborsLabels(l, p);
  if (neighbors.empty()) {
    // Новая метка
    step1_[(l * n_) + p] = label;
    label++;
  } else {
    // Назначаем минимальную метку из соседей
    int min_label = *std::ranges::min_element(neighbors);
    step1_[(l * n_) + p] = min_label;

    // Объединяем метки
    MergeLabels(min_label, neighbors);
  }
}

void laganina_e_component_labeling_seq::TestTaskSequential::MergeLabels(int min_label,
                                                                        const std::vector<int>& neighbors) {
  for (int neighbor_label : neighbors) {
    if (neighbor_label != min_label) {
      UnionSets(min_label, neighbor_label);
    }
  }
}

void laganina_e_component_labeling_seq::TestTaskSequential::SecondPass() {
  for (int l = 0; l < m_; ++l) {
    for (int p = 0; p < n_; ++p) {
      if (binary_[(l * n_) + p] != 0) {
        labeled_binary_[(l * n_) + p] = Find(step1_[(l * n_) + p]);
      }
    }
  }
}
bool laganina_e_component_labeling_seq::TestTaskSequential::PostProcessingImpl() {
  for (int i = 0; i < m_ * n_; ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = labeled_binary_[i];
  }
  return true;
}
