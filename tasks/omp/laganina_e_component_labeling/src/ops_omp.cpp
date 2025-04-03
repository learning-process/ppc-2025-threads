#include "omp/laganina_e_component_labeling/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <memory>
#include <stack>

namespace laganina_e_component_labeling_omp {

TestTaskOpenMP::TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool TestTaskOpenMP::ValidationImpl() {
  if (!task_data || !task_data->inputs[0] || !task_data->outputs[0]) return false;

  const int size = task_data->inputs_count[0] * task_data->inputs_count[1];
  const int* input = reinterpret_cast<int*>(task_data->inputs[0]);

  for (int i = 0; i < size; ++i) {
    if (input[i] != 0 && input[i] != 1) return false;
  }
  return true;
}

bool TestTaskOpenMP::PreProcessingImpl() {
  m_ = task_data->inputs_count[0];
  n_ = task_data->inputs_count[1];

  // Остальная инициализация 4
  binary_.resize(m_ * n_);
  step1_.assign(m_ * n_, 0);

  const int* input = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(input, input + m_ * n_, binary_.begin());

  return true;
}

bool TestTaskOpenMP::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(binary_.begin(), binary_.end(), output);
  return true;
}

bool TestTaskOpenMP::RunImpl() {
  label_connected_components();
  return true;
}

void TestTaskOpenMP::label_connected_components() {
  const int size = m_ * n_;
  std::vector<int> parent(size);

// 1. Инициализация (параллельная)
#pragma omp parallel for schedule(static)
  for (int i = 0; i < size; ++i) {
    parent[i] = (binary_[i] == 1) ? i : -1;
  }

  // 2. Параллельный Union-Find с итеративным уточнением
  bool changed;
  int iterations = 0;
  const int max_iterations = 10;  // Защита от бесконечного цикла

  do {
    changed = false;
    iterations++;

// Проход слева направо, сверху вниз
#pragma omp parallel for reduction(|| : changed) schedule(static)
    for (int i = 0; i < m_; ++i) {
      for (int j = 0; j < n_; ++j) {
        int idx = i * n_ + j;
        if (binary_[idx] != 1) continue;

        // Проверка левого и верхнего соседа
        if (j > 0 && binary_[idx - 1] == 1) {
          int root = idx;
          while (parent[root] != root) {
            parent[root] = parent[parent[root]];
            root = parent[root];
          }

          int neighbor_root = idx - 1;
          while (parent[neighbor_root] != neighbor_root) {
            parent[neighbor_root] = parent[parent[neighbor_root]];
            neighbor_root = parent[neighbor_root];
          }

          if (root != neighbor_root) {
            if (root < neighbor_root) {
              parent[neighbor_root] = root;
            } else {
              parent[root] = neighbor_root;
            }
            changed = true;
          }
        }

        if (i > 0 && binary_[idx - n_] == 1) {
          int root = idx;
          while (parent[root] != root) {
            parent[root] = parent[parent[root]];
            root = parent[root];
          }

          int neighbor_root = idx - n_;
          while (parent[neighbor_root] != neighbor_root) {
            parent[neighbor_root] = parent[parent[neighbor_root]];
            neighbor_root = parent[neighbor_root];
          }

          if (root != neighbor_root) {
            if (root < neighbor_root) {
              parent[neighbor_root] = root;
            } else {
              parent[root] = neighbor_root;
            }
            changed = true;
          }
        }
      }
    }

// Проход справа налево, снизу вверх (для ускорения сходимости)
#pragma omp parallel for reduction(|| : changed) schedule(static)
    for (int i = m_ - 1; i >= 0; --i) {
      for (int j = n_ - 1; j >= 0; --j) {
        int idx = i * n_ + j;
        if (binary_[idx] != 1) continue;

        // Проверка правого и нижнего соседа
        if (j < n_ - 1 && binary_[idx + 1] == 1) {
          int root = idx;
          while (parent[root] != root) {
            parent[root] = parent[parent[root]];
            root = parent[root];
          }

          int neighbor_root = idx + 1;
          while (parent[neighbor_root] != neighbor_root) {
            parent[neighbor_root] = parent[parent[neighbor_root]];
            neighbor_root = parent[neighbor_root];
          }

          if (root != neighbor_root) {
            if (root < neighbor_root) {
              parent[neighbor_root] = root;
            } else {
              parent[root] = neighbor_root;
            }
            changed = true;
          }
        }

        if (i < m_ - 1 && binary_[idx + n_] == 1) {
          int root = idx;
          while (parent[root] != root) {
            parent[root] = parent[parent[root]];
            root = parent[root];
          }

          int neighbor_root = idx + n_;
          while (parent[neighbor_root] != neighbor_root) {
            parent[neighbor_root] = parent[parent[neighbor_root]];
            neighbor_root = parent[neighbor_root];
          }

          if (root != neighbor_root) {
            if (root < neighbor_root) {
              parent[neighbor_root] = root;
            } else {
              parent[root] = neighbor_root;
            }
            changed = true;
          }
        }
      }
    }
  } while (changed && iterations < max_iterations);

// 3. Финальное сжатие путей и разметка
#pragma omp parallel for schedule(static)
  for (int i = 0; i < size; ++i) {
    if (binary_[i] == 1) {
      int root = i;
      while (parent[root] != root) {
        root = parent[root];
      }
      binary_[i] = root + 2;
    }
  }

  // 4. Нормализация меток
  std::vector<int> label_map(size + 2, 0);
  int next_label = 2;

#pragma omp parallel
  {
    std::vector<int> local_labels;

#pragma omp for nowait
    for (int i = 0; i < size; ++i) {
      if (binary_[i] >= 2) {
        local_labels.push_back(binary_[i]);
      }
    }

#pragma omp critical
    {
      for (int val : local_labels) {
        if (label_map[val] == 0) {
          label_map[val] = next_label++;
        }
      }
    }
  }

// 5. Применение нормализованных меток
#pragma omp parallel for schedule(static)
  for (int i = 0; i < size; ++i) {
    if (binary_[i] >= 2) {
      binary_[i] = label_map[binary_[i]];
    }
  }
}

}  // namespace laganina_e_component_labeling_omp