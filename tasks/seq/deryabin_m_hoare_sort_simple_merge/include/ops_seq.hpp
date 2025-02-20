bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential

#pragma once

#include <utility>

#include "core/task/include/task.hpp"

namespace deryabin_m_hoare_sort_simple_merge_seq {

class HoareSortTaskSequential : public ppc::core::Task {
 public:
  explicit HoareSortTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  // сортировка Хоара
  void HoaraSort(double* a, size_type first, size_type last);

  std::vector<double> input_array_A_; // входной массив
  size_t dimension_; // его размер
  size_t min_chunk_size_; // размер частей на которые будет разбиваться исходный массив
  size_t remainder_; // остаток
  size_t chunk_count_; // число таких частей
};
}  // namespace deryabin_m_hoare_sort_simple_merge_seq
