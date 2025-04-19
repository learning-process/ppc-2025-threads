#pragma once

#include <cmath>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace smirnov_i_radix_sort_simple_merge_stl {

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> mas_, output_;
  static void RadixSort(std::vector<int> &mas);
  static std::vector<int> Merge(std::vector<int> &mas1, std::vector<int> &mas2);
  void Merging(std::deque<std::vector<int>> &firstdq, std::deque<std::vector<int>> &seconddq, std::mutex &mtx);
  std::vector<int> Sorting(int id, std::vector<int> &mas_, int max_th);
};

}  // namespace smirnov_i_radix_sort_simple_merge_stl