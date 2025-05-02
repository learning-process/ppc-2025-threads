#pragma once

#include <condition_variable>
#include <mutex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace solovyev_d_shell_sort_simple_stl {

class TaskSTL : public ppc::core::Task {
 public:
  explicit TaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  void ThreadWorker(int t);
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::mutex m;
  std::condition_variable cv;
  bool ready = false;
  bool done = false;
  int gap = 0;
  int num_threads = 0;
};

}  // namespace solovyev_d_shell_sort_simple_stl