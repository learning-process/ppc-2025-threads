#pragma once

#include <atomic>
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
  std::mutex m_;
  std::mutex cout_mutex_;
  std::condition_variable cv_;
  std::condition_variable cv_done_;
  bool ready_ = false;
  bool done_ = false;
  int gap_ = 0;
  int num_threads_ = 0;
  std::atomic<int> threads_completed_{0};
};

}  // namespace solovyev_d_shell_sort_simple_stl