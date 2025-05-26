#include "../include/ops_all.hpp"

#include <tbb/parallel_for.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

namespace kovalchuk_a_shell_sort_all {

ShellSortAll::ShellSortAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool ShellSortAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_.assign(input_ptr, input_ptr + task_data->inputs_count[0]);
  }

  int total_size{};
  boost::mpi::broadcast(world_, total_size, 0);

  int num_procs = world_.size();
  std::vector<int> counts(num_procs, total_size / num_procs);
  std::vector<int> displs(num_procs, 0);
  for (int i = 0; i < total_size % num_procs; ++i) {
    counts[i]++;
  }
  for (int i = 1; i < num_procs; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }


  std::vector<int> buffer;
  if (world_.rank() == 0) {
    buffer = input_;
  }

  input_.resize(counts[world_.rank()]);
  boost::mpi::scatterv(world_, input_.data(), counts, displs, buffer.data(), counts[world_.rank()], 0);

  return true;
}

bool ShellSortAll::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool ShellSortAll::RunImpl() {
  ShellSort();
  return true;
}

void ShellSortAll::ShellSort() {
  int n = static_cast<int>(input_.size());
  for (int gap = n / 2; gap > 0; gap /= 2) {
    tbb::parallel_for(0, gap, [&](int k) {
      for (int i = k + gap; i < n; i += gap) {
        int temp = input_[i];
        int j = i;
        while (j >= gap && input_[j - gap] > temp) {
          input_[j] = input_[j - gap];
          j -= gap;
        }
        input_[j] = temp;
      }
    });
  }
}

bool ShellSortAll::PostProcessingImpl() {
  std::vector<int> gathered;
  boost::mpi::gather(world_, input_.data(), static_cast<int>(input_.size()), gathered, 0);

  if (world_.rank() == 0) {
    std::ranges::sort(gathered.begin(), gathered.end());
    auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    std::ranges::copy(gathered.begin(), gathered.end(), output_ptr);
  }

  return true;
}

}  // namespace kovalchuk_a_shell_sort_all