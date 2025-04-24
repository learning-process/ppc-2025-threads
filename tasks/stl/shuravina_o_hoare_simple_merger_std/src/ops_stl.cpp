#include "stl/shuravina_o_hoare_simple_merger_std/include/ops_stl.hpp"

#include <algorithm>
#include <vector>

namespace shuravina_o_hoare_simple_merger_stl {

bool TestTaskSTL::PreProcessingImpl() {
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }

  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + task_data->inputs_count[0]);
  output_ = std::vector<int>(task_data->outputs_count[0], 0);
  return true;
}

bool TestTaskSTL::ValidationImpl() {
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }
  if (task_data->inputs_count[0] > 0 && task_data->inputs[0] == nullptr) {
    return false;
  }
  if (task_data->outputs_count[0] > 0 && task_data->outputs[0] == nullptr) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskSTL::RunImpl() {
  if (input_.empty()) {
    output_ = input_;
    return true;
  }
  std::ranges::sort(input_);
  output_ = input_;
  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  if (output_.empty()) {
    return true;
  }
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(output_, out_ptr);
  return true;
}

}  // namespace shuravina_o_hoare_simple_merger_stl