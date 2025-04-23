#include <stl/shuravina_o_hoare_simple_merger_std/include/ops_stl.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace shuravina_o_hoare_simple_merger_stl {

bool TestTaskSTL::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + task_data->inputs_count[0]);
  output_ = std::vector<int>(task_data->outputs_count[0], 0);
  return true;
}

bool TestTaskSTL::ValidationImpl() {
  if (task_data->inputs_count.size() != 1 || task_data->outputs_count.size() != 1) {
    return false;
  }
  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  return true;
}

bool TestTaskSTL::RunImpl() {
  std::ranges::sort(input_);
  output_ = input_;
  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(output_, out_ptr);
  return true;
}

}  // namespace shuravina_o_hoare_simple_merger_stl