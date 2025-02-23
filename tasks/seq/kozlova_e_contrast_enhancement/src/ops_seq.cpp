#include "seq/kozlova_e_contrast_enhancement/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

bool kozlova_e_contrast_enhancement_seq::TestTaskSequential::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  size_t size = task_data->inputs_count[0];
  output_.resize(size, 0);
  input_.resize(size);
  std::copy(input_ptr, input_ptr + size, input_.begin());

  return true;
}

bool kozlova_e_contrast_enhancement_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0] && task_data->inputs_count[0] > 0;
}

bool kozlova_e_contrast_enhancement_seq::TestTaskSequential::RunImpl() {
  int min_value = *std::min_element(input_.begin(), input_.end());
  int max_value = *std::max_element(input_.begin(), input_.end());

  if (min_value == max_value) {
    std::copy(input_.begin(), input_.end(), output_.begin());
    return true;
  }

  for (size_t i = 0; i < input_.size(); ++i) {
    output_[i] = static_cast<int>(((input_[i] - min_value) / (double)(max_value - min_value)) * 255);
    output_[i] = std::clamp(output_[i], 0, 255);
  }

  return true;
}

bool kozlova_e_contrast_enhancement_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
