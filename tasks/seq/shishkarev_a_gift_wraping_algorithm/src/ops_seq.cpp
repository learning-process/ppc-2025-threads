#include "seq/shishkarev_a_gift_wraping_algorithm/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <set>
#include <vector>

std::vector<shishkarev_a_gift_wraping_algorithm_seq::Vertex> shishkarev_a_gift_wraping_algorithm_seq::remove_duplicates(
    const std::vector<shishkarev_a_gift_wraping_algorithm_seq::Vertex>& points) {
  std::set<Vertex> unique_points(points.begin(), points.end());
  return std::vector<Vertex>(unique_points.begin(), unique_points.end());
}

bool shishkarev_a_gift_wraping_algorithm_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<Vertex*>(task_data->inputs[0]);
  input_ = std::vector<Vertex>(in_ptr, in_ptr + input_size);

  input_ = remove_duplicates(input_);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<Vertex>(output_size, {0, 0});

  rc_size_ = static_cast<int>(std::sqrt(input_size));
  return true;
}

bool shishkarev_a_gift_wraping_algorithm_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool shishkarev_a_gift_wraping_algorithm_seq::TestTaskSequential::RunImpl() {
  if (input_.size() < 3) {
    output_.clear();
    for (size_t i = 1; i < input_.size(); ++i) {
      output_[i] = input_[i];
    }
    return true;
  }

  output_.clear();

  int start_point = 0;
  for (size_t i = 1; i < input_.size(); ++i) {
    if ((input_[i].y < input_[start_point].y) ||
        ((input_[i].y == input_[start_point].y) && (input_[i].x > input_[start_point].x))) {
      start_point = i;
    }
  }

  int p = start_point;
  do {
    output_.push_back(input_[p]);
    int q = (p + 1) % input_.size();

    for (size_t i = 0; i < input_.size(); ++i) {
      if (input_[p].angle(input_[q], input_[i]) < 0) {
        q = i;
      } else if ((input_[p].angle(input_[q], input_[i]) == 0) &&
                 (input_[p].length(input_[i]) > input_[p].length(input_[q]))) {
        q = i;
      }
    }
    p = q;
  } while (p != start_point);

  return true;
}

bool shishkarev_a_gift_wraping_algorithm_seq::TestTaskSequential::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<Vertex*>(task_data->outputs[0]);

  size_t min_size = std::min(output_.size(), static_cast<size_t>(task_data->outputs_count[0]));
  for (size_t i = 0; i < min_size; ++i) {
    out_ptr[i] = output_[i];
  }

  return true;
}
