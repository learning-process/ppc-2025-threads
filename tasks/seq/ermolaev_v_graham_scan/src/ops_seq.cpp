#include "seq/ermolaev_v_graham_scan/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool ermolaev_v_graham_scan_seq::TestTaskSequential::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<Point *>(task_data->inputs[0]);
  input_ = std::vector<Point>(in_ptr, in_ptr + task_data->inputs_count[0]);
  output_ = std::vector<Point>();
  return true;
}

bool ermolaev_v_graham_scan_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] >= 3 && task_data->inputs_count[0] <= task_data->outputs_count[0];
}

bool ermolaev_v_graham_scan_seq::TestTaskSequential::RunImpl() {
  bool has_different_x = false;
  bool has_different_y = false;
  for (size_t i = 1; i < input_.size(); i++) {
    if (input_[i].x != input_[0].x) {
      has_different_x = true;
    }
    if (input_[i].y != input_[0].y) {
      has_different_y = true;
    }
    if (has_different_x && has_different_y) {
      break;
    }
  }

  if (!has_different_x && !has_different_y) {
    return false;
  }

  size_t base = 0;
  for (size_t i = 1; i < input_.size(); i++) {
    if (input_[i] <= input_[base]) {
      base = i;
    }
  }

  std::swap(input_[0], input_[base]);
  std::sort(input_.begin() + 1, input_.end(), [&](const Point &a, const Point &b) {
    int cross = ((a.x - input_[0].x) * (b.y - input_[0].y)) - ((b.x - input_[0].x) * (a.y - input_[0].y));
    if (cross == 0) {
      return std::pow(a.x - input_[0].x, 2) + std::pow(a.y - input_[0].y, 2) <
             std::pow(b.x - input_[0].x, 2) + std::pow(b.y - input_[0].y, 2);
    }

    return cross > 0;
  });

  output_.clear();
  output_.push_back(input_[0]);
  output_.push_back(input_[1]);

  for (size_t i = 2; i < input_.size(); i++) {
    while (output_.size() >= 2) {
      Point p1 = output_[output_.size() - 2];
      Point p2 = output_[output_.size() - 1];
      Point p3 = input_[i];

      int cross = ((p2.x - p1.x) * (p3.y - p1.y)) - ((p3.x - p1.x) * (p2.y - p1.y));

      if (cross > 0) {
        break;
      }
      output_.pop_back();
    }
    output_.push_back(input_[i]);
  }

  return true;
}

bool ermolaev_v_graham_scan_seq::TestTaskSequential::PostProcessingImpl() {
  task_data->outputs_count.clear();
  task_data->outputs_count.push_back(output_.size());
  std::copy(output_.begin(), output_.end(), reinterpret_cast<Point *>(task_data->outputs[0]));
  return true;
}
