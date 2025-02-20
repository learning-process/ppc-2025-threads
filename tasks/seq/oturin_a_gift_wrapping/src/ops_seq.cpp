#include "seq/oturin_a_gift_wrapping/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

double oturin_a_gift_wrapping_seq::ABTP(coord a, coord b, coord c) {
  coord ab = {b.x - a.x, b.y - a.y};
  coord cb = {b.x - c.x, b.y - c.y};
  double dot = (ab.x * cb.x + ab.y * cb.y);
  double cross = (ab.x * cb.y - ab.y * cb.x);
  return fabs(atan2(cross, dot));
}

double oturin_a_gift_wrapping_seq::ABTP(coord a, coord c) {
  coord b{a.x, a.y - 1};
  return ABTP(b, a, c);
}

double oturin_a_gift_wrapping_seq::distance(coord a, coord b) {
  int t1 = a.x - b.x;
  int t2 = a.y - b.y;
  return sqrt(t1 * t1 + t2 * t2);
}

bool oturin_a_gift_wrapping_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<coord *>(task_data->inputs[0]);
  input_ = std::vector<coord>(in_ptr, in_ptr + input_size);
  n = input_.size();
  output_ = std::vector<coord>(0);
  output_.reserve(n);

  coord t = input_[0];
  for (int i = 1; i < n; i++) {  // check if all points are same
    if (t != input_[i]) return true;
  }

  return false;
}

bool oturin_a_gift_wrapping_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] >= 3;  // task requires 3 or more points to wrap
}

bool oturin_a_gift_wrapping_seq::TestTaskSequential::RunImpl() {
  // find most left point (priority to top)
  coord mostLeft = input_[0];
  int startIndex = 0;
  for (int i = 1; i < n; i++) {
    if (input_[i].x < mostLeft.x) {
      startIndex = i;
      mostLeft = input_[i];
    } else if (input_[i].x == mostLeft.x && input_[i].y > mostLeft.y) {
      startIndex = i;
      mostLeft = input_[i];
    }
  }
  output_.push_back(input_[startIndex]);

  // find second point
  double lineAngle = -5;
  int searchIndex = 0;
  for (int i = 0; i < n; i++) {
    double t = ABTP(input_[startIndex], input_[i]);
    if (t > lineAngle && i != startIndex) {
      lineAngle = t;
      searchIndex = i;
    } else if (t == lineAngle) {
      if (distance(input_[startIndex], input_[i]) < distance(input_[startIndex], input_[searchIndex]) &&
          i != startIndex) {
        searchIndex = i;
        lineAngle = t;
      }
    }
  }

  // main loop
  do {
    output_.push_back(input_[searchIndex]);
    lineAngle = -5;
    for (int i = 0; i < n; i++) {
      double t = ABTP(output_[output_.size() - 2], output_[output_.size() - 1], input_[i]);
      if (t > lineAngle) {
        if (output_.back() != input_[i] && output_[output_.size() - 2] != input_[i]) {
          searchIndex = i;
          lineAngle = t;
        }
      } else if (t == lineAngle) {
        if (distance(output_.back(), input_[i]) < distance(output_.back(), input_[searchIndex]) &&
            output_.back() != input_[i] && output_[output_.size() - 2] != input_[i]) {
          searchIndex = i;
          lineAngle = t;
        }
      }
    }
  } while (searchIndex != startIndex);

  return true;
}

bool oturin_a_gift_wrapping_seq::TestTaskSequential::PostProcessingImpl() {
  auto *result_ptr = reinterpret_cast<coord *>(task_data->outputs[0]);
  std::copy(output_.begin(), output_.end(), result_ptr);
  return true;
}
