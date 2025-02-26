#include "seq/oturin_a_gift_wrapping/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

double oturin_a_gift_wrapping_seq::ABTP(Coord a, Coord b, Coord c) {
  Coord ab = {b.x - a.x, b.y - a.y};
  Coord cb = {b.x - c.x, b.y - c.y};
  double dot = (ab.x * cb.x + ab.y * cb.y);
  double cross = (ab.x * cb.y - ab.y * cb.x);
  return fabs(atan2(cross, dot));
}

double oturin_a_gift_wrapping_seq::ABTP(Coord a, Coord c) {
  Coord b{a.x, a.y - 1};
  return ABTP(b, a, c);
}

double oturin_a_gift_wrapping_seq::Distance(Coord a, Coord b) {
  int t1 = a.x - b.x;
  int t2 = a.y - b.y;
  return sqrt(t1 * t1 + t2 * t2);
}

bool oturin_a_gift_wrapping_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<Coord *>(task_data->inputs[0]);
  input_ = std::vector<Coord>(in_ptr, in_ptr + input_size);
  n_ = int(input_.size());
  output_ = std::vector<Coord>(0);
  output_.reserve(n_);

  Coord t = input_[0];
  for (int i = 1; i < n_; i++) {  // check if all points are same
    if (t != input_[i]) {
      return true;
    }
  }
  return false;
}

bool oturin_a_gift_wrapping_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] >= 3;  // task requires 3 or more points to wrap
}

bool oturin_a_gift_wrapping_seq::TestTaskSequential::RunImpl() {
  // find most left point (priority to top)
  Coord most_left = input_[0];
  int start_index = 0;
  for (int i = 1; i < n_; i++) {
    if (input_[i].x < most_left.x) {
      start_index = i;
      most_left = input_[i];
    } else if (input_[i].x == most_left.x && input_[i].y > most_left.y) {
      start_index = i;
      most_left = input_[i];
    }
  }
  output_.push_back(input_[start_index]);

  // find second point
  double line_angle = -5;
  int search_index = 0;
  for (int i = 0; i < n_; i++) {
    double t = ABTP(input_[start_index], input_[i]);
    if (t > line_angle && i != start_index) {
      line_angle = t;
      search_index = i;
    } else if (t == line_angle) {
      if (Distance(input_[start_index], input_[i]) < Distance(input_[start_index], input_[search_index]) &&
          i != start_index) {
        search_index = i;
        line_angle = t;
      }
    }
  }

  // main loop
  do {
    output_.push_back(input_[search_index]);
    line_angle = -5;
    for (int i = 0; i < n_; i++) {
      double t = ABTP(output_[output_.size() - 2], output_[output_.size() - 1], input_[i]);
      if (t > line_angle) {
        if (output_.back() != input_[i] && output_[output_.size() - 2] != input_[i]) {
          search_index = i;
          line_angle = t;
        }
      } else if (t == line_angle) {
        if (Distance(output_.back(), input_[i]) < Distance(output_.back(), input_[search_index]) &&
            output_.back() != input_[i] && output_[output_.size() - 2] != input_[i]) {
          search_index = i;
          line_angle = t;
        }
      }
    }
  } while (search_index != start_index);

  return true;
}

bool oturin_a_gift_wrapping_seq::TestTaskSequential::PostProcessingImpl() {
  auto *result_ptr = reinterpret_cast<Coord *>(task_data->outputs[0]);
  std::copy(output_.begin(), output_.end(), result_ptr);
  return true;
}
