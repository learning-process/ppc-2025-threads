#include "seq/filateva_e_simpson/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool filateva_e_simpson_seq::Simpson::PreProcessingImpl() {
  auto *temp = reinterpret_cast<double *>(task_data->inputs[0]);
  a_ = temp[0];
  b_ = temp[1];
  alfa_ = temp[2];
  f_ = reinterpret_cast<Func>(task_data->inputs[1]);
  return true;
}

bool filateva_e_simpson_seq::Simpson::ValidationImpl() {
  auto *temp = reinterpret_cast<double *>(task_data->inputs[0]);
  return task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 1 && temp[0] < temp[1] &&
         temp[1] - temp[0] > temp[2] && temp[2] > 0;
}

bool filateva_e_simpson_seq::Simpson::RunImpl() {
  double max_z = 0;
  for (int i = 0; i < (int)((b_ - a_) / alfa_) + 1; ++i) {
    double x = a_ + i * alfa_;
    double temp = std::abs((f_(x - (2 * alfa_)) - 4 * f_(x - alfa_) + 6 * f_(x) - 4 * f_(x + alfa_) + f_(x + (2 * alfa_))) /
                           pow(alfa_, 4));
    max_z = std::max(max_z, temp);
  }

  int n_2 = (int)pow((pow((b_ - a_), 4) * max_z) / (180 * alfa_), 0.25);

  n_2 += ((n_2 % 2) != 0) ? 1 : 0;
  n_2 = (n_2 != 0) ? n_2 : 10;

  double h = (b_ - a_) / n_2;
  res_ = f_(a_) + f_(b_);

  for (int i = 1; i < n_2; i++) {
    double x = a_ + (i * h);
    if (i % 2 == 1) {
      res_ += 4 * f_(x);
    } else {
      res_ += 2 * f_(x);
    }
  }

  res_ *= (h / 3);

  return true;
}

bool filateva_e_simpson_seq::Simpson::PostProcessingImpl() {
  reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  return true;
}
