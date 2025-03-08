#include "stl/filateva_e_simpson/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

double filateva_e_simpson_stl::Simpson::Max_z(int start, int end) {
  double max_z = 0;
  for (int i = start; i <= end; ++i) {
    double x = a_ + (i * alfa_);
    double temp =
        std::abs((f_(x - (2 * alfa_)) - 4 * f_(x - alfa_) + 6 * f_(x) - 4 * f_(x + alfa_) + f_(x + (2 * alfa_))) /
                 pow(alfa_, 4));
    max_z = std::max(max_z, temp);
  }
  return max_z;
}

double filateva_e_simpson_stl::Simpson::Res(int start, int end, double h) {
  double loca_res = 0;
  for (int i = start; i < end; i++) {
    double x = a_ + (i * h);
    if (i % 2 == 1) {
      loca_res += 4 * f_(x);
    } else {
      loca_res += 2 * f_(x);
    }
  }
  return loca_res;
}

bool filateva_e_simpson_stl::Simpson::PreProcessingImpl() {
  auto *temp = reinterpret_cast<double *>(task_data->inputs[0]);
  a_ = temp[0];
  b_ = temp[1];
  alfa_ = temp[2];
  f_ = reinterpret_cast<Func>(task_data->inputs[1]);

  return true;
}

bool filateva_e_simpson_stl::Simpson::ValidationImpl() {
  auto *temp = reinterpret_cast<double *>(task_data->inputs[0]);
  return task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 1 && temp[0] < temp[1] &&
         temp[1] - temp[0] > temp[2] && temp[2] > 0;
}

bool filateva_e_simpson_stl::Simpson::RunImpl() {
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads);
  std::vector<double> temp(num_threads);

  double max_z = 0;
  int n = (int)((b_ - a_) / alfa_) + 1;

  int del = n / num_threads;
  int ost = n % num_threads;

  for (int i = 1; i < num_threads; i++) {
    int start = i * del + std::min(i - 1, ost);
    int end = (i + 1) * del + std::min(i, ost);
    threads[i] = std::thread([&, i]() { temp[i] = Max_z(start, end); });
  }

  max_z = Max_z(0, del);

  for (int i = 1; i < num_threads; i++) {
    threads[i].join();
    max_z = std::max(temp[i], max_z);
  }

  int n_2 = (int)pow((pow((b_ - a_), 4) * max_z) / (180 * alfa_), 0.25);

  n_2 += ((n_2 % 2) != 0) ? 1 : 0;
  n_2 = (n_2 != 0) ? n_2 : 10;

  double h = (b_ - a_) / n_2;
  res_ = f_(a_) + f_(b_);

  // del = (n_2 - 1) / num_threads;
  // ost = (n_2 - 1) % num_threads;

  // for (int i = 1; i < num_threads; i++) {
  //   threads[i] = std::thread([&, i]() {
  //     int start = i * del + std::min(i - 1, ost) + 1;
  //     int end = (i + 1) * del + std::min(i, ost) + 1;
  //     // std::cerr << "\n th: " << i << " start: " << start << " end: " << end << "\n";
  //     temp[i] = Res(start, end, h);
  //   });
  // }

  // res_ += Res(1, del + 1, h);

  // for (int i = 1; i < num_threads; i++) {
  //   threads[i].join();
  //   res_ += temp[i];
  // }

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

bool filateva_e_simpson_stl::Simpson::PostProcessingImpl() {
  reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  return true;
}
