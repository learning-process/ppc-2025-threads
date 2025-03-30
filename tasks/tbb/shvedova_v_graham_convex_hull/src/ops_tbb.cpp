#include "../include/ops_tbb.hpp"

#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/parallel_for.h"

namespace {
bool CheckCollinearity(std::span<double> raw_points) {
  const auto points_count = raw_points.size() / 2;
  if (points_count < 3) {
    return true;
  }
  const auto dx = raw_points[2] - raw_points[0];
  const auto dy = raw_points[3] - raw_points[1];
  for (size_t i = 2; i < points_count; i++) {
    const auto dx_i = raw_points[(i * 2)] - raw_points[0];
    const auto dy_i = raw_points[(i * 2) + 1] - raw_points[1];
    if (std::fabs((dx * dy_i) - (dy * dx_i)) > 1e-9) {
      return false;
    }
  }
  return true;
}

double CalculateAngle(const Point &o, const Point &p) {
  const auto dx = p[0] - o[0];
  const auto dy = p[1] - o[1];

  if (dx == 0. && dy == 0.) {
    return -1.;
  }
  const auto positive_dy = (dx >= 0) ? dy / (dx + dy) : 1 - (dx / (-dx + dy));
  const auto negative_dy = (dx < 0) ? 2 - (dy / (-dx - dy)) : 3 + (dx / (dx - dy));
  return (dy >= 0) ? positive_dy : negative_dy;
}

bool ComparePoints(const Point &p0, const Point &p1, const Point &p2) {
  const auto ang1 = CalculateAngle(p0, p1);
  const auto ang2 = CalculateAngle(p0, p2);
  double exp1 = std::pow(p1[0] - p0[0], 2);
  double exp2 = std::pow(p1[1] - p0[1], 2);
  double exp3 = std::pow(p2[0] - p0[0], 2);
  double exp4 = std::pow(p2[1] - p0[1], 2);

  return (ang1 < ang2) || ((ang1 > ang2) ? false : (exp1 + exp2 - exp3 - exp4 > 0));
}

void ParallelSortStep(oneapi::tbb::task_arena &arena, std::vector<Point> &input, const Point &pivot, int points_count,
                      bool even_step) {
  const int shift = even_step ? 0 : -1;
  const int revshift = even_step ? -1 : 0;

  arena.execute([&] {
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<int>(1, points_count + shift,
                                        (points_count + shift - 1) / tbb::this_task_arena::max_concurrency()),
        [&](const tbb::blocked_range<int> &r) {
          int begin = r.begin();
          if ((begin % 2) == 0) {
            begin++;
          }
          for (int i = begin; i < r.end(); i += 2) {
            if (ComparePoints(pivot, input[i - shift], input[i + revshift])) {
              std::swap(input[i], input[i - (even_step ? 1 : -1)]);
            }
          }
        });
  });
}
}  // namespace

namespace shvedova_v_graham_convex_hull_tbb {

bool GrahamConvexHullTBB::ValidationImpl() {
  return (task_data->inputs.size() == 1 && task_data->inputs_count.size() == 1 && task_data->outputs.size() == 2 &&
          task_data->outputs_count.size() == 2 && (task_data->inputs_count[0] % 2 == 0) &&
          (task_data->inputs_count[0] / 2 > 2) && (task_data->outputs_count[0] == 1) &&
          (task_data->outputs_count[1] >= task_data->inputs_count[0])) &&
         !CheckCollinearity({reinterpret_cast<double *>(task_data->inputs[0]), task_data->inputs_count[0]});
}

bool GrahamConvexHullTBB::PreProcessingImpl() {
  points_count_ = static_cast<int>(task_data->inputs_count[0] / 2);
  input_.resize(points_count_, Point{});

  auto *p_src = reinterpret_cast<double *>(task_data->inputs[0]);
  for (int i = 0; i < points_count_ * 2; i += 2) {
    input_[i / 2][0] = p_src[i];
    input_[i / 2][1] = p_src[i + 1];
  }

  res_.clear();
  res_.reserve(points_count_);

  return true;
}

void GrahamConvexHullTBB::PerformSort() {
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  const auto pivot = *std::ranges::min_element(input_, [](auto &a, auto &b) { return a[1] < b[1]; });

  for (int pt = 0; pt < points_count_; pt++) {
    const bool even_step = pt % 2 == 0;
    ParallelSortStep(arena, input_, pivot, points_count_, even_step);
  }
}

bool GrahamConvexHullTBB::RunImpl() {
  PerformSort();

  for (int i = 0; i < 3; i++) {
    res_.push_back(input_[i]);
  }

  for (int i = 3; i < points_count_; ++i) {
    while (res_.size() > 1) {
      const auto &pv = res_.back();
      const auto dx1 = res_.rbegin()[1][0] - pv[0];
      const auto dy1 = res_.rbegin()[1][1] - pv[1];
      const auto dx2 = input_[i][0] - pv[0];
      const auto dy2 = input_[i][1] - pv[1];
      if (dx1 * dy2 < dy1 * dx2) {
        break;
      }
      res_.pop_back();
    }
    res_.push_back(input_[i]);
  }

  return true;
}

bool GrahamConvexHullTBB::PostProcessingImpl() {
  int res_points_count = static_cast<int>(res_.size());
  *reinterpret_cast<int *>(task_data->outputs[0]) = res_points_count;
  auto *p_out = reinterpret_cast<double *>(task_data->outputs[1]);
  for (int i = 0; i < res_points_count; i++) {
    p_out[2 * i] = res_[i][0];
    p_out[(2 * i) + 1] = res_[i][1];
  }
  return true;
}

}  // namespace shvedova_v_graham_convex_hull_tbb

namespace shvedova_v_graham_convex_hull_seq {

bool GrahamConvexHullSequential::ValidationImpl() {
  return (task_data->inputs.size() == 1 && task_data->inputs_count.size() == 1 && task_data->outputs.size() == 2 &&
          task_data->outputs_count.size() == 2 && (task_data->inputs_count[0] % 2 == 0) &&
          (task_data->inputs_count[0] / 2 > 2) && (task_data->outputs_count[0] == 1) &&
          (task_data->outputs_count[1] >= task_data->inputs_count[0])) &&
         !CheckCollinearity({reinterpret_cast<double *>(task_data->inputs[0]), task_data->inputs_count[0]});
}

bool GrahamConvexHullSequential::PreProcessingImpl() {
  points_count_ = static_cast<int>(task_data->inputs_count[0] / 2);
  input_.resize(points_count_, Point{});

  auto *p_src = reinterpret_cast<double *>(task_data->inputs[0]);
  for (int i = 0; i < points_count_ * 2; i += 2) {
    input_[i / 2][0] = p_src[i];
    input_[i / 2][1] = p_src[i + 1];
  }

  res_.clear();
  res_.reserve(points_count_);

  return true;
}

void GrahamConvexHullSequential::PerformSort() {  // NOLINT(*cognit*)
  const auto cmp = [](const Point &p0, const Point &p1, const Point &p2) {
    const auto calc_ang = [](const Point &o, const Point &p) {
      const auto dx = p[0] - o[0];
      const auto dy = p[1] - o[1];

      if (dx == 0. && dy == 0.) {
        return -1.;
      }
      const auto positive_dy = (dx >= 0) ? dy / (dx + dy) : 1 - (dx / (-dx + dy));
      const auto negative_dy = (dx < 0) ? 2 - (dy / (-dx - dy)) : 3 + (dx / (dx - dy));
      return (dy >= 0) ? positive_dy : negative_dy;
    };
    const auto ang1 = calc_ang(p0, p1);
    const auto ang2 = calc_ang(p0, p2);
    double exp1 = std::pow(p1[0] - p0[0], 2);
    double exp2 = std::pow(p1[1] - p0[1], 2);
    double exp3 = std::pow(p2[0] - p0[0], 2);
    double exp4 = std::pow(p2[1] - p0[1], 2);

    return (ang1 < ang2) || ((ang1 > ang2) ? false : (exp1 + exp2 - exp3 - exp4 > 0));
  };

  const auto pivot = *std::ranges::min_element(input_, [](auto &a, auto &b) { return a[1] < b[1]; });
  for (int pt = 0; pt < points_count_; pt++) {
    const bool ev = pt % 2 == 0;
    const int shift = ev ? 0 : -1;
    const int revshift = ev ? -1 : 0;
    for (int i = 1; i < points_count_ + shift; i += 2) {
      if (cmp(pivot, input_[i - shift], input_[i + revshift])) {
        std::swap(input_[i], input_[i - (ev ? 1 : -1)]);
      }
    }
  }
}

bool GrahamConvexHullSequential::RunImpl() {
  PerformSort();

  for (int i = 0; i < 3; i++) {
    res_.push_back(input_[i]);
  }

  for (int i = 3; i < points_count_; ++i) {
    while (res_.size() > 1) {
      const auto &pv = res_.back();
      const auto dx1 = res_.rbegin()[1][0] - pv[0];
      const auto dy1 = res_.rbegin()[1][1] - pv[1];
      const auto dx2 = input_[i][0] - pv[0];
      const auto dy2 = input_[i][1] - pv[1];
      if (dx1 * dy2 < dy1 * dx2) {
        break;
      }
      res_.pop_back();
    }
    res_.push_back(input_[i]);
  }

  return true;
}

bool GrahamConvexHullSequential::PostProcessingImpl() {
  int res_points_count = static_cast<int>(res_.size());
  *reinterpret_cast<int *>(task_data->outputs[0]) = res_points_count;
  auto *p_out = reinterpret_cast<double *>(task_data->outputs[1]);
  for (int i = 0; i < res_points_count; i++) {
    p_out[2 * i] = res_[i][0];
    p_out[(2 * i) + 1] = res_[i][1];
  }
  return true;
}

}  // namespace shvedova_v_graham_convex_hull_seq
