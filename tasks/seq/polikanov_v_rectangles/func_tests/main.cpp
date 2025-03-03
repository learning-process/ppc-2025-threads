#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/task/include/task.hpp"

namespace {
void RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound> bounds, std::size_t discretization, double ref,
                 const polikanov_v_rectangles::FunctionExecutor &function) {
  double out = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(bounds.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&discretization));
  task_data->inputs_count.emplace_back(bounds.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));

  polikanov_v_rectangles::TaskSEQ task(task_data, function);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_NEAR(out, ref, 0.8);
}
}  // namespace

TEST(polikanov_v_rectangles_seq, exp) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(1, {0, 1}), 75, 1.71,
              [](const auto &p) { return std::exp(p[0]); });
}

TEST(polikanov_v_rectangles_seq, exp_pow_2) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(1, {0, 1}), 75, 3.15,
              [](const auto &p) { return std::pow(std::exp(p[0]), 2); });
}

TEST(polikanov_v_rectangles_seq, exp_arg_pow_2) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(1, {0, 1}), 75, 1.45,
              [](const auto &p) { return std::exp(std::pow(p[0], 2)); });
}

TEST(polikanov_v_rectangles_seq, exp_arg_pow_2_pow_2) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(1, {0, 1}), 75, 0.2,
              [](const auto &p) { return std::pow(std::pow(p[0], 2), 2); });
}

TEST(polikanov_v_rectangles_seq, exp__2D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {0, 1}), 75, 2.9,
              [](const auto &p) { return std::exp(p[0]) * std::exp(p[1]); });
}

TEST(polikanov_v_rectangles_seq, exp_pow_2__2D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {0, 1}), 75, 9.93,
              [](const auto &p) { return std::pow(std::exp(p[0]), 2) * std::pow(std::exp(p[1]), 2); });
}

TEST(polikanov_v_rectangles_seq, exp_arg_pow_2__2D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {0, 1}), 75, 2.10,
              [](const auto &p) { return std::exp(std::pow(p[0], 2)) * std::exp(std::pow(p[1], 2)); });
}

TEST(polikanov_v_rectangles_seq, exp_arg_pow_2_pow_2__2D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(2, {0, 1}), 75, 0.03,
              [](const auto &p) { return std::pow(std::pow(p[0], 2), 2) * std::pow(std::pow(p[1], 2), 2); });
}

TEST(polikanov_v_rectangles_seq, exp__3D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(3, {0, 1}), 75, 4.97,
              [](const auto &p) { return std::exp(p[0]) * std::exp(p[1]) * std::exp(p[2]); });
}

TEST(polikanov_v_rectangles_seq, exp_pow_2__3D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(3, {0, 1}), 75, 31.31, [](const auto &p) {
    return std::pow(std::exp(p[0]), 2) * std::pow(std::exp(p[1]), 2) * std::pow(std::exp(p[2]), 2);
  });
}

TEST(polikanov_v_rectangles_seq, exp_arg_pow_2__3D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(3, {0, 1}), 75, 3.05, [](const auto &p) {
    return std::exp(std::pow(p[0], 2)) * std::exp(std::pow(p[1], 2)) * std::exp(std::pow(p[2], 2));
  });
}

TEST(polikanov_v_rectangles_seq, exp_arg_pow_2_pow_2__3D) {
  RunFuncTest(std::vector<polikanov_v_rectangles::IntegrationBound>(3, {0, 1}), 75, 0.007, [](const auto &p) {
    return std::pow(std::pow(p[0], 2), 2) * std::pow(std::pow(p[1], 2), 2) * std::pow(std::pow(p[2], 2), 2);
  });
}