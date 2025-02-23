#include <gtest/gtest.h>

#include <numbers>

#include "seq/poroshin_v_multi_integral_with_trapez_method/include/ops_seq.hpp"

double poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::area(std::vector<double> &arguments) {
  return 1.0 + arguments.at(0) * 0.0;
}
double poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::f1(std::vector<double> &arguments) {
  return arguments.at(0);
}
double poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::f1cos(std::vector<double> &arguments) {
  return cos(arguments.at(0));
}
double poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::f1Euler(std::vector<double> &arguments) {
  return 2 * cos(arguments.at(0)) * sin(arguments.at(0));
}
double poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::f3(std::vector<double> &arguments) {
  return arguments.at(0) * arguments.at(1) * arguments.at(2);
}
double poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::f3advanced(
    std::vector<double> &arguments) {
  return sin(arguments.at(0)) * tan(arguments.at(1)) * log(arguments.at(2));
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, invalid_size) {
  std::vector<int> n;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> out;
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(n.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmpTaskSeq(
      taskSeq, poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::f1);
  ASSERT_FALSE(tmpTaskSeq.ValidationImpl());
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, invalid_out) {
  size_t dim = 10;
  std::vector<int> n(dim);
  std::vector<double> a(dim);
  std::vector<double> b(dim);
  std::vector<double> out(2);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(n.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmpTaskSeq(
      taskSeq, poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::f1);
  ASSERT_FALSE(tmpTaskSeq.ValidationImpl());
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, minus_0_5_pi_0_5_pi_cos) {
  std::vector<int> n = {1000};
  std::vector<double> a = {-0.5 * std::numbers::pi};
  std::vector<double> b = {0.5 * std::numbers::pi};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(n.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmpTaskSeq(
      taskSeq, poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::f1cos);
  ASSERT_TRUE(tmpTaskSeq.ValidationImpl());
  tmpTaskSeq.PreProcessingImpl();
  tmpTaskSeq.RunImpl();
  tmpTaskSeq.PostProcessingImpl();
  ASSERT_NEAR(2.0, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, Eulers_integral) {
  std::vector<int> n = {1000};
  std::vector<double> a = {0};
  std::vector<double> b = {0.5 * std::numbers::pi};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(n.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmpTaskSeq(
      taskSeq, poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::f1Euler);
  ASSERT_TRUE(tmpTaskSeq.ValidationImpl());
  tmpTaskSeq.PreProcessingImpl();
  tmpTaskSeq.RunImpl();
  tmpTaskSeq.PostProcessingImpl();
  ASSERT_NEAR(1.0, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, 05x05_area) {
  std::vector<int> n = {1000, 1000};
  std::vector<double> a = {0, 0};
  std::vector<double> b = {0.5, 0.5};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(n.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmpTaskSeq(
      taskSeq, poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::area);
  ASSERT_TRUE(tmpTaskSeq.ValidationImpl());
  tmpTaskSeq.PreProcessingImpl();
  tmpTaskSeq.RunImpl();
  tmpTaskSeq.PostProcessingImpl();
  ASSERT_NEAR(0.25, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, 2x2_area) {
  std::vector<int> n = {1000, 1000};
  std::vector<double> a = {0, 0};
  std::vector<double> b = {2.0, 2.0};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(n.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmpTaskSeq(
      taskSeq, poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::area);
  ASSERT_TRUE(tmpTaskSeq.ValidationImpl());
  tmpTaskSeq.PreProcessingImpl();
  tmpTaskSeq.RunImpl();
  tmpTaskSeq.PostProcessingImpl();
  ASSERT_NEAR(4.0, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, 2_3x1_4_area) {
  std::vector<int> n = {1000, 1000};
  std::vector<double> a = {2.0, 1.0};
  std::vector<double> b = {3.0, 4.0};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(n.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmpTaskSeq(
      taskSeq, poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::area);
  ASSERT_TRUE(tmpTaskSeq.ValidationImpl());
  tmpTaskSeq.PreProcessingImpl();
  tmpTaskSeq.RunImpl();
  tmpTaskSeq.PostProcessingImpl();
  ASSERT_NEAR(3.0, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, _0_2xminus2_0_area) {
  std::vector<int> n = {1000, 1000};
  std::vector<double> a = {0.0, -2.0};
  std::vector<double> b = {2.0, 0.0};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(n.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmpTaskSeq(
      taskSeq, poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::area);
  ASSERT_TRUE(tmpTaskSeq.ValidationImpl());
  tmpTaskSeq.PreProcessingImpl();
  tmpTaskSeq.RunImpl();
  tmpTaskSeq.PostProcessingImpl();
  ASSERT_NEAR(4.0, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, minus03_0_x_15_17_x_2_21_area) {
  std::vector<int> n = {100, 100, 100};
  std::vector<double> a = {-0.3, 1.5, 2.0};
  std::vector<double> b = {0.0, 1.7, 2.1};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(n.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmpTaskSeq(
      taskSeq, poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::area);
  ASSERT_TRUE(tmpTaskSeq.ValidationImpl());
  tmpTaskSeq.PreProcessingImpl();
  tmpTaskSeq.RunImpl();
  tmpTaskSeq.PostProcessingImpl();
  ASSERT_NEAR(0.006, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, 08_1_x_15_17_x_18_2_xyz) {
  std::vector<int> n = {100, 100, 100};
  std::vector<double> a = {0.8, 1.5, 1.8};
  std::vector<double> b = {1.0, 1.7, 2.0};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(n.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmpTaskSeq(
      taskSeq, poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::f3);
  ASSERT_TRUE(tmpTaskSeq.ValidationImpl());
  tmpTaskSeq.PreProcessingImpl();
  tmpTaskSeq.RunImpl();
  tmpTaskSeq.PostProcessingImpl();
  ASSERT_NEAR(0.021888, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, 08_1_x_19_2_x_29_3_sinx_tgy_lnz) {
  std::vector<int> n = {100, 100, 100};
  std::vector<double> a = {0.8, 1.9, 2.9};
  std::vector<double> b = {1.0, 2.0, 3.0};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  taskSeq->inputs_count.emplace_back(n.size());
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskSeq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmpTaskSeq(
      taskSeq, poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::f3advanced);
  ASSERT_TRUE(tmpTaskSeq.ValidationImpl());
  tmpTaskSeq.PreProcessingImpl();
  tmpTaskSeq.RunImpl();
  tmpTaskSeq.PostProcessingImpl();
  ASSERT_NEAR(-0.00427191467841401, out[0], eps);
}