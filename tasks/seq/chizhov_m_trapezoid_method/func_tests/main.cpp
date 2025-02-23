#include <gtest/gtest.h>

#include <functional>
#include <vector>

#include "seq/chizhov_m_trapezoid_method/include/ops_seq.hpp"

TEST(chizhov_m_trapezoid_method_seq, one_variable_squared) {
  int div = 20;
  int dim = 1;
  std::vector<double> limits = {0.0, 5.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) { return f_val[0] * f_val[0]; };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  taskDataSeq->inputs_count.emplace_back(sizeof(div));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs_count.emplace_back(sizeof(dim));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  taskDataSeq->inputs_count.emplace_back(limits.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.ValidationImpl());
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], 41.66, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, one_variable_cube) {
  int div = 45;
  int dim = 1;
  std::vector<double> limits = {0.0, 5.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) { return f_val[0] * f_val[0] * f_val[0]; };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  taskDataSeq->inputs_count.emplace_back(sizeof(div));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs_count.emplace_back(sizeof(dim));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  taskDataSeq->inputs_count.emplace_back(limits.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.ValidationImpl());
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], 156.25, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, mul_two_variables) {
  int div = 10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) { return f_val[0] * f_val[1]; };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  taskDataSeq->inputs_count.emplace_back(sizeof(div));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs_count.emplace_back(sizeof(dim));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  taskDataSeq->inputs_count.emplace_back(limits.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.ValidationImpl());
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], 56.25, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, sum_two_variables) {
  int div = 10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) { return f_val[0] + f_val[1]; };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  taskDataSeq->inputs_count.emplace_back(sizeof(div));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs_count.emplace_back(sizeof(dim));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  taskDataSeq->inputs_count.emplace_back(limits.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.ValidationImpl());
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], 60, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, dif_two_variables) {
  int div = 10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) { return f_val[1] - f_val[0]; };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  taskDataSeq->inputs_count.emplace_back(sizeof(div));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs_count.emplace_back(sizeof(dim));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  taskDataSeq->inputs_count.emplace_back(limits.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.ValidationImpl());
  testTaskSequential.PreProcessingImpl();
  testTaskSequential.RunImpl();
  testTaskSequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], -15, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, invalid_value_dim) {
  int div = 10;
  int dim = -2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  taskDataSeq->inputs_count.emplace_back(sizeof(div));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs_count.emplace_back(sizeof(dim));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  taskDataSeq->inputs_count.emplace_back(limits.size());

  chizhov_m_trapezoid_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.ValidationImpl());
}

TEST(chizhov_m_trapezoid_method_seq, invalid_value_div) {
  int div = -10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  taskDataSeq->inputs_count.emplace_back(sizeof(div));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs_count.emplace_back(sizeof(dim));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  taskDataSeq->inputs_count.emplace_back(limits.size());

  chizhov_m_trapezoid_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.ValidationImpl());
}

TEST(chizhov_m_trapezoid_method_seq, invalid_limit_size) {
  int div = -10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  taskDataSeq->inputs_count.emplace_back(sizeof(div));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  taskDataSeq->inputs_count.emplace_back(sizeof(dim));

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  taskDataSeq->inputs_count.emplace_back(limits.size());

  chizhov_m_trapezoid_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_FALSE(testTaskSequential.ValidationImpl());
}
