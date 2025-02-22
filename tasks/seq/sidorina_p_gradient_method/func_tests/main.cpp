#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/sidorina_p_gradient_method/include/ops_seq.hpp"

using Params =
    std::tuple<int, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, double>;

namespace sidorina_p_gradient_method_seq {
class sidorina_p_gradient_method_seq_test : public ::testing::TestWithParam<Params> {
 protected:
};

TEST_P(sidorina_p_gradient_method_seq_test, Test_matrix) {
  const auto &[size, a, b, solution, expected, tolerance] = GetParam();
  std::vector<double> result(expected.size());
  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&size)));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&tolerance)));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(a.data())));
  task->inputs_count.emplace_back(a.size());
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(b.data())));
  task->inputs_count.emplace_back(b.size());
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(solution.data())));
  task->inputs_count.emplace_back(solution.size());
  task->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));

  sidorina_p_gradient_method_seq::GradientMethod gradient_method(task);

  ASSERT_TRUE(gradient_method.ValidationImpl());
  gradient_method.PreProcessingImpl();
  gradient_method.RunImpl();
  gradient_method.PostProcessingImpl();
  for (size_t i = 0; i < expected.size(); i++) {
    ASSERT_NEAR(result[i], expected[i], tolerance);
  }
}

INSTANTIATE_TEST_SUITE_P(sidorina_p_gradient_method_seq_test, sidorina_p_gradient_method_seq_test,
                         ::testing::Values(Params(1, {2}, {4}, {0}, {2}, 1e-6)));

using ParamsVal =
    std::tuple<int, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, double>;

class sidorina_p_gradient_method_seq_test_val : public ::testing::TestWithParam<ParamsVal> {
 protected:
};

TEST_P(sidorina_p_gradient_method_seq_test_val, Test_validation) {
  const auto &[size, a, b, solution, expected, tolerance] = GetParam();
  std::vector<double> result(expected.size());
  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&size)));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(&tolerance)));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(a.data())));
  task->inputs_count.emplace_back(a.size());
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(b.data())));
  task->inputs_count.emplace_back(b.size());
  task->inputs.emplace_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(solution.data())));
  task->inputs_count.emplace_back(solution.size());
  task->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));

  sidorina_p_gradient_method_seq::GradientMethod gradient_method(task);

  ASSERT_FALSE(gradient_method.ValidationImpl());
}

INSTANTIATE_TEST_SUITE_P(sidorina_p_gradient_method_seq_test_val, sidorina_p_gradient_method_seq_test_val,
                         ::testing::Values(Params(0, {2}, {4}, {0}, {2}, 1e-6)));
}  // namespace sidorina_p_gradient_method_seq