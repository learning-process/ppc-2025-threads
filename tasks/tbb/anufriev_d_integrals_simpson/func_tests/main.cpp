#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
// Include the TBB header instead of SEQ
#include "tbb/anufriev_d_integrals_simpson/include/ops_tbb.hpp"

namespace {
const double kPi = std::numbers::pi;

// MakeTaskData remains the same
std::shared_ptr<ppc::core::TaskData> MakeTaskData(const std::vector<double>& elements,
                                                  std::vector<double>& out_buffer) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  auto* input_ptr = reinterpret_cast<uint8_t*>(const_cast<double*>(elements.data()));
  auto* output_ptr = reinterpret_cast<uint8_t*>(out_buffer.data());
  // Ensure counts are correctly set in bytes
  task_data->inputs.push_back(input_ptr);
  task_data->inputs_count.push_back(static_cast<uint32_t>(elements.size() * sizeof(double)));
  task_data->outputs.push_back(output_ptr);
  task_data->outputs_count.push_back(static_cast<uint32_t>(out_buffer.size() * sizeof(double)));
  return task_data;
}
} // namespace

// Rename test suite
TEST(anufriev_d_integrals_simpson_tbb, test_1D_sin) {
  std::vector<double> in = {1, 0.0, kPi / 2.0, 100, 1};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  // Use the TBB class
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run()); // Check Run success
  ASSERT_TRUE(task.PostProcessing()); // Check PostProcessing success
  double result = out_buffer[0];
  EXPECT_NEAR(result, 1.0, 1e-3);
}

// Rename test suite
TEST(anufriev_d_integrals_simpson_tbb, test_2D_sum_of_squares) {
  std::vector<double> in = {2, 0.0, 1.0, 100, 0.0, 1.0, 100, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  // Use the TBB class
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  double result = out_buffer[0];
  EXPECT_NEAR(result, 2.0 / 3.0, 1e-3);
}

// Rename test suite
TEST(anufriev_d_integrals_simpson_tbb, test_2D_sin_cos) {
  // Increase N for better TBB utilization demonstration (optional)
  std::vector<double> in = {2, 0.0, kPi / 2.0, 200, 0.0, kPi / 2.0, 200, 1};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  // Use the TBB class
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  double result = out_buffer[0];
  EXPECT_NEAR(result, 1.0, 1e-3); // Tolerance might need adjustment for large N
}

// Rename test suite
TEST(anufriev_d_integrals_simpson_tbb, test_unknown_func) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 999};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  // Use the TBB class
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  double result = out_buffer[0];
  EXPECT_DOUBLE_EQ(result, 0.0);
}


// --- Negative Tests ---
// Rename test suite
TEST(anufriev_d_integrals_simpson_tbb, test_invalid_empty_input_ptr) {
    auto task_data = std::make_shared<ppc::core::TaskData>();
    // Provide nullptrs or empty vectors explicitly based on PreProcessing logic
    task_data->inputs.push_back(nullptr);
    task_data->inputs_count.push_back(0);
    std::vector<double> out_buffer(1, 0.0); // Need a valid output buffer for Validation
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_buffer.data()));
    task_data->outputs_count.push_back(sizeof(double));

    anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(task_data);
    // PreProcessing should fail due to null input pointer
    ASSERT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_tbb, test_invalid_output_buffer) {
    auto task_data = std::make_shared<ppc::core::TaskData>();
    std::vector<double> in = {1, 0.0, 1.0, 2, 0}; // Valid input
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data->inputs_count.push_back(in.size() * sizeof(double));
    // Invalid output setup
    task_data->outputs.push_back(nullptr);
    task_data->outputs_count.push_back(0);

    anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(task_data);
    // Validation should fail due to invalid output setup
    ASSERT_FALSE(task.ValidationImpl());
}


// Rename test suite
TEST(anufriev_d_integrals_simpson_tbb, test_invalid_dimension_zero) {
  std::vector<double> in = {0, 0.0, 1.0, 2, 999}; // Dimension 0
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  // PreProcessing should fail
  EXPECT_FALSE(task.PreProcessingImpl());
}

// Rename test suite
TEST(anufriev_d_integrals_simpson_tbb, test_invalid_dimension_negative) {
  std::vector<double> in = {-1, 0.0, 1.0, 2, 999}; // Dimension -1
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
   // PreProcessing should fail
  EXPECT_FALSE(task.PreProcessingImpl());
}


// Rename test suite
TEST(anufriev_d_integrals_simpson_tbb, test_invalid_not_enough_data) {
  std::vector<double> in = {2, 0.0, 1.0, 200}; // Missing data for second dimension
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
   // PreProcessing should fail
  EXPECT_FALSE(task.PreProcessingImpl());
}

// Rename test suite
TEST(anufriev_d_integrals_simpson_tbb, test_invalid_odd_n) {
  std::vector<double> in = {1, 0.0, 1.0, 3, 0}; // n=3 (odd)
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
   // PreProcessing should fail
  EXPECT_FALSE(task.PreProcessingImpl());
}

// Rename test suite
TEST(anufriev_d_integrals_simpson_tbb, test_invalid_negative_n) {
  std::vector<double> in = {1, 0.0, 1.0, -2, 0}; // n=-2 (negative)
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
   // PreProcessing should fail
  EXPECT_FALSE(task.PreProcessingImpl());
}

// Rename test suite
TEST(anufriev_d_integrals_simpson_tbb, test_no_output_buffer_in_taskdata) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(in.data())));
  td->inputs_count.push_back(static_cast<std::uint32_t>(in.size() * sizeof(double)));
  // td->outputs is empty
  anufriev_d_integrals_simpson_tbb::IntegralsSimpsonTBB task(td);
  // Validation should fail
  EXPECT_FALSE(task.Validation());
}