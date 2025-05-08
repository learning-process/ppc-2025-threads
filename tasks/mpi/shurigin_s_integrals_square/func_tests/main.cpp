#define USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/shurigin_s_integrals_square/include/ops_mpi.hpp"

namespace shurigin_s_integrals_square_mpi_func_test {
constexpr double kTolerance = 1e-3;

int get_mpi_rank() {
  int rank_val;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_val);
  return rank_val;
}

TEST(ShuriginSIntegralsSquareMPI_Func, TestIntegrationXSquared) {
  const int rank = get_mpi_rank();

  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int intervals = 10000;
  const double expected_value = 1.0 / 3.0;

  std::vector<double> input_data_vec;
  double output_data = 0.0;

  if (rank == 0) {
    input_data_vec = {lower_bound, upper_bound, static_cast<double>(intervals)};
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims = 1;
  const size_t expected_input_bytes = 3 * current_dims * sizeof(double);

  if (rank == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.push_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data->outputs_count.push_back(sizeof(double));
  } else {
    task_data->inputs.push_back(nullptr);
    task_data->inputs_count.push_back(expected_input_bytes);
    task_data->outputs.push_back(nullptr);
    task_data->outputs_count.push_back(sizeof(double));
  }

  shurigin_s_integrals_square_mpi::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return x * x; });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  if (rank == 0) {
    ASSERT_NEAR(output_data, expected_value, kTolerance);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ShuriginSIntegralsSquareMPI_Func, TestIntegrationLinear) {
  const int rank = get_mpi_rank();

  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int intervals = 10000;
  const double expected_value = 0.5;

  std::vector<double> input_data_vec;
  double output_data = 0.0;

  if (rank == 0) {
    input_data_vec = {lower_bound, upper_bound, static_cast<double>(intervals)};
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims = 1;
  const size_t expected_input_bytes = 3 * current_dims * sizeof(double);

  if (rank == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.push_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data->outputs_count.push_back(sizeof(double));
  } else {
    task_data->inputs.push_back(nullptr);
    task_data->inputs_count.push_back(expected_input_bytes);
    task_data->outputs.push_back(nullptr);
    task_data->outputs_count.push_back(sizeof(double));
  }

  shurigin_s_integrals_square_mpi::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return x; });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  if (rank == 0) {
    ASSERT_NEAR(output_data, expected_value, kTolerance);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ShuriginSIntegralsSquareMPI_Func, TestIntegrationSine) {
  const int rank = get_mpi_rank();

  const double lower_bound = 0.0;
  const double upper_bound = std::numbers::pi;
  const int intervals = 20000;
  const double expected_value = 2.0;

  std::vector<double> input_data_vec;
  double output_data = 0.0;

  if (rank == 0) {
    input_data_vec = {lower_bound, upper_bound, static_cast<double>(intervals)};
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims = 1;
  const size_t expected_input_bytes = 3 * current_dims * sizeof(double);

  if (rank == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.push_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data->outputs_count.push_back(sizeof(double));
  } else {
    task_data->inputs.push_back(nullptr);
    task_data->inputs_count.push_back(expected_input_bytes);
    task_data->outputs.push_back(nullptr);
    task_data->outputs_count.push_back(sizeof(double));
  }

  shurigin_s_integrals_square_mpi::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return std::sin(x); });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  if (rank == 0) {
    ASSERT_NEAR(output_data, expected_value, kTolerance);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ShuriginSIntegralsSquareMPI_Func, TestIntegrationExponential) {
  const int rank = get_mpi_rank();

  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int intervals = 10000;
  const double expected_value = std::numbers::e - 1.0;

  std::vector<double> input_data_vec;
  double output_data = 0.0;

  if (rank == 0) {
    input_data_vec = {lower_bound, upper_bound, static_cast<double>(intervals)};
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims = 1;
  const size_t expected_input_bytes = 3 * current_dims * sizeof(double);

  if (rank == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.push_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data->outputs_count.push_back(sizeof(double));
  } else {
    task_data->inputs.push_back(nullptr);
    task_data->inputs_count.push_back(expected_input_bytes);
    task_data->outputs.push_back(nullptr);
    task_data->outputs_count.push_back(sizeof(double));
  }

  shurigin_s_integrals_square_mpi::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return std::exp(x); });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  if (rank == 0) {
    ASSERT_NEAR(output_data, expected_value, kTolerance);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ShuriginSIntegralsSquareMPI_Func, TestFunctionAssignment) {
  auto dummy_task_data = std::make_shared<ppc::core::TaskData>();
  shurigin_s_integrals_square_mpi::Integral integral_task(dummy_task_data);

  std::function<double(double)> func_1d = [](double x) { return x * x; };
  integral_task.SetFunction(func_1d);

  double test_value = 2.0;
  double expected_func_eval = 4.0;
  ASSERT_DOUBLE_EQ(func_1d(test_value), expected_func_eval);

  std::function<double(const std::vector<double>&)> func_nd = [](const std::vector<double>& p) { return p[0] + p[1]; };
  integral_task.SetFunction(func_nd, 2);
  std::vector<double> test_point_nd = {1.0, 2.0};
  double expected_func_eval_nd = 3.0;
  ASSERT_DOUBLE_EQ(func_nd(test_point_nd), expected_func_eval_nd);
}

TEST(ShuriginSIntegralsSquareMPI_Func, TestIntegrationCosine) {
  const int rank = get_mpi_rank();

  const double lower_bound = 0.0;
  const double upper_bound = std::numbers::pi / 2.0;
  const int intervals = 10000;
  const double expected_value = 1.0;

  std::vector<double> input_data_vec;
  double output_data = 0.0;

  if (rank == 0) {
    input_data_vec = {lower_bound, upper_bound, static_cast<double>(intervals)};
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims = 1;
  const size_t expected_input_bytes = 3 * current_dims * sizeof(double);

  if (rank == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.push_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data->outputs_count.push_back(sizeof(double));
  } else {
    task_data->inputs.push_back(nullptr);
    task_data->inputs_count.push_back(expected_input_bytes);
    task_data->outputs.push_back(nullptr);
    task_data->outputs_count.push_back(sizeof(double));
  }

  shurigin_s_integrals_square_mpi::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return std::cos(x); });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  if (rank == 0) {
    ASSERT_NEAR(output_data, expected_value, kTolerance);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ShuriginSIntegralsSquareMPI_Func, TestIntegrationLogarithm) {
  const int rank = get_mpi_rank();

  const double lower_bound = 1.0;
  const double upper_bound = 2.0;
  const int intervals = 10000;
  const double expected_value = (2.0 * std::numbers::ln2) - 1.0;

  std::vector<double> input_data_vec;
  double output_data = 0.0;

  if (rank == 0) {
    input_data_vec = {lower_bound, upper_bound, static_cast<double>(intervals)};
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims = 1;
  const size_t expected_input_bytes = 3 * current_dims * sizeof(double);

  if (rank == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.push_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data->outputs_count.push_back(sizeof(double));
  } else {
    task_data->inputs.push_back(nullptr);
    task_data->inputs_count.push_back(expected_input_bytes);
    task_data->outputs.push_back(nullptr);
    task_data->outputs_count.push_back(sizeof(double));
  }

  shurigin_s_integrals_square_mpi::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return std::log(x); });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  if (rank == 0) {
    ASSERT_NEAR(output_data, expected_value, kTolerance);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ShuriginSIntegralsSquareMPI_Func, TestIntegrationReciprocal) {
  const int rank = get_mpi_rank();

  const double lower_bound = 1.0;
  const double upper_bound = 2.0;
  const int intervals = 10000;
  const double expected_value = std::numbers::ln2;

  std::vector<double> input_data_vec;
  double output_data = 0.0;

  if (rank == 0) {
    input_data_vec = {lower_bound, upper_bound, static_cast<double>(intervals)};
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims = 1;
  const size_t expected_input_bytes = 3 * current_dims * sizeof(double);

  if (rank == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.push_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data->outputs_count.push_back(sizeof(double));
  } else {
    task_data->inputs.push_back(nullptr);
    task_data->inputs_count.push_back(expected_input_bytes);
    task_data->outputs.push_back(nullptr);
    task_data->outputs_count.push_back(sizeof(double));
  }

  shurigin_s_integrals_square_mpi::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return 1.0 / x; });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  if (rank == 0) {
    ASSERT_NEAR(output_data, expected_value, kTolerance);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ShuriginSIntegralsSquareMPI_Func, TestIntegrationSqrt) {
  const int rank = get_mpi_rank();

  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int intervals = 10000;
  const double expected_value = 2.0 / 3.0;

  std::vector<double> input_data_vec;
  double output_data = 0.0;

  if (rank == 0) {
    input_data_vec = {lower_bound, upper_bound, static_cast<double>(intervals)};
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims = 1;
  const size_t expected_input_bytes = 3 * current_dims * sizeof(double);

  if (rank == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.push_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data->outputs_count.push_back(sizeof(double));
  } else {
    task_data->inputs.push_back(nullptr);
    task_data->inputs_count.push_back(expected_input_bytes);
    task_data->outputs.push_back(nullptr);
    task_data->outputs_count.push_back(sizeof(double));
  }

  shurigin_s_integrals_square_mpi::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return std::sqrt(x); });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  if (rank == 0) {
    ASSERT_NEAR(output_data, expected_value, kTolerance);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ShuriginSIntegralsSquareMPI_Func, TestIntegration2DProduct) {
  const int rank = get_mpi_rank();

  const double lower_bound_x = 0.0;
  const double upper_bound_x = 1.0;
  const double lower_bound_y = 0.0;
  const double upper_bound_y = 1.0;
  const int intervals_x = 200;
  const int intervals_y = 200;
  const double expected_value = 1.0 / 4.0;

  std::vector<double> input_data_vec;
  double output_data = 0.0;

  if (rank == 0) {
    input_data_vec = {lower_bound_x,
                      lower_bound_y,
                      upper_bound_x,
                      upper_bound_y,
                      static_cast<double>(intervals_x),
                      static_cast<double>(intervals_y)};
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims_2d = 2;
  const size_t expected_input_bytes_2d = 3 * current_dims_2d * sizeof(double);

  if (rank == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.push_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data->outputs_count.push_back(sizeof(double));
  } else {
    task_data->inputs.push_back(nullptr);
    task_data->inputs_count.push_back(expected_input_bytes_2d);
    task_data->outputs.push_back(nullptr);
    task_data->outputs_count.push_back(sizeof(double));
  }

  shurigin_s_integrals_square_mpi::Integral integral_task(task_data);
  integral_task.SetFunction([](const std::vector<double>& point) { return point[0] * point[1]; }, 2);

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  if (rank == 0) {
    ASSERT_NEAR(output_data, expected_value, kTolerance);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ShuriginSIntegralsSquareMPI_Func, TestIntegration2DSquareSum) {
  const int rank = get_mpi_rank();

  const double lower_bound_x = 0.0;
  const double upper_bound_x = 1.0;
  const double lower_bound_y = 0.0;
  const double upper_bound_y = 1.0;
  const int intervals_x = 200;
  const int intervals_y = 200;
  const double expected_value = 2.0 / 3.0;

  std::vector<double> input_data_vec;
  double output_data = 0.0;

  if (rank == 0) {
    input_data_vec = {lower_bound_x,
                      lower_bound_y,
                      upper_bound_x,
                      upper_bound_y,
                      static_cast<double>(intervals_x),
                      static_cast<double>(intervals_y)};
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims_2d = 2;
  const size_t expected_input_bytes_2d = 3 * current_dims_2d * sizeof(double);

  if (rank == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.push_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data->outputs_count.push_back(sizeof(double));
  } else {
    task_data->inputs.push_back(nullptr);
    task_data->inputs_count.push_back(expected_input_bytes_2d);
    task_data->outputs.push_back(nullptr);
    task_data->outputs_count.push_back(sizeof(double));
  }

  shurigin_s_integrals_square_mpi::Integral integral_task(task_data);
  integral_task.SetFunction(
      [](const std::vector<double>& point) { return (point[0] * point[0]) + (point[1] * point[1]); }, 2);

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  if (rank == 0) {
    ASSERT_NEAR(output_data, expected_value, kTolerance);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ShuriginSIntegralsSquareMPI_Func, TestIntegration2DSinSum) {
  const int rank = get_mpi_rank();

  const double lower_bound_x = 0.0;
  const double upper_bound_x = std::numbers::pi / 2.0;
  const double lower_bound_y = 0.0;
  const double upper_bound_y = std::numbers::pi / 2.0;
  const int intervals_x = 200;
  const int intervals_y = 200;
  const double expected_value = 2.0;

  std::vector<double> input_data_vec;
  double output_data = 0.0;

  if (rank == 0) {
    input_data_vec = {lower_bound_x,
                      lower_bound_y,
                      upper_bound_x,
                      upper_bound_y,
                      static_cast<double>(intervals_x),
                      static_cast<double>(intervals_y)};
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims_2d = 2;
  const size_t expected_input_bytes_2d = 3 * current_dims_2d * sizeof(double);

  if (rank == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.push_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data->outputs_count.push_back(sizeof(double));
  } else {
    task_data->inputs.push_back(nullptr);
    task_data->inputs_count.push_back(expected_input_bytes_2d);
    task_data->outputs.push_back(nullptr);
    task_data->outputs_count.push_back(sizeof(double));
  }

  shurigin_s_integrals_square_mpi::Integral integral_task(task_data);
  integral_task.SetFunction([](const std::vector<double>& point) { return std::sin(point[0] + point[1]); }, 2);

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  if (rank == 0) {
    ASSERT_NEAR(output_data, expected_value, kTolerance);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST(ShuriginSIntegralsSquareMPI_Func, TestIntegration3DProduct) {
  const int rank = get_mpi_rank();

  const double lower_bound_x = 0.0;
  const double upper_bound_x = 1.0;
  const double lower_bound_y = 0.0;
  const double upper_bound_y = 1.0;
  const double lower_bound_z = 0.0;
  const double upper_bound_z = 1.0;
  const int intervals_x = 50;
  const int intervals_y = 50;
  const int intervals_z = 50;
  const double expected_value = 0.125;

  std::vector<double> input_data_vec;
  double output_data = 0.0;

  if (rank == 0) {
    input_data_vec = {lower_bound_x,
                      lower_bound_y,
                      lower_bound_z,
                      upper_bound_x,
                      upper_bound_y,
                      upper_bound_z,
                      static_cast<double>(intervals_x),
                      static_cast<double>(intervals_y),
                      static_cast<double>(intervals_z)};
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  const size_t current_dims_3d = 3;
  const size_t expected_input_bytes_3d = 3 * current_dims_3d * sizeof(double);

  if (rank == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data_vec.data()));
    task_data->inputs_count.push_back(input_data_vec.size() * sizeof(double));
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
    task_data->outputs_count.push_back(sizeof(double));
  } else {
    task_data->inputs.push_back(nullptr);
    task_data->inputs_count.push_back(expected_input_bytes_3d);
    task_data->outputs.push_back(nullptr);
    task_data->outputs_count.push_back(sizeof(double));
  }

  shurigin_s_integrals_square_mpi::Integral integral_task(task_data);
  integral_task.SetFunction([](const std::vector<double>& point) { return point[0] * point[1] * point[2]; }, 3);

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  if (rank == 0) {
    ASSERT_NEAR(output_data, expected_value, kTolerance);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}
}  // namespace shurigin_s_integrals_square_mpi_func_test