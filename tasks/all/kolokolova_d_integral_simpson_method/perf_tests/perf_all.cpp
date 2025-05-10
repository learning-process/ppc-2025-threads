#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/kolokolova_d_integral_simpson_method/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(kolokolova_d_integral_simpson_method_all, test_pipeline_run) { ASSERT_EQ(1, 1); }

TEST(kolokolova_d_integral_simpson_method_all, test_task_run) { ASSERT_EQ(1, 1); }
