#include "all/chizhov_m_trapezoid_method/include/ops_all.hpp"

#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/serialization/vector.hpp>
#include <algorithm>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <functional>
#include <vector>

double chizhov_m_trapezoid_method_all::TrapezoidMethod(Function& f, size_t div, size_t dim,
                                                       std::vector<double>& lower_limits,
                                                       std::vector<double>& upper_limits,
                                                       boost::mpi::communicator world) {
  int int_dim = static_cast<int>(dim);
  std::vector<double> h(int_dim);
  std::vector<int> steps(int_dim);

  for (int i = 0; i < int_dim; i++) {
    steps[i] = static_cast<int>(div);
    h[i] = (upper_limits[i] - lower_limits[i]) / steps[i];
  }

  long long total_nodes = 1;
  for (int i = 0; i < int_dim; ++i) {
    total_nodes *= (steps[i] + 1);
  }

  int rank = world.rank();
  int size = world.size();

  long long base_count = total_nodes / size;
  long long remainder = total_nodes % size;

  long long start;
  if (rank < remainder) {
    start = rank * (base_count + 1);
  } else {
    start = remainder * (base_count + 1) + (rank - remainder) * base_count;
  }

  long long end;
  if (rank < remainder) {
    end = start + base_count + 1;
  } else {
    end = start + base_count;
  }

  double local_result = 0.0;

  const int num_threads = ppc::util::GetPPCNumThreads();
  oneapi::tbb::task_arena arena(num_threads);

  arena.execute([&] {
    local_result = oneapi::tbb::parallel_reduce(
        tbb::blocked_range<long>(start, end, 16), 0.0,
        [&](const tbb::blocked_range<long>& r, double local_res) {
          for (long i = r.begin(); i != r.end(); ++i) {
            int temp = static_cast<int>(i);
            double weight = 1.0;
            std::vector<double> point(int_dim);

            for (int j = 0; j < int_dim; j++) {
              int node_index = temp % (steps[j] + 1);
              point[j] = lower_limits[j] + node_index * h[j];
              temp /= (steps[j] + 1);
            }

            for (int j = 0; j < int_dim; j++) {
              if (point[j] == lower_limits[j] || point[j] == upper_limits[j]) {
                weight *= 1.0;
              } else {
                weight *= 2.0;
              }
            }

            local_res += weight * f(point);
          }
          return local_res;
        },
        [](double a, double b) { return a + b; });
  });

  double global_result = 0.0;
  boost::mpi::reduce(world, local_result, global_result, std::plus<>(), 0);

  if (rank == 0) {
    for (int i = 0; i < int_dim; ++i) {
      global_result *= h[i] / 2.0;
    }
    return std::round(global_result * 100.0) / 100.0;
  }

  return 0.0;
}

bool chizhov_m_trapezoid_method_all::TestTaskMPI::PreProcessingImpl() {
  if (world.rank() == 0) {
    int* divisions_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    div_ = *divisions_ptr;

    int* dimension_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    dim_ = *dimension_ptr;

    auto* limit_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
    for (int i = 0; i < static_cast<int>(task_data->inputs_count[2]); i += 2) {
      lower_limits_.push_back(limit_ptr[i]);
      upper_limits_.push_back(limit_ptr[i + 1]);
    }
  }

  boost::mpi::broadcast(world, div_, 0);
  boost::mpi::broadcast(world, dim_, 0);
  boost::mpi::broadcast(world, lower_limits_, 0);
  boost::mpi::broadcast(world, upper_limits_, 0);

  return true;
}

bool chizhov_m_trapezoid_method_all::TestTaskMPI::ValidationImpl() {
  bool valid = true;
  if (world.rank() == 0) {
    auto* divisions_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    auto* dimension_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    if (*divisions_ptr <= 0 || *dimension_ptr <= 0) {
      valid = false;
    }
    if (task_data->inputs_count[2] % 2 != 0) {
      valid = false;
    }
    auto* limit_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
    for (int i = 0; i < static_cast<int>(task_data->inputs_count[2]); i += 2) {
      if (limit_ptr[i] >= limit_ptr[i + 1]) {
        valid = false;
      }
    }
  }
  
  boost::mpi::broadcast(world, valid, 0);
  return valid;
}

void chizhov_m_trapezoid_method_all::TestTaskMPI::SetFunc(const Function f) { f_ = f; };

bool chizhov_m_trapezoid_method_all::TestTaskMPI::RunImpl() {
  if (!f_) {
    if (world.rank() == 0) {
      std::cerr << "Function not set!" << std::endl;
    }
    return false;
  }
  res_ = TrapezoidMethod(f_, div_, dim_, lower_limits_, upper_limits_, world);
  return true;
}

bool chizhov_m_trapezoid_method_all::TestTaskMPI::PostProcessingImpl() {
  if (world.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = res_;
  }
  return true;
}