#pragma once

#include <cstddef>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>
#include <functional>

#include "core/task/include/task.hpp"

namespace filateva_e_simpson_all {

using Func = double (*)(std::vector<double>);

class Simpson : public ppc::core::Task {
 public:
  explicit Simpson(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void setFunc(Func f);

 private:
  size_t mer_;
  std::vector<double> a_, b_, h_;
  size_t steps_{};
  double res_{};

  Func f_;
  boost::mpi::communicator world_;
  double IntegralFunc(unsigned long start, unsigned long end);
};
}  // namespace filateva_e_simpson_omp