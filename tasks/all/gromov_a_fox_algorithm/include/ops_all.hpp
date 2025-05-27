#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gromov_a_fox_algorithm_all {
void ExtractBlock(const std::vector<double>& sourceMatrix_, double* blockBuffer_, int matrixWidth_, int blockSize_,
                  int blockRowIdx_, int blockColIdx_);
void MultBlocks(const double* matrixA_, const double* matrixB_, double* matrixC_, int blockSize_);
void FoxStep(boost::mpi::communicator& mpiComm_, int processRank_, int activeProcessCount_, int blockSize_,
             std::vector<double>& localMatrixA_, std::vector<double>& localMatrixB_,
             std::vector<double>& localMatrixC_);
std::vector<double> Scatter(const std::vector<double>& sourceMatrix_, std::size_t matrixSize_, int gridSize_,
                                  int blockSize_);
std::vector<double> Gather(const std::vector<double>& gatheredBuffer_, std::size_t matrixSize_, int gridSize_,
                                 int blockSize_);
int ProcessGrid(int totalProcessCount_, std::size_t matrixSize_);
class TestTaskAll : public ppc::core::Task {
 public:
  explicit TestTaskAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> inputMatrixA_, inputMatrixB_;
  std::vector<double> resultMatrix_;
  std::size_t matrixSize_ = 0;
  std::size_t blockDimension_ = 0;
  std::size_t matrixElements_{};
  boost::mpi::communicator mpiCommunicator_;
};

}  // namespace gromov_a_fox_algorithm_all