#include "all/gromov_a_fox_algorithm/include/ops_all.hpp"

#include <mpi.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/collectives/gather.hpp"
#include "boost/mpi/collectives/scatter.hpp"
#include "oneapi/tbb/global_control.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"

namespace gromov_a_fox_algorithm_all {

bool TestTaskAll::PreProcessingImpl() {
  if (mpiCommunicator_.rank() == 0) {
    auto* inputDataPtr_ = reinterpret_cast<double*>(task_data->inputs[0]);
    std::size_t inputDataSize_ = task_data->inputs_count[0];
    matrixSize_ = static_cast<std::size_t>(std::sqrt(inputDataSize_ / 2));
    matrixElements_ = matrixSize_ * matrixSize_;

    if (inputDataSize_ != 2 * matrixElements_) {
      return false;
    }

    inputMatrixA_.resize(matrixElements_);
    inputMatrixB_.resize(matrixElements_);
    resultMatrix_.resize(matrixElements_, 0.0);

    std::copy(inputDataPtr_, inputDataPtr_ + matrixElements_, inputMatrixA_.begin());
    std::copy(inputDataPtr_ + matrixElements_, inputDataPtr_ + 2 * matrixElements_, inputMatrixB_.begin());
  }
  return true;
}

bool TestTaskAll::ValidationImpl() {
  if (mpiCommunicator_.rank() != 0) {
    return true;
  }
  auto& inputCounts_ = task_data->inputs_count;
  auto& outputCounts_ = task_data->outputs_count;

  if (inputCounts_.size() != 1 || outputCounts_.size() != 1) {
    return false;
  }

  std::size_t inputDataSize_ = inputCounts_[0];
  matrixSize_ = static_cast<std::size_t>(std::sqrt(inputDataSize_ / 2));
  matrixElements_ = matrixSize_ * matrixSize_;
  if (inputDataSize_ != 2 * matrixElements_ || outputCounts_[0] != matrixElements_) {
    return false;
  }

  auto* inputDataPtr_ = reinterpret_cast<double*>(task_data->inputs[0]);
  return inputDataPtr_ != nullptr;
}

bool TestTaskAll::RunImpl() {
  int processRank_ = mpiCommunicator_.rank();
  int processCount_ = mpiCommunicator_.size();
  boost::mpi::broadcast(mpiCommunicator_, matrixSize_, 0);
  matrixElements_ = matrixSize_ * matrixSize_;
  boost::mpi::broadcast(mpiCommunicator_, matrixElements_, 0);
  int gridSize_ = ProcessGrid(processCount_, matrixSize_);
  blockDimension_ = matrixSize_ / gridSize_;
  int blockSize_ = static_cast<int>(blockDimension_);
  int processGroup_ = (processRank_ < gridSize_ * gridSize_) ? 1 : MPI_UNDEFINED;
  MPI_Comm computeComm_ = MPI_COMM_NULL;
  MPI_Comm_split(mpiCommunicator_, processGroup_, processRank_, &computeComm_);
  if (processGroup_ == MPI_UNDEFINED) {
    return true;
  }
  boost::mpi::communicator localMpiComm_(computeComm_, boost::mpi::comm_take_ownership);
  processRank_ = localMpiComm_.rank();
  std::vector<double> scatterMatrixA_(matrixElements_);
  std::vector<double> scatterMatrixB_(matrixElements_);
  if (processRank_ == 0) {
    scatterMatrixA_ = Scatter(inputMatrixA_, matrixSize_, gridSize_, blockSize_);
    scatterMatrixB_ = Scatter(inputMatrixB_, matrixSize_, gridSize_, blockSize_);
  }
  std::vector<double> localMatrixA_(blockSize_ * blockSize_);
  std::vector<double> localMatrixB_(blockSize_ * blockSize_);
  std::vector<double> localMatrixC_(blockSize_ * blockSize_, 0.0);
  boost::mpi::scatter(localMpiComm_, scatterMatrixA_, localMatrixA_.data(), static_cast<int>(localMatrixA_.size()), 0);
  boost::mpi::scatter(localMpiComm_, scatterMatrixB_, localMatrixB_.data(), static_cast<int>(localMatrixB_.size()), 0);
  tbb::global_control tbbControl_{tbb::global_control::max_allowed_parallelism, 1};
  tbb::task_arena tbbTaskArena_;
  tbbTaskArena_.execute([&] {
    FoxStep(localMpiComm_, processRank_, gridSize_, blockSize_, localMatrixA_, localMatrixB_, localMatrixC_);
  });
  std::vector<double> gatheredMatrix_(matrixElements_);
  boost::mpi::gather(localMpiComm_, localMatrixC_.data(), static_cast<int>(localMatrixC_.size()), gatheredMatrix_, 0);

  if (processRank_ == 0) {
    resultMatrix_ = Gather(gatheredMatrix_, matrixSize_, gridSize_, blockSize_);
  }
  return true;
}

bool TestTaskAll::PostProcessingImpl() {
  if (mpiCommunicator_.rank() == 0) {
    std::ranges::copy(resultMatrix_, reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}

int ProcessGrid(int totalProcessCount_, std::size_t matrixSize_) {
  int gridSize_ = static_cast<int>(std::floor(std::sqrt(totalProcessCount_)));
  while (gridSize_ > 1 && (totalProcessCount_ % gridSize_ != 0 || (matrixSize_ % gridSize_) != 0)) {
    --gridSize_;
  }
  return std::max(gridSize_, 1);
}

void ExtractBlock(const std::vector<double>& sourceMatrix_, double* blockBuffer_, int matrixWidth_, int blockSize_,
                  int blockRowIdx_, int blockColIdx_) {
  const double* blockStartPtr_ =
      sourceMatrix_.data() + ((blockRowIdx_ * blockSize_) * matrixWidth_) + (blockColIdx_ * blockSize_);
  for (int rowIdx_ = 0; rowIdx_ < blockSize_; ++rowIdx_) {
    std::memcpy(blockBuffer_ + (rowIdx_ * blockSize_), blockStartPtr_ + (rowIdx_ * matrixWidth_),
                blockSize_ * sizeof(double));
  }
}

void MultBlocks(const double* matrixA_, const double* matrixB_, double* matrixC_, int blockSize_) {
  tbb::parallel_for(
      tbb::blocked_range<int>(0, blockSize_),
      [&](const tbb::blocked_range<int>& blockRange_) {
        for (int rowIdx_ = blockRange_.begin(); rowIdx_ < blockRange_.end(); ++rowIdx_) {
          const double* rowA_ = matrixA_ + (rowIdx_ * blockSize_);
          double* rowC_ = matrixC_ + (rowIdx_ * blockSize_);
          for (int innerIdx_ = 0; innerIdx_ < blockSize_; ++innerIdx_) {
            double elementA_ = rowA_[innerIdx_];
            const double* rowB_ = matrixB_ + (innerIdx_ * blockSize_);
            for (int colIdx_ = 0; colIdx_ < blockSize_; ++colIdx_) {
              rowC_[colIdx_] += elementA_ * rowB_[colIdx_];
            }
          }
        }
      },
      tbb::auto_partitioner());
}

std::vector<double> Scatter(const std::vector<double>& sourceMatrix_, std::size_t matrixSize_, int gridSize_,
                            int blockSize_) {
  std::vector<double> scatterBuffer_(matrixSize_ * matrixSize_);
  int bufferIndex_ = 0;
  for (int blockRowIdx_ = 0; blockRowIdx_ < gridSize_; ++blockRowIdx_) {
    for (int blockColIdx_ = 0; blockColIdx_ < gridSize_; ++blockColIdx_) {
      ExtractBlock(sourceMatrix_, scatterBuffer_.data() + bufferIndex_, static_cast<int>(matrixSize_), blockSize_,
                   blockRowIdx_, blockColIdx_);
      bufferIndex_ += blockSize_ * blockSize_;
    }
  }
  return scatterBuffer_;
}

std::vector<double> Gather(const std::vector<double>& gatheredBuffer_, std::size_t matrixSize_, int gridSize_,
                           int blockSize_) {
  std::vector<double> resultMatrix_(matrixSize_ * matrixSize_, 0.0);
  int bufferIndex_ = 0;
  for (int blockRowIdx_ = 0; blockRowIdx_ < gridSize_; ++blockRowIdx_) {
    for (int blockColIdx_ = 0; blockColIdx_ < gridSize_; ++blockColIdx_) {
      for (int rowIdx_ = 0; rowIdx_ < blockSize_; ++rowIdx_) {
        double* destPtr_ = resultMatrix_.data() + (((blockRowIdx_ * blockSize_) + rowIdx_) * matrixSize_) +
                           (blockColIdx_ * blockSize_);
        const double* sourcePtr_ = gatheredBuffer_.data() + bufferIndex_ + (rowIdx_ * blockSize_);
        std::memcpy(destPtr_, sourcePtr_, blockSize_ * sizeof(double));
      }
      bufferIndex_ += blockSize_ * blockSize_;
    }
  }
  return resultMatrix_;
}

void FoxStep(boost::mpi::communicator& mpiComm_, int processRank_, int activeProcessCount_, int blockSize_,
             std::vector<double>& localMatrixA_, std::vector<double>& localMatrixB_,
             std::vector<double>& localMatrixC_) {
  if (activeProcessCount_ == 1) {
    MultBlocks(localMatrixA_.data(), localMatrixB_.data(), localMatrixC_.data(), blockSize_);
    return;
  }

  std::vector<double> tempMatrixA_(blockSize_ * blockSize_);
  std::vector<double> tempMatrixB_(blockSize_ * blockSize_);

  int processRow_ = processRank_ / activeProcessCount_;
  int processCol_ = processRank_ % activeProcessCount_;

  for (int stepIdx_ = 0; stepIdx_ < activeProcessCount_; ++stepIdx_) {
    if (processCol_ == (processRow_ + stepIdx_) % activeProcessCount_) {
      for (int targetColIdx_ = 0; targetColIdx_ < activeProcessCount_; ++targetColIdx_) {
        if (targetColIdx_ == processCol_) {
          continue;
        }
        int targetProcess_ = (processRow_ * activeProcessCount_) + targetColIdx_;
        mpiComm_.send(targetProcess_, 0, localMatrixA_.data(), blockSize_ * blockSize_);
      }
      tempMatrixA_ = localMatrixA_;
    } else {
      int senderProcess_ = (processRow_ * activeProcessCount_) + ((processRow_ + stepIdx_) % activeProcessCount_);
      mpiComm_.recv(senderProcess_, 0, tempMatrixA_.data(), blockSize_ * blockSize_);
    }
    mpiComm_.barrier();
    MultBlocks(tempMatrixA_.data(), localMatrixB_.data(), localMatrixC_.data(), blockSize_);
    int sendToProcess_ =
        (((processRow_ - 1 + activeProcessCount_) % activeProcessCount_) * activeProcessCount_) + processCol_;
    int recvFromProcess_ = (((processRow_ + 1) % activeProcessCount_) * activeProcessCount_) + processCol_;

    mpiComm_.send(sendToProcess_, 0, localMatrixB_.data(), blockSize_ * blockSize_);
    mpiComm_.recv(recvFromProcess_, 0, tempMatrixB_.data(), blockSize_ * blockSize_);

    mpiComm_.barrier();

    localMatrixB_.swap(tempMatrixB_);
  }
}

}  // namespace gromov_a_fox_algorithm_all