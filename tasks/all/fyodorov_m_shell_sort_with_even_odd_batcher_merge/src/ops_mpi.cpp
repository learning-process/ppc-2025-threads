#include "all/fyodorov_m_shell_sort_with_even_odd_batcher_merge/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <vector>

#include "boost/mpi/collectives/gatherv.hpp"
#include "boost/mpi/collectives/scatterv.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>  // для std::copy, std::min
#include <boost/serialization/vector.hpp>
#include <iostream>  // для std::cout, std::endl
#include <string>

namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi {

namespace {
boost::mpi::communicator world;
}  // namespace

void TestTaskMPI::PrepareScatterGather(int n, int size, std::vector<int>& sendcounts, std::vector<int>& displs) {
  int local_n = n / size;
  int remainder = n % size;
  sendcounts.assign(size, local_n);
  for (int i = 0; i < remainder; ++i) {
    sendcounts[i]++;
  }
  displs[0] = 0;
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + sendcounts[i - 1];
  }
}

std::vector<int> TestTaskMPI::MergeBlocks(const std::vector<std::vector<int>>& blocks) {
  std::vector<int> merged;
  for (const auto& block : blocks) {
    if (!block.empty()) {
      if (merged.empty()) {
        merged = block;
      } else {
        std::vector<int> temp(merged.size() + block.size());
        TestTaskMPI::BatcherMerge(merged, const_cast<std::vector<int>&>(block), temp);
        merged.assign(temp.begin(), temp.end());
      }
    }
  }
  return merged;
}

std::vector<std::vector<int>> TestTaskMPI::SplitGatheredToBlocks(const std::vector<int>& gathered,
                                                                 const std::vector<int>& sendcounts) {
  std::vector<std::vector<int>> blocks(sendcounts.size());
  int pos = 0;
  for (size_t i = 0; i < sendcounts.size(); ++i) {
    if (sendcounts[i] > 0 && pos + sendcounts[i] <= static_cast<int>(gathered.size())) {
      auto first = gathered.begin() + pos;
      auto last = first + sendcounts[i];
      blocks[i].assign(first, last);
    } else {
      blocks[i].clear();
    }
    pos += sendcounts[i];
  }
  return blocks;
}

void TestTaskMPI::BroadcastOutput(boost::mpi::communicator& world, int rank, int size, std::vector<int>& output) {
  if (rank == 0) {
    for (int dest = 1; dest < size; ++dest) {
      world.send(dest, 0, output);
    }
  } else {
    world.recv(0, 0, output);
  }
}

void TestTaskMPI::LocalSort(std::vector<int>& local_data, int rank) {
  if (local_data.empty()) {
    return;
  }
  TestTaskMPI::ShellSort(local_data);
  std::cout << "rank " << rank << " local_data (first 10): ";
  for (int i = 0; i < std::min(10, static_cast<int>(local_data.size())); ++i) {
    std::cout << local_data[i] << " ";
  }
  std::cout << '\n';
}

void TestTaskMPI::PrintFirstN(const std::string& label, const std::vector<int>& data, int n) {
  std::cout << label;
  for (int i = 0; i < std::min(n, static_cast<int>(data.size())); ++i) {
    std::cout << data[i] << " ";
  }
  std::cout << '\n';
}

bool TestTaskMPI::PreProcessingImpl() {
  unsigned int input_size = 0;
  if (world.rank() == 0) {
    input_size = task_data->inputs_count[0];
  }
  boost::mpi::broadcast(world, input_size, 0);

  input_.resize(input_size);
  if (world.rank() == 0) {
    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(in_ptr, in_ptr + input_size, input_.begin());
  }
  boost::mpi::broadcast(world, input_, 0);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  PrintFirstN("rank " + std::to_string(world.rank()) + " input_ (first 10): ", input_);

  return true;
}

bool TestTaskMPI::ValidationImpl() {
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }

  if ((task_data->inputs_count[0] > 0 && task_data->inputs[0] == nullptr) ||
      (task_data->outputs_count[0] > 0 && task_data->outputs[0] == nullptr)) {
    return false;
  }

  return true;
}

bool TestTaskMPI::RunImpl() {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  output_.clear();
  output_.shrink_to_fit();

  if (rank != 0) {
    input_.clear();
  }
  boost::mpi::broadcast(world, input_, 0);

  int n = static_cast<int>(input_.size());
  if (rank == 0) {
    PrintFirstN("input_ (first 10): ", input_);
  }

  std::vector<int> local_data;
  int local_size = 0;

  if (n > 0) {
    std::vector<int> sendcounts;
    std::vector<int> displs;
    TestTaskMPI::PrepareScatterGather(n, size, sendcounts, displs);

    local_size = sendcounts[rank];
    local_data.resize(local_size);

    int* send_ptr = (rank == 0 && n > 0) ? input_.data() : nullptr;
    int* recv_ptr = (local_size > 0) ? local_data.data() : nullptr;

    boost::mpi::scatterv(world, send_ptr, sendcounts, displs, recv_ptr, local_size, 0);

    if (local_size > 0) {
      LocalSort(local_data, rank);

      std::vector<int> gathered;
      if (rank == 0) {
        gathered.resize(n);
      }

      int* send_ptr_g = (local_size > 0) ? local_data.data() : nullptr;
      int* recv_ptr_g = (rank == 0 && n > 0) ? gathered.data() : nullptr;

      boost::mpi::gatherv(world, send_ptr_g, local_size, recv_ptr_g, sendcounts, displs, 0);

      if (rank == 0) {
        PrintFirstN("gathered (first 10): ", gathered);

        std::vector<std::vector<int>> blocks = SplitGatheredToBlocks(gathered, sendcounts);
        std::vector<int> merged = MergeBlocks(blocks);
        output_.assign(merged.begin(), merged.end());

        BroadcastOutput(world, rank, size, output_);
      }
    }
  }

  unsigned int output_size = task_data->outputs_count[0];
  output_.resize(output_size, 0);

  if (rank == 0 && !output_.empty()) {
    PrintFirstN("output_ (first 10): ", output_);
  }

  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  unsigned int output_size = task_data->outputs_count[0];
  if (output_.size() == output_size) {
    for (size_t i = 0; i < output_.size(); ++i) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
    }
  } else {
    for (size_t i = 0; i < output_size; ++i) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = 0;
    }
  }
  return true;
}

void TestTaskMPI::ShellSort(std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }
  int n = static_cast<int>(arr.size());
  std::vector<int> gaps;
  for (int k = 1; (1 << k) - 1 < n; ++k) {
    gaps.push_back((1 << k) - 1);
  }
  for (auto it = gaps.rbegin(); it != gaps.rend(); ++it) {
    int gap = *it;
#pragma omp parallel for default(none) shared(arr, n, gap)
    for (int offset = 0; offset < gap; ++offset) {
      for (int i = offset + gap; i < n; i += gap) {
        int temp = arr[i];
        int j = i;
        while (j >= gap && arr[j - gap] > temp) {
          arr[j] = arr[j - gap];
          j -= gap;
        }
        arr[j] = temp;
      }
    }
  }
}

void TestTaskMPI::BatcherMerge(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result) {
  size_t i = 0;
  size_t j = 0;
  size_t k = 0;
  while (i < left.size() && j < right.size()) {
    if (left[i] <= right[j]) {
      result[k++] = left[i++];
    } else {
      result[k++] = right[j++];
    }
  }
  while (i < left.size()) {
    result[k++] = left[i++];
  }
  while (j < right.size()) {
    result[k++] = right[j++];
  }
}

}  // namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi