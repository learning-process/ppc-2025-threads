#include "all/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_all.hpp"

#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <random>

void lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::extract_submatrix_block(const std::vector<double>& matrix,
                                                                                  double* block, int total_columns,
                                                                                  int block_size, int block_row_index,
                                                                                  int block_col_index) {
  const double* src0 = matrix.data() + (block_row_index * block_size) * total_columns + (block_col_index * block_size);
  for (int i = 0; i < block_size; ++i) {
    std::memcpy(block + i * block_size, src0 + i * total_columns, block_size * sizeof(double));
  }
}

void lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::multiply_matrix_blocks(const double* A, const double* B,
                                                                                 double* C, int block_size) {
  tbb::parallel_for(
      tbb::blocked_range<int>(0, block_size),
      [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
          const double* Ai = A + i * block_size;
          double* Ci = C + i * block_size;
          for (int k = 0; k < block_size; ++k) {
            double aik = Ai[k];
            const double* Bk = B + k * block_size;
            for (int j = 0; j < block_size; ++j) {
              Ci[j] += aik * Bk[j];
            }
          }
        }
      },
      tbb::auto_partitioner());
}

void lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::perform_fox_algorithm_step(
    boost::mpi::communicator& my_world, int rank, int cnt_work_process, int K, std::vector<double>& local_A,
    std::vector<double>& local_B, std::vector<double>& local_C) {
  std::vector<double> temp_A(K * K);
  std::vector<double> temp_B(K * K);

  for (int l = 0; l < cnt_work_process; ++l) {
    boost::mpi::request send_request1;
    boost::mpi::request recv_request1;
    boost::mpi::request send_request2;
    boost::mpi::request recv_request2;

    int row = rank / cnt_work_process;
    int col = rank % cnt_work_process;

    if (col == (row + l) % cnt_work_process) {
      for (int target_col = 0; target_col < cnt_work_process; ++target_col) {
        if (target_col != col) {
          int target_rank = row * cnt_work_process + target_col;
          auto request = my_world.isend(target_rank, 0, local_A.data(), K * K);
          request.wait();
        }
      }
      temp_A = local_A;
    } else {
      int sender_rank = row * cnt_work_process + ((row + l) % cnt_work_process);
      auto request = my_world.irecv(sender_rank, 0, temp_A.data(), K * K);
      request.wait();
    }
    send_request1.wait();
    recv_request1.wait();
    my_world.barrier();

    lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb ::multiply_matrix_blocks(temp_A.data(), local_B.data(),
                                                                                 local_C.data(), K);

    int send_to = ((row - 1 + cnt_work_process) % cnt_work_process) * cnt_work_process + col;
    int recv_from = ((row + 1) % cnt_work_process) * cnt_work_process + col;

    send_request2 = my_world.isend(send_to, 0, local_B.data(), K * K);
    recv_request2 = my_world.irecv(recv_from, 0, temp_B.data(), K * K);
    my_world.barrier();
    send_request2.wait();
    recv_request2.wait();

    local_B = temp_B;
  }
}

bool lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb ::TestTaskMPITBB::PreProcessingImpl() {
  if (world_.rank() == 0) {
    n_ = reinterpret_cast<std::size_t*>(task_data->inputs[0])[0];
    block_size_ = reinterpret_cast<std::size_t*>(task_data->inputs[3])[0];
    elements = n_ * n_;
    a_.resize(elements);
    b_.resize(elements);
    resultC.clear();
    b_.resize(elements, 0.0);
    std::copy(reinterpret_cast<double*>(task_data->inputs[1]),
              reinterpret_cast<double*>(task_data->inputs[1]) + (n_ * n_), a_.begin());
    std::copy(reinterpret_cast<double*>(task_data->inputs[2]),
              reinterpret_cast<double*>(task_data->inputs[2]) + (n_ * n_), b_.begin());
    resultC.assign(elements, 0.0);
  }
  return true;
}

bool lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::TestTaskMPITBB::ValidationImpl() {
  if (world_.rank() != 0) return true;
  n_ = *reinterpret_cast<std::size_t*>(task_data->inputs[0]);
  std::size_t total = n_ * n_;
  if (total == 0) return false;
  auto& ic = task_data->inputs_count;
  auto& oc = task_data->outputs_count;
  if (ic.size() != 3 || oc.size() != 1) return false;
  if (ic[0] != total || ic[1] != total || ic[2] != 1) return false;
  if (oc[0] != total) return false;
  auto* ptrA = reinterpret_cast<double*>(task_data->inputs[1]);
  auto* ptrB = reinterpret_cast<double*>(task_data->inputs[2]);
  return ptrA && ptrB;
}

bool lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::TestTaskMPITBB::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();
  boost::mpi::broadcast(world_, n_, 0);
  elements = static_cast<int>(n_ * n_);
  boost::mpi::broadcast(world_, elements, 0);
  int cnt_work_process = static_cast<int>(std::floor(std::sqrt(size)));
  while (cnt_work_process > 1) {
    if (size % cnt_work_process == 0 && (n_ % cnt_work_process) == 0) {
      break;
    }
    --cnt_work_process;
  }
  if (cnt_work_process < 1) cnt_work_process = 1;

  int K = static_cast<int>(n_ / cnt_work_process);
  int process_group = (rank < cnt_work_process * cnt_work_process) ? 1 : MPI_UNDEFINED;
  MPI_Comm computation_comm;
  MPI_Comm_split(world_, process_group, rank, &computation_comm);
  if (process_group == MPI_UNDEFINED) {
    return true;
  }
  boost::mpi::communicator my_comm(computation_comm, boost::mpi::comm_take_ownership);
  rank = my_comm.rank();
  std::vector<double> scatter_A(elements);
  std::vector<double> scatter_B(elements);
  if (rank == 0) {
    int idx = 0;
    for (int br = 0; br < cnt_work_process; ++br) {
      for (int bc = 0; bc < cnt_work_process; ++bc) {
        extract_submatrix_block(a_, scatter_A.data() + idx, n_, K, br, bc);
        extract_submatrix_block(b_, scatter_B.data() + idx, n_, K, br, bc);
        idx += K * K;
      }
    }
  }
  std::vector<double> localA(K * K), localB(K * K), localC(K * K, 0.0);
  boost::mpi::scatter(my_comm, scatter_A, localA.data(), K * K, 0);
  boost::mpi::scatter(my_comm, scatter_B, localB.data(), K * K, 0);
  tbb::global_control fix_ctrl{tbb::global_control::max_allowed_parallelism, 1};
  tbb::task_arena arena;
  arena.execute([&] { perform_fox_algorithm_step(my_comm, rank, cnt_work_process, K, localA, localB, localC); });
  std::vector<double> gathered(elements);
  boost::mpi::gather(my_comm, localC.data(), localC.size(), gathered, 0);

  if (rank == 0) {
    resultC.assign(n_ * n_, 0.0);
    int idx = 0;
    for (int br = 0; br < cnt_work_process; ++br) {
      for (int bc = 0; bc < cnt_work_process; ++bc) {
        for (int i = 0; i < K; ++i) {
          for (int j = 0; j < K; ++j) {
            int gr = br * K + i;
            int gc = bc * K + j;
            resultC[gr * n_ + gc] = gathered[idx + i * K + j];
          }
        }
        idx += K * K;
      }
    }
  }
  return true;
}

bool lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::TestTaskMPITBB::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(resultC, reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}
