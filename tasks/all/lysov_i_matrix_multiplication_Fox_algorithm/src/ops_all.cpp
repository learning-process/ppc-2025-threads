#include "all/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_all.hpp"

#include <tbb/global_control.h>
#include <tbb/task_arena.h>
#include <tbb/tbb.h>
#include <boost/mpi/request.hpp> 
#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstddef>

int lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::compute_process_grid(int world_size, std::size_t n) {
  int q = static_cast<int>(std::floor(std::sqrt(world_size)));
  while (q > 1 && (world_size % q != 0 || (n % q) != 0)) {
    --q;
  }
  return std::max(q, 1);
}
void lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::ExtractSubmatrixBlock(const std::vector<double>& matrix,
                                                                                double* block, int total_columns,
                                                                                int block_size, int block_row_index,
                                                                                int block_col_index) {
  const double* src0 = matrix.data() + ((block_row_index * block_size) * total_columns) + (block_col_index * block_size);
  for (int i = 0; i < block_size; ++i) {
    std::memcpy(block + (i * block_size), src0 + (i * total_columns), block_size * sizeof(double));
  }
}

void lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::MultiplyMatrixBlocks(const double* a, const double* b,
                                                                               double* c, int block_size) {
  tbb::parallel_for(
      tbb::blocked_range<int>(0, block_size),
      [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
          const double* ai = a + (i * block_size);
          double* ci = c + (i * block_size);
          for (int k = 0; k < block_size; ++k) {
            double aik = ai[k];
            const double* bk = b + (k * block_size);
            for (int j = 0; j < block_size; ++j) {
              ci[j] += aik * bk[j];
            }
          }
        }
      },
      tbb::auto_partitioner());
}
std::vector<double> lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::scatter_matrix(const std::vector<double>& m,
                                                                                      std::size_t n, int q, int k) {
  std::vector<double> buf(n * n);
  int idx = 0;
  for (int br = 0; br < q; ++br)
    for (int bc = 0; bc < q; ++bc) {
      ExtractSubmatrixBlock(m, buf.data() + idx, n, k, br, bc);
      idx += k * k;
    }
  return buf;
}

std::vector<double> lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::gather_matrix(const std::vector<double>& buf,
                                                                                       std::size_t n, int q, int k) {
  std::vector<double> c(n * n, 0.0);
  int idx = 0;
  for (int br = 0; br < q; ++br) {
    for (int bc = 0; bc < q; ++bc) {
      for (int i = 0; i < k; ++i) {
        double* dest = c.data() + (((br * k) + i) * n) + (bc * k);
        const double* src = buf.data() + idx + i * k;
        std::memcpy(dest, src, k * sizeof(double));
      }
      idx += k * k;
    }
  }
  return c;
}

void lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::PerformFoxAlgorithmStep(boost::mpi::communicator& my_world,
                                                                                  int rank, int cnt_work_process, int k,
                                                                                  std::vector<double>& local_a,
                                                                                  std::vector<double>& local_b,
                                                                                  std::vector<double>& local_c) {
  std::vector<double> temp_a(k * k);
  std::vector<double> temp_b(k * k);

  for (int l = 0; l < cnt_work_process; ++l) {
    boost::mpi::request send_request2;
    boost::mpi::request recv_request2;

    int row = rank / cnt_work_process;
    int col = rank % cnt_work_process;

    if (col == (row + l) % cnt_work_process) {
      for (int target_col = 0; target_col < cnt_work_process; ++target_col) {
        if (target_col == col) continue;
        int target_rank = (row * cnt_work_process) + target_col;
        boost::mpi::request req = my_world.isend(target_rank, 0, local_a.data(), k * k);
        req.wait();
      }
      temp_a = local_a;
    } else {
      int sender_rank = (row * cnt_work_process) + ((row + l) % cnt_work_process);
      boost::mpi::request req = my_world.irecv(sender_rank, 0, temp_a.data(), k * k);
      req.wait();
    }
    my_world.barrier();

    lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb ::MultiplyMatrixBlocks(temp_a.data(), local_b.data(),
                                                                               local_c.data(), k);

    int send_to = (((row - 1 + cnt_work_process) % cnt_work_process) * cnt_work_process) + col;
    int recv_from = (((row + 1) % cnt_work_process) * cnt_work_process) + col;

    send_request2 = my_world.isend(send_to, 0, local_b.data(), k * k);
    recv_request2 = my_world.irecv(recv_from, 0, temp_b.data(), k * k);
    my_world.barrier();
    send_request2.wait();
    recv_request2.wait();

    local_b = temp_b;
  }
}

bool lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb ::TestTaskMPITBB::PreProcessingImpl() {
  if (world_.rank() == 0) {
    n_ = reinterpret_cast<std::size_t*>(task_data->inputs[0])[0];
    block_size_ = reinterpret_cast<std::size_t*>(task_data->inputs[3])[0];
    elements = n_ * n_;
    a_.resize(elements);
    b_.resize(elements);
    resultC_.clear();
    b_.resize(elements, 0.0);
    std::copy(reinterpret_cast<double*>(task_data->inputs[1]),
              reinterpret_cast<double*>(task_data->inputs[1]) + (n_ * n_), a_.begin());
    std::copy(reinterpret_cast<double*>(task_data->inputs[2]),
              reinterpret_cast<double*>(task_data->inputs[2]) + (n_ * n_), b_.begin());
    resultC_.assign(elements, 0.0);
  }
  return true;
}

bool lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::TestTaskMPITBB::ValidationImpl() {
  if (world_.rank() != 0) {
    return true;
  }
  n_ = *reinterpret_cast<std::size_t*>(task_data->inputs[0]);
  std::size_t total = n_ * n_;
  if (total == 0) {
    return false;
  }
  auto& ic = task_data->inputs_count;
  auto& oc = task_data->outputs_count;
  if (ic.size() != 3 || oc.size() != 1) {
    return false;
  }
  if (ic[0] != total || ic[1] != total || ic[2] != 1) {
    return false;
  }
  if (oc[0] != total) {
    return false;
  }
  auto* ptr_a = reinterpret_cast<double*>(task_data->inputs[1]);
  auto* ptr_b = reinterpret_cast<double*>(task_data->inputs[2]);
  return (ptr_a != nullptr && ptr_b != nullptr);
  ;
}

bool lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::TestTaskMPITBB::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();
  boost::mpi::broadcast(world_, n_, 0);
  elements = static_cast<int>(n_ * n_);
  boost::mpi::broadcast(world_, elements, 0);
  int q = compute_process_grid(size, n_);
  int k = static_cast<int>(n_ / q);
  int process_group = (rank < q * q) ? 1 : MPI_UNDEFINED;
  MPI_Comm computation_comm;
  MPI_Comm_split(world_, process_group, rank, &computation_comm);
  if (process_group == MPI_UNDEFINED) {
    return true;
  }
  boost::mpi::communicator my_comm(computation_comm, boost::mpi::comm_take_ownership);
  rank = my_comm.rank();
  std::vector<double> scatter_a(elements);
  std::vector<double> scatter_b(elements);
  if (rank == 0) {
    scatter_a = scatter_matrix(a_, n_, q, k);
    scatter_b = scatter_matrix(b_, n_, q, k);
  }
  std::vector<double> localA(k * k), localB(k * k), localC(k * k, 0.0);
  boost::mpi::scatter(my_comm, scatter_a, localA.data(), k * k, 0);
  boost::mpi::scatter(my_comm, scatter_b, localB.data(), k * k, 0);
  tbb::global_control fix_ctrl{tbb::global_control::max_allowed_parallelism, 1};
  tbb::task_arena arena;
  arena.execute([&] { PerformFoxAlgorithmStep(my_comm, rank, q, k, localA, localB, localC); });
  std::vector<double> gathered(elements);
  boost::mpi::gather(my_comm, localC.data(), localC.size(), gathered, 0);

  if (rank == 0) {
    resultC_ = gather_matrix(gathered, n_, q, k);
  }
  return true;
}

bool lysov_i_matrix_multiplication_fox_algorithm_mpi_tbb::TestTaskMPITBB::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(resultC_, reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}
