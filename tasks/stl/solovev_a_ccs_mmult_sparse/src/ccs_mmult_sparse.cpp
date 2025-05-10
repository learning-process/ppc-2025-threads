#include "stl/solovev_a_ccs_mmult_sparse/include/ccs_mmult_sparse.hpp"

#include <complex>
#include <vector>

void solovev_a_matrix_stl::SeqMatMultCcs::worker_loop(solovev_a_matrix_stl::SeqMatMultCcs* self) {
  thread_local std::vector<int> available;
  thread_local std::vector<std::complex<double>> cask;

  while (true) {
    if (self->terminate_.load()) return;

    int col = self->next_col_.fetch_add(1);
    if (col >= self->c_n_) {
      std::this_thread::yield();
      continue;
    }

    available.assign(self->r_n_, 0);

    if (self->phase_ == 1) {
      for (int i = self->M2_->col_p[col]; i < self->M2_->col_p[col + 1]; ++i) {
        int r = self->M2_->row[i];
        for (int j = self->M1_->col_p[r]; j < self->M1_->col_p[r + 1]; ++j) {
          available[self->M1_->row[j]] = 1;
        }
      }
      self->counts_[col] = std::accumulate(available.begin(), available.end(), 0);
    } else if (self->phase_ == 2) {
      cask.assign(self->r_n_, {0.0, 0.0});
      for (int i = self->M2_->col_p[col]; i < self->M2_->col_p[col + 1]; ++i) {
        int r = self->M2_->row[i];
        auto v2 = self->M2_->val[i];
        for (int j = self->M1_->col_p[r]; j < self->M1_->col_p[r + 1]; ++j) {
          int rr = self->M1_->row[j];
          cask[rr] += self->M1_->val[j] * v2;
          available[rr] = 1;
        }
      }
      int pos = self->M3_->col_p[col];
      for (int rr = 0; rr < self->r_n_; ++rr) {
        if (available[rr]) {
          self->M3_->row[pos] = rr;
          self->M3_->val[pos++] = cask[rr];
        }
      }
    }

    int done = self->completed_.fetch_add(1) + 1;
    if (done == self->c_n_) {
      std::lock_guard<std::mutex> lk(self->mtx_);
      self->cv_done_.notify_all();
    }
  }
}

bool solovev_a_matrix_stl::SeqMatMultCcs::PreProcessingImpl() {
  M1_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[0]);
  M2_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[1]);
  M3_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->outputs[0]);
  return true;
}

bool solovev_a_matrix_stl::SeqMatMultCcs::ValidationImpl() {
  int m1_c_n = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[0])->c_n;
  int m2_r_n = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[1])->r_n;
  return (m1_c_n == m2_r_n);
}

bool solovev_a_matrix_stl::SeqMatMultCcs::RunImpl() {
  r_n_ = M1_->r_n;
  c_n_ = M2_->c_n;
  M3_->r_n = r_n_;
  M3_->c_n = c_n_;

  std::call_once(init_flag_, [&]() {
    counts_.resize(c_n_);
    unsigned th = std::thread::hardware_concurrency();
    if (!th) th = 1;
    for (unsigned i = 0; i < th; ++i) {
      workers_.emplace_back(worker_loop, this);
    }
  });

  next_col_.store(0);
  completed_.store(0);
  phase_ = 1;
  {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_done_.wait(lk, [&]() { return completed_.load() >= c_n_; });
  }

  M3_->col_p.resize(c_n_ + 1);
  M3_->col_p[0] = 0;
  for (int i = 0; i < c_n_; ++i) {
    M3_->col_p[i + 1] = M3_->col_p[i] + counts_[i];
  }
  int total = M3_->col_p[c_n_];
  M3_->n_z = total;
  M3_->row.resize(total);
  M3_->val.resize(total);

  next_col_.store(0);
  completed_.store(0);
  phase_ = 2;
  {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_done_.wait(lk, [&]() { return completed_.load() >= c_n_; });
  }

  terminate_.store(true);
  for (auto& th : workers_) th.join();
  workers_.clear();

  return true;
}

bool solovev_a_matrix_stl::SeqMatMultCcs::PostProcessingImpl() { return true; }