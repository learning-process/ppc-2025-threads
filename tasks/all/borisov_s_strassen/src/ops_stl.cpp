#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <cstddef>
#include <thread>
#include <vector>

#include "all/borisov_s_strassen/include/ops_all.hpp"
#include "boost/mpi/collectives/broadcast.hpp"

namespace borisov_s_strassen_all {
namespace {

std::vector<double> MultiplyNaive(const std::vector<double>& a, const std::vector<double>& b, int n) {
  std::vector<double> c(n * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int k = 0; k < n; ++k) s += a[(i * n) + k] * b[(k * n) + j];
      c[(i * n) + j] = s;
    }
  return c;
}

std::vector<double> AddMatr(const std::vector<double>& a, const std::vector<double>& b, int n) {
  std::vector<double> c(n * n);
  for (int i = 0; i < n * n; ++i) c[i] = a[i] + b[i];
  return c;
}

std::vector<double> SubMatr(const std::vector<double>& a, const std::vector<double>& b, int n) {
  std::vector<double> c(n * n);
  for (int i = 0; i < n * n; ++i) c[i] = a[i] - b[i];
  return c;
}

std::vector<double> SubMatrix(const std::vector<double>& m, int n, int row, int col, int size) {
  std::vector<double> SubMatr(size * size);
  for (int i = 0; i < size; ++i) {
    std::copy(m.begin() + (row + i) * n + col, m.begin() + (row + i) * n + col + size, SubMatr.begin() + i * size);
  }
  return SubMatr;
}

void SetSubMatrix(std::vector<double>& m, const std::vector<double>& SubMatr, int n, int row, int col, int size) {
  for (int i = 0; i < size; ++i) {
    std::copy(SubMatr.begin() + i * size, SubMatr.begin() + (i + 1) * size, m.begin() + (row + i) * n + col);
  }
}

int NextPowerOfTwo(int n) {
  int r = 1;
  while (r < n) r <<= 1;
  return r;
}

std::vector<double> StrassenRecursive(const std::vector<double>& a, const std::vector<double>& b, int n,
                                      int depth = 0) {
  const int parallel_depth = 2;
  if (n <= 128) return MultiplyNaive(a, b, n);

  int k = n / 2;
  auto a11 = SubMatrix(a, n, 0, 0, k);
  auto a12 = SubMatrix(a, n, 0, k, k);
  auto a21 = SubMatrix(a, n, k, 0, k);
  auto a22 = SubMatrix(a, n, k, k, k);
  auto b11 = SubMatrix(b, n, 0, 0, k);
  auto b12 = SubMatrix(b, n, 0, k, k);
  auto b21 = SubMatrix(b, n, k, 0, k);
  auto b22 = SubMatrix(b, n, k, k, k);

  std::vector<double> m1, m2, m3, m4, m5, m6, m7;

  if (depth < parallel_depth) {
    std::thread t1([&] { m1 = StrassenRecursive(AddMatr(a11, a22, k), AddMatr(b11, b22, k), k, depth + 1); });
    std::thread t2([&] { m2 = StrassenRecursive(AddMatr(a21, a22, k), b11, k, depth + 1); });
    std::thread t3([&] { m3 = StrassenRecursive(a11, SubMatr(b12, b22, k), k, depth + 1); });
    std::thread t4([&] { m4 = StrassenRecursive(a22, SubMatr(b21, b11, k), k, depth + 1); });
    std::thread t5([&] { m5 = StrassenRecursive(AddMatr(a11, a12, k), b22, k, depth + 1); });
    std::thread t6([&] { m6 = StrassenRecursive(SubMatr(a21, a11, k), AddMatr(b11, b12, k), k, depth + 1); });
    std::thread t7([&] { m7 = StrassenRecursive(SubMatr(a12, a22, k), AddMatr(b21, b22, k), k, depth + 1); });
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
    t7.join();
  } else {
    m1 = StrassenRecursive(AddMatr(a11, a22, k), AddMatr(b11, b22, k), k, depth + 1);
    m2 = StrassenRecursive(AddMatr(a21, a22, k), b11, k, depth + 1);
    m3 = StrassenRecursive(a11, SubMatr(b12, b22, k), k, depth + 1);
    m4 = StrassenRecursive(a22, SubMatr(b21, b11, k), k, depth + 1);
    m5 = StrassenRecursive(AddMatr(a11, a12, k), b22, k, depth + 1);
    m6 = StrassenRecursive(SubMatr(a21, a11, k), AddMatr(b11, b12, k), k, depth + 1);
    m7 = StrassenRecursive(SubMatr(a12, a22, k), AddMatr(b21, b22, k), k, depth + 1);
  }

  std::vector<double> c(n * n, 0.0);
  auto c11 = AddMatr(SubMatr(AddMatr(m1, m4, k), m5, k), m7, k);
  auto c12 = AddMatr(m3, m5, k);
  auto c21 = AddMatr(m2, m4, k);
  auto c22 = AddMatr(SubMatr(AddMatr(m1, m3, k), m2, k), m6, k);

  SetSubMatrix(c, c11, n, 0, 0, k);
  SetSubMatrix(c, c12, n, 0, k, k);
  SetSubMatrix(c, c21, n, k, 0, k);
  SetSubMatrix(c, c22, n, k, k, k);
  return c;
}

}  // namespace

bool ParallelStrassenMpiStl::PreProcessingImpl() {
  if (world_.rank() == 0) {
    size_t in_cnt = task_data->inputs_count[0];
    auto* dbl = reinterpret_cast<double*>(task_data->inputs[0]);
    input_.assign(dbl, dbl + in_cnt);

    rowsA_ = static_cast<int>(input_[0]);
    colsA_ = static_cast<int>(input_[1]);
    rowsB_ = static_cast<int>(input_[2]);
    colsB_ = static_cast<int>(input_[3]);

    int max_dim = std::max({rowsA_, colsA_, colsB_});
    m_ = NextPowerOfTwo(max_dim);

    std::vector<double> a(rowsA_ * colsA_);
    std::vector<double> b(rowsB_ * colsB_);
    size_t off = 4;
    for (int i = 0; i < rowsA_ * colsA_; ++i) a[i] = input_[off + i];
    off += rowsA_ * colsA_;
    for (int i = 0; i < rowsB_ * colsB_; ++i) b[i] = input_[off + i];

    a_pad_.assign(m_ * m_, 0.0);
    b_pad_.assign(m_ * m_, 0.0);
    for (int i = 0; i < rowsA_; ++i)
      for (int j = 0; j < colsA_; ++j) a_pad_[(i * m_) + j] = a[(i * colsA_) + j];
    for (int i = 0; i < rowsB_; ++i)
      for (int j = 0; j < colsB_; ++j) b_pad_[(i * m_) + j] = b[(i * colsB_) + j];

    output_.resize(2 + rowsA_ * colsB_);
  }

  boost::mpi::broadcast(world_, rowsA_, 0);
  boost::mpi::broadcast(world_, colsA_, 0);
  boost::mpi::broadcast(world_, rowsB_, 0);
  boost::mpi::broadcast(world_, colsB_, 0);
  boost::mpi::broadcast(world_, m_, 0);

  if (world_.rank() != 0) {
    a_pad_.resize(m_ * m_);
    b_pad_.resize(m_ * m_);
  }
  boost::mpi::broadcast(world_, a_pad_, 0);
  boost::mpi::broadcast(world_, b_pad_, 0);

  return true;
}

bool ParallelStrassenMpiStl::ValidationImpl() {
  bool ok = true;
  if (world_.rank() == 0) {
    ok = (colsA_ == rowsB_) &&
         (task_data->inputs_count[0] >= 4 + static_cast<size_t>(rowsA_ * colsA_ + rowsB_ * colsB_));
  }
  boost::mpi::broadcast(world_, ok, 0);
  return ok;
}

bool ParallelStrassenMpiStl::RunImpl() {
  if (m_ == 1) {
    if (world_.rank() == 0) {
      output_[0] = static_cast<double>(rowsA_);
      output_[1] = static_cast<double>(colsB_);
      output_[2] = a_pad_[0] * b_pad_[0];
    }
    world_.barrier();
    return true;
  }

  constexpr int kNumP = 7;
  int half = m_ / 2;
  auto A11 = SubMatrix(a_pad_, m_, 0, 0, half);
  auto A12 = SubMatrix(a_pad_, m_, 0, half, half);
  auto A21 = SubMatrix(a_pad_, m_, half, 0, half);
  auto A22 = SubMatrix(a_pad_, m_, half, half, half);
  auto B11 = SubMatrix(b_pad_, m_, 0, 0, half);
  auto B12 = SubMatrix(b_pad_, m_, 0, half, half);
  auto B21 = SubMatrix(b_pad_, m_, half, 0, half);
  auto B22 = SubMatrix(b_pad_, m_, half, half, half);

  std::vector<std::vector<double>> local_P(kNumP);
  for (int p = world_.rank(); p < kNumP; p += world_.size()) {
    switch (p) {
      case 0:
        local_P[p] = StrassenRecursive(AddMatr(A11, A22, half), AddMatr(B11, B22, half), half);
        break;
      case 1:
        local_P[p] = StrassenRecursive(AddMatr(A21, A22, half), B11, half);
        break;
      case 2:
        local_P[p] = StrassenRecursive(A11, SubMatr(B12, B22, half), half);
        break;
      case 3:
        local_P[p] = StrassenRecursive(A22, SubMatr(B21, B11, half), half);
        break;
      case 4:
        local_P[p] = StrassenRecursive(AddMatr(A11, A12, half), B22, half);
        break;
      case 5:
        local_P[p] = StrassenRecursive(SubMatr(A21, A11, half), AddMatr(B11, B12, half), half);
        break;
      case 6:
        local_P[p] = StrassenRecursive(SubMatr(A12, A22, half), AddMatr(B21, B22, half), half);
        break;
      default:
        break;
    }
  }

  if (world_.rank() == 0) {
    std::vector<std::vector<double>> P(kNumP);
    for (int p = 0; p < kNumP; ++p) {
      int owner = p % world_.size();
      if (owner != 0) {
        world_.recv(owner, p, P[p]);
      } else {
        P[p] = std::move(local_P[p]);
      }
    }

    std::vector<double> C_pad(m_ * m_, 0.0);
    auto C11 = AddMatr(SubMatr(AddMatr(P[0], P[3], half), P[4], half), P[6], half);
    auto C12 = AddMatr(P[2], P[4], half);
    auto C21 = AddMatr(P[1], P[3], half);
    auto C22 = AddMatr(SubMatr(AddMatr(P[0], P[2], half), P[1], half), P[5], half);
    SetSubMatrix(C_pad, C11, m_, 0, 0, half);
    SetSubMatrix(C_pad, C12, m_, 0, half, half);
    SetSubMatrix(C_pad, C21, m_, half, 0, half);
    SetSubMatrix(C_pad, C22, m_, half, half, half);

    output_[0] = static_cast<double>(rowsA_);
    output_[1] = static_cast<double>(colsB_);
    for (int i = 0; i < rowsA_; ++i)
      for (int j = 0; j < colsB_; ++j) output_[2 + i * colsB_ + j] = C_pad[(i * m_) + j];
  } else {
    for (int p = world_.rank(); p < kNumP; p += world_.size()) world_.send(0, p, local_P[p]);
  }

  world_.barrier();
  return true;
}

bool ParallelStrassenMpiStl::PostProcessingImpl() {
  if (world_.rank() == 0) {
    double* out = reinterpret_cast<double*>(task_data->outputs[0]);
    std::copy(output_.begin(), output_.end(), out);
  }

  return true;
}

}  // namespace borisov_s_strassen_all
