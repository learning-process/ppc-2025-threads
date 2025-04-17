#include "stl/leontev_n_fox/include/ops_stl.hpp"

#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace leontev_n_fox_stl {

double FoxSTL::at_a(size_t i, size_t j) const {
  if (i >= n || j >= n) {
    return 0.0;
  }
  return input_a_[i * n + j];
}

double FoxSTL::at_b(size_t i, size_t j) const {
  if (i >= n || j >= n) {
    return 0.0;
  }
  return input_b_[i * n + j];
}

std::vector<double> mat_mul(std::vector<double>& a, std::vector<double>& b, size_t n) {
  std::vector<double> res(n * n, 0.0);
  for (size_t j = 0; j < n; j++) {
    for (size_t i = 0; i < n; i++) {
      for (size_t l = 0; l < n; l++) {
        res[i * n + j] += a[i * n + l] * b[l * n + j];
      }
    }
  }
  return res;
}

void FoxSTL::mat_mul_blocks(size_t a_posX, size_t a_posY, size_t b_posX, size_t b_posY, size_t c_posX, size_t c_posY,
                            size_t size) {
  for (size_t j = 0; j < std::min(static_cast<size_t>(size), n - c_posX); j++) {
    for (size_t i = 0; i < size; i++) {
      for (size_t l = 0; l < size; l++) {
        output_[(i + c_posY) * n + (j + c_posX)] += at_a(i + a_posY, l + a_posX) * at_b(l + b_posY, j + b_posX);
      }
    }
  }
}

bool FoxSTL::PreProcessingImpl() {
  size_t input_count = task_data->inputs_count[0];
  auto* double_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  n = reinterpret_cast<size_t*>(task_data->inputs[1])[0];
  input_a_.assign(double_ptr, double_ptr + input_count / 2);
  input_b_.assign(double_ptr + input_count / 2, double_ptr + input_count);

  size_t output_count = task_data->outputs_count[0];
  output_.resize(output_count, 0.0);

  return true;
}

bool FoxSTL::ValidationImpl() {
  if (input_a_.size() != n * n || output_.size() != n * n) {
    return false;
  }
  return true;
}

bool FoxSTL::RunImpl() {
  size_t div1;
  const int num_threads = ppc::util::GetPPCNumThreads();
  size_t q = std::min(n, static_cast<size_t>(std::sqrt(num_threads)));
  if (q == 0) {
    return false;
  }
  size_t k;
  std::vector<std::thread> threads(num_threads);
  if (n % q == 0) {
    k = n / q;
  } else {
    k = n / (q - 1);
  }
  for (size_t l = 0; l < q; l++) {
    for (size_t i = 0; i < q; i++) {
      for (size_t j = 0; j < q; j++) {
        div1 = ((i + l) % q) * k;
        threads[i * q + j] = std::thread(&FoxSTL::mat_mul_blocks, this, div1, i * k, j * k, div1, j * k, i * k, k);
      }
    }
    for (size_t i = 0; i < q * q; i++) {
      threads[i].join();
    }
  }
  return true;
}

bool FoxSTL::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}

}  // namespace leontev_n_fox_stl
