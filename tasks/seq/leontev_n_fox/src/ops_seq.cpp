#include "seq/leontev_n_fox/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace leontev_n_fox_seq {

const int k = 4;

double FoxSeq::at_a(size_t i, size_t j) const {
  if (i >= n || j >= n) {
    return 0.0;
  }
  return input_a_[i * n + j];
}

double FoxSeq::at_b(size_t i, size_t j) const {
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

bool FoxSeq::PreProcessingImpl() {
  size_t input_count = task_data->inputs_count[0];
  auto* double_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  n = reinterpret_cast<size_t*>(task_data->inputs[1])[0];
  input_a_.assign(double_ptr, double_ptr + input_count / 2);
  input_b_.assign(double_ptr + input_count / 2, double_ptr + input_count);
  
  size_t output_count = task_data->outputs_count[0];
  output_.resize(output_count, 0.0);

  return true;
}

bool FoxSeq::ValidationImpl() {
  if (input_a_.size() != n * n || output_.size() != n * n) {
    return false;
  }
  return true;
}

bool FoxSeq::RunImpl() {
  size_t q;
  size_t div1;
  if (n % k == 0) {
    q = n / k;
  } else {
    q = n / k + 1;
  }
  for (size_t i = 0; i < q; i++) {
    for (size_t j = 0; j < q; j++) {
      for (size_t l = 0; l < q; l++) {
        div1 = ((i + l) % q) * k;
        // block calc
        for (size_t row = 0; row < k; row++) {
          for (size_t col = 0; col < std::min(static_cast<size_t>(k), n - k * j); col++) {
            for (size_t m = 0; m < k; m++) {
              output_[(row + i * k) * n + (col + j * k)] += at_a(i * k + row, div1 + m) * at_b(div1 + m, j * k + col);
            }
          }
        }
        // block calc end
      }
    }
  }
  return true;
}

bool FoxSeq::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}

}  // namespace leontev_n_fox_seq
