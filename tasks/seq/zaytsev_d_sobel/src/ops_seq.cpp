#include "seq/zaytsev_d_sobel/include/ops_seq.hpp"

#include <cmath>
#include <vector>

bool zaytsev_d_sobel_seq::TestTaskSequential::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + task_data->inputs_count[0]);
  output_ = std::vector<int>(task_data->outputs_count[0], 0);
  rc_size_ = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  return true;
}

bool zaytsev_d_sobel_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool zaytsev_d_sobel_seq::TestTaskSequential::RunImpl() {
  const int gxKernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int gyKernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  for (int i = 1; i < rc_size_ - 1; ++i) {
    for (int j = 1; j < rc_size_ - 1; ++j) {
      int sumGx = 0, sumGy = 0;
      for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
          int ni = i + di;
          int nj = j + dj;
          int kernelRow = di + 1;
          int kernelCol = dj + 1;

          sumGx += input_[ni * rc_size_ + nj] * gxKernel[kernelRow][kernelCol];
          sumGy += input_[ni * rc_size_ + nj] * gyKernel[kernelRow][kernelCol];
        }
      }
      output_[i * rc_size_ + j] = std::min(int(std::sqrt(sumGx * sumGx + sumGy * sumGy)), 255);
    }
  }

  return true;
}

bool zaytsev_d_sobel_seq::TestTaskSequential::PostProcessingImpl() {
  std::copy(output_.begin(), output_.end(), reinterpret_cast<int *>(task_data->outputs[0]));
  return true;
}
