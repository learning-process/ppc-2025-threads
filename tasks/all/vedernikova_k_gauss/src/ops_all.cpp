#include "all/vedernikova_k_gauss/include/ops_all.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <thread>
#include <vector>

#include "boost/mpi/collectives/gatherv.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/util/include/util.hpp"

bool vedernikova_k_gauss_all::Gauss::ValidationImpl() {
  if (task_data->inputs_count.size() != 3 || task_data->outputs_count.empty()) {
    return false;
  }
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];
  channels_ = task_data->inputs_count[2];
  size_ = width_ * height_ * channels_;
  return (!task_data->inputs.empty() && !task_data->outputs.empty() && task_data->outputs_count[0] == size_) ||
         world_.rank() != 0;
}

bool vedernikova_k_gauss_all::Gauss::PreProcessingImpl() {
  if (world_.rank() == 0) {
    width_ = task_data->inputs_count[0];
    height_ = task_data->inputs_count[1];
    channels_ = task_data->inputs_count[2];
    kernel_.resize(9);
    input_.resize(size_);
    output_.resize(size_);
    ComputeKernel();
    auto* tmp_ptr = reinterpret_cast<uint8_t*>(task_data->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + (width_ * height_ * channels_), input_.begin());
  }
  return true;
}

bool vedernikova_k_gauss_all::Gauss::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_.size(); i++) {
      reinterpret_cast<uint8_t*>(task_data->outputs[0])[i] = output_[i];
    }
  }
  return true;
}

void vedernikova_k_gauss_all::Gauss::ComputeKernel(double sigma) {
  // For 3x3 kernel sigma from [1/3; 1/2] is required
  for (int i = 0; i < 9; i++) {
    int ik = (i % 3) - 1;
    int jk = (i / 3) - 1;
    kernel_[i] = std::exp(-1.0 * (ik * ik + jk * jk) / (2 * sigma * sigma)) / (2 * std::numbers::pi * sigma * sigma);
    ;
  }
  double amount = std::accumulate(kernel_.begin(), kernel_.end(), 0.0);
  for (auto&& it : kernel_) {
    it /= amount;
  }
}

uint8_t vedernikova_k_gauss_all::Gauss::GetPixel(uint32_t x, uint32_t y, uint32_t channel) {
  return input_[(y * width_ * channels_) + (x * channels_) + channel];
}

void vedernikova_k_gauss_all::Gauss::SetPixel(uint8_t value, uint32_t x, uint32_t y, uint32_t channel) {
  output_[(y * width_ * channels_) + (x * channels_) + channel] = value;
}

double vedernikova_k_gauss_all::Gauss::GetMultiplier(int i, int j) { return kernel_[(3 * (j + 1)) + (i + 1)]; }

void vedernikova_k_gauss_all::Gauss::ComputePixel(uint32_t x, uint32_t y) {
  for (uint32_t channel = 0; channel < channels_; channel++) {
    double brightness = 0;
    for (int shift_x = -1; shift_x <= 1; shift_x++) {
      for (int shift_y = -1; shift_y <= 1; shift_y++) {
        // if _x or _y out of image bounds, aproximating them with the nearest valid orthogonally adjacent pixels
        int xn = std::clamp((int)x + shift_x, 0, (int)width_ - 1);
        int yn = std::clamp((int)y + shift_y, 0, (int)height_ - 1);
        brightness += GetPixel(xn, yn, channel) * GetMultiplier(shift_x, shift_y);
      }
    }
    SetPixel(std::ceil(brightness), x, y, channel);
  }
}

bool vedernikova_k_gauss_all::Gauss::RunImpl() {
  const int h = static_cast<int>(height_);
  const int w = static_cast<int>(width_);
  const int world_size = world_.size();
  const int rank = world_.rank();

  int base_rows = h / world_size;
  int extra_rows = h % world_size;
  int start_row = (rank * base_rows) + std::min(rank, extra_rows);
  int row_count = base_rows + (rank < extra_rows ? 1 : 0);

  const int tnum = std::min(row_count, ppc::util::GetPPCNumThreads());
  int base = row_count / tnum;
  int extra = row_count % tnum;

  std::vector<std::thread> threads(tnum);
  int cur = start_row;
  for (int k = 0; k < tnum; ++k) {
    int cnt = base + (k < extra ? 1 : 0);
    threads[k] = std::thread(
        [&](int rbegin, int rcnt) {
          for (int y = rbegin; y < rbegin + rcnt; ++y) {
            for (int x = 0; x < w; ++x) {
              ComputePixel(x, y);
            }
          }
        },
        cur, cnt);
    cur += cnt;
  }
  for (auto& t : threads) {
    t.join();
  }

  std::vector<uint8_t> local_output(output_.begin() + start_row * w * channels_,
                                    output_.begin() + (start_row + row_count) * w * channels_);

  if (rank == 0) {
    output_.resize(h * w * channels_);
  }

  std::vector<int> recv_counts(world_size);
  std::vector<int> displs(world_size);
  if (rank == 0) {
    for (int i = 0; i < world_size; ++i) {
      int rows = base_rows + (i < extra_rows ? 1 : 0);
      recv_counts[i] = rows * w * (int)channels_;
    }
    displs[0] = 0;
    for (int i = 1; i < world_size; ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
  }
  boost::mpi::gatherv(world_, local_output, output_.data(), recv_counts, displs, 0);

  return true;
}