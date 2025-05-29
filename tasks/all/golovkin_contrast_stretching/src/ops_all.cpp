// Golovkin Maksims
#include "all/golovkin_contrast_stretching/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

template <typename PixelType>
bool golovkin_contrast_stretching::ContrastStretchingMPI_OMP<PixelType>::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

template <typename PixelType>
bool golovkin_contrast_stretching::ContrastStretchingMPI_OMP<PixelType>::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);

  image_size_ = task_data->inputs_count[0] / sizeof(PixelType);

  if (image_size_ == 0) {
    return true;
  }

  if (rank_ == 0) {
    auto* input_ptr = reinterpret_cast<PixelType*>(task_data->inputs[0]);
    input_image_.assign(input_ptr, input_ptr + image_size_);
  }

  MPI_Bcast(&image_size_, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  input_image_.resize(image_size_);
  output_image_.resize(image_size_);

  MPI_Bcast(input_image_.data(), image_size_, MPI_BYTE, 0, MPI_COMM_WORLD);

  PixelType local_min, local_max;
  auto [min_it, max_it] = std::minmax_element(input_image_.begin(), input_image_.end());
  local_min = *min_it;
  local_max = *max_it;

  MPI_Allreduce(&local_min, &min_val_, 1, MPI_BYTE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &max_val_, 1, MPI_BYTE, MPI_MAX, MPI_COMM_WORLD);

  return true;
}

template <typename PixelType>
bool golovkin_contrast_stretching::ContrastStretchingMPI_OMP<PixelType>::RunImpl() {
  if (image_size_ == 0) {
    return true;
  }
  if (min_val_ == max_val_) {
    std::fill(output_image_.begin(), output_image_.end(), 0);
    return true;
  }

  const double scale = 255.0 / (max_val_ - min_val_);

  size_t chunk_size = image_size_ / num_procs_;
  size_t remainder = image_size_ % num_procs_;

  size_t start = static_cast<size_t>(rank_) * chunk_size + std::min(static_cast<size_t>(rank_), remainder);
  size_t end = start + chunk_size + (static_cast<size_t>(rank_) < remainder ? 1 : 0);

  const int signed_start = static_cast<int>(start);
  const int signed_end = static_cast<int>(end);

#pragma omp parallel for
  for (int i = signed_start; i < signed_end; ++i) {
    double stretched = (input_image_[i] - min_val_) * scale;

    if constexpr (std::is_same_v<PixelType, uint8_t>) {
      output_image_[i] = static_cast<uint8_t>(std::clamp(static_cast<int>(stretched + 1e-9), 0, 255));
    } else if constexpr (std::is_same_v<PixelType, uint16_t>) {
      output_image_[i] = static_cast<uint16_t>(std::clamp(static_cast<int>(stretched + 1e-9), 0, 255));
    } else {
      output_image_[i] = static_cast<PixelType>(stretched);
    }
  }

  if (rank_ == 0) {
    for (int proc = 1; proc < num_procs_; ++proc) {
      size_t proc_start = static_cast<size_t>(proc) * chunk_size + std::min(static_cast<size_t>(proc), remainder);
      size_t proc_end = proc_start + chunk_size + (static_cast<size_t>(proc) < remainder ? 1 : 0);
      MPI_Recv(output_image_.data() + proc_start, proc_end - proc_start, MPI_BYTE, proc, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  } else {
    MPI_Send(output_image_.data() + start, end - start, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
  }

  return true;
}

template <typename PixelType>
bool golovkin_contrast_stretching::ContrastStretchingMPI_OMP<PixelType>::PostProcessingImpl() {
  if (image_size_ == 0) {
    return true;
  }

  if (rank_ == 0) {
    auto* output_ptr = reinterpret_cast<PixelType*>(task_data->outputs[0]);
    std::memcpy(output_ptr, output_image_.data(), output_image_.size() * sizeof(PixelType));
  }

  return true;
}

template class golovkin_contrast_stretching::ContrastStretchingMPI_OMP<uint8_t>;
template class golovkin_contrast_stretching::ContrastStretchingMPI_OMP<uint16_t>;
template class golovkin_contrast_stretching::ContrastStretchingMPI_OMP<float>;