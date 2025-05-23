// Golovkin Maksim
#include "seq/golovkin_contrast_stretching/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

template <typename PixelType>
bool golovkin_contrast_stretching::ContrastStretchingSeq<PixelType>::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

template <typename PixelType>
bool golovkin_contrast_stretching::ContrastStretchingSeq<PixelType>::PreProcessingImpl() {
  image_size_ = task_data->inputs_count[0] / sizeof(PixelType);

  if (image_size_ == 0) {
    return true;
  }
  auto* input_ptr = reinterpret_cast<PixelType*>(task_data->inputs[0]);
  input_image_.assign(input_ptr, input_ptr + image_size_);
  output_image_.resize(image_size_);

  auto [min_it, max_it] = std::minmax_element(input_image_.begin(), input_image_.end());
  min_val_ = *min_it;
  max_val_ = *max_it;

  return true;
}

template <typename PixelType>
bool golovkin_contrast_stretching::ContrastStretchingSeq<PixelType>::RunImpl() {
  if (image_size_ == 0) return true;
  if (min_val_ == max_val_) {
    std::fill(output_image_.begin(), output_image_.end(), 0);
    return true;
  }

  const double scale = 255.0 / (max_val_ - min_val_);

  for (size_t i = 0; i < image_size_; ++i) {
    double stretched = (input_image_[i] - min_val_) * scale;

    if constexpr (std::is_same_v<PixelType, uint8_t>) {
      output_image_[i] = static_cast<uint8_t>(std::clamp(static_cast<int>(stretched + 1e-9), 0, 255));
    } else if constexpr (std::is_same_v<PixelType, uint16_t>) {
      output_image_[i] = static_cast<uint16_t>(std::clamp(static_cast<int>(stretched + 1e-9), 0, 255));
    } else {
      output_image_[i] = static_cast<PixelType>(stretched);
    }
  }

  return true;
}

template <typename PixelType>
bool golovkin_contrast_stretching::ContrastStretchingSeq<PixelType>::PostProcessingImpl() {
  if (image_size_ == 0) {
    return true;
  }

  auto* output_ptr = reinterpret_cast<PixelType*>(task_data->outputs[0]);
  std::memcpy(output_ptr, output_image_.data(), output_image_.size() * sizeof(PixelType));
  return true;
}

template class golovkin_contrast_stretching::ContrastStretchingSeq<uint8_t>;
template class golovkin_contrast_stretching::ContrastStretchingSeq<uint16_t>;
template class golovkin_contrast_stretching::ContrastStretchingSeq<float>;