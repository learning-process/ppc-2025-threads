#include "seq/tarakanov_d_contrast_enhancement_by_linear_histogram_stretching/include/ops_seq.hpp"

#include <cmath>
#include <cstdint>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif

bool tarakanov_d_linear_stretching::TaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<uchar *>(task_data->inputs[0]);

  rc_size_ = static_cast<int>(std::sqrt(input_size));

  inputImage_ = cv::Mat(rc_size_, rc_size_, CV_8UC1, in_ptr).clone();

  outputImage_ = cv::Mat::zeros(rc_size_, rc_size_, CV_8UC1);

  return true;
}

bool tarakanov_d_linear_stretching::TaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool tarakanov_d_linear_stretching::TaskSequential::RunImpl() {
  uchar minVal = 255;
  uchar maxVal = 0;

  for (int i = 0; i < inputImage_.rows; ++i) {
    for (int j = 0; j < inputImage_.cols; ++j) {
      uchar pixel = inputImage_.at<uchar>(i, j);
      if (pixel < minVal) minVal = pixel;
      if (pixel > maxVal) maxVal = pixel;
    }
  }

  if (minVal == maxVal) {
    outputImage_ = inputImage_.clone();
    return true;
  }

  for (int i = 0; i < inputImage_.rows; ++i) {
    for (int j = 0; j < inputImage_.cols; ++j) {
      uchar pixel = inputImage_.at<uchar>(i, j);
      uchar newPixel = static_cast<uchar>((pixel - minVal) * 255.0 / (maxVal - minVal));
      outputImage_.at<uchar>(i, j) = newPixel;
    }
  }
  return true;
}

bool tarakanov_d_linear_stretching::TaskSequential::PostProcessingImpl() {
  size_t total_elements = outputImage_.rows * outputImage_.cols;
  auto *out_ptr = reinterpret_cast<uchar *>(task_data->outputs[0]);

  std::memcpy(out_ptr, outputImage_.data, total_elements * sizeof(uchar));

  return true;
}
