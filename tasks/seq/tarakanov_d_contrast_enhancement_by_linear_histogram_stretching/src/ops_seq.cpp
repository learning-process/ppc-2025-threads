#include "seq/tarakanov_d_contrast_enhancement_by_linear_histogram_stretching/include/ops_seq.hpp"

#include <cmath>
#include <cstdint>
#include <vector>
#include <cstring>

bool tarakanov_d_linear_stretching::TaskSequential::PreProcessingImpl() {
    unsigned int input_size = task_data->inputs_count[0];
    auto *in_ptr = reinterpret_cast<uchar*>(task_data->inputs[0]);

    rc_size_ = static_cast<int>(std::sqrt(input_size));

    inputImage_.resize(input_size);
    std::memcpy(inputImage_.data(), in_ptr, input_size * sizeof(uchar));

    outputImage_.resize(input_size, 0);

    return true;
}

bool tarakanov_d_linear_stretching::TaskSequential::ValidationImpl() {
    return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool tarakanov_d_linear_stretching::TaskSequential::RunImpl() {
    uchar minVal = 255;
    uchar maxVal = 0;
    size_t total_pixels = inputImage_.size();

    for (size_t idx = 0; idx < total_pixels; ++idx) {
        uchar pixel = inputImage_[idx];
        if (pixel < minVal) {
            minVal = pixel;
        }
        if (pixel > maxVal) {
            maxVal = pixel;
        }
    }

    if (minVal == maxVal) {
        outputImage_ = inputImage_;
        return true;
    }

    for (size_t idx = 0; idx < total_pixels; ++idx) {
        uchar pixel = inputImage_[idx];
        uchar newPixel = static_cast<uchar>((pixel - minVal) * 255.0 / (maxVal - minVal));
        outputImage_[idx] = newPixel;
    }

    return true;
}

bool tarakanov_d_linear_stretching::TaskSequential::PostProcessingImpl() {
    size_t total_elements = outputImage_.size();
    auto *out_ptr = reinterpret_cast<uchar*>(task_data->outputs[0]);

    std::memcpy(out_ptr, outputImage_.data(), total_elements * sizeof(uchar));

    return true;
}
