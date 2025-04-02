#include "seq/khovansky_d_double_radix_batcher/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace khovansky_d_double_radix_batcher_seq {
namespace {
  uint64_t EncodeDoubleToUint64(double value) {
    uint64_t bitRepresentation = 0;
    std::memcpy(&bitRepresentation, &value, sizeof(value));

    if ((bitRepresentation >> 63) != 0) {
        return ~bitRepresentation;
    }
    return bitRepresentation ^ (1ULL << 63);
}

double DecodeUint64ToDouble(uint64_t encoded) {
    if ((encoded >> 63) != 0) {
        encoded ^= (1ULL << 63);
    } else {
        encoded = ~encoded;
    }

    double result = 0.0;
    std::memcpy(&result, &encoded, sizeof(result));
    return result;
}

void RadixSort(std::vector<uint64_t>& array) {
    const int BITS_IN_BYTE = 8;
    const int TOTAL_BITS = 64;
    const int BUCKET_COUNT = 256;

    std::vector<uint64_t> buffer(array.size(), 0);
    std::vector<int> frequency(BUCKET_COUNT, 0);

    for (int shift = 0; shift < TOTAL_BITS; shift += BITS_IN_BYTE) {
        std::fill(frequency.begin(), frequency.end(), 0);

        for (uint64_t number : array) {
            uint8_t bucket = static_cast<uint8_t>((number >> shift) & 0xFF);
            frequency[bucket]++;
        }

        for (int i = 1; i < BUCKET_COUNT; i++) {
            frequency[i] += frequency[i - 1];
        }

        for (int i = static_cast<int>(array.size()) - 1; i >= 0; i--) {
            uint8_t bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
            buffer[--frequency[bucket]] = array[i];
        }

        array.swap(buffer);
    }
}

void OddEvenMergeSort(std::vector<uint64_t>& array, int left, int right) {
    if (right - left <= 1) {
        return;
    }

    int middle = left + (right - left) / 2;

    OddEvenMergeSort(array, left, middle);
    OddEvenMergeSort(array, middle, right);

    for (int i = left; i + 1 < right; i += 2) {
        if (array[i] > array[i + 1]) {
            std::swap(array[i], array[i + 1]);
        }
    }
}

void RadixBatcherSort(std::vector<double>& data) {
    std::vector<uint64_t> TransformedData(data.size(), 0);

    for (std::size_t i = 0; i < data.size(); i++) {
        TransformedData[i] = EncodeDoubleToUint64(data[i]);
    }

    RadixSort(TransformedData);
    OddEvenMergeSort(TransformedData, 0, static_cast<int>(TransformedData.size()));

    for (std::size_t i = 0; i < data.size(); i++) {
        data[i] = DecodeUint64ToDouble(TransformedData[i]);
    }
}
}  // namespace
}  // namespace khovansky_d_double_radix_batcher_seq

bool khovansky_d_double_radix_batcher_seq::RadixSeq::PreProcessingImpl() {
  if (!task_data) {
    return false;
  }

  if (task_data->inputs[0] == nullptr && task_data->inputs_count[0] == 0) {
    return false;
  }

  auto* in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);

  unsigned int input_size = task_data->inputs_count[0];
  unsigned int output_size = task_data->outputs_count[0];

  input_ = std::vector<double>(in_ptr, in_ptr + input_size);
  output_ = std::vector<double>(output_size, 0);

  return true;
}

bool khovansky_d_double_radix_batcher_seq::RadixSeq::ValidationImpl() {
  if (!task_data) {
    return false;
  }

  if (task_data->inputs[0] == nullptr && task_data->inputs_count[0] == 0) {
    return false;
  }

  if (task_data->outputs[0] == nullptr) {
    return false;
  }

  if (task_data->inputs_count[0] < 2) {
    return false;
  }

  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool khovansky_d_double_radix_batcher_seq::RadixSeq::RunImpl() {
  output_ = input_;
  khovansky_d_double_radix_batcher_seq::RadixBatcherSort(output_);
  return true;
}

bool khovansky_d_double_radix_batcher_seq::RadixSeq::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
  }

  return true;
}