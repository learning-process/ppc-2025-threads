#include "seq/zaitsev_a_bw_labeling/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

bool zaitsev_a_labeling::Labeler::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];
  size_ = height_ * width_;
  image_ = std::vector<int>(size_);
  labels_ = std::vector<int>(size_);
  auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + size_, image_.begin());
  return true;
}

bool zaitsev_a_labeling::Labeler::ValidationImpl() {
  return task_data->inputs_count.size() == 2 && (!task_data->inputs.empty()) &&
         (task_data->outputs_count[0] == task_data->inputs_count[0] * task_data->inputs_count[1]);
}

void zaitsev_a_labeling::Labeler::ComputeLabel(unsigned int i, int& current_label) {
  if (image_[i] == 0) {
    labels_[i] = 0;
    return;
  }

  int top = (i % width_ > 0) ? labels_[i - 1] : 0;
  int left = (i >= width_) ? labels_[i - width_] : 0;

  if (top == 0 && left == 0) {
    labels_[i] = ++current_label;
  } else if (top == 0 && left != 0) {
    labels_[i] = left;
  } else if ((top != 0 && left == 0) || top == left) {
    labels_[i] = top;
  } else {
    labels_[i] = std::min(top, left);
    equivalencies_[std::max(top, left)] = std::min(top, left);
  }
}

void zaitsev_a_labeling::Labeler::LabelingRasterScan() {
  int current_label = 0;
  for (unsigned int i = 0; i < size_; i++) {
    ComputeLabel(i, current_label);
  }
}

void zaitsev_a_labeling::Labeler::EquivReplaceRasterScan() {
  for (unsigned int i = 0; i < size_; i++) {
    auto replacement = equivalencies_.find(labels_[i]);
    do {
      if (replacement != equivalencies_.end()) {
        labels_[i] = replacement->second;
      }
      replacement = equivalencies_.find(labels_[i]);
    } while (replacement != equivalencies_.end());
  }
}

bool zaitsev_a_labeling::Labeler::RunImpl() {
  LabelingRasterScan();
  EquivReplaceRasterScan();
  return true;
}

bool zaitsev_a_labeling::Labeler::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(labels_, out_ptr);
  return true;
}
