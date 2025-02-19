#include "seq/zaitsev_a_bw_labeling/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

bool zaitsev_a_labeling::Labeler::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];
  size_ = height_ * width_;
  current_label_ = 0;
  equivalences_.push_back(0);
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

void zaitsev_a_labeling::Labeler::ComputeLabel(unsigned int i) {
  if (image_[i] == 0) {
    labels_[i] = 0;
    return;
  }

  int top = (i % width_ > 0) ? labels_[i - 1] : 0;
  int left = (i >= width_) ? labels_[i - width_] : 0;

  if (top == 0 && left == 0) {
    labels_[i] = ++current_label_;
    equivalences_.push_back(current_label_);
  } else if (top == 0 && left != 0) {
    labels_[i] = left;
  } else if ((top != 0 && left == 0) || top == left) {
    labels_[i] = top;
  } else {
    labels_[i] = std::min(top, left);
    equivalences_[std::max(top, left)] = equivalences_[std::min(top, left)];
  }
}

void zaitsev_a_labeling::Labeler::LabelingRasterScan() {
  current_label_ = 0;
  for (unsigned int i = 0; i < size_; i++) {
    ComputeLabel(i);
  }
}

void zaitsev_a_labeling::Labeler::PrepareReplacements() {
  if (equivalences_.empty()) {
    return;
  }
  int label = -1;
  replacements_ = std::vector<int>(equivalences_.size(), -1);
  for (unsigned int i = 0; i < equivalences_.size(); i++) {
    if (replacements_[equivalences_[i]] == -1) {
      replacements_[i] = ++label;
    } else {
      replacements_[i] = replacements_[equivalences_[i]];
    }
  }
}

void zaitsev_a_labeling::Labeler::PerformReplacements() {
  for (int i = 0; i < (int)size_; i++) {
    labels_[i] = replacements_[labels_[i]];
  }
}

bool zaitsev_a_labeling::Labeler::RunImpl() {
  LabelingRasterScan();
  PrepareReplacements();
  PerformReplacements();
  return true;
}

bool zaitsev_a_labeling::Labeler::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(labels_, out_ptr);
  return true;
}
