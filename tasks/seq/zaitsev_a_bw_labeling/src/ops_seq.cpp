#include "seq/zaitsev_a_bw_labeling/include/ops_seq.hpp"

bool zaitsev_a_labeling::Labeler::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];                                            // width of input image
  height_ = task_data->inputs_count[1];                                           // height of input image
  image_ = std::vector<std::vector<int>>(height_, std::vector<int>(width_));      // preparing image_
  labels_ = std::vector<std::vector<int>>(height_, std::vector<int>(width_, 0));  // preparing labels_
  auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  for (unsigned int j = 0; j < height_; j++) {
    std::copy(tmp_ptr, tmp_ptr + width_, image_[j].begin());
    tmp_ptr += width_;
  }
  return true;
}

bool zaitsev_a_labeling::Labeler::ValidationImpl() {
  return task_data->inputs_count.size() == 2 && task_data->inputs.size() > 0;  // && task_data->outputs_count[0] == 1;
}

void zaitsev_a_labeling::Labeler::ComputeLabel(unsigned int i, unsigned int j, int& current_label) {
  if (image_[j][i] == 0) {
    labels_[j][i] = 0;
  } else {
    int top = (i > 0) ? labels_[j][i - 1] : 0;
    int left = (j > 0) ? labels_[j - 1][i] : 0;
    if (top == 0 && left == 0) {
      labels_[j][i] = ++current_label;
    } else if (top == 0 && left != 0) {
      labels_[j][i] = left;
    } else if ((top != 0 && left == 0) || top == left) {
      labels_[j][i] = top;
    } else {
      labels_[j][i] = std::min(top, left);
      equivalencies_[std::max(top, left)] = std::min(top, left);
    }
  }
}

void zaitsev_a_labeling::Labeler::LabelingRasterScan() {
  int current_label = 0;
  for (unsigned int j = 0; j < height_; j++) {
    for (unsigned int i = 0; i < width_; i++) {
      ComputeLabel(i, j, current_label);
    }
  }
}

void zaitsev_a_labeling::Labeler::EquivReplaceRasterScan() {
  for (unsigned int j = 0; j < height_; j++) {
    for (unsigned int i = 0; i < width_; i++) {
      // auto replacement = equivalencies_.find(labels_[j][i]);
      // if (replacement != equivalencies_.end()) {
      //   labels_[j][i] = replacement->second;
      // }
      auto replacement1 = equivalencies_.find(labels_[j][i]);
      do {
        if (replacement1 != equivalencies_.end()) {
          labels_[j][i] = replacement1->second;
        }
        replacement1 = equivalencies_.find(labels_[j][i]);
      } while (replacement1 != equivalencies_.end());
    }
  }
}

bool zaitsev_a_labeling::Labeler::RunImpl() {
  LabelingRasterScan();
  EquivReplaceRasterScan();
  return true;
}

bool zaitsev_a_labeling::Labeler::PostProcessingImpl() {
  std::vector<int> out(width_ * height_);
  for (unsigned int j = 0; j < height_; j++) {
    for (unsigned int i = 0; i < width_; i++) {
      out[j * width_ + i] = labels_[j][i];
    }
  }
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(out.begin(), out.end(), out_ptr);
  return true;
}
