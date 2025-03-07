#include "seq/zaitsev_a_bw_labeling/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "seq/zaitsev_a_bw_labeling/include/disjoint_set.hpp"

bool zaitsev_a_labeling::Labeler::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];
  size_ = height_ * width_;
  current_label_ = 0;
  image_.resize(size_, 0);
  labels_.resize(size_, 0);
  std::copy(task_data->inputs[0], task_data->inputs[0] + size_, image_.begin());
  return true;
}

bool zaitsev_a_labeling::Labeler::ValidationImpl() {
  return task_data->inputs_count.size() == 2 && (!task_data->inputs.empty()) &&
         (task_data->outputs_count[0] == task_data->inputs_count[0] * task_data->inputs_count[1]);
}

void zaitsev_a_labeling::Labeler::ComputeLabel(unsigned int i) {
  if (image_[i] == 0) {
    return;
  }
  std::vector<uint16_t> neighbours;
  neighbours.reserve(4);

  for (int shift = 0; shift < 4; shift++) {
    long x = ((long)i % width_) + (shift % 3 - 1);
    long y = ((long)i / width_) + (shift / 3 - 1);
    long neighbour_index = x + y * width_;
    uint16_t value = 0;
    if (x >= 0 && x < width_ && y >= 0) {
      value = labels_[neighbour_index];
    }
    if (value != 0) {
      neighbours.push_back(value);
    }
  }

  if (neighbours.empty()) {
    labels_[i] = ++current_label_;
    eqs_[current_label_].insert(current_label_);
  } else {
    labels_[i] = *std::min(neighbours.begin(), neighbours.end());
    for (auto& first : neighbours) {
      for (auto& second : neighbours) {
        eqs_[first].insert(second);
      }
    }
  }
}

void zaitsev_a_labeling::Labeler::LabelingRasterScan() {
  for (uint32_t i = 0; i < image_.size(); i++) {
    ComputeLabel(i);
  }
}

void zaitsev_a_labeling::Labeler::CalculateReplacements() {
  zaitsev_a_disjoint_set::DisjointSet<uint16_t> disjoint_labels(current_label_);
  for (auto& statement : eqs_) {
    for (const auto& equal : statement.second) {
      disjoint_labels.UnionRank(statement.first, equal);
    }
  }

  replacements_.resize(current_label_ + 1);
  std::set<uint16_t> unique_labels;

  for (uint16_t tmp_label = 1; tmp_label < current_label_ + 1; tmp_label++) {
    replacements_[tmp_label] = disjoint_labels.FindParent(tmp_label);
    unique_labels.insert(replacements_[tmp_label]);
  }

  uint16_t true_label = 0;
  std::map<uint16_t, uint16_t> reps;
  for (const auto& it : unique_labels) {
    reps[it] = ++true_label;
  }

  for (uint32_t i = 0; i < replacements_.size(); i++) {
    replacements_[i] = reps[replacements_[i]];
  }
}

void zaitsev_a_labeling::Labeler::PerformReplacements() {
  for (uint32_t i = 0; i < size_; i++) {
    labels_[i] = replacements_[labels_[i]];
  }
}

bool zaitsev_a_labeling::Labeler::RunImpl() {
  LabelingRasterScan();
  CalculateReplacements();
  PerformReplacements();
  return true;
}

bool zaitsev_a_labeling::Labeler::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<uint16_t*>(task_data->outputs[0]);
  std::ranges::copy(labels_, out_ptr);
  return true;
}
