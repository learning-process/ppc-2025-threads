#include "tbb/zaitsev_a_bw_labeling/include/ops_tbb.hpp"

#include <oneapi/tbb/mutex.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/detail/_range_common.h"

using zaitsev_a_labeling_tbb::Labeler;

namespace {

oneapi::tbb::mutex my_mutex;

// NOLINTBEGIN(readability-identifier-naming) - required here to make HardRange class compatible with tbb::parallel_for
class HardRange {
  Length start_;
  Length end_;
  Length chunk_;

 public:
  HardRange(Length end, Length chunk) : start_(0), end_(end), chunk_(chunk) {}
  HardRange(const HardRange& r) = default;
  HardRange(HardRange& r, tbb::detail::split) : start_(r.start_ + r.chunk_), end_(r.end_), chunk_(r.chunk_) {
    r.end_ = std::min(r.start_ + r.chunk_, r.end_);
  }

  [[nodiscard]] bool empty() const { return start_ >= end_; }

  [[nodiscard]] bool is_divisible() const { return end_ > start_ + chunk_; };

  [[nodiscard]] Length begin() const { return start_; }

  [[nodiscard]] Length end() const { return end_; }
};
// NOLINTEND(readability-identifier-naming)

class FirstScan {
  LabelsList& labels_list_;
  Ordinals& ordinals_;
  unsigned int width_;
  long chunk_;

 public:
  FirstScan(LabelsList& labels_list, Equivalencies& eqs, Ordinals& ordinals, Length width, Length chunk)
      : labels_list_(labels_list), ordinals_(ordinals), width_(width), chunk_(chunk) {}

  void operator()(const HardRange& r) const {
    Length id = r.begin() / chunk_;
    DisjointSet dsj;
    auto& ordinal = ordinals_[id];
    auto& labels = labels_list_[id];
    for (Length i = 0; i < r.end() - r.begin(); i++) {
      if (labels[i] == 0) {
        continue;
      }

      std::vector<Ordinal> neighbours;
      for (int shift = 0; shift < 4; shift++) {
        long x = ((long)i % width_) + (shift % 3 - 1);
        long y = ((long)i / width_) + (shift / 3 - 1);
        long neighbour_index = x + (y * width_);
        if (x >= 0 && x < static_cast<long>(width_) && y >= 0 && labels[neighbour_index] != 0) {
          for (const auto& it : neighbours) {
            dsj.UnionRank(it, labels[neighbour_index]);
          }
          neighbours.push_back(labels[neighbour_index]);
        }
      }

      if (neighbours.empty()) {
        labels[i] = ++ordinal;
      } else {
        labels[i] = std::ranges::min(neighbours);
      }
    }

    for (Length i = 0; i < r.end() - r.begin(); i++) {
      labels[i] = dsj.FindParent(labels[i]);
    }
    std::set<Ordinal> unique_labels(labels.begin(), labels.end());
    std::map<Ordinal, Ordinal> replacements;
    ordinal = 0;
    for (const auto& it : unique_labels) {
      replacements[it] = ordinal++;
    }
    for (Length i = 0; i < r.end() - r.begin(); i++) {
      labels[i] = replacements[labels[i]];
    }
  }
};

}  // namespace

bool Labeler::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];
  size_ = height_ * width_;
  image_.resize(size_, 0);
  std::copy(task_data->inputs[0], task_data->inputs[0] + size_, image_.begin());
  return true;
}

bool Labeler::ValidationImpl() {
  return task_data->inputs_count.size() == 2 && (!task_data->inputs.empty()) &&
         (task_data->outputs_count[0] == task_data->inputs_count[0] * task_data->inputs_count[1]);
}

void Labeler::LabelingRasterScan(Equivalencies& eqs, Ordinals& ordinals) {
  LabelsList lbls(ppc::util::GetPPCNumThreads(), Labels(chunk_, 0));
  for (Length i = 0; i < size_; i++) {
    lbls[i / chunk_][i % chunk_] = image_[i];
  }
  oneapi::tbb::parallel_for(HardRange(size_, chunk_), FirstScan(lbls, eqs, ordinals, width_, chunk_));

  for (Length i = 0; i < size_; i++) {
    labels_[i] = lbls[i / chunk_][i % chunk_];
  }
}

void Labeler::UniteChunks() {
  DisjointSet dsj(width_);
  long start_pos = 0;
  long end_pos = width_;
  for (long i = 1; i < ppc::util::GetPPCNumThreads(); i++) {
    start_pos += chunk_;
    end_pos += chunk_;
    for (long pos = start_pos; pos < end_pos; pos++) {
      if (pos >= static_cast<long>(size_) || pos < 0 || labels_[pos] == 0) {
        continue;
      }
      Ordinal lower = labels_[pos];
      for (long shift = -1; shift < 2; shift++) {
        long neighbour_pos = std::clamp(pos + shift, start_pos, end_pos - 1) - width_;
        if (neighbour_pos < 0 || neighbour_pos >= static_cast<long>(size_) || labels_[neighbour_pos] == 0) {
          continue;
        }
        Ordinal upper = labels_[neighbour_pos];
        dsj.UnionRank(upper, lower);
      }
    }
  }

  for (Length i = 0; i < size_; i++) {
    labels_[i] = dsj.FindParent(labels_[i]);
  }
}

void Labeler::CalculateReplacements(Replacements& replacements, Equivalencies& eqs, Ordinals& ordinals) {}

void Labeler::PerformReplacements(Replacements& replacements) {
  for (Length i = 0; i < size_; i++) {
    labels_[i] = replacements[labels_[i]];
  }
}

void Labeler::GlobalizeLabels(Ordinals& ordinals) {
  Length shift = 0;
  for (Length i = chunk_; i < size_; i++) {
    if (i % chunk_ == 0) {
      shift += std::max(ordinals[(i / chunk_) - 1] - 1, 0);
    }
    if (labels_[i] != 0) {
      labels_[i] += shift;
    }
  }
}

bool Labeler::RunImpl() {
  labels_.clear();
  labels_.resize(size_, 0);
  Equivalencies eqs(ppc::util::GetPPCNumThreads());
  Ordinals ordinals(ppc::util::GetPPCNumThreads(), 0);
  chunk_ = static_cast<long>(std::ceil(static_cast<double>(height_) / ppc::util::GetPPCNumThreads())) * width_;

  LabelingRasterScan(eqs, ordinals);
  GlobalizeLabels(ordinals);
  UniteChunks();
  return true;
}

bool Labeler::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<std::uint16_t*>(task_data->outputs[0]);
  std::ranges::copy(labels_, out_ptr);
  return true;
}