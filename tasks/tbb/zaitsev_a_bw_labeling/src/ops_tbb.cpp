#include "tbb/zaitsev_a_bw_labeling/include/ops_tbb.hpp"

#include <oneapi/tbb/mutex.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/detail/_range_common.h"
#include "oneapi/tbb/task_arena.h"

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

Length GetChunk(Length width, Length height, int n_threads) {
  return static_cast<Length>(std::ceil(static_cast<double>(height) / n_threads)) * width;
}

class FirstScan {
  Image& image_;
  Labels& labels_;
  unsigned int width_;
  unsigned int height_;

  [[nodiscard]] bool IsPointValid(long x, long y, const HardRange& r) const {
    return 0 <= x && x < width_ && y >= 0 && (y * width_) + x >= r.begin() && (y * width_) + x < r.end();
  }

  void GetNeighbours(Ordinals& neighbours, Length pos, const HardRange& r) const {
    std::vector<std::pair<long, long>> shifts = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0}};
    for (const auto& shift : shifts) {
      long x = (pos % width_) + shift.first;
      long y = (pos / width_) + shift.second;
      if (IsPointValid(x, y, r) && labels_[(y * width_) + x] != 0) {
        neighbours.push_back(labels_[(y * width_) + x]);
      }
    }
  }

 public:
  FirstScan(Image& image, Labels& labels, Length width, Length height)
      : image_(image), labels_(labels), width_(width), height_(height) {}

  void operator()(const HardRange& r) const {
    //   (void)image_;
    //   (void)height_;
    int n_threads = oneapi::tbb::this_task_arena::max_concurrency();
    Ordinal ordinal = 0;
    long chunk = GetChunk(width_, height_, n_threads);

    // (void)image_;
    // (void)height_;
    // (void)ordinal;
    // (void)chunk;

    DisjointSet dsj(chunk);
    for (Length i = r.begin(); i < r.end(); i++) {
      if (image_[i] == 0) {
        continue;
      }

      std::vector<Ordinal> neighbours;
      GetNeighbours(neighbours, i, r);

      if (neighbours.empty()) {
        labels_[i] = ++ordinal;
      } else {
        labels_[i] = std::ranges::min(neighbours);
        std::ranges::for_each(neighbours, [&](Ordinal& x) { dsj.UnionRank(labels_[i], x); });
      }
    }

    for (Length i = r.begin(); i < r.end(); i++) {
      labels_[i] = dsj.FindParent(labels_[i]);
    }

    std::set<Ordinal> unique_labels(labels_.begin() + r.begin(), labels_.begin() + r.end());
    std::map<Ordinal, Ordinal> replacements;
    Ordinal true_label = 0;
    for (const auto& it : unique_labels) {
      replacements[it] = true_label++;
      if (r.begin() == 0) {
        std::cout << std::format("\n{} {}", it, true_label);
      }
    }

    for (Length i = r.begin(); i < r.end(); i++) {
      labels_[i] = replacements[labels_[i]];
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

void Labeler::LabelingRasterScan() {
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute(
      [&] { oneapi::tbb::parallel_for(HardRange(size_, chunk_), FirstScan(image_, labels_, width_, height_)); });
}

void Labeler::UniteChunks() {
  DisjointSet dsj(width_);

  long start_pos = 0;
  long end_pos = width_;
  for (long i = 0; i < (long)std::ceil(((double)height_ * width_) / (chunk_)) - 1; i++) {
    start_pos += chunk_;
    end_pos += chunk_;
    for (long pos = start_pos; pos < end_pos; pos++) {
      if (labels_[pos] == 0) {
        continue;
      }
      Ordinal lower = labels_[pos];
      for (long shift = -1; shift < 2; shift++) {
        if ((pos == start_pos && shift == -1) || (pos == end_pos - 1 && shift == 1)) {
          continue;
        }
        long neighbour_pos = std::clamp(pos + shift, start_pos, end_pos - 1) - width_;
        if (neighbour_pos < 0 || neighbour_pos >= static_cast<long>(size_) || labels_[neighbour_pos] == 0) {
          continue;
        }
        Ordinal upper = labels_[neighbour_pos];
        dsj.UnionRank(lower, upper);
      }
    }
  }

  for (Length i = 0; i < size_; i++) {
    labels_[i] = dsj.FindParent(labels_[i]);
  }
}

void Labeler::GlobalizeLabels(Ordinals& ordinals) {
  Length shift = 0;
  for (Length i = chunk_; i < size_; i++) {
    if (i % chunk_ == 0) {
      shift += ordinals[(i / chunk_) - 1];
    }
    if (labels_[i] != 0) {
      labels_[i] += shift;
    }
  }
}

bool Labeler::RunImpl() {
  chunk_ = GetChunk(width_, height_, ppc::util::GetPPCNumThreads());
  labels_.clear();
  labels_.resize(size_, 0);
  Ordinals ordinals(ppc::util::GetPPCNumThreads(), 0);

  LabelingRasterScan();
  // std::cout << std::endl << "After FirstScan prallel" << std::endl;
  // for (int i = 0; i < (int)labels_.size(); i++) {
  //   if (i % chunk_ == 0) {
  //     std::cout << std::endl;
  //   }
  //   if (i != 0 && i % width_ == 0) {
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::setw(3) << labels_[i];
  // }

  Length begin = 0;
  Length end = chunk_;
  while (end < size_) {
    ordinals[begin / chunk_] =
        *std::ranges::max_element(labels_.begin() + begin, labels_.begin() + std::min(end, size_));
    begin += chunk_;
    end += chunk_;
  }

  // Length begin = 0;
  // Length end = begin + chunk_ - 1;
  // for (int i = 0; i < ; i++) {
  //   ordinals[i] = *std::ranges::max_element(labels_.begin() + begin, labels_.begin() + end);
  //   begin += chunk_;
  //   end = std::min(begin + chunk_, size_ - 1);
  // }

  // std::cout << std::endl << "[INFO] Ordinals: ";
  // for (const auto& it : ordinals) {
  //   std::cout << (int)it << " ";
  // }

  GlobalizeLabels(ordinals);

  // std::cout << std::endl << std::endl << "After GlobalizeLabels" << std::endl;
  // for (int i = 0; i < (int)labels_.size(); i++) {
  //   if (i % chunk_ == 0) {
  //     std::cout << std::endl;
  //   }
  //   if (i != 0 && i % width_ == 0) {
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::setw(3) << labels_[i];
  // }

  UniteChunks();

  // std::cout << std::endl << "After UniteChunks" << std::endl;
  // for (int i = 0; i < (int)labels_.size(); i++) {
  //   if (i % chunk_ == 0) {
  //     std::cout << std::endl;
  //   }
  //   if (i != 0 && i % width_ == 0) {
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::setw(3) << labels_[i];
  // }

  std::set<Ordinal> unique_labels(labels_.begin(), labels_.end());
  std::map<Ordinal, Ordinal> replacements;
  Ordinal ordinal = 0;
  for (const auto& it : unique_labels) {
    replacements[it] = ordinal++;
  }
  for (Length i = 0; i < size_; i++) {
    labels_[i] = replacements[labels_[i]];
  }

  // std::cout << std::endl << "After Sequenize" << std::endl;
  // for (int i = 0; i < (int)labels_.size(); i++) {
  //   if (i % chunk_ == 0) {
  //     std::cout << std::endl;
  //   }
  //   if (i != 0 && i % width_ == 0) {
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::setw(3) << labels_[i];
  // }
  return true;
}

bool Labeler::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<std::uint16_t*>(task_data->outputs[0]);
  std::ranges::copy(labels_, out_ptr);
  return true;
}