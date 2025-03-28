#include "omp/zaitsev_a_bw_labeling/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <vector>

#include "core/util/include/util.hpp"
#include "omp/zaitsev_a_bw_labeling/include/disjoint_set.hpp"

#ifndef _WIN32
// NOLINTBEGIN(modernize-use-ranges)
#pragma omp declare reduction(merge : std::vector<std::uint16_t> : std::transform( \
        omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(),           \
            [](const auto& x, const auto& y){return std::max(x, y);}))             \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
// NOLINTEND(modernize-use-ranges)
#endif

bool zaitsev_a_labeling_omp::Labeler::PreProcessingImpl() {
  width_ = task_data->inputs_count[0];
  height_ = task_data->inputs_count[1];
  size_ = height_ * width_;
  image_.resize(size_, 0);
  std::copy(task_data->inputs[0], task_data->inputs[0] + size_, image_.begin());
  return true;
}

bool zaitsev_a_labeling_omp::Labeler::ValidationImpl() {
  return task_data->inputs_count.size() == 2 && (!task_data->inputs.empty()) &&
         (task_data->outputs_count[0] == task_data->inputs_count[0] * task_data->inputs_count[1]);
}

void zaitsev_a_labeling_omp::Labeler::LabelingRasterScan(
    std::vector<std::map<std::uint16_t, std::set<std::uint16_t>>>& eqs, std::vector<std::uint16_t>& current_label) {
#ifndef _WIN32
#pragma omp parallel if (static_cast<int>(height_) >= 2 * ppc::util::GetPPCNumThreads()) reduction(merge : labels_)
#else
#pragma omp parallel if (static_cast<int>(height_) >= 2 * ppc::util::GetPPCNumThreads())
#endif
  {
    std::uint8_t id = omp_get_thread_num();
    std::uint32_t i_begin = id * chunk_;
    std::uint32_t i_end = i_begin + chunk_;
    i_end = std::min(i_end, size_);

    for (uint32_t i = i_begin; i != i_end; i++) {
      if (image_[i] == 0) {
        continue;
      }

      std::vector<std::uint16_t> neighbours;
      neighbours.reserve(4);

      for (int shift = 0; shift < 4; shift++) {
        long x = ((long)i % width_) + (shift % 3 - 1);
        long y = ((long)i / width_) + (shift / 3 - 1);
        long neighbour_index = x + (y * width_);
        std::uint16_t value = 0;
        if (x >= 0 && x < (long)width_ && y >= 0) {
          value = labels_[neighbour_index];
        }
        if (value != 0) {
          neighbours.push_back(value);
        }
      }

      if (neighbours.empty()) {
        labels_[i] = ++current_label[id];
        eqs[id][current_label[id]].insert(current_label[id]);
      } else {
        labels_[i] = *std::ranges::min_element(neighbours);
        for (auto& first : neighbours) {
          for (auto& second : neighbours) {
            eqs[id][first].insert(second);
          }
        }
      }
    }
  }
}

void zaitsev_a_labeling_omp::Labeler::UniteChunks(zaitsev_a_disjoint_set::DisjointSet<uint16_t>& dsj,
                                                  std::vector<uint16_t>& current_label) {
  long start_pos = 0;
  long end_pos = width_;
  for (long i = 1; i < ppc::util::GetPPCNumThreads(); i++) {
    start_pos += chunk_;
    end_pos += chunk_;
    for (long pos = start_pos; pos < end_pos; pos++) {  // +
      if (pos >= static_cast<long>(size_) || pos < 0 || labels_[pos] == 0) {
        continue;
      }
      uint16_t lower = labels_[pos];
      for (long shift = -1; shift != 2; shift++) {
        long neighbour_pos = std::clamp(pos + shift, start_pos, end_pos - 1) - width_;
        if (neighbour_pos < 0 || neighbour_pos >= static_cast<long>(size_) || labels_[neighbour_pos] == 0) {
          continue;
        }
        uint16_t upper = labels_[neighbour_pos];
        dsj.UnionRank(upper, lower);
      }
    }
  }
}

void zaitsev_a_labeling_omp::Labeler::CalculateReplacements(
    std::vector<std::uint16_t>& replacements, std::vector<std::map<std::uint16_t, std::set<std::uint16_t>>>& eqs,
    std::vector<std::uint16_t>& current_label) {
  uint16_t labels_amount = std::reduce(current_label.begin(), current_label.end(), 0);
  std::uint16_t shift = 0;

  zaitsev_a_disjoint_set::DisjointSet<std::uint16_t> disjoint_labels(labels_amount + 1);

  for (int i = 0; i != ppc::util::GetPPCNumThreads(); i++) {
    for (auto& eqv : eqs[i]) {
      for (const auto& equal : eqv.second) {
        disjoint_labels.UnionRank(eqv.first + shift, equal + shift);
      }
    }
    shift += current_label[i];
  }

  UniteChunks(disjoint_labels, current_label);

  replacements.resize(labels_amount + 1);
  std::set<std::uint16_t> unique_labels;

  for (std::uint16_t tmp_label = 1; tmp_label != labels_amount + 1; tmp_label++) {
    replacements[tmp_label] = disjoint_labels.FindParent(tmp_label);
    unique_labels.insert(replacements[tmp_label]);
  }

  std::uint16_t true_label = 0;
  std::map<std::uint16_t, std::uint16_t> reps;
  for (const auto& it : unique_labels) {
    reps[it] = ++true_label;
  }

  for (uint32_t i = 0; i < replacements.size(); i++) {
    replacements[i] = reps[replacements[i]];
  }
}

void zaitsev_a_labeling_omp::Labeler::PerformReplacements(std::vector<std::uint16_t>& replacements) {
  for (uint32_t i = 0; i != size_; i++) {
    labels_[i] = replacements[labels_[i]];
  }
}

void zaitsev_a_labeling_omp::Labeler::GlobalizeLabels(std::vector<std::uint16_t>& current_label) {
  uint16_t shift = 0;
  for (uint32_t i = chunk_; i != size_; i++) {
    if (i % chunk_ == 0) {
      shift += current_label[(i / chunk_) - 1];
    }
    if (labels_[i] != 0) {
      labels_[i] += shift;
    }
  }
}

bool zaitsev_a_labeling_omp::Labeler::RunImpl() {
  labels_.clear();
  labels_.resize(size_);
  std::vector<std::map<std::uint16_t, std::set<std::uint16_t>>> eqs(ppc::util::GetPPCNumThreads());
  std::vector<std::uint16_t> current_label(ppc::util::GetPPCNumThreads(), 0);
  std::vector<std::uint16_t> replacements;

  chunk_ = static_cast<long>(std::ceil(static_cast<double>(height_) / ppc::util::GetPPCNumThreads())) * width_;

  LabelingRasterScan(eqs, current_label);
  GlobalizeLabels(current_label);

  CalculateReplacements(replacements, eqs, current_label);
  PerformReplacements(replacements);
  return true;
}

bool zaitsev_a_labeling_omp::Labeler::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<std::uint16_t*>(task_data->outputs[0]);
  std::ranges::copy(labels_, out_ptr);
  return true;
}