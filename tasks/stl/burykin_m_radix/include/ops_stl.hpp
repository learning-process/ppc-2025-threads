#pragma once

#include <array>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace burykin_m_radix_stl {

class RadixSTL : public ppc::core::Task {
 public:
  explicit RadixSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::array<int, 256> ComputeFrequency(const std::vector<int>& a, int shift);
  static std::array<int, 256> ComputeIndices(const std::array<int, 256>& count);
  static void DistributeElements(const std::vector<int>& a, std::vector<int>& b, std::array<int, 256> index, int shift);

 private:
  std::vector<int> input_, output_;
};

void ComputeFrequencyParallel(const std::vector<int>& a, int shift, std::array<int, 256>& count, int start_idx,
                              int end_idx);
void DistributeElementsParallel(const std::vector<int>& a, std::vector<int>& b,
                                const std::array<int, 256>& global_index,
                                std::vector<std::array<int, 256>>& local_counts, int shift, int thread_id,
                                int start_idx, int end_idx);

}  // namespace burykin_m_radix_stl