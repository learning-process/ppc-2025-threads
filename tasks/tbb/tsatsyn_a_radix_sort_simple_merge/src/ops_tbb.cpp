#include "tbb/tsatsyn_a_radix_sort_simple_merge/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "oneapi/tbb/concurrent_vector.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"




namespace {
// 1. Разделение данных на положительные/отрицательные (параллельно)
void SplitData(tbb::concurrent_vector<uint64_t>& positive, tbb::concurrent_vector<uint64_t>& negative, std::vector<double> input_data_) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, input_data_.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      double num = input_data_[i];
      uint64_t bits;
      ::memcpy(&bits, &num, sizeof(double));  // Используем memcpy вместо bit_cast для совместимости
      if (num >= 0) {
        positive.push_back(bits);
      } else {
        negative.push_back(bits);
      }
    }
  });
}

// Исправленный метод RadixSort с инверсией битов для отрицательных чисел
void RadixSort(tbb::concurrent_vector<uint64_t>& data, bool invert_order) {
  for (int bit = 0; bit < 64; ++bit) {
    tbb::enumerable_thread_specific<std::vector<uint64_t>> local_group0;
    tbb::enumerable_thread_specific<std::vector<uint64_t>> local_group1;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, data.size()), [&](const tbb::blocked_range<size_t>& r) {
      auto& g0 = local_group0.local();
      auto& g1 = local_group1.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        // Инвертируем биты для отрицательных чисел
        uint64_t current_bit = (invert_order) ? ~data[i] : data[i];
        if (((current_bit >> bit) & 1) != 0) {
          g1.push_back(data[i]);
        } else {
          g0.push_back(data[i]);
        }
      }
    });

    // Сборка групп
    std::vector<uint64_t> group0, group1;
    for (const auto& vec : local_group0) {
      group0.insert(group0.end(), vec.begin(), vec.end());
    }
    for (const auto& vec : local_group1) {
      group1.insert(group1.end(), vec.begin(), vec.end());
    }

    // Обновление данных
    data.clear();
    if (invert_order) {
      data.grow_by(group1.begin(), group1.end());  // Сначала group1
      data.grow_by(group0.begin(), group0.end());  // Потом group0
    } else {
      data.grow_by(group0.begin(), group0.end());  // Сначала group0
      data.grow_by(group1.begin(), group1.end());  // Потом group1
    }
  }
}

// 3. Слияние результатов (параллельно)
void MergeResults(const tbb::concurrent_vector<uint64_t>& negative, const tbb::concurrent_vector<uint64_t>& positive, std::vector<double>& output_) {
  // Запись отрицательных чисел
  tbb::parallel_for(tbb::blocked_range<size_t>(0, negative.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      double value;
      ::memcpy(&value, &negative[i], sizeof(double));
      output_[i] = value;
    }
  });

  // Запись положительных чисел
  tbb::parallel_for(tbb::blocked_range<size_t>(0, positive.size()), [&](const tbb::blocked_range<size_t>& r) {
    size_t offset = negative.size();
    for (size_t i = r.begin(); i < r.end(); ++i) {
      double value;
      ::memcpy(&value, &positive[i], sizeof(double));
      output_[offset + i] = value;
    }
  });
}
}  // namespace

bool tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB::PreProcessingImpl() {
  // Init value for input and output
  auto* temp_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_ = std::vector<double>(temp_ptr, temp_ptr + task_data->inputs_count[0]);
  output_.resize(task_data->inputs_count[0]);

  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB::RunImpl() {
  tbb::concurrent_vector<uint64_t> pozitive_copy;
  tbb::concurrent_vector<uint64_t> negative_copy;
  size_t num_threads = ppc::util::GetPPCNumThreads();
  tbb::task_arena arena(num_threads);
  arena.execute([&] {
    SplitData(pozitive_copy, negative_copy, input_);
    tbb::task_group tasks;
    tasks.run([&] { RadixSort(negative_copy, true); });
    tasks.run([&] { RadixSort(pozitive_copy, false); });
    tasks.wait();
    MergeResults(negative_copy, pozitive_copy,output_);
  });
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_tbb::TestTaskTBB::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
