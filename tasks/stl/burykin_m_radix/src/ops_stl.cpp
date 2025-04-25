#include "stl/burykin_m_radix/include/ops_stl.hpp"

#include <array>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"  // для получения числа потоков

std::array<int, 256> burykin_m_radix_stl::RadixSTL::ComputeFrequency(const std::vector<int>& a, const int shift) {
  std::array<int, 256> count = {};
  for (const int v : a) {
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }
    ++count[key];
  }
  return count;
}

// Функция для параллельного расчета частот
std::array<int, 256> burykin_m_radix_stl::RadixSTL::ComputeFrequencyParallel(const std::vector<int>& a, const int shift,
                                                                             int num_threads) {
  std::vector<std::array<int, 256>> thread_counts(num_threads);
  std::vector<std::thread> threads(num_threads);

  const int chunk_size = a.size() / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    threads[t] = std::thread([&, t]() {
      const int start = t * chunk_size;
      const int end = (t == num_threads - 1) ? a.size() : (t + 1) * chunk_size;

      for (int i = start; i < end; ++i) {
        unsigned int key = ((static_cast<unsigned int>(a[i]) >> shift) & 0xFFU);
        if (shift == 24) {
          key ^= 0x80;
        }
        ++thread_counts[t][key];
      }
    });
  }

  // Ждем завершения всех потоков
  for (auto& thread : threads) {
    thread.join();
  }

  // Объединяем результаты
  std::array<int, 256> total_count = {};
  for (const auto& count : thread_counts) {
    for (int i = 0; i < 256; ++i) {
      total_count[i] += count[i];
    }
  }

  return total_count;
}

std::array<int, 256> burykin_m_radix_stl::RadixSTL::ComputeIndices(const std::array<int, 256>& count) {
  std::array<int, 256> index = {0};
  for (int i = 1; i < 256; ++i) {
    index[i] = index[i - 1] + count[i - 1];
  }
  return index;
}

void burykin_m_radix_stl::RadixSTL::DistributeElements(const std::vector<int>& a, std::vector<int>& b,
                                                       std::array<int, 256> index, const int shift) {
  for (const int v : a) {
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }
    b[index[key]++] = v;
  }
}

// Функция для параллельного распределения элементов
void burykin_m_radix_stl::RadixSTL::DistributeElementsParallel(const std::vector<int>& a, std::vector<int>& b,
                                                               const std::array<int, 256>& global_index,
                                                               const int shift, int num_threads) {
  // Копируем индексы для каждого потока
  std::vector<std::array<int, 256>> thread_indices(num_threads);
  std::vector<std::array<int, 256>> thread_counts(num_threads);

  // Расчет частот для каждого потока
  const int chunk_size = a.size() / num_threads;
  std::vector<std::thread> freq_threads(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    freq_threads[t] = std::thread([&, t]() {
      const int start = t * chunk_size;
      const int end = (t == num_threads - 1) ? a.size() : (t + 1) * chunk_size;

      for (int i = start; i < end; ++i) {
        unsigned int key = ((static_cast<unsigned int>(a[i]) >> shift) & 0xFFU);
        if (shift == 24) {
          key ^= 0x80;
        }
        ++thread_counts[t][key];
      }
    });
  }

  // Ждем завершения подсчета частот
  for (auto& thread : freq_threads) {
    thread.join();
  }

  // Вычисляем смещения для каждого потока
  for (int k = 0; k < 256; ++k) {
    int offset = global_index[k];
    for (int t = 0; t < num_threads; ++t) {
      thread_indices[t][k] = offset;
      offset += thread_counts[t][k];
    }
  }

  // Распределяем элементы параллельно
  std::vector<std::thread> dist_threads(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    dist_threads[t] = std::thread([&, t]() {
      const int start = t * chunk_size;
      const int end = (t == num_threads - 1) ? a.size() : (t + 1) * chunk_size;

      for (int i = start; i < end; ++i) {
        unsigned int key = ((static_cast<unsigned int>(a[i]) >> shift) & 0xFFU);
        if (shift == 24) {
          key ^= 0x80;
        }
        b[thread_indices[t][key]++] = a[i];
      }
    });
  }

  // Ждем завершения распределения
  for (auto& thread : dist_threads) {
    thread.join();
  }
}

bool burykin_m_radix_stl::RadixSTL::PreProcessingImpl() {
  const unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  output_.resize(input_size);
  return true;
}

bool burykin_m_radix_stl::RadixSTL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool burykin_m_radix_stl::RadixSTL::RunImpl() {
  if (input_.empty()) {
    return true;
  }

  std::vector<int> a = std::move(input_);
  std::vector<int> b(a.size());

  // Получаем количество потоков
  const int num_threads = ppc::util::GetPPCNumThreads();

  // Определяем минимальный размер данных для параллельной обработки
  const int min_parallel_size = 1000;  // Настройте этот параметр согласно вашим требованиям

  for (int shift = 0; shift < 32; shift += 8) {
    std::array<int, 256> count;

    // Используем параллельный расчет частот только если размер данных достаточно велик
    if (a.size() > min_parallel_size && num_threads > 1) {
      count = ComputeFrequencyParallel(a, shift, num_threads);
    } else {
      count = ComputeFrequency(a, shift);
    }

    const auto index = ComputeIndices(count);

    // Используем параллельное распределение только если размер данных достаточно велик
    if (a.size() > min_parallel_size && num_threads > 1) {
      DistributeElementsParallel(a, b, index, shift, num_threads);
    } else {
      DistributeElements(a, b, index, shift);
    }

    a.swap(b);
  }

  output_ = std::move(a);
  return true;
}

bool burykin_m_radix_stl::RadixSTL::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}