#include "all/deryabin_m_hoare_sort_simple_merge/include/ops_all.hpp"

#include <oneapi/tbb/task_group.h>

#include <algorithm>
#include <bit>
#include <boost/mpi/collectives/broadcast.hpp>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <vector>

void deryabin_m_hoare_sort_simple_merge_mpi::HoaraSort(std::vector<double>& a, size_t first, size_t last) {
  if (first >= last) {
    return;
  }
  const size_t mid = (first + last) >> 1;
  const double x = std::max(std::min(a[first], a[mid]), std::min(std::max(a[first], a[mid]), a[last]));
  double* pi = &a[first];
  double* pj = &a[last];
  do {
    while (*pi < x) {
      pi++;
    }
    while (*pj > x) {
      pj--;
    }
    const double tmp = *pi;
    *pi = *pj;
    *pj = tmp;
  } while (pi < pj);
  const size_t j = pj - a.data();
  const size_t i = pi - a.data();
  HoaraSort(a, first, j);
  HoaraSort(a, i + 1, last);
}

void deryabin_m_hoare_sort_simple_merge_mpi::HoaraSort(std::vector<double>& a, size_t first, size_t last,
                                                       oneapi::tbb::task_group& tg, size_t available_threads) {
  if (first >= last) {
    return;
  }
  const size_t mid = (first + last) >> 1;
  const double x = std::max(std::min(a[first], a[mid]), std::min(std::max(a[first], a[mid]), a[last]));
  double* pi = &a[first];
  double* pj = &a[last];
  do {
    while (*pi < x) {
      pi++;
    }
    while (*pj > x) {
      pj--;
    }
    const double tmp = *pi;
    *pi = *pj;
    *pj = tmp;
  } while (pi < pj);
  const size_t j = pj - a.data();
  const size_t i = pi - a.data();
  if (available_threads > 1) {
    tg.run([&a, &first, &j, &tg, &available_threads]() { HoaraSort(a, first, j, tg, available_threads >> 1); });
    tg.run([&a, &i, &last, &tg, &available_threads]() {
      HoaraSort(a, i + 1, last, tg, available_threads - (available_threads >> 1));
    });
  } else {
    HoaraSort(a, first, j, tg, 1);
    HoaraSort(a, i + 1, last, tg, 1);
  }
}

void deryabin_m_hoare_sort_simple_merge_mpi::MergeTwoParts(std::vector<double>& a, size_t first, size_t last,
                                                           oneapi::tbb::task_group& tg, size_t available_threads) {
  if (last - first <= 1) {
    return;
  }
  const size_t size = last - first;
  const size_t mid = first + size / 2;
  if (available_threads > 1) {
    tg.run([&, first, mid, available_threads]() { MergeTwoParts(a, first, mid, tg, available_threads / 2); });
    tg.run([&, last, mid, available_threads]() {
      MergeTwoParts(a, mid, last, tg, available_threads - available_threads / 2);
    });
    tg.wait();
    std::inplace_merge(a.begin() + first, a.begin() + mid, a.begin() + last);
  } else {
    std::inplace_merge(a.begin() + first, a.begin() + mid, a.begin() + last);
  }
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskSequential::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskSequential::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskSequential::RunImpl() {
  size_t count = 0;
  while (count != chunk_count_) {
    HoaraSort(input_array_A_, count * min_chunk_size_, ((count + 1) * min_chunk_size_) - 1);
    count++;
  }
  size_t chunk_count = chunk_count_;
  for (size_t i = 0; i < static_cast<size_t>(std::bit_width(chunk_count_) - 1); i++) {
    for (size_t j = 0; j < chunk_count; j++) {
      std::inplace_merge(input_array_A_.begin() + static_cast<long>(j * min_chunk_size_ << (i + 1)),
                         input_array_A_.begin() + static_cast<long>(((j << 1 | 1) * (min_chunk_size_ << i))),
                         input_array_A_.begin() + static_cast<long>((j + 1) * min_chunk_size_ << (i + 1)));
      chunk_count--;
    }
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskSequential::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI::PreProcessingImpl() {
  if (world.rank() == 0) {
    input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
    dimension_ = task_data->inputs_count[0];
    chunk_count_ = task_data->inputs_count[1];
    min_chunk_size_ = dimension_ / chunk_count_;
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI::ValidationImpl() {
  if (world.rank() == 0) {
    return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
           static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
           task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI::RunImpl() {
  size_t num_chunk_per_proc = 0;
  if (world.rank() == 0) {
    if (chunk_count_ < static_cast<size_t>(world.size())) {
      // Увеличиваем число кусочков до ближайшей степени двойки >= world.size(),
      // чтобы эффективно загрузить все доступные процессы
      chunk_count_ = 1ULL << std::bit_width(static_cast<size_t>(world.size()) - 1);
      min_chunk_size_ = dimension_ / chunk_count_;
    }
    num_chunk_per_proc = chunk_count_ / static_cast<size_t>(world.size());
  }
  boost::mpi::broadcast(world, num_chunk_per_proc, 0);
  oneapi::tbb::task_group tg;
  const size_t num_threads = ppc::util::GetPPCNumThreads();
  if (world.rank() != world.size() - 1) {
    HoaraSort(input_array_A_, static_cast<size_t>(world.rank()) * num_chunk_per_proc * min_chunk_size_,
              ((static_cast<size_t>(world.rank()) + 1) * num_chunk_per_proc * min_chunk_size_) - 1, tg, num_threads);
  } else {
    HoaraSort(input_array_A_, static_cast<size_t>(world.rank()) * num_chunk_per_proc * min_chunk_size_, dimension_ - 1,
              tg, num_threads);
  }
  tg.wait();
  for (size_t i = 0; i < static_cast<size_t>(std::bit_width(chunk_count_) -
                                             1);  // Вычисялем сколько уровней слияния потребуется как логарифм по
                                                  // основанию 2 от числа частей chunk_count_
       ++i) {  // На каждом уровне сливаются пары соседних блоков размером min_chunk_size_ × 2^i
    size_t step = 1ULL << i;
    if ((static_cast<size_t>(world.rank()) & step) == 0) {
      world.send(
          world.rank() + 1, world.rank(),
          input_array_A_.data() + (static_cast<size_t>(world.rank() * num_chunk_per_proc * min_chunk_size_) << (i + 1)),
          static_cast<size_t>(num_chunk_per_proc * min_chunk_size_) << i);
    } else {
      world.recv(
          world.rank() - 1, world.rank() - 1,
          input_array_A_.data() +
              (static_cast<unsigned short>((world.rank() - 1) * num_chunk_per_proc * min_chunk_size_) << (i + 1)),
          static_cast<unsigned short>(num_chunk_per_proc * min_chunk_size_) << i);
      if (world.rank() != world.size() - 1) {
        MergeTwoParts(input_array_A_,
                      static_cast<size_t>((world.rank() - 1) * num_chunk_per_proc * min_chunk_size_) << (i + 1),
                      (((world.rank() + 1) * num_chunk_per_proc * min_chunk_size_) << (i + 1)) - 1, tg, num_threads);
      } else {
        MergeTwoParts(input_array_A_,
                      static_cast<size_t>((world.rank() - 1) * num_chunk_per_proc * min_chunk_size_) << (i + 1),
                      dimension_ - 1, tg, num_threads);
      }
    }
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_mpi::HoareSortTaskMPI::PostProcessingImpl() {
  if (world.rank() == 0) {
    reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  }
  return true;
}
