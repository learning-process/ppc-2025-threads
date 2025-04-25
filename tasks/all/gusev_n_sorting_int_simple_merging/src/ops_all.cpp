#include "all/gusev_n_sorting_int_simple_merging/include/ops_all.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // IWYU pragma: keep
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

void gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::SplitBySign(const std::vector<int>& arr,
                                                                                     std::vector<int>& negatives,
                                                                                     std::vector<int>& positives) {
  for (int num : arr) {
    if (num < 0) {
      negatives.push_back(-num);
    } else {
      positives.push_back(num);
    }
  }
}

void gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::MergeResults(
    std::vector<int>& arr, const std::vector<int>& negatives, const std::vector<int>& positives) {
  arr.clear();
  arr.insert(arr.end(), negatives.begin(), negatives.end());
  arr.insert(arr.end(), positives.begin(), positives.end());
}

std::vector<std::vector<int>> gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::DistributeArray(
    const std::vector<int>& arr, int num_procs) {
  std::vector<std::vector<int>> chunks(num_procs);

  if (arr.empty()) {
    return chunks;
  }

  size_t chunk_size = arr.size() / num_procs;
  size_t remainder = arr.size() % num_procs;

  size_t start = 0;
  for (size_t i = 0; i < static_cast<size_t>(num_procs); ++i) {
    size_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);

    if (current_chunk_size > 0) {
      size_t end = start + current_chunk_size;
      chunks[i].insert(chunks[i].end(), arr.begin() + start, arr.begin() + end);
      start = end;
    }
  }

  return chunks;
}

std::vector<int> gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::MergeSortedArrays(
    const std::vector<std::vector<int>>& arrays) {
  std::vector<int> result;

  size_t total_size = 0;
  for (const auto& arr : arrays) {
    total_size += arr.size();
  }
  result.reserve(total_size);

  for (const auto& arr : arrays) {
    if (arr.empty()) continue;

    if (result.empty()) {
      result = arr;
    } else {
      std::vector<int> merged(result.size() + arr.size());
      std::merge(result.begin(), result.end(), arr.begin(), arr.end(), merged.begin());
      result = std::move(merged);
    }
  }

  return result;
}

void gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::RadixSort(std::vector<int>& arr) {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  if (arr.empty()) {
    return;
  }

  std::vector<int> negatives;
  std::vector<int> positives;

  if (rank == 0) {
    SplitBySign(arr, negatives, positives);
  }

  size_t negatives_size = negatives.size();
  size_t positives_size = positives.size();
  boost::mpi::broadcast(world, negatives_size, 0);
  boost::mpi::broadcast(world, positives_size, 0);

  std::vector<int> sorted_negatives;
  std::vector<int> sorted_positives;

  if (negatives_size > 0) {
    if (rank != 0) {
      negatives.resize(negatives_size);
    }
    boost::mpi::broadcast(world, negatives, 0);

    std::vector<std::vector<int>> neg_chunks;
    if (rank == 0) {
      neg_chunks = DistributeArray(negatives, size);
    }

    std::vector<int> my_neg_chunk;
    if (size > 1) {
      if (rank == 0) {
        for (int i = 1; i < size; ++i) {
          world.send(i, 0, neg_chunks[i]);
        }
        my_neg_chunk = std::move(neg_chunks[0]);
      } else {
        world.recv(0, 0, my_neg_chunk);
      }
    } else {
      my_neg_chunk = negatives;
    }

    if (!my_neg_chunk.empty()) {
      RadixSortForNonNegative(my_neg_chunk);
    }

    if (size > 1) {
      std::vector<std::vector<int>> gathered_neg_chunks;
      boost::mpi::gather(world, my_neg_chunk, gathered_neg_chunks, 0);

      if (rank == 0) {
        sorted_negatives = MergeSortedArrays(gathered_neg_chunks);

        std::ranges::reverse(sorted_negatives);
        std::ranges::transform(sorted_negatives, sorted_negatives.begin(), std::negate{});
      }
    } else {
      sorted_negatives = std::move(my_neg_chunk);
      std::ranges::reverse(sorted_negatives);
      std::ranges::transform(sorted_negatives, sorted_negatives.begin(), std::negate{});
    }
  }

  if (positives_size > 0) {
    if (rank != 0) {
      positives.resize(positives_size);
    }
    boost::mpi::broadcast(world, positives, 0);

    std::vector<std::vector<int>> pos_chunks;
    if (rank == 0) {
      pos_chunks = DistributeArray(positives, size);
    }

    std::vector<int> my_pos_chunk;
    if (size > 1) {
      if (rank == 0) {
        for (int i = 1; i < size; ++i) {
          world.send(i, 1, pos_chunks[i]);
        }
        my_pos_chunk = std::move(pos_chunks[0]);
      } else {
        world.recv(0, 1, my_pos_chunk);
      }
    } else {
      my_pos_chunk = positives;
    }

    if (!my_pos_chunk.empty()) {
      RadixSortForNonNegative(my_pos_chunk);
    }

    if (size > 1) {
      std::vector<std::vector<int>> gathered_pos_chunks;
      boost::mpi::gather(world, my_pos_chunk, gathered_pos_chunks, 0);

      if (rank == 0) {
        sorted_positives = MergeSortedArrays(gathered_pos_chunks);
      }
    } else {
      sorted_positives = std::move(my_pos_chunk);
    }
  }

  if (rank == 0) {
    MergeResults(arr, sorted_negatives, sorted_positives);
  }

  boost::mpi::broadcast(world, arr, 0);

  world.barrier();
}

void gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::RadixSortForNonNegative(
    std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }

  int max = *std::ranges::max_element(arr);
  for (int exp = 1; max / exp > 0; exp *= 10) {
    CountingSort(arr, exp);
  }
}

void gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::CountingSort(std::vector<int>& arr, int exp) {
  boost::mpi::communicator world;

  std::vector<int> output(arr.size());
  std::vector<int> count(10, 0);

  oneapi::tbb::enumerable_thread_specific<std::vector<int>> tl_counts([&] { return std::vector<int>(10, 0); });

  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, arr.size()),
                            [&](const oneapi::tbb::blocked_range<size_t>& r) {
                              auto& local_counts = tl_counts.local();
                              for (size_t i = r.begin(); i < r.end(); ++i) {
                                int digit = (arr[i] / exp) % 10;
                                local_counts[digit]++;
                              }
                            });

  for (const auto& lc : tl_counts) {
    for (int d = 0; d < 10; ++d) {
      count[d] += lc[d];
    }
  }

  std::partial_sum(count.begin(), count.end(), count.begin());

  for (auto i = arr.size(); i > 0; --i) {
    int digit = (arr[i - 1] / exp) % 10;
    output[--count[digit]] = arr[i - 1];
  }

  arr = output;
}

bool gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::PreProcessingImpl() {
  input_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
                            reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  output_.resize(input_.size());
  return true;
}

bool gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::RunImpl() {
  RadixSort(input_);
  return true;
}

bool gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL::PostProcessingImpl() {
  std::ranges::copy(input_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
