#include "all/gusev_n_sorting_int_simple_merging/include/ops_all.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

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
    for (int num : arr) {
      if (num < 0) {
        negatives.push_back(-num);
      } else {
        positives.push_back(num);
      }
    }
  }

  int negatives_size = negatives.size();
  int positives_size = positives.size();

  boost::mpi::broadcast(world, negatives_size, 0);
  boost::mpi::broadcast(world, positives_size, 0);

  if (rank != 0) {
    negatives.resize(negatives_size);
    positives.resize(positives_size);
  }

  boost::mpi::broadcast(world, negatives, 0);
  boost::mpi::broadcast(world, positives, 0);

  if (size > 1) {
    if (rank == 0 && !negatives.empty()) {
      RadixSortForNonNegative(negatives);
      std::ranges::reverse(negatives);
      std::ranges::transform(negatives, negatives.begin(), std::negate{});
    } else if (rank == 1 && !positives.empty()) {
      RadixSortForNonNegative(positives);
    }
  } else {
    oneapi::tbb::parallel_invoke(
        [&] {
          if (!negatives.empty()) {
            RadixSortForNonNegative(negatives);
            std::ranges::reverse(negatives);
            std::ranges::transform(negatives, negatives.begin(), std::negate{});
          }
        },
        [&] {
          if (!positives.empty()) {
            RadixSortForNonNegative(positives);
          }
        });
  }

  if (size > 1) {
    if (rank == 1) {
      world.send(0, 0, positives);
    } else if (rank == 0) {
      if (!positives.empty()) {
        world.recv(1, 0, positives);
      }
    }
  }

  if (rank == 0) {
    arr.clear();
    arr.insert(arr.end(), negatives.begin(), negatives.end());
    arr.insert(arr.end(), positives.begin(), positives.end());
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
  // int rank = world.rank();
  int size = world.size();

  std::vector<int> output(arr.size());
  std::vector<int> count(10, 0);

  // Using tbb for parallel numeric
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

  if (size > 1) {
    std::vector<int> global_count(10, 0);
    boost::mpi::all_reduce(world, count.data(), 10, global_count.data(), std::plus<int>());
    count = global_count;
  }

  std::partial_sum(count.begin(), count.end(), count.begin());

  for (auto i = arr.size(); i > 0; --i) {
    int digit = (arr[i - 1] / exp) % 10;
    output[--count[digit]] = arr[i - 1];
  }

  arr = output;

  if (size > 1) {
    boost::mpi::broadcast(world, arr, 0);
  }
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
