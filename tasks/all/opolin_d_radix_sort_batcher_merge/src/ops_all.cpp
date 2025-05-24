#include "all/opolin_d_radix_sort_batcher_merge/include/ops_all.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/enumerable_thread_specific.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/parallel_invoke.h"

bool opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll::PreProcessingImpl() {
  int rank = world_.rank();
  int num_procs = world_.size();

  std::vector<int> global_input_data_on_root;
  if (rank == 0) {
    if (!task_data || task_data->inputs.empty() || task_data->inputs_count.empty()) {
      global_original_size_ = 0;
    } else {
      global_original_size_ = task_data->inputs_count[0];
      if (this->global_original_size_ > 0) {
        auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
        global_input_data_on_root.assign(in_ptr, in_ptr + this->global_original_size_);
      }
    }
  }
  boost::mpi::broadcast(world_, this->global_original_size_, 0);

  if (this->global_original_size_ == 0) {
    size_ = 0;
    if (rank == 0 && task_data && !task_data->outputs_count.empty()) {
      task_data->outputs_count[0] = 0;
    }
    return true;
  }
  size_t padded_chunk_size_calc = (global_original_size_ + num_procs - 1) / num_procs;
  size_ = static_cast<int>(padded_chunk_size_calc);
  size_t padded_global_size_calc = padded_chunk_size_calc * num_procs;

  this->input_.resize(this->size_);
  if (rank == 0) {
    std::vector<int> temp_padded_global_input_on_root = global_input_data_on_root;
    if (global_input_data_on_root.size() < padded_global_size_calc) {
      temp_padded_global_input_on_root.resize(padded_global_size_calc, std::numeric_limits<int>::max());
    }
    boost::mpi::scatter(world_, temp_padded_global_input_on_root, this->input_.data(), this->size_, 0);
  } else {
    boost::mpi::scatter(world_, this->input_.data(), this->size_, 0);
  }
  return true;
}

bool opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll::ValidationImpl() {
  if (world_.rank() == 0) {
    global_original_size_ = static_cast<int>(task_data->inputs_count[0]);
    if (global_original_size_ <= 0 || task_data->inputs.empty() || task_data->outputs.empty()) {
      return false;
    }
    if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
      return false;
    }
    return task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll::RunImpl() {
  if (size_ == 0 && global_original_size_ == 0) {
    return true;
  }
  if (size_ == 0 && global_original_size_ > 0) {
    output_.assign(size_, std::numeric_limits<int>::max());
  }
  if (output_.size() != static_cast<size_t>(size_)) {
    output_.resize(size_);
  }
  if (size_ > 0) {
    std::vector<uint32_t> keys(size_);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, static_cast<size_t>(size_)),
                      [&](const tbb::blocked_range<size_t>& r) {
                        for (size_t i = r.begin(); i < r.end(); ++i) {
                          keys[i] = ConvertIntToUint(input_[i]);
                        }
                      });
    RadixSort(keys);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, static_cast<size_t>(size_)),
                      [&](const tbb::blocked_range<size_t>& r) {
                        for (size_t i = r.begin(); i < r.end(); ++i) {
                          output_[i] = ConvertUintToInt(keys[i]);
                        }
                      });
  }
  int rank = world_.rank();
  int num_procs = world_.size();
  if (num_procs > 1 && size_ > 0) {
    int stages_mpi_merge = 0;
    if (num_procs > 1) {
      stages_mpi_merge = static_cast<int>(std::ceil(std::log2(static_cast<double>(num_procs))));
    }

    for (int stage_idx = 0; stage_idx < stages_mpi_merge; ++stage_idx) {
      int offset_val = 1 << (stages_mpi_merge - stage_idx - 1);
      for (int step_val = offset_val; step_val > 0; step_val >>= 1) {
        int partner = rank ^ step_val;
        if (partner >= num_procs) {
          continue;
        }

        std::vector<int> received_data(size_);
        std::vector<int> merged_data;
        merged_data.reserve(static_cast<size_t>(size_) * 2);

        boost::mpi::request reqs[2];
        if (rank < partner) {
          reqs[0] = world_.isend(partner, 0, output_.data(), size_);
          reqs[1] = world_.irecv(partner, 0, received_data.data(), size_);
          boost::mpi::wait_all(reqs, reqs + 2);

          std::merge(output_.begin(), output_.end(), received_data.begin(), received_data.end(),
                     std::back_inserter(merged_data));
          std::copy(merged_data.begin(), merged_data.begin() + size_, output_.begin());
        } else {
          reqs[0] = world_.irecv(partner, 0, received_data.data(), size_);
          reqs[1] = world_.isend(partner, 0, output_.data(), size_);
          boost::mpi::wait_all(reqs, reqs + 2);

          std::merge(received_data.begin(), received_data.end(), output_.begin(), output_.end(),
                     std::back_inserter(merged_data));
          std::copy(merged_data.begin() + size_, merged_data.end(), output_.begin());
        }
      }
      world_.barrier();
    }
  }
  return true;
}

bool opolin_d_radix_batcher_sort_all::RadixBatcherSortTaskAll::PostProcessingImpl() {
  int rank = world_.rank();
  int num_procs = world_.size();
  if (global_original_size_ == 0) {
    if (rank == 0 && task_data && !task_data->outputs_count.empty()) {
      task_data->outputs_count[0] = 0;
    }
    return true;
  }
  std::vector<int> final_gathered_data;
  if (rank == 0) {
    final_gathered_data.resize(static_cast<size_t>(size_) * num_procs);
  }

  if (size_ > 0 && output_.empty()) {
    output_.assign(size_, std::numeric_limits<int>::max());
  }

  boost::mpi::gather(world_, output_.data(), size_, final_gathered_data.data(), 0);

  if (rank == 0) {
    if (task_data && task_data->outputs[0] != nullptr) {
      auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
      size_t elements_to_copy = std::min(global_original_size_, final_gathered_data.size());
      for (size_t i = 0; i < elements_to_copy; ++i) {
        out_ptr[i] = final_gathered_data[i];
      }
      if (!task_data->outputs_count.empty()) {
        task_data->outputs_count[0] = global_original_size_;
      }
    } else if (global_original_size_ > 0) {
      return false;
    }
  }
  return true;
}

uint32_t opolin_d_radix_batcher_sort_all::ConvertIntToUint(int num) { return static_cast<uint32_t>(num) ^ 0x80000000U; }

int opolin_d_radix_batcher_sort_all::ConvertUintToInt(uint32_t unum) { return static_cast<int>(unum ^ 0x80000000U); }

void opolin_d_radix_batcher_sort_all::RadixSort(std::vector<uint32_t>& uns_vec) {
  size_t sz = uns_vec.size();
  if (sz <= 1) {
    return;
  }
  const int rad = 256;
  std::vector<uint32_t> res(sz);
  for (int stage = 0; stage < 4; stage++) {
    tbb::enumerable_thread_specific<std::vector<size_t>> local_counts([&] { return std::vector<size_t>(rad, 0); });
    int shift = stage * 8;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, sz), [&](const tbb::blocked_range<size_t>& r) {
      auto& lc = local_counts.local();
      for (size_t i = r.begin(); i < r.end(); ++i) {
        const uint8_t byte = (uns_vec[i] >> shift) & (rad - 1);
        lc[byte]++;
      }
    });
    std::vector<size_t> pref(rad, 0);
    for (auto& lc_instance : local_counts) {
      for (int j = 0; j < rad; ++j) {
        pref[j] += lc_instance[j];
      }
    }
    for (int j = 1; j < rad; ++j) {
      pref[j] += pref[j - 1];
    }
    for (int i = static_cast<int>(sz) - 1; i >= 0; --i) {
      const uint8_t byte = (uns_vec[i] >> shift) & (rad - 1);
      res[--pref[byte]] = uns_vec[i];
    }
    uns_vec.swap(res);
  }
}