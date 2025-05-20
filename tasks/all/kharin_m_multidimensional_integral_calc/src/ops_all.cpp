#include "all/kharin_m_multidimensional_integral_calc/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <functional>
#include <thread>
#include <utility>

#include "boost/mpi/collectives/reduce.hpp"
#include "core/include/utils.hpp"

bool kharin_m_multidimensional_integral_calc_all::TaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs.size() != 3 || task_data->outputs.size() != 1) return false;
    if (task_data->inputs_count[1] != task_data->inputs_count[2]) return false;
    if (task_data->outputs_count[0] != 1) return false;
  }
  return true;
}

bool kharin_m_multidimensional_integral_calc_all::TaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto* input_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    size_t input_size = task_data->inputs_count[0];
    input_ = std::vector<double>(input_ptr, input_ptr + input_size);
    auto* sizes_ptr = reinterpret_cast<size_t*>(task_data->inputs[1]);
    size_t d = task_data->inputs_count[1];
    grid_sizes_ = std::vector<size_t>(sizes_ptr, sizes_ptr + d);
    auto* steps_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
    step_sizes_ = std::vector<double>(steps_ptr, steps_ptr + d);
    size_t total_size = 1;
    for (auto n : grid_sizes_) total_size *= n;
    if (total_size != input_size) return false;
  }
  boost::mpi::broadcast(world_, grid_sizes_, 0);
  boost::mpi::broadcast(world_, step_sizes_, 0);
  size_t total_size = 1;
  for (auto n : grid_sizes_) total_size *= n;
  size_t P = world_.size();
  size_t chunk_size = total_size / P;
  size_t remainder = total_size % P;
  size_t rank = world_.rank();
  size_t local_size = (rank < remainder) ? chunk_size + 1 : chunk_size;
  local_input_.resize(local_size);
  if (world_.rank() == 0) {
    std::vector<int> send_counts(P);
    std::vector<int> displacements(P);
    size_t offset = 0;
    for (size_t i = 0; i < P; ++i) {
      size_t size = (i < remainder) ? chunk_size + 1 : chunk_size;
      send_counts[i] = static_cast<int>(size);
      displacements[i] = static_cast<int>(offset);
      offset += size;
    }
    boost::mpi::scatterv(world_, input_, send_counts, displacements, local_input_.data(),
                         static_cast<int>(local_input_.size()), 0);
  } else {
    boost::mpi::scatterv(world_, local_input_.data(), static_cast<int>(local_input_.size()), 0);
  }
  for (const auto& h : step_sizes_) {
    if (h <= 0.0) return false;
  }
  return true;
}

bool kharin_m_multidimensional_integral_calc_all::TaskALL::RunImpl() {
  if (!local_input_.empty()) {
    num_threads_ = std::min(static_cast<size_t>(ppc::util::GetPPCNumThreads()), local_input_.size());
    std::vector<std::thread> threads;
    threads.reserve(num_threads_);
    std::vector<double> partial_sums(num_threads_, 0.0);
    auto input_chunk_size = local_input_.size() / num_threads_;
    auto remainder = local_input_.size() % num_threads_;
    auto chunk_plus = [&](std::vector<double>::iterator it_begin, size_t size, double& result_location) {
      double local = 0.0;
      for (size_t i = 0; i < size; ++i) {
        local += *(it_begin + static_cast<std::vector<double>::difference_type>(i));
      }
      result_location = local;
    };
    size_t current_start_index = 0;
    for (size_t i = 0; i < num_threads_; ++i) {
      size_t size = (i < remainder) ? (input_chunk_size + 1) : input_chunk_size;
      auto it_begin = local_input_.begin() + static_cast<std::vector<double>::difference_type>(current_start_index);
      std::thread th(chunk_plus, it_begin, size, std::ref(partial_sums[i]));
      threads.push_back(std::move(th));
      current_start_index += size;
    }
    for (auto& th : threads) {
      if (th.joinable()) th.join();
    }
    double local_sum = 0.0;
    for (const auto& partial : partial_sums) {
      local_sum += partial;
    }
    double total_sum = 0.0;
    boost::mpi::reduce(world_, local_sum, total_sum, std::plus<double>(), 0);
    if (world_.rank() == 0) {
      double volume_element = 1.0;
      for (const auto& h : step_sizes_) {
        volume_element *= h;
      }
      output_result_ = total_sum * volume_element;
    }
  } else {
    double local_sum = 0.0;
    double total_sum = 0.0;
    boost::mpi::reduce(world_, local_sum, total_sum, std::plus<double>(), 0);
    if (world_.rank() == 0) {
      double volume_element = 1.0;
      for (const auto& h : step_sizes_) {
        volume_element *= h;
      }
      output_result_ = total_sum * volume_element;
    }
  }
  return true;
}

bool kharin_m_multidimensional_integral_calc_all::TaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = output_result_;
  }
  return true;
}