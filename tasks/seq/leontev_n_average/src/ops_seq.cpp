// Copyright 2024 Nesterov Alexander
#include "seq/leontev_n_average/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

template <class InOutType>
bool leontev_n_average_seq::VecAvgSequential<InOutType>::PreProcessingImpl() {
  input_ = std::vector<InOutType>(task_data->inputs_count[0]);
  auto* vec_ptr = reinterpret_cast<InOutType*>(task_data->inputs[0]);
  for (size_t i = 0; i < task_data->inputs_count[0]; i++) {
    input_[i] = vec_ptr[i];
  }
  res_ = 0;
  return true;
}

template <class InOutType>
bool leontev_n_average_seq::VecAvgSequential<InOutType>::ValidationImpl() {
  // Input vector exists and output is a single number
  return task_data->inputs_count[0] != 0 && task_data->outputs_count[0] == 1;
}

template <class InOutType>
bool leontev_n_average_seq::VecAvgSequential<InOutType>::RunImpl() {
  res_ = std::accumulate(input_.begin(), input_.end(), InOutType(0)) / input_.size();
  return true;
}

template <class InOutType>
bool leontev_n_average_seq::VecAvgSequential<InOutType>::PostProcessingImpl() {
  reinterpret_cast<InOutType*>(task_data->outputs[0])[0] = res_;
  return true;
}

template class leontev_n_average_seq::VecAvgSequential<int32_t>;
template class leontev_n_average_seq::VecAvgSequential<uint32_t>;
template class leontev_n_average_seq::VecAvgSequential<float>;
template class leontev_n_average_seq::VecAvgSequential<double>;
