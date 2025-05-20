#include "all/kolokolova_d_integral_simpson_method/include/ops_all.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

bool kolokolova_d_integral_simpson_method_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    nums_variables_ = int(task_data->inputs_count[0]);

    steps_ = std::vector<int>(task_data->inputs_count[0]);
    auto* input_steps = reinterpret_cast<int*>(task_data->inputs[0]);
    for (unsigned i = 0; i < task_data->inputs_count[0]; i++) {
      steps_[i] = input_steps[i];
    }

    borders_ = std::vector<int>(task_data->inputs_count[1]);
    auto* input_borders = reinterpret_cast<int*>(task_data->inputs[1]);
    for (unsigned i = 0; i < task_data->inputs_count[1]; i++) {
      borders_[i] = input_borders[i];
    }

    result_output_ = 0;
  }
  return true;
}

bool kolokolova_d_integral_simpson_method_all::TestTaskALL::ValidationImpl() {
  // Check inputs and outputs
  if (world_.rank() == 0) {
    std::vector<int> bord = std::vector<int>(task_data->inputs_count[1]);
    auto* input_bord = reinterpret_cast<int*>(task_data->inputs[1]);
    for (unsigned i = 0; i < task_data->inputs_count[1]; i++) {
      bord[i] = input_bord[i];
    }
    int num_var = int(task_data->inputs_count[0]);
    int num_bord = int(task_data->inputs_count[1]) / 2;
    return (task_data->inputs_count[0] != 0 && task_data->inputs_count[1] != 0 && task_data->outputs_count[0] != 0 &&
            CheckBorders(bord) && num_var == num_bord);
  }
  return true;
}

bool kolokolova_d_integral_simpson_method_all::TestTaskALL::RunImpl() {
  if (world_.rank() == 0) {
    //  Find size of step
    size_step.resize(nums_variables_);
    for (int i = 0; i < nums_variables_; i++) {
      double a = (double((borders_[(2 * i) + 1] - borders_[2 * i])) / double(steps_[i]));
      size_step[i] = a;
    }
    //  Create vector of points
    for (int i = 0; i < nums_variables_; i++) {
      std::vector<double> vec;
      for (int j = 0; j < steps_[i] + 1; j++) {
        auto num = double(borders_[2 * i] + (double(j) * size_step[i]));
        vec.push_back(num);
      }
      points.push_back(vec);
    }
    results_func.resize(int(points.size()));
    coeff.resize(steps_[0]);
    results_func = FindFunctionValue(points, func_);
    coeff = FindCoeff(steps_[0]);

    vec_coeff = std::vector<double>(int(results_func.size()));
    for (int i = 0; i < int(vec_coeff.size()); ++i) {
      vec_coeff[i] = coeff[i % int(coeff.size())];
    }

    while (int(results_func.size()) % world_.size() != 0) {
      results_func.push_back(0.0);
    }

    while (int(vec_coeff.size()) % world_.size() != 0) {
      vec_coeff.push_back(0.0);
    }

    size_local_results_func = int(results_func.size()) / int(world_.size());
    size_local_coeff = int(vec_coeff.size()) / int(world_.size());
    size_local_size_step = int(size_step.size());
  }

  broadcast(world_, size_local_results_func, 0);
  broadcast(world_, nums_variables_, 0);
  broadcast(world_, size_local_coeff, 0);
  broadcast(world_, size_local_size_step, 0);

  if (world_.rank() == 0) {
    for (int proc = 1; proc < world_.size(); proc++) {
      world_.send(proc, 0, results_func.data() + proc * size_local_results_func, size_local_results_func);
    }
  }
  local_results_func = std::vector<double>(size_local_results_func);

  if (world_.rank() == 0) {
    local_results_func = std::vector<double>(results_func.begin(), results_func.begin() + size_local_results_func);
  } else {
    world_.recv(0, 0, local_results_func.data(), size_local_results_func);
  }

  if (world_.rank() == 0) {
    for (int proc = 1; proc < world_.size(); proc++) {
      world_.send(proc, 0, vec_coeff.data() + proc * size_local_coeff, size_local_coeff);
    }
  }

  local_coeff = std::vector<double>(size_local_coeff);

  if (world_.rank() == 0) {
    local_coeff = std::vector<double>(vec_coeff.begin(), vec_coeff.begin() + size_local_coeff);
  } else {
    world_.recv(0, 0, local_coeff.data(), size_local_coeff);
  }

  int function_vec_size = int(local_results_func.size());

  // initial multiplication
  for (int i = 0; i < function_vec_size; ++i) {
    local_results_func[i] *= local_coeff[i];
  }

  gather(world_, local_results_func.data(), size_local_results_func, results_func, 0);

  if (world_.rank() == 0) {
    for (int iteration = 1; iteration < nums_variables_; ++iteration) {
      for (int i = 0; i < int(results_func.size()); ++i) {
        int block_size = iteration * int(coeff.size());
        int current_n_index = (i / block_size) % int(coeff.size());
        results_func[i] *= coeff[current_n_index];
      }
    }
  }

  if (world_.rank() == 0) {
    for (int proc = 1; proc < world_.size(); proc++) {
      world_.send(proc, 0, results_func.data() + proc * size_local_results_func, size_local_results_func);
    }
  }

  local_results_func = std::vector<double>(size_local_results_func);

  if (world_.rank() == 0) {
    local_results_func = std::vector<double>(results_func.begin(), results_func.begin() + size_local_results_func);
  } else {
    world_.recv(0, 0, local_results_func.data(), size_local_results_func);
  }

  if (world_.rank() == 0) {
    for (int proc = 1; proc < world_.size(); proc++) {
      world_.send(proc, 0, size_step.data(), size_local_size_step);
    }
  }

  local_size_step = std::vector<double>(size_local_size_step);

  if (world_.rank() == 0) {
    local_size_step = std::vector<double>(size_step.begin(), size_step.begin() + size_local_size_step);
  } else {
    world_.recv(0, 0, local_size_step.data(), size_local_size_step);
  }

  for (size_t i = 0; i < local_results_func.size(); i++) {
    local_results_output += local_results_func[i];
  }

  for (size_t i = 0; i < local_size_step.size(); i++) {
    local_results_output *= local_size_step[i];
  }

  reduce(world_, local_results_output, result_output_, std::plus(), 0);

  if (world_.rank() == 0) {
    // divided by 3 to the power
    result_output_ /= pow(3, nums_variables_);
  }
  return true;
}

bool kolokolova_d_integral_simpson_method_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = result_output_;
  }
  return true;
}

std::vector<double> kolokolova_d_integral_simpson_method_all::TestTaskALL::FindFunctionValue(
    const std::vector<std::vector<double>>& coordinates, const std::function<double(std::vector<double>)>& f) {
  std::vector<double> results;                                     // result of function
  std::vector<double> current;                                     // current point
  GeneratePointsAndEvaluate(coordinates, 0, current, results, f);  // recursive function
  return results;
}

void kolokolova_d_integral_simpson_method_all::TestTaskALL::GeneratePointsAndEvaluate(
    const std::vector<std::vector<double>>& coordinates, int index, std::vector<double>& current,
    std::vector<double>& results, const std::function<double(const std::vector<double>)>& f) {
  // if it the end of vector
  if (index == int(coordinates.size())) {
    double result = f(current);  // find value of function
    results.push_back(result);   // save result
    return;
  }

  // sort through the coordinates
  for (double coord : coordinates[index]) {
    current.push_back(coord);
    GeneratePointsAndEvaluate(coordinates, index + 1, current, results, func_);  // recursive
    current.pop_back();                                                          // delete for next coordinat
  }
}

std::vector<double> kolokolova_d_integral_simpson_method_all::TestTaskALL::FindCoeff(int count_step) {
  std::vector<double> result_coeff(1, 1.0);  // first coeff is always 1
  for (int i = 1; i < count_step; i++) {
    if (i % 2 != 0) {
      result_coeff.push_back(4.0);  // odd coeff is 4
    } else {
      result_coeff.push_back(2.0);  // even coeff is 2
    }
  }
  result_coeff.push_back(1.0);  // last coeff is always 1
  return result_coeff;
}

void kolokolova_d_integral_simpson_method_all::TestTaskALL::MultiplyCoeffandFunctionValue(
    std::vector<double>& function_val, const std::vector<double>& coeff_vec, int a) {}

double kolokolova_d_integral_simpson_method_all::TestTaskALL::CreateOutputResult(std::vector<double> vec,
                                                                                 std::vector<double> size_steps) const {
  double sum = 0;

  // sum all of vector elements
  for (size_t i = 0; i < vec.size(); i++) {
    sum += vec[i];
  }

  // multiply by the length of steps
  for (size_t i = 0; i < size_steps.size(); i++) {
    sum *= size_steps[i];
  }

  // divided by 3 to the power
  sum /= pow(3, nums_variables_);

  return sum;
}
bool kolokolova_d_integral_simpson_method_all::TestTaskALL::CheckBorders(std::vector<int> vec) {
  size_t i = 0;
  while (i < vec.size()) {
    if (vec[i] > vec[i + 1]) {
      return false;
    }
    i += 2;
  }
  return true;
}