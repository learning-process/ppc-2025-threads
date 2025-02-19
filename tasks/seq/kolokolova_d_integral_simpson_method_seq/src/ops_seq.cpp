#include "seq/kolokolova_d_integral_simpson_method_seq/include/ops_seq.hpp"

#include <math.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

bool kolokolova_d_integral_simpson_method_seq::TestTaskSequential::PreProcessingImpl() {
  nums_variables = task_data->inputs_count[0];

  steps = std::vector<int>(task_data->inputs_count[0]);
  auto* input_steps = reinterpret_cast<int*>(task_data->inputs[0]);
  for (unsigned i = 0; i < task_data->inputs_count[0]; i++) {
    steps[i] = input_steps[i];
  }

  borders = std::vector<int>(task_data->inputs_count[1]);
  auto* input_borders = reinterpret_cast<int*>(task_data->inputs[1]);
  for (unsigned i = 0; i < task_data->inputs_count[1]; i++) {
    borders[i] = input_borders[i];
  }

  result_output = 0;
  return true;
}

bool kolokolova_d_integral_simpson_method_seq::TestTaskSequential::ValidationImpl() {
  // Check inputs and outputs
  std::vector<int> bord = std::vector<int>(task_data->inputs_count[1]);
  auto* input_bord = reinterpret_cast<int*>(task_data->inputs[1]);
  for (unsigned i = 0; i < task_data->inputs_count[1]; i++) {
    bord[i] = input_bord[i];
  }
  return (task_data->inputs_count[0] != 0 && task_data->inputs_count[1] != 0 && task_data->outputs_count[0] != 0 &&
          checkBorders(bord));
  return true;
}

bool kolokolova_d_integral_simpson_method_seq::TestTaskSequential::RunImpl() {
  //  Find size of step
  std::vector<double> size_step(nums_variables);
  for (int i = 0; i < nums_variables; i++) {
    double a = (double((borders[2 * i + 1] - borders[2 * i])) / double(steps[i]));
    size_step[i] = a;
  }

  std::vector<std::vector<double>> points;

  //  Create vector of points
  for (int i = 0; i < nums_variables; i++) {
    std::vector<double> vec;
    for (double j = 0.0; j < steps[i] + 1; j++) {
      double num = double(borders[2 * i] + j * size_step[i]);
      vec.push_back(num);
    }
    points.push_back(vec);
  }

  std::vector<double> results_func = findFunctionValue(points, func);
  std::vector<double> coeff = findCoeff(steps[0]);
  multiplyCoeffandFunctionValue(results_func, coeff, nums_variables);
  result_output = createOutputResult(results_func, size_step);
  return true;
}

bool kolokolova_d_integral_simpson_method_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_output;
  return true;
}

std::vector<double> kolokolova_d_integral_simpson_method_seq::TestTaskSequential::findFunctionValue(
    const std::vector<std::vector<double>>& coordinates, std::function<double(std::vector<double>)> f) {
  std::vector<double> results;                                        // result of function
  std::vector<double> current;                                        // current point
  generatePointsAndEvaluate(coordinates, 0, current, results, func);  // recursive function
  return results;
}

void kolokolova_d_integral_simpson_method_seq::TestTaskSequential::generatePointsAndEvaluate(
    const std::vector<std::vector<double>>& coordinates, int index, std::vector<double>& current,
    std::vector<double>& results, const std::function<double(const std::vector<double>)> f) {
  // if it the end of vector
  if (index == int(coordinates.size())) {
    double result = func(current);  // find value of function
    results.push_back(result);      // save result
    return;
  }

  // sort through the coordinates
  for (double coord : coordinates[index]) {
    current.push_back(coord);
    generatePointsAndEvaluate(coordinates, index + 1, current, results, func);  // recursive
    current.pop_back();                                                         // delete for next coordinat
  }
}

std::vector<double> kolokolova_d_integral_simpson_method_seq::TestTaskSequential::findCoeff(int count_step) {
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

void kolokolova_d_integral_simpson_method_seq::TestTaskSequential::multiplyCoeffandFunctionValue(
    std::vector<double>& function_val, const std::vector<double>& coeff_vec, int a) {
  int coeff_vec_size = coeff_vec.size();
  int function_vec_size = function_val.size();

  // initial multiplication
  for (int i = 0; i < function_vec_size; ++i) {
    function_val[i] *= coeff_vec[i % coeff_vec_size];
  }

  // perform additional iterations on a
  for (int iteration = 1; iteration < a; ++iteration) {
    for (int i = 0; i < function_vec_size; ++i) {
      int blockSize = iteration * coeff_vec_size;
      int currentNIndex = (i / blockSize) % coeff_vec_size;
      function_val[i] *= coeff_vec[currentNIndex];
    }
  }
}

double kolokolova_d_integral_simpson_method_seq::TestTaskSequential::createOutputResult(
    std::vector<double> vec, std::vector<double> size_steps) {
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
  sum /= pow(3, nums_variables);

  return sum;
}
bool kolokolova_d_integral_simpson_method_seq::TestTaskSequential::checkBorders(std::vector<int> vec) {
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i] > vec[i + 1]) return false;
    i++;
  }
  return true;
}