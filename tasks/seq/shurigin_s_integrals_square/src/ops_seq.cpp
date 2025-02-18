#include "seq/shurigin_s_integrals_square/include/ops_seq.hpp"

#include < cmath>
#include < cstddef>
#include < iostream>
#include < vector>

namespace shurigin_s_integrals_square_seq {

Integral::Integral(std::shared_ptr<ppc::core::TaskData> taskData_)
    : Task(taskData_),
      down_limit(0.0),
      up_limit(0.0),
      count(0),
      result_(0.0),
      func_(nullptr),
      taskData(std::move(taskData_)) {}

bool Integral::PreProcessingImpl() {
  try {
    if (!taskData || taskData->inputs.empty() || taskData->inputs[0] == nullptr) {
      throw std::invalid_argument("Неверные входные данные.");
    }

    double* inputs = reinterpret_cast<double*>(taskData->inputs[0]);

    if (!inputs) {
      throw std::invalid_argument("Указатель inputs равен nullptr.");
    }

    down_limit = inputs[0];
    up_limit = inputs[1];
    count = static_cast<int>(inputs[2]);

    if (count <= 0) {
      throw std::invalid_argument("Количество разбиений должно быть положительным числом.");
    }
    if (up_limit <= down_limit) {
      throw std::invalid_argument("Верхний предел должен быть больше нижнего предела.");
    }

    result_ = 0.0;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in PreProcessingImpl: " << e.what() << std::endl;
    return false;
  }
}

bool Integral::ValidationImpl() {
  try {
    if (!taskData) {
      throw std::invalid_argument("taskData is null");
    }
    if (taskData->inputs_count.empty() || taskData->outputs_count.empty()) {
      throw std::invalid_argument("Неверные размеры входных или выходных данных.");
    }
    if (taskData->inputs_count[0] != 3 * sizeof(double)) {
      throw std::invalid_argument("Ожидалось 3 числа типа double во входных данных.");
    }
    if (taskData->outputs_count[0] != sizeof(double)) {
      throw std::invalid_argument("Ожидалось одно число типа double в выходных данных.");
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in ValidationImpl: " << e.what() << std::endl;
    return false;
  }
}

bool Integral::RunImpl() {
  try {
    if (!func_) {
      throw std::runtime_error("Функция не задана. Используйте setFunction() для задания функции.");
    }
    result_ = compute(func_, down_limit, up_limit, count);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in RunImpl: " << e.what() << std::endl;
    return false;
  }
}

bool Integral::PostProcessingImpl() {
  try {
    if (!taskData || taskData->outputs.empty() || taskData->outputs[0] == nullptr) {
      throw std::invalid_argument("Неверные выходные данные.");
    }
    double* outputs = reinterpret_cast<double*>(taskData->outputs[0]);
    outputs[0] = result_;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in PostProcessingImpl: " << e.what() << std::endl;
    return false;
  }
}

void Integral::setFunction(const std::function<double(double)>& func) { func_ = func; }

double Integral::compute(const std::function<double(double)>& f, double a, double b, int n) {
  double step = (b - a) / n;
  double area = 0.0;

  for (int i = 0; i < n; ++i) {
    double x = a + (i + 0.5) * step;
    area += f(x) * step;
  }

  return area;
}

}  // namespace shurigin_s_integrals_square_seq
