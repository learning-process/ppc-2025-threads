#include "seq/burykin_m_radix/include/ops_seq.hpp"

#include <array>

bool burykin_m_radix_seq::RadixSequential::PreProcessingImpl() {
  // Считываем входные данные как массив int
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  // Инициализируем вектор для результата нужного размера
  output_.resize(input_size);
  return true;
}

bool burykin_m_radix_seq::RadixSequential::ValidationImpl() {
  // Проверяем, что число элементов на входе и выходе совпадает
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool burykin_m_radix_seq::RadixSequential::RunImpl() {
  if (input_.empty()) {
	return true;
  }

  // Будем сортировать копию входного массива
  std::vector<int> a = input_;
  std::vector<int> b(a.size());

  // Обрабатываем 4 байта (32 бита) по 8 бит за проход
  for (int shift = 0; shift < 32; shift += 8) {
    // Подсчёт частот для каждого возможного значения байта (0..255)
    std::array<int, 256> count = {0};
    for (int v : a) {
      // Извлекаем нужный байт
      unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
      // Для самого значащего байта корректируем ключ, чтобы отрицательные числа сортировались правильно
      if (shift == 24) {
        key ^= 0x80;
      }
      ++count[key];
    }

    // Вычисляем индексы для распределения (накопленная сумма)
    std::array<int, 256> index = {0};
    for (int i = 1; i < 256; ++i) {
      index[i] = index[i - 1] + count[i - 1];
    }

    // Стабильное распределение элементов по новому порядку
    for (int v : a) {
      unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
      if (shift == 24) {
        key ^= 0x80;
      }
      b[index[key]++] = v;
    }
    // Подготавливаемся к следующему проходу
    a.swap(b);
  }

  // Результат сортировки теперь находится в 'a'
  output_ = std::move(a);
  return true;
}

bool burykin_m_radix_seq::RadixSequential::PostProcessingImpl() {
  // Записываем результат в выделенную для задачи память
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
