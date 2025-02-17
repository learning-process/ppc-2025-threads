#include "seq/shulpin_i_Jarvis_passage/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

int shulpin_i_Jarvis_seq::JarvisSequential::orientation(const Point& p, const Point& q, const Point& r) {
  int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
  if (val == 0) return 0;
  return (val > 0) ? 1 : 2;
}

void shulpin_i_Jarvis_seq::JarvisSequential::makeJarvisPassage(std::vector<shulpin_i_Jarvis_seq::Point>& input_jar,
                                                               std::vector<shulpin_i_Jarvis_seq::Point>& output_jar) {
  int total = input_jar.size();
  std::vector<bool> used(total, false);
  output_jar.clear();

  int start = std::min_element(input_jar.begin(), input_jar.end(),
                               [](const auto& a, const auto& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }) -
              input_jar.begin();

  int active = start;

  do {
    output_jar.emplace_back(input_jar[active]);
    used[active] = true;
    int candidate = (active + 1) % total;

    for (int index = 0; index < total; ++index) {
      if (!used[index] && orientation(input_jar[active], input_jar[index], input_jar[candidate]) == 2) {
        candidate = index;
      }
    }

    active = candidate;

  } while (active != start);
}

bool shulpin_i_Jarvis_seq::JarvisSequential::PreProcessingImpl() {
  std::vector<shulpin_i_Jarvis_seq::Point> tmp_input;
    
  shulpin_i_Jarvis_seq::Point* tmp_data = reinterpret_cast<shulpin_i_Jarvis_seq::Point*>(task_data->inputs[0]);
  int tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input = tmp_input;

  int output_size = task_data->outputs_count[0];
  output.resize(output_size);

  return true;
}

bool shulpin_i_Jarvis_seq::JarvisSequential::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_Jarvis_seq::JarvisSequential::RunImpl() { 
  makeJarvisPassage(input, output);
  return true;
}

bool shulpin_i_Jarvis_seq::JarvisSequential::PostProcessingImpl() { 
  int* result = reinterpret_cast<int*>(task_data->outputs[0]);
  std::copy(reinterpret_cast<int*>(output.data()), reinterpret_cast<int*>(output.data() + output.size()), result);
  return true;
}
