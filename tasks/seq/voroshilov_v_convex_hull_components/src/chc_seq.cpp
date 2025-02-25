#include "seq/voroshilov_v_convex_hull_components/include/chc_seq.hpp"

#include <algorithm>
#include <vector>

#include "seq/voroshilov_v_convex_hull_components/include/chc.hpp"

using namespace voroshilov_v_convex_hull_components_seq;

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::ValidationImpl() {
  int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  int height = *ptr;
  ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  int width = *ptr;
  return height > 0 && width > 0;
}

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::PreProcessingImpl() {
  int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  int height = *ptr;

  ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  int width = *ptr;

  std::vector<int> pixels(task_data->inputs_count[0]);
  ptr = reinterpret_cast<int *>(task_data->inputs[2]);
  std::copy(ptr, ptr + task_data->inputs_count[0], pixels.begin());

  Image image(height, width, pixels);
  imageIn_ = image;

  return true;
}

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::RunImpl() {
  std::vector<Component> components = FindComponents(imageIn_);

  hullsOut_ = QuickHullAll(components);

  return true;
}

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::PostProcessingImpl() {
  std::vector<int> out = PackHulls(hullsOut_);

  std::ranges::copy(out.data(), reinterpret_cast<int *>(task_data->outputs[0]));
  task_data->outputs_count[0] = out.size();

  return true;
}
