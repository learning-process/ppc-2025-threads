#include "all/naumov_b_marc_on_bin_image/include/ops_all.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"


bool naumov_b_marc_on_bin_image_all::TestTaskALL::PreProcessingImpl() {
  return true;
}

bool naumov_b_marc_on_bin_image_all::TestTaskALL::ValidationImpl() {
  return true;
}

bool naumov_b_marc_on_bin_image_all::TestTaskALL::RunImpl() {
  return true;
}

bool naumov_b_marc_on_bin_image_all::TestTaskALL::PostProcessingImpl() {

  return true;
}
