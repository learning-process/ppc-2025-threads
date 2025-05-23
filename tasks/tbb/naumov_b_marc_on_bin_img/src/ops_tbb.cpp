#include "tbb/naumov_b_marc_on_bin_img/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"


bool naumov_b_marc_on_bin_img_tbb::TestTaskTBB::PreProcessingImpl() {
  return true;
}

bool naumov_b_marc_on_bin_img_tbb::TestTaskTBB::ValidationImpl() {

  return true;
}

bool naumov_b_marc_on_bin_img_tbb::TestTaskTBB::RunImpl() {
  
  return true;
}

bool naumov_b_marc_on_bin_img_tbb::TestTaskTBB::PostProcessingImpl() {
  
  return true;
}
