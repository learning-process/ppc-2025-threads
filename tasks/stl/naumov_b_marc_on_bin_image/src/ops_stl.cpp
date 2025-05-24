#include "stl/naumov_b_marc_on_bin_image/include/ops_stl.hpp"

#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"


bool naumov_b_marc_on_bin_image_stl::TestTaskSTL::PreProcessingImpl() {
  
  return true;
}

bool naumov_b_marc_on_bin_image_stl::TestTaskSTL::ValidationImpl() {
  // Check equality of counts elements
  return true;
}

bool naumov_b_marc_on_bin_image_stl::TestTaskSTL::RunImpl() {

  return true;
}

bool naumov_b_marc_on_bin_image_stl::TestTaskSTL::PostProcessingImpl() {

  return true;
}
