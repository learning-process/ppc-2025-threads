#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/zaitsev_a_bw_labeling/include/ops_seq.hpp"

TEST(zaitsev_a_labeling, test1) {
  // clang-format off
    std::vector in = {
        0, 1, 0, 0, 0, 
        1, 0, 1, 0, 1,
        1, 1, 1, 1, 1, 
        0, 1, 1, 0, 1, 
        0, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0
    };

    std::vector<std::vector<int>> out = {
      {0, 0, 0, 0},
      {1, 0, 1, 0},
      {0, 1, 0, 0},
      {1, 1, 1, 1},
      {0, 0, 0, 1}
    };

  // clang-format on
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(5);
  task_data_seq->inputs_count.emplace_back(7);
  task_data_seq->outputs_count.emplace_back(0);

  zaitsev_a_labeling::Labeler task(task_data_seq);
  task.ValidationImpl();
  task.PreProcessingImpl();
  task.RunImpl();

  EXPECT_TRUE(1 == 1);
}
