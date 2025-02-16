#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/zaitsev_a_bw_labeling/include/ops_seq.hpp"

using Params = std::tuple<int, int, std::vector<int>, std::vector<int>>;

namespace {

class ZaitsevALabelingSeqTest : public ::testing::TestWithParam<Params> {
 protected:
};

TEST_P(ZaitsevALabelingSeqTest, returns_correct_label_map) {
  const auto& [width, height, in, expected] = GetParam();
  std::vector<int> out(in.size(), -1);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(in.data())));
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs_count.emplace_back(out.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));

  zaitsev_a_labeling::Labeler task(task_data_seq);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_EQ(expected, out);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(ZaitsevALabelingSeqTest, ZaitsevALabelingSeqTest, ::testing::Values(
    Params(
      3, 3, 
      {
        0, 1, 0, 
        1, 0, 1, 
        0, 1, 0
      },
      {
        0, 1, 0,
        2, 0, 3, 
        0, 4, 0
      }
    ),
    Params(
      4, 4,
      {
        0, 1, 1, 0,
        1, 1, 0, 1,
        1, 0, 1, 1,
        0, 1, 1, 0
      },
      {
        0, 1, 1, 0,
        1, 1, 0, 3, 
        1, 0, 3, 3, 
        0, 3, 3, 0
      }
    ), 
    Params(
      7, 7,
      {
        1, 0, 1, 0, 1, 0, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 0, 0, 0, 0, 0, 
        0, 0, 1, 1, 1, 0, 0, 
        0, 1, 1, 0, 1, 1, 0,
        1, 0, 0, 1, 0, 1, 1,
        0, 1, 1, 1, 1, 0, 0
      },
      {
        1, 0, 1, 0, 1, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 
        1, 1, 0, 0, 0, 0, 0,
        0, 0, 5, 5, 5, 0, 0,
        0, 5, 5, 0, 5, 5, 0,
        7, 0, 0, 8, 0, 5, 5, 
        0, 8, 8, 8, 8, 0, 0
      }
    ),
    Params(
      9, 9, 
      {
        0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 1, 1, 1, 0, 1, 1, 1, 0, 
        0, 1, 0, 0, 0, 0, 0, 1, 0, 
        1, 1, 0, 1, 1, 1, 0, 1, 1, 
        1, 0, 0, 1, 0, 1, 0, 0, 1, 
        1, 1, 0, 1, 1, 1, 0, 1, 1, 
        0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 1, 1, 0, 1, 1, 1, 0, 
        0, 0, 0, 1, 1, 1, 0, 0, 0
      }, 
      {
        0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 1, 1, 1, 0, 1, 1, 1, 0, 
        0, 1, 0, 0, 0, 0, 0, 1, 0, 
        1, 1, 0, 4, 4, 4, 0, 1, 1, 
        1, 0, 0, 4, 0, 4, 0, 0, 1, 
        1, 1, 0, 4, 4, 4, 0, 1, 1, 
        0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 1, 1, 0, 1, 1, 1, 0, 
        0, 0, 0, 1, 1, 1, 0, 0, 0
      }
    )
  )
);
//clang-format on

} // namespace