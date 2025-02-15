#include <gtest/gtest.h>

#include <cstdint>
#include <format>
#include <memory>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "opencv2/core/persistence.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "seq/zaitsev_a_bw_labeling/include/ops_seq.hpp"

using Params = std::tuple<std::string>;

namespace {

class ZaitsevALabelingTestSeq : public ::testing::TestWithParam<Params> {
 protected:
};

TEST_P(ZaitsevALabelingTestSeq, returns_correct_on_real_image) {
  const auto& [filename] = GetParam();

  cv::Mat img8 = cv::imread(ppc::util::GetAbsolutePath(std::format("seq/zaitsev_a_bw_labeling/data/{}.bmp", filename)),
                            cv::IMREAD_GRAYSCALE);
  cv::Mat img;
  cv::threshold(img8, img, 128, 1, cv::THRESH_BINARY_INV);
  const int width = img.cols;
  const int height = img.rows;
  img = img.reshape(1, 1);

  std::vector<int> in(width * height);
  img.row(0).copyTo(in);
  std::vector<int> out(width * height, 0);

  cv::FileStorage fs(ppc::util::GetAbsolutePath(std::format("seq/zaitsev_a_bw_labeling/data/{}.json", filename)),
                     cv::FileStorage::READ);
  cv::Mat exp;
  fs["exp"] >> exp;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  zaitsev_a_labeling::Labeler task(task_data_seq);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  cv::Mat out_cv(out);
  out_cv = out_cv.reshape(1, height);

  cv::Mat compared;
  cv::bitwise_xor(out_cv, exp, compared);

  EXPECT_EQ(cv::countNonZero(compared), 0);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(zaitsev_a_labeling_test_seq, ZaitsevALabelingTestSeq, ::testing::Values(
  Params("bodies"),
  Params("kittens"),
  Params("rombs"),
  Params("rosenrot"),
  Params("small")
  )
);
//clang-format on

} // namespace