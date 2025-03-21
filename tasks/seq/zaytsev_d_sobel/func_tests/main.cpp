#include <gtest/gtest.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/zaytsev_d_sobel/include/ops_seq.hpp"

static std::vector<int> matToVector(const cv::Mat &img) {
  std::vector<int> vec;
  
  vec.reserve(img.rows * img.cols);
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      vec.push_back(static_cast<int>(img.at<uchar>(i, j)));
    }
  }
  return vec;
}

TEST(zaytsev_d_sobel_seq, test_validation_fail) {
  std::vector<int> in(9, 0);
  std::vector<int> out(10, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  zaytsev_d_sobel_seq::TestTaskSequential task(task_data);
  ASSERT_EQ(task.Validation(), false);
}

TEST(zaytsev_d_sobel_seq, SobelEdgeDetection_5x5) {
  constexpr size_t kSize = 5;
  std::vector<int> input = {0, 0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 10, 10, 10, 0, 0, 10, 10, 10, 0, 0, 0, 0, 0, 0};
  std::vector<int> expected_output = {0,  0, 0, 0,  0,  0,  42, 40, 42, 0, 0, 40, 0, 40, 0, 0, 42, 40, 42, 0,  0,  0,  0, 0, 0};
  std::vector<int> output(kSize * kSize, 0);
  
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.push_back(output.size());

  zaytsev_d_sobel_seq::TestTaskSequential sobel_task(task_data);
  ASSERT_TRUE(sobel_task.Validation());
  sobel_task.PreProcessing();
  sobel_task.Run();
  sobel_task.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(zaytsev_d_sobel_seq, SobelEdgeDetection_UniformImage) {
  constexpr size_t kSize = 5;
  std::vector<int> input(kSize * kSize, 10);
  std::vector<int> expected_output(kSize * kSize, 0);
  std::vector<int> output(kSize * kSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.push_back(output.size());

  zaytsev_d_sobel_seq::TestTaskSequential sobel_task(task_data);
  ASSERT_TRUE(sobel_task.Validation());
  sobel_task.PreProcessing();
  sobel_task.Run();
  sobel_task.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(zaytsev_d_sobel_seq, SobelEdgeDetection_OpenCVImage) {
  cv::Mat inputImg = cv::imread(ppc::util::GetAbsolutePath("seq/zaytsev_d_sobel/data/inwhite.png"), cv::IMREAD_GRAYSCALE);

  cv::Mat expectedImg = cv::imread(ppc::util::GetAbsolutePath("seq/zaytsev_d_sobel/data/outputwhite.png"), cv::IMREAD_GRAYSCALE);

  std::vector<int> input = matToVector(inputImg);
  std::vector<int> expected = matToVector(expectedImg);
  std::vector<int> output(input.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.push_back(input.size());
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.push_back(output.size());

  zaytsev_d_sobel_seq::TestTaskSequential sobel_task(task_data);
  ASSERT_TRUE(sobel_task.Validation());
  sobel_task.PreProcessing();
  sobel_task.Run();
  sobel_task.PostProcessing();

  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_EQ(output[i], expected[i]);
  }
}