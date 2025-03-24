#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/zaytsev_d_sobel/include/ops_seq.hpp"

#ifndef _WIN32
namespace {
std::vector<int> MatToVector(const cv::Mat &img) {
  std::vector<int> vec;

  vec.reserve(img.rows * img.cols);
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      vec.push_back(static_cast<int>(img.at<uchar>(i, j)));
    }
  }
  return vec;
}
}  // namespace
#endif

TEST(zaytsev_d_sobel_seq, test_validation_fail1) {
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

TEST(zaytsev_d_sobel_seq, test_validation_fail2) {
  std::vector<int> in(5, 0);
  std::vector<int> out(5, 0);

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
  std::vector<int> expected_output = {0,  0, 0, 0,  0,  0,  42, 40, 42, 0, 0, 40, 0,
                                      40, 0, 0, 42, 40, 42, 0,  0,  0,  0, 0, 0};
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

TEST(zaytsev_d_sobel_seq, Sobel_12x12) {
  constexpr size_t kSize = 12;
  std::vector<int> input = {0,   0,   0,   0,   0,   0,   50,  50,  50,  0,   0,   0,   0,   50,  100, 150, 200, 255,
                            255, 200, 150, 100, 50,  0,   0,   100, 150, 200, 255, 255, 255, 255, 200, 150, 100, 0,
                            0,   150, 200, 255, 255, 255, 255, 255, 255, 200, 150, 0,   0,   200, 255, 255, 255, 255,
                            255, 255, 255, 255, 200, 0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,
                            50,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 50,  100, 200, 255, 255, 255, 255,
                            255, 255, 255, 255, 200, 100, 150, 150, 200, 255, 255, 255, 255, 255, 255, 200, 150, 150,
                            200, 100, 150, 200, 255, 255, 255, 200, 150, 100, 150, 200, 255, 50,  100, 150, 200, 255,
                            200, 150, 100, 50,  50,  255, 0,   0,   0,   50,  100, 150, 150, 100, 50,  0,   0,   0};
  std::vector<int> expected_output = {
      0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,
      0, 255, 255, 255, 255, 77,  77,  255, 255, 255, 255, 0, 0, 255, 255, 255, 77,  0,   0,   77,  255, 255, 255, 0,
      0, 255, 255, 77,  0,   0,   0,   0,   77,  255, 255, 0, 0, 255, 77,  0,   0,   0,   0,   0,   0,   77,  255, 0,
      0, 255, 77,  0,   0,   0,   0,   0,   0,   77,  255, 0, 0, 255, 255, 77,  0,   0,   0,   0,   77,  255, 255, 0,
      0, 255, 255, 255, 77,  0,   77,  239, 255, 255, 219, 0, 0, 255, 255, 255, 255, 110, 255, 255, 255, 255, 255, 0,
      0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};
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

#ifndef _WIN32
TEST(zaytsev_d_sobel_seq, SobelEdgeDetection_OpenCVImage) {
  cv::Mat input_img =
      cv::imread(ppc::util::GetAbsolutePath("seq/zaytsev_d_sobel/data/inwhite.png"), cv::IMREAD_GRAYSCALE);
  cv::Mat expected_img =
      cv::imread(ppc::util::GetAbsolutePath("seq/zaytsev_d_sobel/data/outputwhite.png"), cv::IMREAD_GRAYSCALE);

  std::vector<int> input = MatToVector(input_img);
  std::vector<int> expected = MatToVector(expected_img);
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
#endif
