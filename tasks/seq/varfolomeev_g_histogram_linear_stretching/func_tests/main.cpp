#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/varfolomeev_g_histogram_linear_stretching/include/ops_seq.hpp"

namespace {
std::vector<uint8_t> GetRandomImage(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dis(0, 255);
  std::vector<uint8_t> res(sz);

  for (int i = 0; i < sz; ++i) {
    res[i] = dis(gen);
  }

  return res;
}
}  // namespace

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_manual_opencv_64x64) {
  // loading template orginal img
  cv::Mat inputImage =
      cv::imread(ppc::util::GetAbsolutePath("seq/varfolomeev_g_histogram_linear_stretching/data/cobble_orig.jpg"),
                 cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(inputImage.empty());

  // loading template modified img
  cv::Mat referenceImage =
      cv::imread(ppc::util::GetAbsolutePath("seq/varfolomeev_g_histogram_linear_stretching/data/cobble_mod.jpg"),
                 cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(referenceImage.empty());

  // images validation
  ASSERT_EQ(inputImage.size(), referenceImage.size());
  ASSERT_EQ(inputImage.type(), referenceImage.type());

  std::vector<uint8_t> inputVector(inputImage.total());
  std::vector<uint8_t> outputVector(inputImage.total());
  std::vector<uint8_t> expectedOutputVector(referenceImage.total());

  for (size_t i = 0; i < inputImage.total(); ++i) {
    inputVector[i] = static_cast<uint8_t>(inputImage.data[i]);
    expectedOutputVector[i] = static_cast<uint8_t>(referenceImage.data[i]);
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  task_data_seq->inputs_count.emplace_back(inputVector.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputVector.data()));
  task_data_seq->outputs_count.emplace_back(outputVector.size());

  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  EXPECT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  // check quality
  cv::Mat resultImage(inputImage.rows, inputImage.cols, CV_8UC1, outputVector.data());
  double mse = cv::norm(resultImage, referenceImage, cv::NORM_L2) / (resultImage.rows * resultImage.cols);
  double psnr = 10.0 * log10((255.0 * 255.0) / mse);
  EXPECT_GT(psnr, 30.0);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_manual_9) {
  // Create data
  std::vector<uint8_t> in = {100, 50, 200, 75, 150, 25, 175, 125, 225};
  std::vector<uint8_t> out(in.size());
  std::vector<uint8_t> expected_out = {96, 32, 223, 64, 159, 0, 191, 128, 255};
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_manual_10) {
  // Create data
  std::vector<uint8_t> in = {30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
  std::vector<uint8_t> out(in.size());
  std::vector<uint8_t> expected_out = {0, 28, 57, 85, 113, 142, 170, 198, 227, 255};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_manual_25) {
  // Create data
  std::vector<uint8_t> in = {12,  25,  88, 14,  65,  79, 64, 128, 122, 220, 138, 147, 215,
                             211, 189, 89, 167, 181, 2,  12, 34,  25,  85,  75,  77};
  std::vector<uint8_t> out(in.size());
  std::vector<uint8_t> expected_out = {12,  27,  101, 14,  74,  90, 73, 147, 140, 255, 159, 170, 249,
                                       244, 219, 102, 193, 209, 0,  12, 37,  27,  97,  85,  88};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_stretched) {
  // Create data (already has 0 and 255 in it)
  std::vector<uint8_t> in = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 255, 19, 20, 21, 22, 23};
  std::vector<uint8_t> out(in.size());
  std::vector<uint8_t> expected_out = in;  // nothing changes

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_boundary_values) {
  // Create data
  std::vector<uint8_t> in = {0, 1, 254, 255};
  std::vector<uint8_t> out(in.size());
  std::vector<uint8_t> expected_out = {0, 1, 254, 255};  // nothing changes
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_flat) {
  // Create data
  std::vector<uint8_t> in = {128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                             128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
  std::vector<uint8_t> out(in.size());
  std::vector<uint8_t> expected_out = in;  // nothing changes

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_single_non_flat_pixel) {
  // Create data
  std::vector<uint8_t> in = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0};
  std::vector<uint8_t> out(in.size());
  std::vector<uint8_t> expected_out = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0};
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_single_non_flat_pixel_2) {
  // Create data
  std::vector<uint8_t> in = {36, 36, 36, 36, 36,  36, 36, 36, 36, 36, 36, 36, 36, 36,
                             36, 36, 36, 36, 128, 36, 36, 36, 36, 36, 36, 36, 36, 36};
  std::vector<uint8_t> out(in.size());
  std::vector<uint8_t> expected_out = {0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_10k_generated) {
  // Create data
  const int sz = 10000;
  std::vector<uint8_t> in = GetRandomImage(sz);
  std::vector<uint8_t> out(sz);
  std::vector<uint8_t> expected_out(sz);

  int min_pix = *std::ranges::min_element(in);
  int max_pix = *std::ranges::max_element(in);
  if (min_pix != max_pix) {
    for (size_t i = 0; i < in.size(); ++i) {
      expected_out[i] = static_cast<int>(std::round((in[i] - min_pix) / static_cast<double>(max_pix - min_pix) * 255));
    }
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_empty) {
  // Create data
  std::vector<uint8_t> in;
  std::vector<uint8_t> out;
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_in_out_NE) {
  // Create data
  std::vector<uint8_t> in(12, 0);
  std::vector<uint8_t> out(11, 0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}
