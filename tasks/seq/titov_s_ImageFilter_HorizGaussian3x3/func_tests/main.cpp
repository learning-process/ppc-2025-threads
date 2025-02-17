#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/titov_s_ImageFilter_HorizGaussian3x3/include/ops_seq.hpp"

TEST(titov_s_image_filter_horiz_gaussian3x3_seq, test_10_diag1) {
  constexpr size_t kwidth = 10;
  constexpr size_t kheight = 10;
  std::vector<double> input_image(kwidth * kheight, 0.0);
  std::vector<double> output_image(kwidth * kheight, 0.0);
  std::vector<double> expected_output(kwidth * kheight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kheight; ++i) {
    input_image[(i * kwidth) + i] = 1;
  }

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth; ++j) {
      if (((j == 0 && i == 0) || i == j) || (j == kwidth - 1 && i == kheight - 1)) {
        expected_output[((i * kwidth)) + j] = 0.5;
      } else if (j == i + 1 || j == i - 1) {
        expected_output[((i * kwidth)) + j] = 0.25;
      }
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth; ++j) {
      ASSERT_NEAR(output_image[((i * kwidth)) + j], expected_output[((i * kwidth)) + j], 1e-5);
    }
  }
}

TEST(titov_s_image_filter_horiz_gaussian3x3_seq, test_10_1) {
  constexpr size_t kwidth = 10;
  constexpr size_t kheight = 10;
  std::vector<double> input_image(kwidth * kheight, 1.0);
  std::vector<double> output_image(kwidth * kheight, 1.0);
  std::vector<double> expected_output(kwidth * kheight, 1.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth; ++j) {
      if (j == 0 || j == kwidth - 1) {
        expected_output[((i * kwidth)) + j] = 0.75;
      } else {
        expected_output[((i * kwidth)) + j] = 1.0;
      }
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth; ++j) {
      ASSERT_NEAR(output_image[((i * kwidth)) + j], expected_output[((i * kwidth)) + j], 1e-5);
    }
  }
}

TEST(titov_s_image_filter_horiz_gaussian3x3_seq, test_10_vertical_lines) {
  constexpr size_t kwidth = 10;
  constexpr size_t kheight = 10;
  std::vector<double> input_image(kwidth * kheight, 0.0);
  std::vector<double> output_image(kwidth * kheight, 0.0);
  std::vector<double> expected_output(kwidth * kheight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kheight; ++i) {
    input_image[((i * kwidth)) + 2] = 1.0;
    input_image[((i * kwidth)) + 7] = 1.0;
  }

  for (size_t i = 0; i < kheight; ++i) {
    expected_output[((i * kwidth)) + 1] = 0.25;
    expected_output[((i * kwidth)) + 2] = 0.5;
    expected_output[((i * kwidth)) + 3] = 0.25;
    expected_output[((i * kwidth)) + 6] = 0.25;
    expected_output[((i * kwidth)) + 7] = 0.5;
    expected_output[((i * kwidth)) + 8] = 0.25;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth; ++j) {
      ASSERT_NEAR(output_image[((i * kwidth)) + j], expected_output[((i * kwidth)) + j], 1e-5);
    }
  }
}

TEST(titov_s_image_filter_horiz_gaussian3x3_seq, test_horizontal_lines) {
  constexpr size_t kwidth = 10;
  constexpr size_t kheight = 10;
  std::vector<double> input_image(kwidth * kheight, 0.0);
  std::vector<double> output_image(kwidth * kheight, 0.0);
  std::vector<double> expected_output(kwidth * kheight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t j = 0; j < kwidth; ++j) {
    input_image[(2 * kwidth) + j] = 1.0;
    input_image[(7 * kwidth) + j] = 1.0;
  }

  expected_output[2 * kwidth] = expected_output[(3 * kwidth) - 1] = expected_output[7 * kwidth] =
      expected_output[(8 * kwidth) - 1] = 0.75;
  for (size_t i = 1; i < kwidth - 1; ++i) {
    expected_output[(2 * kwidth) + i] = 1.0;
    expected_output[(7 * kwidth) + i] = 1.0;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth; ++j) {
      ASSERT_NEAR(output_image[((i * kwidth)) + j], expected_output[((i * kwidth)) + j], 1e-5);
    }
  }
}

TEST(titov_s_image_filter_horiz_gaussian3x3_seq, test_noise) {
  constexpr size_t kwidth = 10;
  constexpr size_t kheight = 10;
  std::vector<double> input_image(kwidth * kheight, 0.0);
  std::vector<double> output_image(kwidth * kheight, 0.0);
  std::vector<double> expected_output(kwidth * kheight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth; ++j) {
      input_image[((i * kwidth)) + j] = (i + j) % 2;
    }
  }

  for (size_t i = 0; i < kheight; ++i) {
    if (i % 2 == 0) {
      for (size_t j = 0; j < kwidth; ++j) {
        expected_output[((i * kwidth)) + j] = 0.5;
      }
      expected_output[(i * kwidth)] = 0.25;
    } else {
      for (size_t j = 0; j < kwidth; ++j) {
        expected_output[((i * kwidth)) + j] = 0.5;
      }
      expected_output[((i + 1) * kwidth) - 1] = 0.25;
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth; ++j) {
      ASSERT_NEAR(output_image[((i * kwidth)) + j], expected_output[((i * kwidth)) + j], 1e-5);
    }
  }
}

TEST(titov_s_image_filter_horiz_gaussian3x3_seq, test_empty_image) {
  constexpr size_t kwidth = 10;
  constexpr size_t kheight = 10;
  std::vector<double> input_image(kwidth * kheight, 0.0);
  std::vector<double> output_image(kwidth * kheight, 0.0);
  std::vector<double> expected_output(kwidth * kheight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth; ++j) {
      ASSERT_NEAR(output_image[(i * kwidth) + j], expected_output[(i * kwidth) + j], 1e-5);
    }
  }
}

TEST(titov_s_image_filter_horiz_gaussian3x3_seq, test_sharp_transitions) {
  constexpr size_t kwidth = 10;
  constexpr size_t kheight = 10;
  std::vector<double> input_image(kwidth * kheight, 0.0);
  std::vector<double> output_image(kwidth * kheight, 0.0);
  std::vector<double> expected_output(kwidth * kheight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth / 2; ++j) {
      input_image[(i * kwidth) + j] = 0.0;
    }
    for (size_t j = kwidth / 2; j < kwidth; ++j) {
      input_image[(i * kwidth) + j] = 1.0;
    }
  }

  for (size_t i = 0; i < kheight; ++i) {
    expected_output[(i * kwidth) + 4] = 0.25;
    expected_output[(i * kwidth) + 5] = 0.75;
    expected_output[(i * kwidth) + 6] = 1.0;
    expected_output[(i * kwidth) + 7] = 1.0;
    expected_output[(i * kwidth) + 8] = 1.0;
    expected_output[(i * kwidth) + 9] = 0.75;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth; ++j) {
      ASSERT_NEAR(output_image[(i * kwidth) + j], expected_output[(i * kwidth) + j], 1e-5);
    }
  }
}

TEST(titov_s_image_filter_horiz_gaussian3x3_seq, test_smooth_gradients) {
  constexpr size_t kwidth = 10;
  constexpr size_t kheight = 10;
  std::vector<double> input_image(kwidth * kheight, 0.0);
  std::vector<double> output_image(kwidth * kheight, 0.0);
  std::vector<double> expected_output(kwidth * kheight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth; ++j) {
      input_image[(i * kwidth) + j] = expected_output[(i * kwidth) + j] = static_cast<double>(j) / (kwidth - 1);
    }
  }

  for (size_t i = 0; i < kheight; ++i) {
    expected_output[(i * kwidth)] = 0.03;
    expected_output[((i + 1) * kwidth) - 1] = 0.72;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth; ++j) {
      ASSERT_NEAR(output_image[(i * kwidth) + j], expected_output[(i * kwidth) + j], 0.5);
    }
  }
}

TEST(titov_s_image_filter_horiz_gaussian3x3_seq, test_all_max) {
  constexpr size_t kwidth = 10;
  constexpr size_t kheight = 10;
  std::vector<double> input_image(kwidth * kheight, 255.0);
  std::vector<double> output_image(kwidth * kheight, 0.0);
  std::vector<double> expected_output(kwidth * kheight, 255.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kheight; ++i) {
    expected_output[(i * kwidth)] = 191.25;
    expected_output[((i + 1) * kwidth) - 1] = 191.25;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < kheight; ++i) {
    for (size_t j = 0; j < kwidth; ++j) {
      ASSERT_NEAR(output_image[(i * kwidth) + j], expected_output[(i * kwidth) + j], 1e-5);
    }
  }
}
