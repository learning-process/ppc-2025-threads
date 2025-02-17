#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/titov_s_ImageFilter_HorizGaussian3x3/include/ops_seq.hpp"

TEST(titov_s_ImageFilter_HorizGaussian3x3_seq, test_10_diag1) {
  constexpr size_t width = 10;
  constexpr size_t height = 10;
  std::vector<double> input_image(width * height, 0.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_output(width * height, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < height; ++i) {
    input_image[i * width + i] = 1;
  }

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      if ((j == 0 && i == 0) || i == j) {
        expected_output[i * width + j] = 0.5;
      } else if (j == width - 1 && i == height - 1) {
        expected_output[i * width + j] = 0.5;
      } else if (j == i + 1 || j == i - 1) {
        expected_output[i * width + j] = 0.25;
      }
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_ImageFilter_HorizGaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      ASSERT_NEAR(output_image[i * width + j], expected_output[i * width + j], 1e-5);
    }
  }
}

TEST(titov_s_ImageFilter_HorizGaussian3x3_seq, test_10_1) {
  constexpr size_t width = 10;
  constexpr size_t height = 10;
  std::vector<double> input_image(width * height, 1.0);
  std::vector<double> output_image(width * height, 1.0);
  std::vector<double> expected_output(width * height, 1.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      if (j == 0 || j == width - 1) {
        expected_output[i * width + j] = 0.75;
      } else {
        expected_output[i * width + j] = 1.0;
      }
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_ImageFilter_HorizGaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      ASSERT_NEAR(output_image[i * width + j], expected_output[i * width + j], 1e-5);
    }
  }
}

TEST(titov_s_ImageFilter_HorizGaussian3x3_seq, test_10_vertical_lines) {
  constexpr size_t width = 10;
  constexpr size_t height = 10;
  std::vector<double> input_image(width * height, 0.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_output(width * height, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < height; ++i) {
    input_image[i * width + 2] = 1.0;
    input_image[i * width + 7] = 1.0;
  }

  for (size_t i = 0; i < height; ++i) {
    expected_output[i * width + 1] = 0.25;
    expected_output[i * width + 2] = 0.5;
    expected_output[i * width + 3] = 0.25;
    expected_output[i * width + 6] = 0.25;
    expected_output[i * width + 7] = 0.5;
    expected_output[i * width + 8] = 0.25;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_ImageFilter_HorizGaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      ASSERT_NEAR(output_image[i * width + j], expected_output[i * width + j], 1e-5);
    }
  }
}

TEST(titov_s_ImageFilter_HorizGaussian3x3_seq, test_horizontal_lines) {
  constexpr size_t width = 10;
  constexpr size_t height = 10;
  std::vector<double> input_image(width * height, 0.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_output(width * height, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t j = 0; j < width; ++j) {
    input_image[2 * width + j] = 1.0;
    input_image[7 * width + j] = 1.0;
  }

  expected_output[2 * width] = expected_output[3 * width - 1] = expected_output[7 * width] =
      expected_output[8 * width - 1] = 0.75;
  for (size_t i = 1; i < width - 1; ++i) {
    expected_output[2 * width + i] = 1.0;
    expected_output[7 * width + i] = 1.0;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_ImageFilter_HorizGaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      ASSERT_NEAR(output_image[i * width + j], expected_output[i * width + j], 1e-5);
    }
  }
}

TEST(titov_s_ImageFilter_HorizGaussian3x3_seq, test_noise) {
  constexpr size_t width = 10;
  constexpr size_t height = 10;
  std::vector<double> input_image(width * height, 0.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_output(width * height, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      input_image[i * width + j] = (i + j) % 2;
    }
  }

  for (size_t i = 0; i < height; ++i) {
    if (i % 2 == 0) {
      for (size_t j = 0; j < width; ++j) {
        expected_output[i * width + j] = 0.5;
      }
      expected_output[i * width] = 0.25;
    } else {
      for (size_t j = 0; j < width; ++j) {
        expected_output[i * width + j] = 0.5;
      }
      expected_output[(i + 1) * width - 1] = 0.25;
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_ImageFilter_HorizGaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      ASSERT_NEAR(output_image[i * width + j], expected_output[i * width + j], 1e-5);
    }
  }
}

TEST(titov_s_ImageFilter_HorizGaussian3x3_seq, test_empty_image) {
  constexpr size_t width = 10;
  constexpr size_t height = 10;
  std::vector<double> input_image(width * height, 0.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_output(width * height, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_ImageFilter_HorizGaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      ASSERT_NEAR(output_image[i * width + j], expected_output[i * width + j], 1e-5);
    }
  }
}

TEST(titov_s_ImageFilter_HorizGaussian3x3_seq, test_sharp_transitions) {
  constexpr size_t width = 10;
  constexpr size_t height = 10;
  std::vector<double> input_image(width * height, 0.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_output(width * height, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width / 2; ++j) {
      input_image[i * width + j] = 0.0;
    }
    for (size_t j = width / 2; j < width; ++j) {
      input_image[i * width + j] = 1.0;
    }
  }

  for (size_t i = 0; i < height; ++i) {
    expected_output[i * width + 4] = 0.25;
    expected_output[i * width + 5] = 0.75;
    expected_output[i * width + 6] = 1.0;
    expected_output[i * width + 7] = 1.0;
    expected_output[i * width + 8] = 1.0;
    expected_output[i * width + 9] = 0.75;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_ImageFilter_HorizGaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      ASSERT_NEAR(output_image[i * width + j], expected_output[i * width + j], 1e-5);
    }
  }
}

TEST(titov_s_ImageFilter_HorizGaussian3x3_seq, test_smooth_gradients) {
  constexpr size_t width = 10;
  constexpr size_t height = 10;
  std::vector<double> input_image(width * height, 0.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_output(width * height, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      input_image[i * width + j] = expected_output[i * width + j] = static_cast<double>(j) / (width - 1);
    }
  }

  for (size_t i = 0; i < height; ++i) {
    expected_output[i * width] = 0.03;
    expected_output[(i + 1) * width - 1] = 0.72;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_ImageFilter_HorizGaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      ASSERT_NEAR(output_image[i * width + j], expected_output[i * width + j], 0.5);
    }
  }
}

TEST(titov_s_ImageFilter_HorizGaussian3x3_seq, test_all_max) {
  constexpr size_t width = 10;
  constexpr size_t height = 10;
  std::vector<double> input_image(width * height, 255.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_output(width * height, 255.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < height; ++i) {
    expected_output[i * width] = 191.25;
    expected_output[(i + 1) * width - 1] = 191.25;
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  titov_s_ImageFilter_HorizGaussian3x3_seq::ImageFilterSequential image_filter_sequential(task_data_seq);

  ASSERT_EQ(image_filter_sequential.Validation(), true);

  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      ASSERT_NEAR(output_image[i * width + j], expected_output[i * width + j], 1e-5);
    }
  }
}
