#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/frolova_e_Sobel_filter/include/ops_seq.hpp"

TEST(frolova_e_Sobel_filter_seq, test_1) {
  std::vector<int> value_1 = {10, 10};
  std::vector<int> pict = {
      172, 47,  117, 192, 67,  251, 195, 103, 9,   211, 21,  242, 36,  87,  70,  216, 88,  140, 58,  193, 230, 39,
      87,  174, 88,  81,  165, 25,  77,  72,  9,   148, 115, 208, 243, 197, 254, 79,  175, 192, 82,  99,  216, 177,
      243, 29,  147, 147, 142, 167, 32,  193, 9,   185, 127, 32,  31,  202, 244, 151, 163, 254, 203, 114, 183, 28,
      34,  128, 128, 164, 53,  133, 38,  232, 244, 17,  79,  132, 105, 42,  186, 31,  120, 1,   65,  231, 169, 57,
      35,  102, 119, 11,  174, 82,  91,  128, 142, 99,  53,  140, 121, 170, 84,  203, 68,  6,   196, 47,  127, 244,
      131, 204, 100, 180, 232, 78,  143, 148, 227, 186, 23,  207, 141, 117, 85,  48,  49,  69,  169, 163, 192, 95,
      197, 94,  0,   113, 178, 36,  162, 48,  93,  131, 98,  42,  205, 112, 231, 149, 201, 127, 0,   138, 114, 43,
      186, 127, 23,  187, 130, 121, 98,  62,  163, 222, 123, 195, 82,  174, 227, 148, 209, 50,  155, 14,  41,  58,
      193, 36,  10,  86,  43,  104, 11,  2,   51,  80,  32,  182, 128, 38,  19,  174, 42,  115, 184, 188, 232, 77,
      30,  24,  125, 2,   3,   94,  226, 107, 13,  112, 40,  72,  19,  95,  72,  154, 194, 248, 180, 67,  236, 61,
      14,  96,   4,   195, 237, 139, 252, 86,  205, 121, 109, 75,  184, 16,  152, 157, 149, 110, 25,  208, 188, 121,
      118, 117, 189, 83,  161, 104, 160, 228, 251, 251, 121, 70,  213, 31,  13,  71,  184, 152, 79,  41,  18,  40,
      182, 207, 11,  166, 111, 93,  249, 129, 223, 118, 44,  216, 125, 24,  67,  210, 239, 3,   234, 204, 230, 35,
      214, 254, 189, 197, 215, 43,  32,  11,  104, 212, 138, 182, 235, 165};

  std::vector<int> res(100, 0);

  std::vector<int> reference = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 146, 255, 151, 138, 155, 244,
                                135, 255, 255, 255, 255, 255, 255, 95,  206, 171, 239, 221, 255, 255, 232, 116, 218,
                                84,  107, 118, 46,  194, 255, 255, 157, 179, 188, 69,  39,  105, 153, 255, 255, 255,
                                129, 70,  255, 205, 132, 255, 255, 246, 255, 255, 209, 183, 189, 255, 153, 255, 134,
                                244, 255, 255, 255, 255, 255, 255, 238, 255, 234, 168, 255, 255, 184, 156, 255, 255,
                                104, 196, 135, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(value_1.data()));
  task_data_seq->inputs_count.emplace_back(value_1.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data_seq->inputs_count.emplace_back(pict.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  // Create Task
  frolova_e_Sobel_filter_seq::SobelFilterSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(reference, res);
}

TEST(frolova_e_Sobel_filter_seq, small_image_1) {
  std::vector<int> value_1 = {3, 3};
  std::vector<int> pict = {172, 47, 117, 192, 67, 251, 195, 103, 9, 211, 21, 242, 3, 87, 70,
      216, 88, 140, 58, 193, 230, 39, 87, 174, 88, 81, 165};

  std::vector<int> res(9, 0);

  std::vector<int> reference = {255, 255, 255, 255, 53, 255, 255, 255, 255};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(value_1.data()));
  task_data_seq->inputs_count.emplace_back(value_1.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data_seq->inputs_count.emplace_back(pict.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  // Create Task
  frolova_e_Sobel_filter_seq::SobelFilterSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(reference, res);
}

TEST(frolova_e_Sobel_filter_seq, small_image_2) {
  std::vector<int> value_1 = {5, 5};
  std::vector<int> pict = {172, 47,  117, 192, 67,  251, 195, 103, 9,   211, 21,  242, 36,  87,  70,  216, 88,  140, 58,
                           193, 230, 39,  87,  174, 88,  81,  165, 25,  77,  72,  9,   148, 115, 208, 243, 197, 254, 79,
                           175, 192, 82,  99,  216, 177, 243, 29,  147, 147, 142, 167, 32,  193, 9,   185, 127, 32,  31,
                           202, 244, 151, 163, 254, 203, 114, 183, 28,  34,  128, 128, 164, 53,  133, 38,  232, 244};

  std::vector<int> res(25, 0);

  std::vector<int> reference = {255, 255, 255, 255, 255, 255, 239, 255, 180, 255, 255, 43, 255,
                                242, 255, 255, 162, 255, 255, 255, 255, 255, 255, 255, 255};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(value_1.data()));
  task_data_seq->inputs_count.emplace_back(value_1.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
  task_data_seq->inputs_count.emplace_back(pict.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size());

  // Create Task
  frolova_e_Sobel_filter_seq::SobelFilterSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(reference, res);
}


TEST(frolova_e_Sobel_filter_seq, one_pixel) {
  std::vector<int> value_1 = {1, 1};
  std::vector<int> pict = {100, 0, 0};

  std::vector<int> res(1, 0);

  std::vector<int> reference = {0};
  
    // Create task_data
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(value_1.data()));
    task_data_seq->inputs_count.emplace_back(value_1.size());

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data_seq->inputs_count.emplace_back(pict.size());

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_seq->outputs_count.emplace_back(res.size());

    // Create Task
    frolova_e_Sobel_filter_seq::SobelFilterSequential test_task_sequential(task_data_seq);
    ASSERT_EQ(test_task_sequential.Validation(), true);
    test_task_sequential.PreProcessing();
    test_task_sequential.Run();
    test_task_sequential.PostProcessing();
    EXPECT_EQ(reference, res);
}

//______ASSERT_FALSE________________

TEST(frolova_e_Sobel_filter_seq, not_correct_value) {
    std::vector<int> value_1 = {-1, 1};
    std::vector<int> pict = {100, 0, 0};

    std::vector<int> res(1, 0);

    std::vector<int> reference = {0};

    // Create task_data
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(value_1.data()));
    task_data_seq->inputs_count.emplace_back(value_1.size());

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data_seq->inputs_count.emplace_back(pict.size());

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_seq->outputs_count.emplace_back(res.size());

    // Create Task
    frolova_e_Sobel_filter_seq::SobelFilterSequential test_task_sequential(task_data_seq);
    ASSERT_EQ(test_task_sequential.Validation(), false);

}

TEST(frolova_e_Sobel_filter_seq, vector_is_not_multiple_of_three) {
    std::vector<int> value_1 = {1, 1};
    std::vector<int> pict = {100, 0};

    std::vector<int> res(1, 0);

    std::vector<int> reference = {0};

    // Create task_data
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(value_1.data()));
    task_data_seq->inputs_count.emplace_back(value_1.size());

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data_seq->inputs_count.emplace_back(pict.size());

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_seq->outputs_count.emplace_back(res.size());

    // Create Task
    frolova_e_Sobel_filter_seq::SobelFilterSequential test_task_sequential(task_data_seq);
    ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(frolova_e_Sobel_filter_seq, vector_element_is_not_included_the_range) {
    std::vector<int> value_1 = {1, 1};
    std::vector<int> pict = {100, 0, 270};

    std::vector<int> res(1, 0);

    std::vector<int> reference = {0};

    // Create task_data
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(value_1.data()));
    task_data_seq->inputs_count.emplace_back(value_1.size());

    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(pict.data()));
    task_data_seq->inputs_count.emplace_back(pict.size());

    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    task_data_seq->outputs_count.emplace_back(res.size());

    // Create Task
    frolova_e_Sobel_filter_seq::SobelFilterSequential test_task_sequential(task_data_seq);
    ASSERT_EQ(test_task_sequential.Validation(), false);
}