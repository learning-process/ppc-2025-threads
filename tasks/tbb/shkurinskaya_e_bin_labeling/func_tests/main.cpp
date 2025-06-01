#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "tbb/shkurinskaya_e_bin_labeling/include/ops_tbb.hpp"

TEST(shkurinskaya_e_bin_labeling_tbb, empty_input) {
  int height = 5000;
  int width = 5000;
  int size = width * height;
  // Create data
  std::vector<int> in;
  std::vector<int> out(size);
  std::vector<int> ans(size, 1);
  // Create TaskData
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  shkurinskaya_e_bin_labeling_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), false);
}

TEST(shkurinskaya_e_bin_labeling_tbb, empty_output) {
  int height = 5000;
  int width = 5000;
  int size = width * height;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out;
  std::vector<int> ans(size, 1);
  // Create TaskData
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  shkurinskaya_e_bin_labeling_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), false);
}

TEST(shkurinskaya_e_bin_labeling_tbb, test_diag_object) {
  int height = 100;
  int width = 100;
  int size = width * height;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out(size);
  std::vector<int> ans(size);
  for (int i = 0; i < height; ++i) {
    in[(i * width) + i] = 1;
    ans[(i * width) + i] = 1;
  }
  // Create TaskData
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  shkurinskaya_e_bin_labeling_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();
  ASSERT_EQ(ans, out);
}

TEST(shkurinskaya_e_bin_labeling_tbb, test_two_components) {
  int height = 100;
  int width = 100;
  int size = width * height;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out(size);
  std::vector<int> ans(size);
  in[0] = 1;
  in[9999] = 1;
  ans[0] = 1;
  ans[9999] = 2;
  // Create TaskData
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  shkurinskaya_e_bin_labeling_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();
  ASSERT_EQ(ans, out);
}

TEST(shkurinskaya_e_bin_labeling_tbb, test_horizontal_stripe) {
  int height = 100;
  int width = 100;
  int size = width * height;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out(size);
  std::vector<int> ans(size);
  for (int j = 0; j < width; ++j) {
    in[j] = 1;
    ans[j] = 1;
  }
  // Create TaskData
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  shkurinskaya_e_bin_labeling_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();
  ASSERT_EQ(ans, out);
}

TEST(shkurinskaya_e_bin_labeling_tbb, test_vertical_stripe) {
  int height = 100;
  int width = 100;
  int size = width * height;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out(size);
  std::vector<int> ans(size);
  for (int j = 0; j < height; ++j) {
    in[j * width] = 1;
    ans[j * width] = 1;
  }
  // Create TaskData
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  shkurinskaya_e_bin_labeling_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();
  ASSERT_EQ(ans, out);
}

TEST(shkurinskaya_e_bin_labeling_tbb, test_horizontal_stripe_dif_size) {
  int height = 50;
  int width = 100;
  int size = width * height;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out(size);
  std::vector<int> ans(size);
  for (int j = 0; j < width; ++j) {
    in[j] = 1;
    ans[j] = 1;
  }
  // Create TaskData
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  shkurinskaya_e_bin_labeling_tbb::TaskTBB task_tbb(task_data_tbb);
  ASSERT_EQ(task_tbb.Validation(), true);
  task_tbb.PreProcessing();
  task_tbb.Run();
  task_tbb.PostProcessing();
  ASSERT_EQ(ans, out);
}
