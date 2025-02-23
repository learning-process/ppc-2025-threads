
#include "core/task/include/task.hpp"
#include <algorithm>
#include <utility>
#include <vector>

namespace laganina_e_component_labeling_seq{

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int m;  // строки
  int n;  // столбцы
  std::vector<int> binary;
  std::vector<int> labaled_binary;
  std::vector<std::pair<int, int>> parent;
  std ::vector<int> step1;
  std::pair<int,int> find(int x);
  bool Union_sets(int x, int y);
  std::vector<int> neighbors_labels(int x, int y);
};

}  // namespace laganina_e_component_labeling_seq