#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace laganina_e_component_labeling_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int m_;  // строки
  int n_;  // столбцы
  std::vector<int> binary_;
  std::vector<int> labeled_binary_;
  std::vector<int> parent_;
  std ::vector<int> step1_;
  int Find(int x);
  bool UnionSets(int x, int y);
  std::vector<int> NeighborsLabels(int x, int y);
};

}  // namespace laganina_e_component_labeling_seq