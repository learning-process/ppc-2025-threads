#pragma once

#include <boost/mpi.hpp>
#include <cstddef>
#include <vector>

namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi {
class TestTaskMPI {
 public:
    bool PreProcessingImpl();
    bool ValidationImpl();
    bool RunImpl();
    bool PostProcessingImpl();

 private:
    void ShellSort();
    static void BatcherMerge(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result);
    std::vector<int> input_;
    std::vector<int> output_;
    std::vector<int> local_input_;
    boost::mpi::communicator world_;
};
}  // namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi