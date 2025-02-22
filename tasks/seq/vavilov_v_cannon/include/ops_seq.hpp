#pragma once

#include <utility>
#include <vector>
#include <memory>
#include <cmath>
#include "core/task/include/task.hpp"

namespace vavilov_v_cannon_seq {

    class CannonSequential : public ppc::core::Task {
    public:
        explicit CannonSequential(std::shared_ptr<ppc::core::TaskData> task_data)
            : Task(std::move(task_data)) {}

        bool PreProcessingImpl() override;
        bool ValidationImpl() override;
        bool RunImpl() override;
        bool PostProcessingImpl() override;

    private:
        unsigned int N;
        unsigned int block_size;
        unsigned int num_blocks;
        std::vector<double> A_;
        std::vector<double> B_;
        std::vector<double> C_;

        void InitialShift();
        void BlockMultiply();
        void ShiftBlocks();
    };

}  // namespace vavilov_v_cannon_seq