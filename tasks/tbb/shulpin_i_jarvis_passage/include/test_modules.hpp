#include "tbb/shulpin_i_jarvis_passage/include/ops_tbb.hpp"

#include <vector>

namespace shulpin_tbb_test_module {
void VerifyResults(const std::vector<shulpin_i_jarvis_tbb::Point> &expected,
                   const std::vector<shulpin_i_jarvis_tbb::Point> &result_seq,
                   const std::vector<shulpin_i_jarvis_tbb::Point> &result_tbb);

[[maybe_unused]] void MainTestBody(std::vector<shulpin_i_jarvis_tbb::Point> &input,
                                   std::vector<shulpin_i_jarvis_tbb::Point> &expected);

std::vector<shulpin_i_jarvis_tbb::Point> GeneratePointsInCircle(size_t num_points,
                                                                const shulpin_i_jarvis_tbb::Point &center,
                                                                double radius);

inline size_t CalculateIndex(size_t i, size_t tmp);

inline void ExpectEqualPoints(const shulpin_i_jarvis_tbb::Point &expected, const shulpin_i_jarvis_tbb::Point &seq,
                              const shulpin_i_jarvis_tbb::Point &tbb);

void VerifyResultsCircle(const std::vector<shulpin_i_jarvis_tbb::Point> &expected,
                         const std::vector<shulpin_i_jarvis_tbb::Point> &result_seq,
                         const std::vector<shulpin_i_jarvis_tbb::Point> &result_tbb, size_t &num_points);

[[maybe_unused]] void TestBodyRandomCircle(std::vector<shulpin_i_jarvis_tbb::Point> &input,
                                           std::vector<shulpin_i_jarvis_tbb::Point> &expected, size_t &num_points);

[[maybe_unused]] void TestBodyFalse(std::vector<shulpin_i_jarvis_tbb::Point> &input,
                                    std::vector<shulpin_i_jarvis_tbb::Point> &expected);

int Orientation(const shulpin_i_jarvis_tbb::Point &p, const shulpin_i_jarvis_tbb::Point &q,
                const shulpin_i_jarvis_tbb::Point &r);

[[maybe_unused]] std::vector<shulpin_i_jarvis_tbb::Point> ComputeConvexHull(
    std::vector<shulpin_i_jarvis_tbb::Point> raw_points);

void VerifyResultsRandom(const std::vector<shulpin_i_jarvis_tbb::Point> &expected,
                         const std::vector<shulpin_i_jarvis_tbb::Point> &result_tbb);

std::vector<shulpin_i_jarvis_tbb::Point> GenerateRandomPoints(size_t num_points);

[[maybe_unused]] void RandomTestBody(std::vector<shulpin_i_jarvis_tbb::Point> &input,
                                     std::vector<shulpin_i_jarvis_tbb::Point> &expected);
}  // namespace shulpin_tbb_test_module