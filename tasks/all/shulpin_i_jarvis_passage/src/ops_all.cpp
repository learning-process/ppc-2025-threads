#include "all/shulpin_i_jarvis_passage/include/ops_all.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
int Orientation(const shulpin_i_jarvis_all::Point& p, const shulpin_i_jarvis_all::Point& q,
                const shulpin_i_jarvis_all::Point& r) {
  double val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y));
  if (std::fabs(val) < 1e-9) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}

shulpin_i_jarvis_all::Point findLocalCandidate(const shulpin_i_jarvis_all::Point& current,
                                               const std::vector<shulpin_i_jarvis_all::Point>& points,
                                               int num_threads) {
  std::vector<shulpin_i_jarvis_all::Point> local_cand(num_threads, points.front());

  auto worker = [&](int tid) {
    size_t chunk = points.size() / num_threads;
    size_t start = tid * chunk;
    size_t end = (tid == num_threads - 1 ? points.size() : start + chunk);
    shulpin_i_jarvis_all::Point& candidate = local_cand[tid];
    for (size_t i = start; i < end; ++i) {
      const auto& p = points[i];
      if (p == current) continue;
      double cross = ((p.y - current.y) * (candidate.x - current.x)) - ((p.x - current.x) * (candidate.y - current.y));
      double d_p = std::pow(p.x - current.x, 2) + std::pow(p.y - current.y, 2);
      double d_c = std::pow(candidate.x - current.x, 2) + std::pow(candidate.y - current.y, 2);
      if (cross > 0 || (cross == 0 && d_p > d_c)) {
        candidate = p;
      }
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(worker, t);
  }
  for (auto& th : threads) {
    if (th.joinable()) th.join();
  }

  shulpin_i_jarvis_all::Point best = local_cand[0];
  for (int t = 1; t < num_threads; ++t) {
    const auto& cand = local_cand[t];
    double cross = ((cand.y - current.y) * (best.x - current.x)) - ((cand.x - current.x) * (best.y - current.y));
    double d_c = std::pow(cand.x - current.x, 2) + std::pow(cand.y - current.y, 2);
    double d_b = std::pow(best.x - current.x, 2) + std::pow(best.y - current.y, 2);
    if (cross > 0 || (cross == 0 && d_c > d_b)) {
      best = cand;
    }
  }
  return best;
}

}  // namespace

void shulpin_i_jarvis_all::JarvisSequential::MakeJarvisPassage(std::vector<shulpin_i_jarvis_all::Point>& input_jar,
                                                               std::vector<shulpin_i_jarvis_all::Point>& output_jar) {
  size_t total = input_jar.size();
  output_jar.clear();

  size_t start = 0;
  for (size_t i = 1; i < total; ++i) {
    if (input_jar[i].x < input_jar[start].x ||
        (input_jar[i].x == input_jar[start].x && input_jar[i].y < input_jar[start].y)) {
      start = i;
    }
  }

  size_t active = start;
  do {
    output_jar.emplace_back(input_jar[active]);
    size_t candidate = (active + 1) % total;

    for (size_t index = 0; index < total; ++index) {
      if (Orientation(input_jar[active], input_jar[index], input_jar[candidate]) == 2) {
        candidate = index;
      }
    }

    active = candidate;
  } while (active != start);
}

bool shulpin_i_jarvis_all::JarvisSequential::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_all::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_all::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_seq_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_seq_.resize(output_size);

  return true;
}

bool shulpin_i_jarvis_all::JarvisSequential::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_all::JarvisSequential::RunImpl() {
  MakeJarvisPassage(input_seq_, output_seq_);
  return true;
}

bool shulpin_i_jarvis_all::JarvisSequential::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_seq_.begin(), output_seq_.end(), result);
  return true;
}

// this whole nolint block is for NOLINT(readability-function-cognitive-complexity). using it as end-of-line comment
// doesn't work. all other linter warnings have been resolved
// NOLINTBEGIN
/*
void shulpin_i_jarvis_all::JarvisSTLParallel::MakeJarvisPassageSTL(
    std::vector<shulpin_i_jarvis_all::Point>& input_jar, std::vector<shulpin_i_jarvis_all::Point>& output_jar) {
  output_jar.clear();

  std::unordered_set<Point, PointHash, PointEqual> unique_points;

  size_t most_left = 0;
  for (size_t i = 1; i < input_jar.size(); ++i) {
    if (input_jar[i].x < input_jar[most_left].x ||
        (input_jar[i].x == input_jar[most_left].x && input_jar[i].y < input_jar[most_left].y)) {
      most_left = i;
    }
  }

  const Point& min_point = input_jar[most_left];
  std::vector<Point> convex_hull = {min_point};
  Point prev_point = min_point;
  Point next_point;

  int num_threads = ppc::util::GetPPCNumThreads();
  int chunk_size = static_cast<int>(input_jar.size() / num_threads);

  std::vector<std::thread> threads;
  std::vector<Point> candidates(num_threads, input_jar[0]);
  std::vector<bool> thread_ready(num_threads, false);
  std::vector<bool> thread_done(num_threads, false);
  std::mutex mtx;
  std::condition_variable cv;
  bool stop = false;

  auto findNextPointThread = [&](int tid) {
    while (true) {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] { return thread_ready[tid] || stop; });

      if (stop) {
        return;
      }

      int start = tid * chunk_size;
      int end = (tid == num_threads - 1) ? static_cast<int>(input_jar.size()) : (tid + 1) * chunk_size;
      Point candidate = input_jar[start];

      for (int i = start; i < end; ++i) {
        const auto& point = input_jar[i];
        if (point == prev_point) {
          continue;
        }

        double cross_product = ((point.y - prev_point.y) * (candidate.x - prev_point.x)) -
                               ((point.x - prev_point.x) * (candidate.y - prev_point.y));
        double dist1 = std::pow(point.x - prev_point.x, 2) + std::pow(point.y - prev_point.y, 2);
        double dist2 = std::pow(candidate.x - prev_point.x, 2) + std::pow(candidate.y - prev_point.y, 2);

        if (cross_product > 0 || (cross_product == 0 && dist1 > dist2)) {
          candidate = point;
        }
      }

      candidates[tid] = candidate;
      thread_ready[tid] = false;
      thread_done[tid] = true;
      cv.notify_all();
    }
  };

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(findNextPointThread, i);
  }

  do {
    next_point = input_jar[0];

    {
      std::unique_lock<std::mutex> lock(mtx);
      for (int i = 0; i < num_threads; ++i) {
        thread_ready[i] = true;
        thread_done[i] = false;
      }
    }
    cv.notify_all();

    {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] {
        return std::ranges::all_of(thread_done.begin(), thread_done.end(), [](bool done) { return done; });
      });
    }

    for (const auto& candidate : candidates) {
      double cross_product = ((candidate.y - prev_point.y) * (next_point.x - prev_point.x)) -
                             ((candidate.x - prev_point.x) * (next_point.y - prev_point.y));
      double dist1 = std::pow(candidate.x - prev_point.x, 2) + std::pow(candidate.y - prev_point.y, 2);
      double dist2 = std::pow(next_point.x - prev_point.x, 2) + std::pow(next_point.y - prev_point.y, 2);
      if (cross_product > 0 || (cross_product == 0 && dist1 > dist2)) {
        next_point = candidate;
      }
    }

    if (unique_points.find(next_point) == unique_points.end()) {
      output_jar.push_back(next_point);
      unique_points.insert(next_point);
    }

    prev_point = next_point;

  } while (next_point != min_point);

  {
    std::unique_lock<std::mutex> lock(mtx);
    stop = true;
    cv.notify_all();
  }

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}
*/
// NOLINTEND
void shulpin_i_jarvis_all::JarvisALLParallel::MakeJarvisPassageALL(
    std::vector<shulpin_i_jarvis_all::Point>& input_jar, std::vector<shulpin_i_jarvis_all::Point>& output_jar) {
  output_jar.clear();

  MPI_Datatype MPI_POINT;
  MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_POINT);
  MPI_Type_commit(&MPI_POINT);

  size_t N = input_jar.size();
  MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (rank_ != 0) {
    input_jar.resize(N);
  }

  MPI_Bcast(input_jar.data(), N, MPI_POINT, 0, MPI_COMM_WORLD);

  size_t most_left = 0;
  if (rank_ == 0) {
    for (size_t i = 1; i < N; ++i) {
      if (input_jar[i].x < input_jar[most_left].x ||
          (input_jar[i].x == input_jar[most_left].x && input_jar[i].y < input_jar[most_left].y)) {
        most_left = i;
      }
    }
  }
  MPI_Bcast(&most_left, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  Point min_point = input_jar[most_left];

  std::vector<int> counts(world_size_), displs(world_size_);
  int base = static_cast<int>(N / world_size_);
  int rem = static_cast<int>(N % world_size_);
  int offset = 0;
  for (int i = 0; i < world_size_; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }

  std::vector<Point> local_points(counts[rank_]);
  MPI_Scatterv(input_jar.data(), counts.data(), displs.data(), MPI_POINT, local_points.data(), counts[rank_], MPI_POINT,
               0, MPI_COMM_WORLD);

  std::unordered_set<Point, PointHash, PointEqual> unique_points;
  if (rank_ == 0) {
    output_jar.push_back(min_point);
    unique_points.insert(min_point);
  }

  Point prev_point = min_point;
  Point next_point;
  int num_threads = ppc::util::GetPPCNumThreads();

  bool done = false;

  do {
    MPI_Bcast(&prev_point, 1, MPI_POINT, 0, MPI_COMM_WORLD);

    Point local_candidate = findLocalCandidate(prev_point, local_points, num_threads);

    std::vector<Point> all_cand;
    if (rank_ == 0) {
      all_cand.resize(world_size_);
    }
    MPI_Gather(&local_candidate, 1, MPI_POINT, all_cand.data(), 1, MPI_POINT, 0, MPI_COMM_WORLD);

    if (rank_ == 0) {
      next_point = all_cand[0];
      for (int i = 1; i < world_size_; ++i) {
        const auto& cand = all_cand[i];
        double cross = ((cand.y - prev_point.y) * (next_point.x - prev_point.x)) -
                       ((cand.x - prev_point.x) * (next_point.y - prev_point.y));
        double d_c = std::pow(cand.x - prev_point.x, 2) + std::pow(cand.y - prev_point.y, 2);
        double d_n = std::pow(next_point.x - prev_point.x, 2) + std::pow(next_point.y - prev_point.y, 2);
        if (cross > 0 || (cross == 0 && d_c > d_n)) {
          next_point = cand;
        }
      }

      done = (next_point == min_point);

      if (!done && unique_points.find(next_point) == unique_points.end()) {
        output_jar.push_back(next_point);
        unique_points.insert(next_point);
      }
    }

    MPI_Bcast(&next_point, 1, MPI_POINT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&done, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    prev_point = next_point;

  } while (!done);

  MPI_Type_free(&MPI_POINT);
}

bool shulpin_i_jarvis_all::JarvisALLParallel::PreProcessingImpl() {
  if (rank_ == 0) {
    std::vector<shulpin_i_jarvis_all::Point> tmp_input;

    auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_all::Point*>(task_data->inputs[0]);
    size_t tmp_size = task_data->inputs_count[0];
    tmp_input.assign(tmp_data, tmp_data + tmp_size);

    input_stl_ = tmp_input;

    size_t output_size = task_data->outputs_count[0];
    output_stl_.resize(output_size);
  }
  return true;
}

bool shulpin_i_jarvis_all::JarvisALLParallel::ValidationImpl() {
  if (rank_ == 0) {
    return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
  }
  return true;
}

bool shulpin_i_jarvis_all::JarvisALLParallel::RunImpl() {
  MakeJarvisPassageALL(input_stl_, output_stl_);
  return true;
}

bool shulpin_i_jarvis_all::JarvisALLParallel::PostProcessingImpl() {
  if (rank_ == 0) {
    auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
    std::ranges::copy(output_stl_.begin(), output_stl_.end(), result);
  }
  return true;
}