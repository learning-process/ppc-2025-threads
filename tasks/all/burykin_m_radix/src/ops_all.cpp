#include "all/burykin_m_radix/include/ops_all.hpp"

#include <algorithm>
#include <array>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/util/include/util.hpp"

std::array<int, 256> burykin_m_radix_all::RadixALL::ComputeFrequency(const std::vector<int>& a, const int shift) {
  std::array<int, 256> count = {};
  for (const int v : a) {
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }
    ++count[key];
  }
  return count;
}

std::array<int, 256> burykin_m_radix_all::RadixALL::ComputeIndices(const std::array<int, 256>& count) {
  std::array<int, 256> index = {0};
  for (int i = 1; i < 256; ++i) {
    index[i] = index[i - 1] + count[i - 1];
  }
  return index;
}

void burykin_m_radix_all::RadixALL::DistributeElements(const std::vector<int>& a, std::vector<int>& b,
                                                       std::array<int, 256> index, const int shift) {
  for (const int v : a) {
    unsigned int key = ((static_cast<unsigned int>(v) >> shift) & 0xFFU);
    if (shift == 24) {
      key ^= 0x80;
    }
    b[index[key]++] = v;
  }
}

namespace {
std::vector<std::span<int>> Distribute(std::span<int> arr, std::size_t n) {
  std::vector<std::span<int>> chunks(n);
  const std::size_t delta = arr.size() / n;
  const std::size_t extra = arr.size() % n;

  auto* cur = arr.data();
  for (std::size_t i = 0; i < n; i++) {
    const std::size_t sz = delta + ((i < extra) ? 1 : 0);
    chunks[i] = std::span{cur, cur + sz};
    cur += sz;
  }

  return chunks;
}

void RadixSortLocal(std::span<int> data) {
  if (data.empty()) {
    return;
  }

  std::vector<int> a(data.begin(), data.end());
  std::vector<int> b(a.size());

  for (int shift = 0; shift < 32; shift += 8) {
    auto count = burykin_m_radix_all::RadixALL::ComputeFrequency(a, shift);
    const auto index = burykin_m_radix_all::RadixALL::ComputeIndices(count);
    burykin_m_radix_all::RadixALL::DistributeElements(a, b, index, shift);
    a.swap(b);
  }

  std::copy(a.begin(), a.end(), data.begin());
}
}  // namespace

void burykin_m_radix_all::RadixALL::Merge(boost::mpi::communicator& group) {
  const auto numprocs = static_cast<std::size_t>(group.size());
  for (std::size_t i = 1; i < numprocs; i *= 2) {
    if (group.rank() % (2 * i) == 0) {
      const int slave = group.rank() + static_cast<int>(i);
      if (slave < static_cast<int>(numprocs)) {
        int size{};
        group.recv(int(slave), 0, size);

        const std::size_t threshold = procchunk_.size();
        procchunk_.resize(threshold + size);
        group.recv(int(slave), 0, procchunk_.data() + threshold, size);

        std::ranges::inplace_merge(procchunk_, procchunk_.begin() + std::int64_t(threshold));
      }
    } else if ((group.rank() % i) == 0) {
      const int size = static_cast<int>(procchunk_.size());
      const int master = group.rank() - static_cast<int>(i);
      group.send(master, 0, size);
      group.send(master, 0, procchunk_.data(), size);
      break;
    }
  }
}

bool burykin_m_radix_all::RadixALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    const unsigned int input_size = task_data->inputs_count[0];
    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);
    output_.reserve(input_.size());
  }
  return true;
}

bool burykin_m_radix_all::RadixALL::ValidationImpl() {
  return world_.rank() != 0 || (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool burykin_m_radix_all::RadixALL::RunImpl() {
  std::size_t totalsize{};
  if (world_.rank() == 0) {
    totalsize = input_.size();
  }
  boost::mpi::broadcast(world_, totalsize, 0);

  if (totalsize == 0) {
    return true;
  }

  const auto numprocs = std::min<std::size_t>(totalsize, world_.size());

  if (world_.rank() >= int(numprocs)) {
    world_.split(1);
    return true;
  }
  auto group = world_.split(0);

  if (group.rank() == 0) {
    std::vector<std::span<int>> procchunks = Distribute(input_, numprocs);
    procchunk_.assign(procchunks[0].begin(), procchunks[0].end());
    for (int i = 1; i < int(procchunks.size()); i++) {
      const auto& chunk = procchunks[i];
      const int chunksize = int(chunk.size());
      group.send(i, 0, chunksize);
      group.send(i, 0, chunk.data(), chunksize);
    }
  } else {
    int chunksize{};
    group.recv(0, 0, chunksize);
    procchunk_.clear();
    procchunk_.resize(chunksize);
    group.recv(0, 0, procchunk_.data(), chunksize);
  }

  const auto numthreads = std::min<std::size_t>(procchunk_.size(), ppc::util::GetPPCNumThreads());
  std::vector<std::span<int>> chunks = Distribute(procchunk_, numthreads);

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(numthreads); i++) {
    RadixSortLocal(chunks[i]);
  }

  for (std::size_t i = 1; i < numthreads; i *= 2) {
    const auto multithreaded = chunks.front().size() > 48;
    const auto active_threads = numthreads - i;

#pragma omp parallel for if (multithreaded)
    for (int j = 0; j < static_cast<int>(active_threads); j += 2 * static_cast<int>(i)) {
      auto& left = chunks[j];
      auto& right = chunks[j + i];

      std::inplace_merge(left.begin(), left.end(), right.end());
      left = std::span{left.begin(), right.end()};
    }
  }

  Merge(group);

  return true;
}

bool burykin_m_radix_all::RadixALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    output_ = procchunk_;
    for (size_t i = 0; i < output_.size(); ++i) {
      reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
    }
  }
  return true;
}