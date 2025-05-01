bool RunImpl() override {
  const auto size = input_.size();
  std::ranges::copy_n(input_.begin(), size, res_.begin());

  const auto parallelism = std::min<std::size_t>(size, ppc::util::GetPPCNumThreads());
  if (size < 1) {
    return true;
  }

  const auto fair = size / parallelism;
  const auto extra = size % parallelism;

  std::vector<Block> blocks(parallelism);
  T* pointer = res_.data();
  for (std::size_t i = 0; i < parallelism; i++) {
    blocks[i] = {.e = pointer, .sz = fair + ((i < extra) ? 1 : 0)};
    pointer += blocks[i].sz;
  }

#pragma omp parallel for
  for (int i = 0; i < int(parallelism); i++) {
    DoSort(blocks[i].e, 0, blocks[i].sz - 1, reverse_ ? ReverseComp : StandardComp);
  }

  auto partind = parallelism;
  for (auto wide = decltype(parallelism){1}; partind > 1; wide *= 2, partind /= 2) {
#pragma omp parallel for if (blocks[wide].sz >= 40)
    for (std::size_t k = 0; k < static_cast<std::size_t>(partind / 2); ++k) {
      const auto idx = 2 * wide * k;
      DoInplaceMerge(blocks.data() + idx, blocks.data() + idx + wide, reverse_ ? ReverseComp : StandardComp);
    }
    if ((partind / 2) == 1) {
      DoInplaceMerge(&blocks.front(), &blocks.back(), reverse_ ? ReverseComp : StandardComp);
    } else if ((partind / 2) % 2 != 0) {
      DoInplaceMerge(blocks.data() + (2 * wide * ((partind / 2) - 2)), blocks.data() + (2 * wide * ((partind / 2) - 1)),
                     reverse_ ? ReverseComp : StandardComp);
    }
  }

  return true;
}
