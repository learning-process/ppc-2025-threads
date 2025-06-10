#include "../include/chc.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/utility.hpp>  // NOLINT(misc-include-cleaner)
#include <boost/serialization/vector.hpp>   // NOLINT(misc-include-cleaner)
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace voroshilov_v_convex_hull_components_all;

Pixel::Pixel(int y_param, int x_param) : y(y_param), x(x_param), value(0) {}
Pixel::Pixel(int y_param, int x_param, int value_param) : y(y_param), x(x_param), value(value_param) {}

bool Pixel::operator==(const int value_param) const { return value == value_param; }
bool Pixel::operator==(const Pixel& other) const { return (y == other.y) && (x == other.x); }

Image::Image(int hght, int wdth, std::vector<int> pxls) {
  height = hght;
  width = wdth;
  pixels.resize(height * width);

#pragma omp parallel for
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      pixels[(y * width) + x] = Pixel(y, x, pxls[(y * width) + x]);
    }
  }
}

Pixel& Image::GetPixel(int y, int x) { return pixels[(y * width) + x]; }

LineSegment::LineSegment(Pixel& a_param, Pixel& b_param) : a(a_param), b(b_param) {}

UnionFind::UnionFind(int n) {
  roots.resize(n);
  ranks.resize(n);
  for (int i = 0; i < n; i++) {
    roots[i] = i;
    ranks[i] = 1;
  }
}

int UnionFind::FindRoot(int x) {
  while (roots[x] != x) {
    roots[x] = roots[roots[x]];
    x = roots[x];
  }
  return x;
}

void UnionFind::Union(int x, int y) {
  int root_x = FindRoot(x);
  int root_y = FindRoot(y);
  if (root_x == root_y) {
    return;
  }
  if (ranks[root_x] < ranks[root_y]) {
    std::swap(root_x, root_y);
  }
  roots[root_y] = root_x;
  if (ranks[root_x] == ranks[root_y]) {
    ranks[root_x]++;
  }
}

void voroshilov_v_convex_hull_components_all::CheckBoundaryPixels(UnionFind& union_find, Image& image, int y, int x) {
  Pixel p1 = image.GetPixel(y, x);

  Pixel p2 = image.GetPixel(y + 1, x);
  if (p1.value > 1 && p2.value > 1) {
    union_find.Union(p1.value, p2.value);
  }

  if (x > 0) {
    Pixel p3 = image.GetPixel(y + 1, x - 1);
    if (p1.value > 1 && p3.value > 1) {
      union_find.Union(p1.value, p3.value);
    }
  }
  if (x < image.width - 1) {
    Pixel p4 = image.GetPixel(y + 1, x + 1);
    if (p1.value > 1 && p4.value > 1) {
      union_find.Union(p1.value, p4.value);
    }
  }
}

void voroshilov_v_convex_hull_components_all::MergeComponentsAcrossAreas(std::vector<Component>& components,
                                                                         Image& image, int area_height,
                                                                         std::vector<int>& end_y) {
  if (components.empty()) {
    return;
  }

  int num_threads = omp_get_max_threads();
  UnionFind union_find((num_threads * 1000) + 3);

  int width = image.width;
  int height = image.height;

  for (int endy : end_y) {
    int y = endy - 1;
    if (y != height - 1) {
      for (int x = 0; x < width; x++) {
        CheckBoundaryPixels(union_find, image, y, x);
      }
    }
  }

  int n = (int)components.size();
  std::vector<int> all_roots;
  all_roots.reserve(n);
  for (int i = 0; i < n; i++) {
    int label = components[i][0].value;
    all_roots.push_back(union_find.FindRoot(label));
  }

  std::vector<int> roots_unique = all_roots;
  std::ranges::sort(roots_unique);
  auto it = std::ranges::unique(roots_unique).begin();
  roots_unique.erase(it, roots_unique.end());
  int r = (int)roots_unique.size();

  int max_root = -1;
  if (!roots_unique.empty()) {
    max_root = roots_unique.back();
  }
  std::vector<int> root_to_indx(max_root + 1, -1);
  for (int i = 0; i < r; i++) {
    root_to_indx[roots_unique[i]] = i;
  }

  std::vector<std::vector<int>> comps_by_root(r);
  for (int i = 0; i < n; i++) {
    int b = root_to_indx[all_roots[i]];
    comps_by_root[b].push_back(i);
  }

  std::vector<Component> merged(r);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < r; i++) {
    size_t total = 0;
    for (int indx : comps_by_root[i]) {
      total += components[indx].size();
    }
    merged[i].reserve(total);

    for (int indx : comps_by_root[i]) {
      Component& src = components[indx];
      std::ranges::move(src, std::back_inserter(merged[i]));
    }
  }

  components = std::move(merged);
}

template <typename T>
std::vector<T> voroshilov_v_convex_hull_components_all::MergeVectors(std::vector<std::vector<T>>& vectors) {
  int size = (int)vectors.size();

  std::vector<int> sizes(size);
  std::vector<int> offsets(size + 1);
  for (int i = 0; i < size; i++) {
    sizes[i] = (int)vectors[i].size();
  }
  if (!offsets.empty()) {
    offsets[0] = 0;
  }
  for (int i = 0; i < size; i++) {
    offsets[i + 1] = offsets[i] + sizes[i];
  }

  std::vector<T> vec(offsets[size]);

  for (int i = 0; i < size; i++) {
    std::vector<T>& src = vectors[i];
    auto dst_it = vec.begin() + offsets[i];
    std::move(src.begin(), src.end(), dst_it);
  }

  return vec;
}

Component voroshilov_v_convex_hull_components_all::DepthComponentSearchInArea(Pixel start_pixel, Image& image,
                                                                              int index, int start_y, int end_y) {
  const int step_y[8] = {1, 1, 1, 0, 0, -1, -1, -1};  // Offsets by Y (up, stand, down)
  const int step_x[8] = {-1, 0, 1, -1, 1, -1, 0, 1};  // Offsets by X (left, stand, right)
  std::stack<Pixel> stack;
  std::vector<Pixel> component_pixels;
  stack.push(start_pixel);
  image.GetPixel(start_pixel.y, start_pixel.x).value = index;                // Mark start pixel as visited
  component_pixels.push_back(image.GetPixel(start_pixel.y, start_pixel.x));  // Add start pixel to component

  while (!stack.empty()) {
    Pixel current_pixel = stack.top();
    stack.pop();
    for (int i = 0; i < 8; i++) {
      int next_y = current_pixel.y + step_y[i];
      int next_x = current_pixel.x + step_x[i];
      if (next_y >= start_y && next_y < end_y && next_x >= 0 && next_x < image.width &&
          image.GetPixel(next_y, next_x) == 1) {
        stack.push(image.GetPixel(next_y, next_x));
        image.GetPixel(next_y, next_x).value = index;                // Mark neighbour pixel as visited
        component_pixels.push_back(image.GetPixel(next_y, next_x));  // Add neighbour pixel to component
      }
    }
  }

  Component component(component_pixels);

  return component;
}

std::vector<Component> voroshilov_v_convex_hull_components_all::FindComponentsInArea(Image& image, int start_y,
                                                                                     int end_y, int index_offset) {
  std::vector<Component> components;
  int index = index_offset;  // unique index in this area

  for (int y = start_y; y < end_y; y++) {
    for (int x = 0; x < image.width; x++) {
      if (image.GetPixel(y, x) == 1) {
        Component component = DepthComponentSearchInArea(image.GetPixel(y, x), image, index, start_y, end_y);
        components.push_back(component);
        index++;
      }
    }
  }

  if (components.empty()) {
    return {};
  }

  return components;
}

std::vector<Component> voroshilov_v_convex_hull_components_all::FindComponentsOMP(Image& image) {
  int num_threads = omp_get_max_threads();

  std::vector<std::vector<Component>> threads_components(num_threads);

  int height = image.height;

  int area_height = height / num_threads;
  int remainder = height % num_threads;
  std::vector<int> start_y(num_threads);
  std::vector<int> end_y(num_threads);
  std::vector<int> index_offset(num_threads);

  if (num_threads == 1) {
    start_y[0] = 0;
    end_y[0] = height;
    index_offset[0] = 2;
  } else {
    for (size_t i = 1; i < start_y.size(); i++) {
      start_y[i] = start_y[i - 1] + area_height;
      if (remainder > 0) {
        start_y[i]++;
        remainder--;
      }
    }

    for (size_t i = 0; i < end_y.size() - 1; i++) {
      end_y[i] = start_y[i + 1];
    }
    end_y[end_y.size() - 1] = height;

    for (int i = 0; i < num_threads; i++) {
      index_offset[i] = (i * 1000) + 2;
    }
  }

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();

    threads_components[thread_id] =
        FindComponentsInArea(image, start_y[thread_id], end_y[thread_id], index_offset[thread_id]);
  }

  std::vector<Component> components = MergeVectors<Component>(threads_components);

  MergeComponentsAcrossAreas(components, image, area_height, end_y);

  return components;
}

std::unordered_map<int, std::vector<Pixel>> voroshilov_v_convex_hull_components_all::UnionComponents(
    int start_y, int end_y, std::vector<Component>& local_components, std::vector<Component>& from_up,
    std::unordered_map<int, std::vector<int>>& boundary_map) {
  int loc_size = (int)local_components.size();
  int up_size = (int)from_up.size();

  UnionFind uf(loc_size + up_size);

  for (int i = 0; i < (int)from_up.size(); i++) {
    for (Pixel& p : from_up[i]) {
      if (p.y == start_y - 1) {
        for (int dx = -1; dx <= 1; dx++) {
          int xx = p.x + dx;
          auto it = boundary_map.find(xx);
          if (it != boundary_map.end()) {
            for (int i_loc : it->second) {
              uf.Union(i_loc, loc_size + i);
            }
          }
        }
      }
    }
  }

  std::unordered_map<int, std::vector<Pixel>> merged_pixels;
  merged_pixels.reserve(loc_size + up_size);
  for (int i = 0; i < loc_size; i++) {
    int r = uf.FindRoot(i);
    merged_pixels[r].insert(merged_pixels[r].end(), std::make_move_iterator(local_components[i].begin()),
                            std::make_move_iterator(local_components[i].end()));
  }
  for (int i = 0; i < up_size; i++) {
    int idx = loc_size + i;
    int r = uf.FindRoot(idx);
    merged_pixels[r].insert(merged_pixels[r].end(), std::make_move_iterator(from_up[i].begin()),
                            std::make_move_iterator(from_up[i].end()));
  }

  return merged_pixels;
}

std::vector<Component> voroshilov_v_convex_hull_components_all::SendExtraComponents(
    boost::mpi::communicator& world, int start_y, int end_y, std::vector<Component>& local_components) {
  int rank = world.rank();
  int num_procs = world.size();

  std::vector<Component> from_up;
  if (rank > 0) {
    world.recv(rank - 1, 2, from_up);
  }

  std::unordered_map<int, std::vector<int>> boundary_map;
  for (int i = 0; i < (int)local_components.size(); i++) {
    for (Pixel& p : local_components[i]) {
      if (p.y == start_y) {
        boundary_map[p.x].push_back(i);
      }
    }
  }

  std::unordered_map<int, std::vector<Pixel>> merged_pixels =
      UnionComponents(start_y, end_y, local_components, from_up, boundary_map);

  std::vector<Component> to_keep;
  std::vector<Component> to_send;

  for (auto& [root, idxs] : merged_pixels) {
    Component comp(std::move(idxs));
    int max_y = comp.front().y;
    for (Pixel& p : comp) {
      max_y = std::max(max_y, p.y);
    }
    if (rank == num_procs - 1 || max_y < end_y - 1) {
      to_keep.push_back(std::move(comp));
    } else {
      to_send.push_back(std::move(comp));
    }
  }

  if (rank + 1 < num_procs) {
    world.send(rank + 1, 2, to_send);
  }

  return to_keep;
}

std::vector<Component> voroshilov_v_convex_hull_components_all::FindComponentsMPIOMP(int height, int width,
                                                                                     std::vector<int>& pixels_in) {
  boost::mpi::communicator world;

  int num_procs = world.size();
  int rank = world.rank();

  // NOLINTNEXTLINE(misc-include-cleaner)
  boost::mpi::broadcast(world, height, 0);
  // NOLINTNEXTLINE(misc-include-cleaner)
  boost::mpi::broadcast(world, width, 0);

  int area_height = height / num_procs;
  int remainder = height % num_procs;
  std::vector<int> start_y(num_procs, 0);
  std::vector<int> end_y(num_procs);
  std::vector<int> index_offset(num_procs);

  if (num_procs == 1) {
    end_y[0] = height;
    index_offset[0] = 2;
  } else {
    for (size_t i = 1; i < start_y.size(); i++) {
      start_y[i] = start_y[i - 1] + area_height;
      if (remainder > 0) {
        start_y[i]++;
        remainder--;
      }
    }
    for (size_t i = 0; i < end_y.size() - 1; i++) {
      end_y[i] = start_y[i + 1];
    }
    end_y[end_y.size() - 1] = height;

    for (int i = 0; i < num_procs; i++) {
      index_offset[i] = (i * 1000) + 2;
    }
  }

  std::vector<int> sizes(num_procs);
  std::vector<int> displs(num_procs);
  for (int i = 0; i < num_procs; i++) {
    sizes[i] = (end_y[i] - start_y[i]) * width;
    displs[i] = start_y[i] * width;
  }

  std::vector<int> local_pixels(sizes[rank]);
  // NOLINTNEXTLINE(misc-include-cleaner)
  boost::mpi::scatterv(world, pixels_in, sizes, displs, local_pixels.data(), static_cast<int>(local_pixels.size()), 0);

  int local_height = end_y[rank] - start_y[rank];
  Image local_image(local_height, width, local_pixels);

  std::vector<Component> local_components = FindComponentsOMP(local_image);

  int y_offset = start_y[rank];
  for (Component& comp : local_components) {
    for (Pixel& p : comp) {
      p.y += y_offset;
    }
  }

  std::vector<Component> local_full_components =
      SendExtraComponents(world, start_y[rank], end_y[rank], local_components);

  return local_full_components;
}

int voroshilov_v_convex_hull_components_all::CheckRotation(Pixel& first, Pixel& second, Pixel& third) {
  return ((second.x - first.x) * (third.y - second.y)) - ((second.y - first.y) * (third.x - second.x));
}

Pixel voroshilov_v_convex_hull_components_all::FindFarthestPixel(std::vector<Pixel>& pixels,
                                                                 LineSegment& line_segment) {
  Pixel farthest_pixel(-1, -1, -1);
  double max_dist = 0.0;

  for (Pixel& c : pixels) {
    Pixel a = line_segment.a;
    Pixel b = line_segment.b;
    if (CheckRotation(a, b, c) < 0) {  // left rotation
      double distance = std::abs(((b.x - a.x) * (a.y - c.y)) - ((a.x - c.x) * (b.y - a.y)));
      if (distance > max_dist) {
        max_dist = distance;
        farthest_pixel = c;
      }
    }
  }

  return farthest_pixel;
}

std::vector<Pixel> voroshilov_v_convex_hull_components_all::QuickHull(Component& component) {
  if (component.size() < 3) {
    return component;
  }

  Pixel left = component[0];
  Pixel right = component[0];

  for (Pixel& pixel : component) {
    if ((pixel.x < left.x) || (pixel.x == left.x && pixel.y < left.y)) {
      left = pixel;
    }
    if ((pixel.x > right.x) || (pixel.x == right.x && pixel.y > right.y)) {
      right = pixel;
    }
  }

  std::vector<Pixel> hull;
  std::stack<LineSegment> stack;

  LineSegment line_segment1(left, right);
  LineSegment line_segment2(right, left);
  stack.push(line_segment1);
  stack.push(line_segment2);

  while (!stack.empty()) {
    LineSegment line_segment = stack.top();
    Pixel a = line_segment.a;
    Pixel b = line_segment.b;
    stack.pop();

    Pixel c = FindFarthestPixel(component, line_segment);
    if (c == -1) {
      hull.push_back(a);
    } else {
      LineSegment new_line1(a, c);
      stack.push(new_line1);
      LineSegment new_line2(c, b);
      stack.push(new_line2);
    }
  }

  std::ranges::reverse(hull);

  std::vector<Pixel> res_hull;
  for (size_t i = 0; i < hull.size(); i++) {
    if (i == 0 || i == hull.size() - 1 || CheckRotation(hull[i - 1], hull[i], hull[i + 1]) != 0) {
      res_hull.push_back(hull[i]);
    }
  }

  return res_hull;
}

std::vector<Hull> voroshilov_v_convex_hull_components_all::QuickHullAllOMP(std::vector<Component>& components) {
  if (components.empty()) {
    return {};
  }

  int components_size = static_cast<int>(components.size());
  std::vector<Hull> hulls(components.size());

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < components_size; i++) {
    hulls[i] = QuickHull(components[i]);
  }

  return hulls;
}

std::vector<Hull> voroshilov_v_convex_hull_components_all::QuickHullAllMPIOMP(
    std::vector<Component>& local_components) {
  boost::mpi::communicator world;
  int rank = world.rank();

  std::vector<Hull> local_hulls = QuickHullAllOMP(local_components);

  std::vector<std::vector<Hull>> gathered_hulls;
  // NOLINTNEXTLINE(misc-include-cleaner)
  boost::mpi::gather(world, local_hulls, gathered_hulls, 0);

  if (rank == 0) {
    std::vector<Hull> result_hulls = MergeVectors<Hull>(gathered_hulls);
    return result_hulls;
  }

  return {};
}

void voroshilov_v_convex_hull_components_all::PackHulls(std::vector<Hull>& hulls, int width, int height,
                                                        int* hulls_indxs, int* pixels_indxs) {
  std::fill(hulls_indxs, hulls_indxs + (height * width), 0);
  std::fill(pixels_indxs, pixels_indxs + (height * width), 0);

  int hull_index = 1;
  for (Hull& hull : hulls) {
    int pixel_index = 1;
    for (Pixel& p : hull) {
      int pos = (p.y * width) + p.x;
      hulls_indxs[pos] = hull_index;
      pixels_indxs[pos] = pixel_index;
      pixel_index++;
    }
    hull_index++;
  }
}

std::vector<Hull> voroshilov_v_convex_hull_components_all::UnpackHulls(std::vector<int>& hulls_indexes,
                                                                       std::vector<int>& pixels_indexes, int height,
                                                                       int width, size_t hulls_size) {
  std::vector<Hull> hulls(hulls_size);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int hull_index = hulls_indexes[(y * width) + x];
      if (hull_index > 0) {
        int pixel_index = pixels_indexes[(y * width) + x];
        Pixel pixel(y, x, pixel_index);
        hulls[hull_index - 1].push_back(pixel);
      }
    }
  }

  for (Hull& hull : hulls) {
    for (size_t p1 = 0; p1 < hull.size() - 1; p1++) {
      for (size_t p2 = p1 + 1; p2 < hull.size(); p2++) {
        if (hull[p1].value > hull[p2].value) {
          Pixel tmp = hull[p1];
          hull[p1] = hull[p2];
          hull[p2] = tmp;
        }
      }
    }
  }

  if (hulls.empty()) {
    return {};
  }

  return hulls;
}
