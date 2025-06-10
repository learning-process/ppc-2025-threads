#include "../include/chc.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stack>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

using namespace voroshilov_v_convex_hull_components_stl;

Pixel::Pixel(int y_param, int x_param) : y(y_param), x(x_param), value(0) {}
Pixel::Pixel(int y_param, int x_param, int value_param) : y(y_param), x(x_param), value(value_param) {}

bool Pixel::operator==(const int value_param) const { return value == value_param; }
bool Pixel::operator==(const Pixel& other) const { return (y == other.y) && (x == other.x); }

Image::Image(int hght, int wdth, std::vector<int> pxls) {
  height = hght;
  width = wdth;
  pixels.resize(height * width);

  int num_threads = ppc::util::GetPPCNumThreads();
  int chunk = (height + num_threads - 1) / num_threads;
  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; t++) {
    int y1 = t * chunk;
    int y2 = std::min(y1 + chunk, height);
    threads.emplace_back([this, y1, y2, &pxls]() {
      for (int y = y1; y < y2; y++) {
        for (int x = 0; x < width; x++) {
          pixels[(y * width) + x] = Pixel(y, x, pxls[(y * width) + x]);
        }
      }
    });
  }

  for (auto& th : threads) {
    th.join();
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

void voroshilov_v_convex_hull_components_stl::CheckBoundaryPixels(UnionFind& union_find, Image& image, int y, int x) {
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

void voroshilov_v_convex_hull_components_stl::MergeComponentsAcrossAreas(std::vector<Component>& components,
                                                                         Image& image, int area_height,
                                                                         std::vector<int>& end_y) {
  if (components.empty()) {
    return;
  }

  int num_threads = ppc::util::GetPPCNumThreads();
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

  std::vector<std::thread> threads;
  int chunk = (r + num_threads - 1) / num_threads;
  for (int t = 0; t < num_threads; t++) {
    int r1 = t * chunk;
    int r2 = std::min(r1 + chunk, r);
    threads.emplace_back([&, r1, r2]() {
      for (int i = r1; i < r2; i++) {
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
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  components = std::move(merged);
}

template <typename T>
std::vector<T> voroshilov_v_convex_hull_components_stl::MergeVectors(std::vector<std::vector<T>>& vectors) {
  int size = (int)vectors.size();

  std::vector<int> sizes(size);
  std::vector<int> offsets(size + 1);
  for (int i = 0; i < size; i++) {
    sizes[i] = (int)vectors[i].size();
  }

  offsets[0] = 0;
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

Component voroshilov_v_convex_hull_components_stl::DepthComponentSearchInArea(Pixel start_pixel, Image& image,
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

std::vector<Component> voroshilov_v_convex_hull_components_stl::FindComponentsInArea(Image& image, int start_y,
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

std::vector<Component> voroshilov_v_convex_hull_components_stl::FindComponentsSTL(Image& image) {
  int height = image.height;

  std::vector<std::thread> threads;
  int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::vector<Component>> threads_components(num_threads);
  int chunk_height = (height + num_threads - 1) / num_threads;
  std::vector<int> y2(num_threads);

  std::vector<int> index_offset(num_threads);
  for (int i = 0; i < num_threads; i++) {
    index_offset[i] = (i * 1000) + 2;
  }

  for (int t = 0; t < num_threads; t++) {
    int y1 = t * chunk_height;
    threads.emplace_back([&, t, y1]() {
      y2[t] = std::min(y1 + chunk_height, height);
      threads_components[t] = FindComponentsInArea(image, y1, y2[t], index_offset[t]);
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  std::vector<Component> components = MergeVectors<Component>(threads_components);

  MergeComponentsAcrossAreas(components, image, chunk_height, y2);

  return components;
}

int voroshilov_v_convex_hull_components_stl::CheckRotation(Pixel& first, Pixel& second, Pixel& third) {
  return ((second.x - first.x) * (third.y - second.y)) - ((second.y - first.y) * (third.x - second.x));
}

Pixel voroshilov_v_convex_hull_components_stl::FindFarthestPixel(std::vector<Pixel>& pixels,
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

std::vector<Pixel> voroshilov_v_convex_hull_components_stl::QuickHull(Component& component) {
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

std::vector<Hull> voroshilov_v_convex_hull_components_stl::QuickHullAllSTL(std::vector<Component>& components) {
  if (components.empty()) {
    return {};
  }
  std::vector<Hull> hulls(components.size());

  std::vector<std::thread> threads;
  size_t num_threads = ppc::util::GetPPCNumThreads();
  size_t chunk = (components.size() + num_threads - 1) / num_threads;

  for (size_t t = 0; t < num_threads; t++) {
    size_t c1 = t * chunk;
    size_t c2 = std::min(c1 + chunk, components.size());
    threads.emplace_back([=, &components, &hulls]() {
      for (size_t c = c1; c < c2; c++) {
        hulls[c] = QuickHull(components[c]);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  return hulls;
}

void voroshilov_v_convex_hull_components_stl::PackHulls(std::vector<Hull>& hulls, int width, int height,
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

std::vector<Hull> voroshilov_v_convex_hull_components_stl::UnpackHulls(std::vector<int>& hulls_indexes,
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
