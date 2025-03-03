#include "seq/frolova_e_Sobel_filter/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

std::vector<int> frolova_e_Sobel_filter_seq::genRGBpicture(size_t width, size_t height, size_t seed) {
  std::vector<int> image(width * height * 3);
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> rgb(0, 255);

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      size_t index = (y * width + x) * 3;
      image[index] = rgb(gen);
      image[index + 1] = rgb(gen);
      image[index + 2] = rgb(gen);
    }
  }

  return image;
}

std::vector<int> frolova_e_Sobel_filter_seq::toGrayScaleImg(std::vector<RGB>& colorImg, size_t width, size_t height) {
  std::vector<int> grayScaleImage(width * height);
  for (size_t i = 0; i < width * height; i++) {
    grayScaleImage[i] = static_cast<int>(0.299 * colorImg[i].R + 0.587 * colorImg[i].G + 0.114 * colorImg[i].B);
  }

  return grayScaleImage;
}

int frolova_e_Sobel_filter_seq::Clamp(int value, int minVal, int maxVal) {
  if (value < minVal) return minVal;
  if (value > maxVal) return maxVal;
  return value;
}

bool frolova_e_Sobel_filter_seq::SobelFilterSequential::PreProcessingImpl() {
  int* value_1 = reinterpret_cast<int*>(task_data->inputs[0]);
  width = static_cast<size_t>(value_1[0]);

  height = static_cast<size_t>(value_1[1]);

  int* value_2 = reinterpret_cast<int*>(task_data->inputs[1]);
  std::vector<int> pictureVector;
  pictureVector.assign(value_2, value_2 + task_data->inputs_count[1]);
  for (size_t i = 0; i < pictureVector.size(); i += 3) {
    RGB pixel;
    pixel.R = pictureVector[i];
    pixel.G = pictureVector[i + 1];
    pixel.B = pictureVector[i + 2];

    picture.push_back(pixel);
  }

  grayscaleImage = frolova_e_Sobel_filter_seq::toGrayScaleImg(picture, width, height);
  resImage.resize(width * height);

  return true;
}

bool frolova_e_Sobel_filter_seq::SobelFilterSequential::ValidationImpl() {
  int* value_1 = reinterpret_cast<int*>(task_data->inputs[0]);

  if (task_data->inputs_count[0] != 2) {
    return false;
  }

  if (value_1[0] <= 0 || value_1[1] <= 0) {
    return false;
  }

  auto width_1 = static_cast<size_t>(value_1[0]);
  auto height_1 = static_cast<size_t>(value_1[1]);

  int* value_2 = reinterpret_cast<int*>(task_data->inputs[1]);
  std::vector<int> pictureVector;
  pictureVector.assign(value_2, value_2 + task_data->inputs_count[1]);
  if (task_data->inputs_count[1] != width_1 * height_1 * 3) {
    return false;
  }

  for (size_t i = 0; i < pictureVector.size(); i++) {
    if (pictureVector[i] < 0 || pictureVector[i] > 255) {
      return false;
    }
  }

  return true;
}

bool frolova_e_Sobel_filter_seq::SobelFilterSequential::RunImpl() {
  const std::vector<int> Gx = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const std::vector<int> Gy = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int resX = 0;
      int resY = 0;

      for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
          int px = x + kx;
          int py = y + ky;

          int pixelValue = 0;

          if (px >= 0 && px < width && py >= 0 && py < height) {
            pixelValue = grayscaleImage[py * width + px];
          }

          int kernelInd = (ky + 1) * 3 + (kx + 1);
          resX += pixelValue * Gx[kernelInd];
          resY += pixelValue * Gy[kernelInd];
        }
      }
      int gradient = static_cast<int>(sqrt(resX * resX + resY * resY));
      resImage[y * width + x] = Clamp(gradient, 0, 255);
    }
  }
  return true;
}

bool frolova_e_Sobel_filter_seq::SobelFilterSequential::PostProcessingImpl() {
  for (size_t i = 0; i < width * height; i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = resImage[i];
  }
  return true;
}