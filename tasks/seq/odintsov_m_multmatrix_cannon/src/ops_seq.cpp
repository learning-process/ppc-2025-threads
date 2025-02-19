
#include "seq/odintsov_m_multmatrix_cannon/include/ops_seq.hpp"

#include <math.h>

#include <iostream>
using namespace std;
void odintsov_m_mulmatix_cannon_seq::MulMatrixCannonSequential::shiftRow(std::vector<double>& matrix, int root, int row,
                                                                         int shift) {
  shift = shift % root;
  std::vector<double> tmp(root);
  for (int j = 0; j < root; j++) {
    tmp[j] = matrix[row * root + (j + shift) % root];
  }
  for (int j = 0; j < root; j++) {
    matrix[row * root + j] = tmp[j];
  }
}

void odintsov_m_mulmatix_cannon_seq::MulMatrixCannonSequential::shiftColumn(std::vector<double>& matrix, int root,
                                                                            int col, int shift) {
  shift = shift % root;
  std::vector<double> tmp(root);

  for (int i = 0; i < root; i++) {
    tmp[i] = matrix[((i + shift) % root) * root + col];
  }
  for (int i = 0; i < root; i++) {
    matrix[i * root + col] = tmp[i];
  }
}
void odintsov_m_mulmatix_cannon_seq::MulMatrixCannonSequential::shiftBlocksUp(std::vector<double>& matrix, int root,
                                                                              int sz) {
  int p = root / block_sz;
  for (int bj = 0; bj < p; bj++) {
    std::vector<double> first_block(block_sz * block_sz);

    for (int i = 0; i < block_sz; i++) {
      for (int j = 0; j < block_sz; j++) {
        first_block[i * block_sz + j] = matrix[i * root + (bj * block_sz + j)];
      }
    }

    for (int bi = 0; bi < p - 1; bi++) {
      for (int i = 0; i < block_sz; i++) {
        for (int j = 0; j < block_sz; j++) {
          matrix[(bi * block_sz + i) * root + bj * block_sz + j] =
              matrix[(bi + 1) * block_sz * root + i * root + bj * block_sz + j];
        }
      }
    }

    for (int i = 0; i < block_sz; i++) {
      for (int j = 0; j < block_sz; j++) {
        matrix[(p - 1) * block_sz * root + i * root + bj * block_sz + j] = first_block[i * block_sz + j];
      }
    }
  }
}

void odintsov_m_mulmatix_cannon_seq::MulMatrixCannonSequential::shiftBlocksLeft(std::vector<double>& matrix, int root,
                                                                                int sz) {
  int p = root / block_sz;
  for (int bi = 0; bi < p; bi++) {
    std::vector<double> first_block(block_sz * block_sz);

    for (int i = 0; i < block_sz; i++) {
      for (int j = 0; j < block_sz; j++) {
        first_block[i * block_sz + j] = matrix[(bi * block_sz + i) * root + j];
      }
    }

    for (int bj = 0; bj < p - 1; bj++) {
      for (int i = 0; i < block_sz; i++) {
        for (int j = 0; j < block_sz; j++) {
          matrix[(bi * block_sz + i) * root + bj * block_sz + j] =
              matrix[(bi * block_sz + i) * root + (bj + 1) * block_sz + j];
        }
      }
    }

    for (int i = 0; i < block_sz; i++) {
      for (int j = 0; j < block_sz; j++) {
        matrix[(bi * block_sz + i) * root + (p - 1) * block_sz + j] = first_block[i * block_sz + j];
      }
    }
  }
}

bool odintsov_m_mulmatix_cannon_seq::MulMatrixCannonSequential::is_squere(int num) {
  if (num < 0) return false;
  int root = static_cast<int>(sqrt(num));

  return root * root == num;
}

int odintsov_m_mulmatix_cannon_seq::MulMatrixCannonSequential::get_block_size(int N) {
  for (int k = N / 2; k >= 2; k--) {
    if (N % k == 0) {
      return k;
    }
  }
  return 1;
}

bool odintsov_m_mulmatix_cannon_seq::MulMatrixCannonSequential::PreProcessingImpl() {
  szA = task_data->inputs_count[0];
  szB = task_data->inputs_count[1];
  matrixA.assign(reinterpret_cast<double*>(task_data->inputs[0]),
                 reinterpret_cast<double*>(task_data->inputs[0]) + szA);
  matrixB.assign(reinterpret_cast<double*>(task_data->inputs[1]),
                 reinterpret_cast<double*>(task_data->inputs[1]) + szB);
  matrixC.assign(szA, 0);

  block_sz = get_block_size(sqrt(szA));
  return true;
}

bool odintsov_m_mulmatix_cannon_seq::MulMatrixCannonSequential::ValidationImpl() {
  if (task_data->inputs_count[0] != task_data->inputs_count[1]) return false;

  if ((!(is_squere(task_data->inputs_count[0]))) || (!(is_squere(task_data->inputs_count[0])))) return false;
  return true;
}

bool odintsov_m_mulmatix_cannon_seq::MulMatrixCannonSequential::RunImpl() {
  int root = sqrt(szA);

  std::vector<double> blockA(block_sz * block_sz, 0), blockB(block_sz * block_sz, 0);
  int grid_size = root / block_sz;

  for (int bi = 0; bi < grid_size; ++bi) {
    for (int row = bi * block_sz; row < (bi + 1) * block_sz; ++row) {
      for (int shift = 0; shift < bi; ++shift) {
        shiftRow(matrixA, root, row, block_sz);
      }
    }
  }

  for (int bj = 0; bj < grid_size; ++bj) {
    for (int col = bj * block_sz; col < (bj + 1) * block_sz; ++col) {
      for (int shift = 0; shift < bj; ++shift) {
        shiftColumn(matrixB, root, col, block_sz);
      }
    }
  }

  int p = root / block_sz;
  for (int step = 0; step < p; step++) {
    for (int bi = 0; bi < root / block_sz; bi++) {
      for (int bj = 0; bj < root / block_sz; bj++) {
        int start = (bi * block_sz) * root + (bj * block_sz);

        for (int i = 0; i < block_sz; i++) {
          for (int j = 0; j < block_sz; j++) {
            int index = start + i * root + j;
            blockA[i * block_sz + j] = matrixA[index];
            blockB[i * block_sz + j] = matrixB[index];
          }
        }

        for (int i = 0; i < block_sz; i++) {
          for (int j = 0; j < block_sz; j++) {
            for (int k = 0; k < block_sz; k++) {
              int index = (bi * block_sz + i) * root + (bj * block_sz + j);

              matrixC[index] += blockA[i * block_sz + k] * blockB[k * block_sz + j];
            }
          }
        }
      }
    }
    if (step < p - 1) {
      shiftBlocksLeft(matrixA, root, block_sz);
      shiftBlocksUp(matrixB, root, block_sz);
    }
  }

  return true;
}

bool odintsov_m_mulmatix_cannon_seq::MulMatrixCannonSequential::PostProcessingImpl() {
  int szC = matrixC.size();
  for (int i = 0; i < szC; i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = matrixC[i];
  }
  return true;
}
