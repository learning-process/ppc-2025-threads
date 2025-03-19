#include "seq/solovev_a_ccs_mmult_sparse/include/ccs_mmult_sparse.hpp"

#include <iostream>

bool solovev_a_matrix::Seq_MatMultCCS::PreProcessingImpl() {
  M1 = reinterpret_cast<MatrixInCCS_Sparse*>(task_data->inputs[0]);
  M2 = reinterpret_cast<MatrixInCCS_Sparse*>(task_data->inputs[1]);
  M3 = reinterpret_cast<MatrixInCCS_Sparse*>(task_data->outputs[0]);
  return true;
}

bool solovev_a_matrix::Seq_MatMultCCS::ValidationImpl() {
  int m1_c_n = reinterpret_cast<MatrixInCCS_Sparse*>(task_data->inputs[0])->c_n;
  int m2_r_n = reinterpret_cast<MatrixInCCS_Sparse*>(task_data->inputs[1])->r_n;
  return (m1_c_n == m2_r_n);
}

bool solovev_a_matrix::Seq_MatMultCCS::RunImpl() {
  M3->r_n = M1->r_n;
  M3->c_n = M2->c_n;
  M3->col_p.resize(M3->c_n + 1);
  M3->col_p[0] = 0;
  std::vector<int> available_el(M3->r_n);
  for (int m2_c = 0; m2_c < M3->c_n; ++m2_c) {
    for (int m3_r = 0; m3_r < M3->r_n; ++m3_r) {
      available_el[m3_r] = 0;
    }
    for (int m2_i = M2->col_p[m2_c]; m2_i < M2->col_p[m2_c + 1]; ++m2_i) {
      int m2_r = M2->row[m2_i];
      for (int m1_i = M1->col_p[m2_r]; m1_i < M1->col_p[m2_r + 1]; ++m1_i) {
        available_el[M1->row[m1_i]] = 1;
      }
    }
    int n_z_c_cnt = 0;
    for (int m3_r = 0; m3_r < M3->r_n; ++m3_r) {
      n_z_c_cnt += available_el[m3_r];
    }
    M3->col_p[m2_c + 1] = n_z_c_cnt + M3->col_p[m2_c];
  }

  int n_z_full = M3->col_p[M3->c_n];
  M3->n_z = n_z_full;
  M3->row.resize(n_z_full);
  M3->val.resize(n_z_full);

  std::complex<double> nought;
  std::complex<double> m2_val;
  std::vector<std::complex<double>> cask(M3->r_n);
  for (int m2_c = 0; m2_c < M3->c_n; ++m2_c) {
    for (int m3_r = 0; m3_r < M3->r_n; ++m3_r) {
      cask[m3_r] = nought;
      available_el[m3_r] = 0;
    }

    for (int m2_i = M2->col_p[m2_c]; m2_i < M2->col_p[m2_c + 1]; ++m2_i) {
      int m2_r = M2->row[m2_i];
      m2_val = M2->val[m2_i];
      for (int m1_i = M1->col_p[m2_r]; m1_i < M1->col_p[m2_r + 1]; ++m1_i) {
        int m1_row = M1->row[m1_i];
        cask[m1_row] += M1->val[m1_i] * m2_val;
        available_el[m1_row] = 1;
      }
    }

    int c_pos = M3->col_p[m2_c];
    for (int m3_r = 0; m3_r < M3->r_n; ++m3_r) {
      if (available_el[m3_r] != 0) {
        M3->row[c_pos] = m3_r;
        M3->val[c_pos++] = cask[m3_r];
      }
    }
  }

  return true;
}

bool solovev_a_matrix::Seq_MatMultCCS::PostProcessingImpl() { return true; }
