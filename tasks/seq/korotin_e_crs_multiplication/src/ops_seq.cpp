#include "seq/example/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

bool korotin_e_crs_multiplication_seq::CrsMultiplicationSequential::PreProcessingImpl() {
  A_N_ = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[0]);
  A_rI_ = std::vector<unsigned int>(in_ptr, in_ptr + A_N_);

  A_Nz_ = task_data->inputs_count[1];
  in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[1]);
  A_col_ = std::vector<int>(in_ptr, in_ptr + A_Nz_);

  auto *val_ptr = reinterpret_cast<double *>(task_data->inputs[2]);
  A_val_ = std::vector<double>(val_ptr, val_ptr + A_Nz_);

  B_N_ = task_data->inputs_count[3];
  in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[3]);
  B_rI_ = std::vector<int>(in_ptr, in_ptr + B_N_);

  B_Nz_ = task_data->inputs_count[4];
  in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[4]);
  B_col_ = std::vector<int>(in_ptr, in_ptr + B_Nz_);

  val_ptr = reinterpret_cast<double *>(task_data->inputs[5]);
  B_val_ = std::vector<double>(val_ptr, val_ptr + B_Nz_);

  unsigned int output_size = task_data->outputs_count[0];
  output_rI_ = std::vector<unsigned int>(output_size, 0);
  output_col_.clear();
  output_val_.clear();

  return true;
}

bool korotin_e_crs_multiplication_seq::CrsMultiplicationSequential::ValidationImpl() {
  return task_data->inputs_count[1] == task_data->inputs_count[2] &&  task_data->inputs_count[4] == task_data->inputs_count[5] && task_data->inputs_count[0] == task_data->outputs_count[0] && *std::max_element(task_data->inputs_[1].begin(), task_data->inpouts[1].end()) == task_data->inputs_count[3]-2;
}

bool korotin_e_crs_multiplication_seq::CrsMultiplicationSequential::RunImpl() {
  std::vector<unsigned int> trI(*std::max_element(B_col_.begin(), B_col_.end()) + 2, 0);
  unsigned int i;
  unsigned int j;
  for (i = 0; i < B_Nz_; i++)
    trI[B_col_[i] + 1]++;
  for (i = 1; i < trI.size(); i++)
    trI[i] += trI[i - 1];

  std::vector<unsigned int> tcol(B_Nz_, 0);
  std::vector<double> tval(B_Nz_, 0);
  for (i = 0; i < B_N_ - 1; i++)
    for (j = B_rI_[i]; i < B_rI_[i + 1]; j++} {
      tval[trI[B_col_[j]]] = B_val_[j];
      tcol[trI[B_col_[j]]] = i;
      trI[B_col_[j]]++;
    }
  for (i = trI.size() - 1; i > 0; i--)
    trI[i] = trI[i-1];
  trI[0] = 0;

  unsigned int ai;
  unsigned int bt;
  double sum;
  for (i = 0; i < A_N_ - 1; i++)
    for (j = 0; j < trI.size() - 1; j++) {
      sum = 0;
      ai = A_rI_[i];
      bt = trI[j];
      while (ai < A_rI_[i + 1] && bt < trI[j + 1]) {
        if (A_col_[ai] == tcol[bt]) {
	  sum += A_val_[ai] * tval[bt];
	  ai++;
	  bt++;
	}
	else if (col[ai] < tcol[bt]) ai++;
	else bt++;
      }
      if (sum != 0) {
        output_val_.push_back(sum);
	output_col_.push_back(j);
	output_rI_[i + 1]++;
      }
    }
  for (i = 1; i < A_N_; i++)
    output_rI_[i] += output_rI_[i - 1];
  return true;
}

bool korotin_e_crs_multiplication_seq::CrsMultiplicationSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_rI_.size(); i++) {
    reinterpret_cast<unsigned int *>(task_data->outputs[0])[i] = output_rI_[i];
  }
  for (size_t i = 0; i < output_col_,size(); i++) {
    reinterpret_cast<unsigned int *>(task_data->outputs[1])[i] = output_col_[i];
    reinterpret_cast<double *>(task_data->outputs[2])[i] = output_val_[i];
  }
  task_data->outputs_count.emplace_back(output_col_.size());
  task_data->outputs_count.emplace_back(output_val_.size());
  return true;
}
