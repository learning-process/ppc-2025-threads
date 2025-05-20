#include "all/kolodkin_g_multiplication_matrix_CRS/include/ops_all.hpp"

#include <cmath>
#include <complex>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <iostream>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

void kolodkin_g_multiplication_matrix_all::SparseMatrixCRS::AddValue(int row, Complex value, int col) {
  for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j) {
    if (colIndices[j] == col) {
      values[j] += value;
      return;
    }
  }
  colIndices.emplace_back(col);
  values.emplace_back(value);
  for (int i = row + 1; i <= numRows; ++i) {
    rowPtr[i]++;
  }
}

void kolodkin_g_multiplication_matrix_all::SparseMatrixCRS::PrintSparseMatrix(
    const kolodkin_g_multiplication_matrix_all::SparseMatrixCRS& matrix) {
  for (int i = 0; i < matrix.numRows; ++i) {
    for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; ++j) {
      std::cout << "Element at (" << i << ", " << matrix.colIndices[j] << ") = " << matrix.values[j] << '\n';
    }
  }
}

bool kolodkin_g_multiplication_matrix_all::AreEqualElems(const Complex& a, const Complex& b, double epsilon) {
  return std::abs(a.real() - b.real()) < epsilon && std::abs(a.imag() - b.imag()) < epsilon;
}

void kolodkin_g_multiplication_matrix_all::AddResult(std::vector<CoordVal>& results, int row, int col, Complex val) {
  results.push_back({row, col, val});
}

std::vector<Complex> kolodkin_g_multiplication_matrix_all::ParseMatrixIntoVec(const SparseMatrixCRS& mat) {
  std::vector<Complex> res = {};
  res.reserve(5 + mat.values.size() + mat.colIndices.size() + mat.rowPtr.size());
  res.emplace_back((double)mat.numRows);
  res.emplace_back((double)mat.numCols);
  res.emplace_back((double)mat.values.size());
  res.emplace_back((double)mat.colIndices.size());
  res.emplace_back((double)mat.rowPtr.size());
  for (unsigned int i = 0; i < (unsigned int)mat.values.size(); i++) {
    res.emplace_back(mat.values[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)mat.colIndices.size(); i++) {
    res.emplace_back(mat.colIndices[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)mat.rowPtr.size(); i++) {
    res.emplace_back(mat.rowPtr[i]);
  }
  return res;
}
bool kolodkin_g_multiplication_matrix_all::CheckMatrixesEquality(
    const kolodkin_g_multiplication_matrix_all::SparseMatrixCRS& a,
    const kolodkin_g_multiplication_matrix_all::SparseMatrixCRS& b) {
  if (a.numCols != b.numCols || a.numRows != b.numRows) {
    return false;
  }
  for (unsigned int i = 0; i < (unsigned int)a.numRows; ++i) {
    unsigned int this_row_start = a.rowPtr[i];
    unsigned int this_row_end = a.rowPtr[i + 1];
    unsigned int other_row_start = b.rowPtr[i];
    unsigned int other_row_end = b.rowPtr[i + 1];
    if ((this_row_end - this_row_start) != (other_row_end - other_row_start)) {
      return false;
    }
    for (unsigned int j = this_row_start; j < this_row_end; ++j) {
      bool found = false;
      for (unsigned int k = other_row_start; k < other_row_end; ++k) {
        if (a.colIndices[j] == b.colIndices[k] && AreEqualElems(a.values[j], b.values[k], 0.000001)) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
  }
  return true;
}
kolodkin_g_multiplication_matrix_all::SparseMatrixCRS kolodkin_g_multiplication_matrix_all::ParseVectorIntoMatrix(
    std::vector<Complex>& vec) {
  SparseMatrixCRS res;
  res.numRows = (int)vec[0].real();
  res.numCols = (int)vec[1].real();
  auto values_size = (unsigned int)vec[2].real();
  auto col_indices_size = (unsigned int)vec[3].real();
  auto row_ptr_size = (unsigned int)vec[4].real();
  res.values.reserve(values_size);
  res.colIndices.reserve(col_indices_size);
  res.rowPtr.reserve(row_ptr_size);
  for (unsigned int i = 0; i < values_size; i++) {
    res.values.emplace_back(vec[5 + i]);
  }
  for (unsigned int i = 0; i < col_indices_size; i++) {
    res.colIndices.emplace_back((int)vec[5 + values_size + i].real());
  }
  for (unsigned int i = 0; i < row_ptr_size; i++) {
    res.rowPtr.emplace_back((int)vec[5 + values_size + col_indices_size + i].real());
  }
  return res;
}

bool kolodkin_g_multiplication_matrix_all::TestTaskALL::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<Complex*>(task_data->inputs[0]);
  input_ = std::vector<Complex>(in_ptr, in_ptr + input_size);
  std::vector<Complex> matrix_a = {};
  std::vector<Complex> matrix_b = {};
  matrix_a.reserve(5 + (unsigned int)(input_[2].real() + input_[3].real() + input_[4].real()));
  matrix_b.reserve(input_.size() - (unsigned int)(5 + input_[2].real() + input_[3].real() + input_[4].real()));
  for (unsigned int i = 0; i < (unsigned int)(5 + input_[2].real() + input_[3].real() + input_[4].real()); i++) {
    matrix_a.emplace_back(input_[i]);
  }
  for (auto i = (unsigned int)(5 + input_[2].real() + input_[3].real() + input_[4].real());
       i < (unsigned int)input_.size(); i++) {
    matrix_b.emplace_back(input_[i]);
  }
  A_ = ParseVectorIntoMatrix(matrix_a);
  B_ = ParseVectorIntoMatrix(matrix_b);
  return true;
}

bool kolodkin_g_multiplication_matrix_all::TestTaskALL::ValidationImpl() {
  // Check equality of counts elements
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<Complex*>(task_data->inputs[0]);
  std::vector<Complex> vec = std::vector<Complex>(in_ptr, in_ptr + input_size);
  return !(vec[1] != vec[5 + (int)(vec[2].real() + vec[3].real() + vec[4].real())].real());
}

bool kolodkin_g_multiplication_matrix_all::TestTaskALL::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int aNumRows = A_.numRows;
  int aNumCols = A_.numCols;
  int bNumRows = B_.numRows;
  int bNumCols = B_.numCols;

  MPI_Bcast(&aNumRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&aNumCols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bNumRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bNumCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int rowPerProc = aNumRows / size;
  int remainder = aNumRows % size;

  int startRow = rank * rowPerProc + std::min(rank, remainder);
  int endRow = startRow + rowPerProc + (rank < remainder ? 1 : 0);

  std::vector<CoordVal> localResults;

  if (rank == 0) {
    int numThreads = ppc::util::GetPPCNumThreads();

    std::vector<std::thread> threads(numThreads);
    std::vector<std::vector<CoordVal>> threadResults(numThreads);
    int chunkSize = (endRow - startRow) / numThreads;
    int currentStart = startRow;

    auto processPart = [&](int startI, int endI, int threadIndex) {
      std::vector<CoordVal>& localThreadResults = threadResults[threadIndex];
      for (int i = startI; i < endI; ++i) {
        for (int jIdx = A_.rowPtr[i]; jIdx < A_.rowPtr[i + 1]; ++jIdx) {
          int colA = A_.colIndices[jIdx];
          Complex valueA = A_.values[jIdx];

          for (int kIdx = B_.rowPtr[colA]; kIdx < B_.rowPtr[colA + 1]; ++kIdx) {
            int colB = B_.colIndices[kIdx];
            Complex valueB = B_.values[kIdx];

            AddResult(localThreadResults, i, colB, valueA * valueB);
          }
        }
      }
    };

    for (int t = 0; t < numThreads; ++t) {
      int threadStart = currentStart + (chunkSize * t);
      int threadEnd = (t == numThreads - 1) ? endRow : threadStart + chunkSize;
      threads[t] = std::thread(processPart, threadStart, threadEnd, t);
    }
    for (auto& th : threads) {
      th.join();
    }
    for (const auto& vec : threadResults) {
      localResults.insert(localResults.end(), vec.begin(), vec.end());
    }

  } else {
    for (int i = startRow; i < endRow; ++i) {
      for (int jIdx = A_.rowPtr[i]; jIdx < A_.rowPtr[i + 1]; ++jIdx) {
        int colA = A_.colIndices[jIdx];
        Complex valueA = A_.values[jIdx];

        for (int kIdx = B_.rowPtr[colA]; kIdx < B_.rowPtr[colA + 1]; ++kIdx) {
          int colB = B_.colIndices[kIdx];
          Complex valueB = B_.values[kIdx];

          AddResult(localResults, i, colB, valueA * valueB);
        }
      }
    }
  }

  int localSizeBytes = static_cast<int>(localResults.size() * sizeof(CoordVal));

  std::vector<int> recvCounts(size);
  MPI_Gather(&localSizeBytes, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<int> displs(size);
    if (!displs.empty()) {
      displs[0] = 0;
      for (int i = 1; i < size; ++i) {
        displs[i] = displs[i - 1] + recvCounts[i - 1];
      }
    }
    int totalBytes = displs[size - 1] + recvCounts[size - 1];

    std::vector<char> recvBuffer(totalBytes);

    std::vector<CoordVal> sendBuffer(localResults.begin(), localResults.end());

    MPI_Gatherv(sendBuffer.data(), static_cast<int>(sendBuffer.size() * sizeof(CoordVal)), MPI_BYTE, recvBuffer.data(),
                recvCounts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    std::vector<CoordVal> allResults;
    allResults.reserve(totalBytes / sizeof(CoordVal));
    CoordVal* ptr = reinterpret_cast<CoordVal*>(recvBuffer.data());
    size_t countCoords = totalBytes / sizeof(CoordVal);
    for (size_t i = 0; i < countCoords; ++i) {
      allResults.push_back(ptr[i]);
    }
    SparseMatrixCRS C(aNumRows, bNumCols);
    std::map<std::pair<int, int>, Complex> resultMap;

    for (const auto& rv : allResults) {
      resultMap[{rv.row, rv.col}] += rv.value;
    }

    C.rowPtr.resize(aNumRows + 1);
    C.values.reserve(resultMap.size());
    C.colIndices.reserve(resultMap.size());

    C.rowPtr[0] = 0;

    for (int i = 0; i < aNumRows; ++i) {
      for (auto& kv : resultMap) {
        if (kv.first.first == i && kv.second != Complex(0)) {
          C.colIndices.push_back(kv.first.second);
          C.values.push_back(kv.second);
        }
      }
      C.rowPtr[i + 1] = static_cast<int>(C.colIndices.size());

      for (auto it = resultMap.begin(); it != resultMap.end();) {
        if (it->first.first == i)
          it = resultMap.erase(it);
        else
          ++it;
      }
    }
    output_ = ParseMatrixIntoVec(C);

  } else {
    MPI_Gatherv(localResults.data(), static_cast<int>(localResults.size() * sizeof(CoordVal)), MPI_BYTE, nullptr,
                nullptr, nullptr, MPI_BYTE, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool kolodkin_g_multiplication_matrix_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_.size(); i++) {
      reinterpret_cast<Complex*>(task_data->outputs[0])[i] = output_[i];
    }
  }
  return true;
}
