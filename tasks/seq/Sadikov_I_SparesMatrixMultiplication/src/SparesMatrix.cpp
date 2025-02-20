#include "seq/Sadikov_I_SparesMatrixMultiplication/include/SparesMatrix.hpp"

SparesMatrix SparesMatrix::Transpose(const SparesMatrix& matrix) {
  std::vector<double> val;
  std::vector<int> rows;
  std::vector<int> elemSum;
  std::vector<std::vector<double>> intermediateValues(matrix.GetRowsCount());
  std::vector<std::vector<int>> intermediateIndexes(matrix.GetColumnsCount());
  auto columnNumber = 0;
  auto columnCounter = 0;
  for (auto i = 0; i < matrix.GetValues().size(); ++i) {
    if (columnCounter == matrix.GetElementsSum()[columnNumber]) {
      columnNumber++;
    }
    columnCounter++;
    intermediateValues[matrix.GetRows()[i]].emplace_back(matrix.GetValues()[i]);
    intermediateIndexes[matrix.GetRows()[i]].emplace_back(columnNumber);
  }
  for (auto i = 0; i < intermediateValues.size(); ++i) {
    for (auto j = 0; j < intermediateValues[i].size(); ++j) {
      val.emplace_back(intermediateValues[i][j]);
      rows.emplace_back(intermediateIndexes[i][j]);
    }
    if (i > 0) {
      elemSum.emplace_back(intermediateValues[i].size() + elemSum[i - 1]);
    } else {
      elemSum.emplace_back(intermediateValues[i].size());
    }
  }
  return SparesMatrix(matrix.GetColumnsCount(), matrix.GetRowsCount(), val, rows, elemSum);
}

SparesMatrix SparesMatrix::operator*(const SparesMatrix& smatrix) {
  if (smatrix.GetColumnsCount() == 0 || smatrix.GetRowsCount() == 0 || this->GetColumnsCount() == 0 ||
      this->GetRowsCount() == 0) {
    return SparesMatrix();
  }
  std::vector<double> values;
  std::vector<int> rows;
  std::vector<int> elementsSum(smatrix.GetColumnsCount());
  auto fmatrix = Transpose(*this);
  for (auto i = 0; i < smatrix.GetElementsSum().size(); ++i) {
    for (auto j = 0; j < fmatrix.GetElementsSum().size(); ++j) {
      auto sum = 0.0;
      auto fmatrixElementsCount =
          j == 0 ? fmatrix.GetElementsSum()[j] : fmatrix.GetElementsSum()[j] - fmatrix.GetElementsSum()[j - 1];
      auto smatrixElementsCount =
          i == 0 ? smatrix.GetElementsSum()[i] : smatrix.GetElementsSum()[i] - smatrix.GetElementsSum()[i - 1];
      auto fmatrixStartIndex = j != 0 ? fmatrix.GetElementsSum()[j] - fmatrixElementsCount : 0;
      auto smatrixStartIndex = i != 0 ? smatrix.GetElementsSum()[i] - smatrixElementsCount : 0;
      for (auto n = 0; n < fmatrixElementsCount; n++) {
        for (auto n2 = 0; n2 < smatrixElementsCount; n2++) {
          if (fmatrix.GetRows()[fmatrixStartIndex + n] == smatrix.GetRows()[smatrixStartIndex + n2]) {
            sum += fmatrix.GetValues()[n + fmatrixStartIndex] * smatrix.GetValues()[n2 + smatrixStartIndex];
          }
        }
      }
      if (sum > m_epsilon) {
        // Write sum data
        values.emplace_back(sum);
        rows.emplace_back(j);
        elementsSum[i]++;
      }
    }
  }
  for (auto i = 1; i < elementsSum.size(); ++i) {
    elementsSum[i] = elementsSum[i] + elementsSum[i - 1];
  }
  return SparesMatrix(smatrix.GetColumnsCount(), smatrix.GetColumnsCount(), values, rows, elementsSum);
}

SparesMatrix MatrixToSpares(int rowsCount, int columnsCount, const std::vector<double>& values) {
  std::vector<double> val;
  std::vector<int> sums(columnsCount, 0);
  std::vector<int> rows;
  for (auto i = 0; i < columnsCount; ++i) {
    for (auto j = 0; j < rowsCount; ++j) {
      if (values[i + columnsCount * j] != 0) {
        val.emplace_back(values[i + columnsCount * j]);
        rows.emplace_back(j);
        sums[i] += 1;
      }
    }
    if (i != columnsCount - 1) {
      sums[i + 1] = sums[i];
    }
  }
  return SparesMatrix(rowsCount, columnsCount, val, rows, sums);
}

std::vector<double> FromSparesMatrix(const SparesMatrix& matrix) {
  std::vector<double> simplMatrix(matrix.GetRowsCount() * matrix.GetColumnsCount(), 0.0);
  auto columnNumber = 0;
  auto columnCounter = 0;
  for (auto i = 0; i < matrix.GetValues().size(); ++i) {
    if (columnCounter >= matrix.GetElementsSum()[columnNumber]) {
      columnNumber++;
    }
    columnCounter++;
    if (columnNumber > 0 && matrix.GetElementsSum()[columnNumber] - matrix.GetElementsSum()[columnNumber - 1] == 0) {
      columnNumber++;
    }
    simplMatrix[columnNumber + matrix.GetRows()[i] * matrix.GetColumnsCount()] = matrix.GetValues()[i];
  }
  return simplMatrix;
}

std::ostream& operator<<(std::ostream& os, const SparesMatrix& matrix) {
  os << "VALUES" << std::endl;
  for (auto i = 0; i < matrix.GetValues().size(); ++i) {
    os << matrix.GetValues()[i] << " ";
  }
  os << std::endl << "ROWS" << std::endl;
  for (auto i = 0; i < matrix.GetRows().size(); ++i) {
    os << matrix.GetRows()[i] << " ";
  }
  os << std::endl << "ElementsSum" << std::endl;
  for (auto i = 0; i < matrix.GetElementsSum().size(); ++i) {
    os << matrix.GetElementsSum()[i] << " ";
  }
  os << std::endl;
  return os;
}