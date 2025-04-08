/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __LIBTENSOR__IMPL__FILTER__
#define __LIBTENSOR__IMPL__FILTER__

#include "libtensor.hh"

namespace libtensor {
enum class BorderType { INTERNAL, CONSTANT, REPLICATE, REFLECT, WRAP };

template <typename T, BorderType BT = BorderType::REFLECT>
void conv2d(const Tensor<T, 2> &tensor, const Tensor<T, 2> &filter, Tensor<T, 2> &ret,
            [[maybe_unused]] T cst = T{}) {
  const auto &t_shape = tensor.shape();
  const auto &f_shape = filter.shape();
  if (f_shape != Shape<2>({3, 3})) {
    throw std::invalid_argument("invalid shape of kernel given");
  }
  if (ret.shape() != tensor.shape()) {
    throw std::invalid_argument("shape result does not match between give & result");
  }
  ret.fill(0.0);

// internal
#pragma omp parallel for
  for (std::size_t i = 1; i < (t_shape[0] - 1); ++i) {
#pragma omp parallel for
    for (std::size_t j = 1; j < t_shape[1] - 1; ++j) {
      for (int64_t m = 0; m < static_cast<int64_t>(f_shape[0]); ++m) {
        for (int64_t n = 0; n < static_cast<int64_t>(f_shape[1]); ++n) {
          ret[i][j] += (tensor[i + m - 1][j + n - 1] * filter[n][m]);
        }
      }
    }
  }

  // boundary
  if constexpr (BT == BorderType::INTERNAL) {
    // do nothing
    return;
  }
  if constexpr (BT == BorderType::CONSTANT) {
    // TODO(anyone): Implement it
    throw std::invalid_argument("Not supported yet");
  }
  if constexpr (BT == BorderType::REPLICATE) {
    // TODO(anyone): Implement it
    throw std::invalid_argument("Not supported yet");
  }
  if constexpr (BT == BorderType::REFLECT) {
    const std::size_t y_lim = t_shape[0] - 1;
    const std::size_t x_lim = t_shape[1] - 1;
    // y direction
#pragma omp parallel for
    for (std::size_t i = 1; i < y_lim; ++i) {
      for (int64_t m = 0; m < static_cast<int64_t>(f_shape[0]); ++m) {
        for (int64_t n = 0; n < static_cast<int64_t>(f_shape[1]); ++n) {
          const int64_t y = i + m - 1;
          const int64_t xl = (n - 1) >= 0 ? n - 1 : 1;
          const int64_t xu = (n - 1) <= 0 ? x_lim + n - 1 : x_lim - 1;
          ret[i][0] += (tensor[y][xl] * filter[m][n]);
          ret[i][x_lim] += (tensor[y][xu] * filter[m][n]);
        }
      }
    }
    // x direction
#pragma omp parallel for
    for (std::size_t i = 1; i < x_lim; ++i) {
      for (int64_t m = 0; m < static_cast<int64_t>(f_shape[0]); ++m) {
        for (int64_t n = 0; n < static_cast<int64_t>(f_shape[1]); ++n) {
          const int64_t x = i + n - 1;
          const int64_t yl = (m - 1) >= 0 ? m - 1 : 1;
          const int64_t yu = (m - 1) <= 0 ? y_lim + m - 1 : y_lim - 1;
          ret[0][i] += (tensor[yl][x] * filter[m][n]);
          ret[y_lim][i] += (tensor[yu][x] * filter[m][n]);
        }
      }
    }
    // corner
    for (int64_t m = 0; m < static_cast<int64_t>(f_shape[0]); ++m) {
      for (int64_t n = 0; n < static_cast<int64_t>(f_shape[1]); ++n) {
        const int64_t yl = (m - 1) >= 0 ? m - 1 : 1;
        const int64_t xl = (n - 1) >= 0 ? n - 1 : 1;
        const int64_t yu = (m - 1) <= 0 ? y_lim + m - 1 : y_lim - 1;
        const int64_t xu = (n - 1) <= 0 ? x_lim + n - 1 : x_lim - 1;
        ret[0][0] += (tensor[yl][xl] * filter[m][n]);
        ret[0][x_lim] += (tensor[yl][xu] * filter[m][n]);
        ret[y_lim][0] += (tensor[yu][xl] * filter[m][n]);
        ret[y_lim][x_lim] += (tensor[yu][xu] * filter[m][n]);
      }
    }
  }
  if constexpr (BT == BorderType::WRAP) {
    throw std::invalid_argument("Not supported yet");
  }
}
} // namespace libtensor
#endif
