/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __LIBTENSOR__CORE__SHAPE__
#define __LIBTENSOR__CORE__SHAPE__

#include "decl.hh"
#include <array>
#include <ostream>

namespace libtensor {
template <std::size_t N>
class Shape : public std::array<std::size_t, N> {
  friend std::ostream &operator<<(std::ostream &os, const Shape<N> &s) {
    os << "(";
    for (std::size_t i = 0; i < N; ++i) {
      os << s[i];
      if (i < N - 1) {
        os << ", ";
      }
    }
    os << ")";
    return os;
  }
};
} // namespace libtensor

#endif
