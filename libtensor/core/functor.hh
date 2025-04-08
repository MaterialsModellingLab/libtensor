/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __LIBTENSOR__CORE__FUNCTOR__
#define __LIBTENSOR__CORE__FUNCTOR__

#include <type_traits>

namespace libtensor::functor {
/* Nullary Functor  */
template <typename T>
struct FillFunctor {
  using value_type = T;
  const T &xpr;
  FillFunctor(const T &x) : xpr(x) {}

  inline void operator()(T &ret) const { ret = this->xpr; }
};

/* Unary Operator */
template <typename T>
struct NegSingFunctor {
  using value_type = T;
  inline void operator()(T &ret) const { ret = -ret; }
};

/* Binary Functor */
template <typename T>
struct DiffFunctor {
  using value_type = T;
  inline void operator()(T &ret, const T &lhs, const T &rhs) const { ret = lhs - rhs; }
};

template <typename T>
struct DivFunctor {
  using value_type = T;
  inline void operator()(T &ret, const T &lhs, const T &rhs) const { ret = lhs / rhs; }
};

template <typename F>
struct BindLhsWrapper {
  using T = typename F::value_type;
  const T &lhs;
  const F functor;
  BindLhsWrapper(const T &l, const F f = {}) : lhs(l), functor(f) {}
  inline void operator()(T &ret, const T &rhs) const { this->functor.operator()(ret, lhs, rhs); }
};

template <typename F>
struct BindRhsWrapper {
  using T = typename F::value_type;
  const T &rhs;
  const F functor;
  BindRhsWrapper(const T &r, const F f = {}) : rhs(r), functor(f) {}
  inline void operator()(T &ret, const T &lhs) const { this->functor.operator()(ret, lhs, rhs); }
};

/* Variadic Functor  */
template <typename T>
struct SumFunctor {
  using value_type = T;
  template <typename... Args>
  inline void operator()(T &ret, const Args &...args) const {
    static_assert((std::is_same_v<std::decay_t<Args>, T> && ...));
    ret = (args + ...);
  }
};

/* Variadic Functor  */
template <typename T>
struct ProdFunctor {
  using value_type = T;
  template <typename... Args>
  inline void operator()(T &ret, const Args &...args) const {
    static_assert((std::is_same_v<std::decay_t<Args>, T> && ...));
    ret = (args * ...);
  }
};
} // namespace libtensor::functor

#endif