/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __LIBTENSOR__CORE__BASE__
#define __LIBTENSOR__CORE__BASE__

#include "decl.hh"
#include "functor.hh"
#include "shape.hh"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

#include <omp.h>

namespace libtensor {
template <typename T, std::size_t N>
class Tensor {
public:
  static const std::size_t n_dims = N;
  using scalar_type = T;
  using value_type = typename std::conditional<(n_dims == 1), T, Tensor<T, n_dims - 1>>::type;
  using Shape = libtensor::Shape<N>;
  using SubShape = libtensor::Shape<N - 1>;

  static Tensor fromShape(const Shape &s) { return Tensor(s); }
  static Tensor fromShape(Shape &&s) { return Tensor(std::forward<Shape>(s)); }
  static Tensor like(const Tensor &t) { return Tensor(t.shape()); }

private:
  Shape dims = {0};
  std::vector<value_type> data = {value_type{}};

public:
  Tensor(const Tensor &&t) {
    this->resize(t.shape());
    *this = t;
  }
  Tensor(const Tensor &t) {
    this->resize(t.shape());
    *this = t;
  }
  Tensor(const Shape &s) { this->resize(s); }
  Tensor(Shape &&s) { this->resize(std::forward<Shape>(s)); }
  Tensor() {}

  Tensor &resize(const Shape &s) {
    if (s == this->shape()) {
      return *this;
    }

    this->resize_recurse(s);

    return *this;
  }

  void resize_recurse(const Shape &s) {
    this->dims = s;
    this->data.resize(this->dims[0]);

    // If the tensor is not the base, resize recursively
    if constexpr (n_dims > 1) {
      SubShape ss;
      std::copy(std::next(this->dims.begin()), this->dims.end(), ss.begin());
#pragma omp parallel for
      for (std::size_t i = 0; i < this->dims[0]; ++i) {
        this->data.at(i).resize_recurse(ss);
      }
    }
  }

  inline const Shape &shape() const { return this->dims; }

  template <typename F, typename... Tensors>
  Tensor &map(F &&f, const Tensors &...others) noexcept {
    static_assert((std::is_same_v<std::decay_t<Tensors>, Tensor> && ...));

    map_recurse(std::forward<F>(f), others...);

    return *this;
  }

  template <typename F, typename... Tensors>
  Tensor &map_safe(F &&f, const Tensors &...others) {
    static_assert((std::is_same_v<std::decay_t<Tensors>, Tensor> && ...));
    if (((this->shape() != others.shape()) || ...)) {
      throw std::invalid_argument("invalid dimensions");
    }

    map_recurse_safe(std::forward<F>(f), others...);

    return *this;
  }

  template <typename F, typename... Tensors>
  void map_recurse(F &&f, const Tensors &...others) {
#pragma omp parallel for
    for (std::size_t i = 0; i < this->dims[0]; ++i) {
      if constexpr (n_dims > 1) {
        (*this)[i].map_recurse(std::forward<F>(f), (others[i])...);
      } else {
        f((*this)[i], (others[i])...);
      }
    }
  }

  template <typename F, typename... Tensors>
  void map_recurse_safe(F &&f, const Tensors &...others) {
#pragma omp parallel for
    for (std::size_t i = 0; i < this->dims.at(0); ++i) {
      if constexpr (n_dims > 1) {
        (*this).at(i).map_recurse(std::forward<F>(f), (others.at(i))...);
      } else {
        f((*this).at(i), (others.at(i))...);
      }
    }
  }

  Tensor &fill(const T &v) { return this->map(functor::FillFunctor(v)); }

  /* Getter and Setter */
  inline value_type &operator[](const std::size_t i) noexcept { return this->data[i]; }
  inline value_type &at(const std::size_t i) { return this->data.at(i); }
  inline const value_type &operator[](const std::size_t i) const noexcept { return this->data[i]; };
  inline const value_type at(const std::size_t i) const { return this->data.at(i); };

  /* Unary operators */
  Tensor operator+() const { return (*this); }
  Tensor operator-() const {
    auto ret = Tensor::fromShape(this->dims);
    ret.map([](T &t, const T &other) { t = -other; }, (*this));
    return ret;
  }

  /* Binary operators */
  Tensor &operator=(const Tensor &other) {
    if (this->shape() != other.shape()) {
      throw std::invalid_argument("invalid dimensions");
    }

    if (this == &other) {
      return *this;
    }
    for (std::size_t i = 0; i < this->dims[0]; ++i) {
      (*this)[i] = other[i];
    }
    return (*this);
  }

  bool operator==(const Tensor &rhs) const {
    if (this->dims != rhs.dims) {
      return false;
    }

    for (std::size_t i = 0; i < this->dims[0]; ++i) {
      if (!((*this)[i] == rhs[i])) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const Tensor &rhs) const { return !((*this) == rhs); }

  friend Tensor operator+(const T &lhs, const Tensor &rhs) {
    auto ret = Tensor::fromShape(rhs.dims);
    using Functor = functor::BindLhsWrapper<functor::SumFunctor<T>>;
    ret.map(Functor(lhs), rhs);
    return ret;
  }
  friend Tensor operator+(const Tensor &lhs, const T &rhs) {
    auto ret = Tensor::fromShape(lhs.dims);
    using Functor = functor::BindRhsWrapper<functor::SumFunctor<T>>;
    ret.map(Functor(rhs), lhs);
    return ret;
  }
  friend Tensor operator+(const Tensor &lhs, const Tensor &rhs) {
    auto ret = Tensor::fromShape(lhs.dims);
    ret.map(functor::SumFunctor<T>(), lhs, rhs);
    return ret;
  }
  friend Tensor operator-(const T &lhs, const Tensor &rhs) {
    auto ret = Tensor::fromShape(rhs.dims);
    using Functor = functor::BindLhsWrapper<functor::DiffFunctor<T>>;
    ret.map(Functor(lhs), rhs);
    return ret;
  }
  friend Tensor operator-(const Tensor &lhs, const T &rhs) {
    auto ret = Tensor::fromShape(lhs.dims);
    using Functor = functor::BindRhsWrapper<functor::DiffFunctor<T>>;
    ret.map(Functor(rhs), lhs);
    return ret;
  }
  friend Tensor operator-(const Tensor &lhs, const Tensor &rhs) {
    auto ret = Tensor::fromShape(lhs.dims);
    ret.map(functor::DiffFunctor<T>(), lhs, rhs);
    return ret;
  }
  friend Tensor operator*(const T &lhs, const Tensor &rhs) {
    auto ret = Tensor::fromShape(rhs.dims);
    using Functor = functor::BindLhsWrapper<functor::ProdFunctor<T>>;
    ret.map(Functor(lhs), rhs);
    return ret;
  }
  friend Tensor operator*(const Tensor &lhs, const T &rhs) {
    auto ret = Tensor::fromShape(lhs.dims);
    using Functor = functor::BindRhsWrapper<functor::ProdFunctor<T>>;
    ret.map(Functor(rhs), lhs);
    return ret;
  }
  friend Tensor operator*(const Tensor &lhs, const Tensor &rhs) {
    auto ret = Tensor::fromShape(lhs.dims);
    ret.map(functor::ProdFunctor<T>(), lhs, rhs);
    return ret;
  }
  friend Tensor operator/(const T &lhs, const Tensor &rhs) {
    auto ret = Tensor::fromShape(rhs.dims);
    using Functor = functor::BindLhsWrapper<functor::DivFunctor<T>>;
    ret.map(Functor(lhs), rhs);
    return ret;
  }
  friend Tensor operator/(const Tensor &lhs, const T &rhs) {
    auto ret = Tensor::fromShape(lhs.dims);
    using Functor = functor::BindRhsWrapper<functor::DivFunctor<T>>;
    ret.map(Functor(rhs), lhs);
    return ret;
  }
  friend Tensor operator/(const Tensor &lhs, const Tensor &rhs) {
    auto ret = Tensor::fromShape(lhs.dims);
    ret.map(functor::DivFunctor<T>(), lhs, rhs);
    return ret;
  }

  friend std::ostream &operator<<(std::ostream &os, const Tensor &t) {
    os << "{";
    for (std::size_t i = 0; i < t.dims[0]; ++i) {
      os << t[i];
      if (i < t.dims[0] - 1) {
        if (t.n_dims > 1) {
          os << std::endl;
        } else {
          os << " ";
        }
      }
    }
    os << "}";
    return os;
  }
};
} // namespace libtensor

#endif
