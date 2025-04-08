/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <libtensor/filter.hh>
#include <libtensor/libtensor.hh>

using Tensor2D = libtensor::Tensor<double, 2>;

Tensor2D get_tensor() {
  auto t = Tensor2D::fromShape({3, 3});
  t[0][0] = 0.0, t[0][1] = 0.0, t[0][2] = 0.0;
  t[1][0] = 0.0, t[1][1] = 5.0, t[1][2] = 0.0;
  t[2][0] = 0.0, t[2][1] = 0.0, t[2][2] = 0.0;
  return t;
}

Tensor2D get_2d_filter() {
  auto f = Tensor2D::fromShape({3, 3});
  f[0][0] = 0.0, f[0][1] = 1.0, f[0][2] = 0.0;
  f[1][0] = 2.0, f[1][1] = -6.0, f[1][2] = 2.0;
  f[2][0] = 0.0, f[2][1] = 1.0, f[2][2] = 0.0;
  return f;
}

TEST(filter, conv2d) {
  const auto filter = get_2d_filter();
  const auto t1 = Tensor2D::fromShape({10, 10}).fill(1.0);
  const auto t2 = get_tensor();
  const auto expect1 = Tensor2D::like(t1).fill(0.0);
  auto expect2 = Tensor2D::like(t2);

  expect2[0][0] = 0.0, expect2[0][1] = 10.0, expect2[0][2] = 0.0;
  expect2[1][0] = 20.0, expect2[1][1] = -30.0, expect2[1][2] = 20.0;
  expect2[2][0] = 0.0, expect2[2][1] = 10.0, expect2[2][2] = 0.0;

  auto actual1 = Tensor2D::like(t1);
  libtensor::conv2d<double>(t1, filter, actual1);
  ASSERT_EQ(expect1, actual1);

  auto actual2 = Tensor2D::like(t2);
  libtensor::conv2d<double>(t2, filter, actual2);
  ASSERT_EQ(expect2, actual2);
}
