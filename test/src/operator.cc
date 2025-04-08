/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <libtensor/libtensor.hh>

using Tensor2D = libtensor::Tensor<double, 2>;

Tensor2D get_t1() {
  auto t = Tensor2D::fromShape({2, 2});
  t[0][0] = 0.0, t[0][1] = 1.0;
  t[1][0] = 2.0, t[1][1] = 3.0;
  return t;
}

Tensor2D get_t2() {
  auto t = Tensor2D::fromShape({2, 2});
  t[0][0] = 3.0, t[0][1] = 2.0;
  t[1][0] = 1.0, t[1][1] = 0.0;
  return t;
}

TEST(operator, sing) {
  const auto t1 = get_t1();
  auto expect1 = Tensor2D::like(t1);
  expect1[0][0] = 0.0, expect1[0][1] = -1.0;
  expect1[1][0] = -2.0, expect1[1][1] = -3.0;

  ASSERT_EQ(get_t1(), +t1);
  ASSERT_EQ(expect1, -t1);
}

TEST(operator, operator_plus) {
  const auto t1 = get_t1();
  const auto t2 = get_t2();
  auto expect1 = Tensor2D::like(t1);
  const auto expect2 = Tensor2D::like(t2).fill(3.0);
  expect1[0][0] = 1.0, expect1[0][1] = 2.0;
  expect1[1][0] = 3.0, expect1[1][1] = 4.0;

  ASSERT_EQ(expect1, (t1 + 1.0));
  ASSERT_EQ(expect1, (1.0 + t1));
  ASSERT_EQ(expect2, (t1 + t2));
  ASSERT_EQ(expect2, (t2 + t1));
}

TEST(operator, operator_minus) {
  const auto t1 = get_t1();
  const auto t2 = get_t2();
  auto expect1 = Tensor2D::like(t1);
  auto expect2 = Tensor2D::like(t1);
  auto expect3 = Tensor2D::like(t1);
  auto expect4 = Tensor2D::like(t1);

  expect1[0][0] = -1.0, expect1[0][1] = 0.0;
  expect1[1][0] = 1.0, expect1[1][1] = 2.0;

  expect2[0][0] = 1.0, expect2[0][1] = 0.0;
  expect2[1][0] = -1.0, expect2[1][1] = -2.0;

  expect3[0][0] = -3.0, expect3[0][1] = -1.0;
  expect3[1][0] = 1.0, expect3[1][1] = 3.0;

  expect4[0][0] = 3.0, expect4[0][1] = 1.0;
  expect4[1][0] = -1.0, expect4[1][1] = -3.0;

  ASSERT_EQ(expect1, (t1 - 1.0));
  ASSERT_EQ(expect2, (1.0 - t1));
  ASSERT_EQ(expect3, (t1 - t2));
  ASSERT_EQ(expect4, (t2 - t1));
}

TEST(operator, operator_multi) {
  const auto t1 = get_t1();
  const auto t2 = get_t2();
  auto expect1 = Tensor2D::like(t1);
  auto expect2 = Tensor2D::like(t2);

  expect1[0][0] = 0.0, expect1[0][1] = 2.0;
  expect1[1][0] = 4.0, expect1[1][1] = 6.0;

  expect2[0][0] = 0.0, expect2[0][1] = 2.0;
  expect2[1][0] = 2.0, expect2[1][1] = 0.0;

  ASSERT_EQ(expect1, (t1 * 2.0));
  ASSERT_EQ(expect1, (2.0 * t1));
  ASSERT_EQ(expect2, (t1 * t2));
  ASSERT_EQ(expect2, (t2 * t1));
}

TEST(operator, operator_div) {
  const auto t1 = get_t1();
  const auto t2 = get_t2();
  auto expect1 = Tensor2D::like(t1);
  auto expect2 = Tensor2D::like(t1);
  auto expect3 = Tensor2D::like(t1);
  auto expect4 = Tensor2D::like(t1);

  expect1[0][0] = 0.0, expect1[0][1] = 0.5;
  expect1[1][0] = 1.0, expect1[1][1] = 1.5;

  expect2[0][0] = std::numeric_limits<double>::infinity(), expect2[0][1] = 2.0;
  expect2[1][0] = 1.0, expect2[1][1] = 2.0 / 3.0;

  expect3[0][0] = 0.0, expect3[0][1] = 0.5;
  expect3[1][0] = 2.0, expect3[1][1] = std::numeric_limits<double>::infinity();

  expect4[0][0] = std::numeric_limits<double>::infinity(), expect4[0][1] = 2.0;
  expect4[1][0] = 0.5, expect4[1][1] = 0.0;

  ASSERT_EQ(expect1, (t1 / 2.0));
  ASSERT_EQ(expect2, (2.0 / t1));
  ASSERT_EQ(expect3, (t1 / t2));
  ASSERT_EQ(expect4, (t2 / t1));
}
