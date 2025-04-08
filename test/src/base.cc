/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <libtensor/libtensor.hh>

using Tensor1D = libtensor::Tensor<double, 1>;
using Tensor2D = libtensor::Tensor<double, 2>;
using Tensor3D = libtensor::Tensor<double, 3>;

TEST(base, initialize) {
  Tensor1D t1({3});
  Tensor2D t2({2, 2});
  Tensor3D t3({1, 2, 2});
  Tensor3D t4({1, 3, 2});

  ASSERT_THAT(t1.shape(), testing::ElementsAre(3));
  ASSERT_THAT(t2.shape(), testing::ElementsAre(2, 2));
  ASSERT_THAT(t3.shape(), testing::ElementsAre(1, 2, 2));
  ASSERT_THAT(t4.shape(), testing::ElementsAre(1, 3, 2));

  ASSERT_NO_THROW((t3[0] = t2));
  ASSERT_THROW((t4[0] = t2), std::invalid_argument);
}

TEST(base, resize) {
  Tensor1D t1({3});
  Tensor2D t2({2, 2});
  Tensor3D t3({1, 2, 2});
  Tensor3D t4({1, 3, 2});
  auto t1_expect = Tensor1D::like(t1);
  auto t2_expect = Tensor2D::like(t2);
  auto t3_expect = Tensor3D::like(t3);
  auto t4_expect = Tensor3D::like(t4);

  ASSERT_THAT(t1_expect.shape(), testing::ElementsAre(3));
  ASSERT_THAT(t2_expect.shape(), testing::ElementsAre(2, 2));
  ASSERT_THAT(t3_expect.shape(), testing::ElementsAre(1, 2, 2));
  ASSERT_THAT(t4_expect.shape(), testing::ElementsAre(1, 3, 2));
}

TEST(base, fill) {
  const auto t1 = Tensor1D(Tensor1D::Shape{3}).fill(3.0);
  const auto t2 = Tensor2D(Tensor2D::Shape{2, 2}).fill(3.0);
  const auto t3 = Tensor3D(Tensor3D::Shape{1, 2, 2}).fill(3.0);
  const auto t4 = Tensor3D(Tensor3D::Shape{1, 3, 2}).fill(3.0);
  auto t1_expect = Tensor1D::like(t1);
  auto t2_expect = Tensor2D::like(t2);
  auto t3_expect = Tensor3D::like(t3);
  auto t4_expect = Tensor3D::like(t4);

  ASSERT_NE(t1, t1_expect);
  ASSERT_NE(t2, t2_expect);
  ASSERT_NE(t3, t3_expect);

  t1_expect[0] = 3.0, t1_expect[1] = 3.0, t1_expect[2] = 3.0;

  t2_expect[0][0] = 3.0, t2_expect[0][1] = 3.0;
  t2_expect[1][0] = 3.0, t2_expect[1][1] = 3.0;

  t3_expect[0][0][0] = 3.0, t3_expect[0][0][1] = 3.0;
  t3_expect[0][1][0] = 3.0, t3_expect[0][1][1] = 3.0;

  ASSERT_EQ(t1, t1_expect);
  ASSERT_EQ(t2, t2_expect);
  ASSERT_EQ(t3, t3_expect);
}

TEST(base, map) {
  auto t1 = Tensor2D::fromShape({2, 2});
  const auto tmp1 = Tensor2D::like(t1).fill(3.0);
  const auto tmp2 = Tensor2D::like(t1).fill(2.0);
  const auto expected1 = Tensor2D::like(t1).fill(3.0);
  const auto expected2 = Tensor2D::like(t1).fill(5.0);
  const auto invalid_shape = Tensor2D::fromShape({1, 1});

  t1.map([](double &v1, const double v2) { v1 = v2; }, tmp1);
  ASSERT_EQ(t1, expected1);

  t1.map([](double &v1, const double v2, const double v3) { v1 = (v2 + v3); }, tmp1, tmp2);
  ASSERT_EQ(t1, expected2);

  // Set invalid dimension
  ASSERT_THROW((t1.map_safe([](double &v1, const double v2) { v1 = v2; }, invalid_shape)),
               std::invalid_argument);
}
