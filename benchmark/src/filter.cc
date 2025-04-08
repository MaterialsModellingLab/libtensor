/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark/benchmark.h>

#include <iostream>

#include <libtensor/filter.hh>
#include <libtensor/libtensor.hh>

using Tensor2D = libtensor::Tensor<double, 2>;

Tensor2D get_filter() {
  auto t = Tensor2D::fromShape({3, 3});
  t[0][0] = 0.0, t[0][1] = 1.0, t[0][2] = 0.0;
  t[1][0] = 1.0, t[1][1] = -4.0, t[1][2] = 0.0;
  t[2][0] = 0.0, t[2][1] = 1.0, t[2][2] = 0.0;
  return t;
}

static void BM_conv2d(benchmark::State &state) {
  const auto filter = get_filter();
  Tensor2D tensor, ret;
  for (auto _ : state) {
    state.PauseTiming();
    const auto size = static_cast<std::size_t>(128);
    const auto shape = Tensor2D::Shape({size, size});
    tensor.resize(shape);
    ret.resize(shape);
    state.ResumeTiming();
    libtensor::conv2d<double>(tensor, filter, ret);
  }
}
BENCHMARK(BM_conv2d)->Iterations(1000);

BENCHMARK_MAIN();
