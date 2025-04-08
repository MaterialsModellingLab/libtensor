/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark/benchmark.h>
#include <iostream>
#include <libtensor/libtensor.hh>

using Tensor2D = libtensor::Tensor<double, 2>;
using Tensor3D = libtensor::Tensor<double, 3>;
using Functor = libtensor::functor::SumFunctor<double>;

static void BM_map3D(benchmark::State &state) {
  Tensor3D t1, t2, t3, ret;
  for (auto _ : state) {
    state.PauseTiming();
    const auto size = static_cast<std::size_t>(32);
    const auto shape = Tensor3D::Shape{size, size, size};
    t1.resize(shape).fill(1.0);
    t2.resize(shape).fill(2.0);
    t3.resize(shape).fill(3.0);
    ret.resize(shape);
    state.ResumeTiming();
    ret.map(Functor(), t1, t2, t3);
  }
}
BENCHMARK(BM_map3D)->Iterations(1000);

static void BM_map2D(benchmark::State &state) {
  Tensor2D t1, t2, t3, ret;
  for (auto _ : state) {
    state.PauseTiming();
    const auto size = static_cast<std::size_t>(32);
    const auto shape = Tensor2D::Shape{size, size};
    t1.resize(shape).fill(1.0);
    t2.resize(shape).fill(2.0);
    t3.resize(shape).fill(3.0);
    ret.resize(shape);
    state.ResumeTiming();
    ret.map(Functor(), t1, t2, t3);
  }
}
BENCHMARK(BM_map2D)->Iterations(1000);

BENCHMARK_MAIN();
