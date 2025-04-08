[![CI](https://github.com/MaterialsModellingLab/libtensor/actions/workflows/ci.yaml/badge.svg)](https://github.com/MaterialsModellingLab/libtensor/actions/workflows/ci.yaml)

# libtensor
A simple, header-only, and parallelized C++ Tensor Library for handling the Phase-Field Model.

## Sample Program

Here is a simple example of how to use the C++ Tensor Library.

```cpp
#include <iostream>
#include <libtensor/libtensor.hh>

int main() {
  // Initialize 3D tensor
  using Tensor3D = libtensor::Tensor<double, 3>;
  Tensor3D tensor1({3, 10, 10});

  // Create a tensor with the same shape as tensor1
  Tensor3D tensor2 = Tensor3D::like(tensor1);

  // Fill tensor with some values
  tensor1.fill(1.0);

  // Print tensor values
  std::cout << tensor1 << std::endl;

  // Apply a functor to each element of the tensor
  tensor1.map([](double& x) { x *= 2.0; });
  std::cout << tensor1 << std::endl;

  return 0;
}
```

## Requirements
- `CMake`
- `Ninja`
- C++ Compiler
  - `C++17`
- OpenMP

## Build
```shell
cmake --preset debug # or release
cmake --build build/debug # or build/release
```

## Run Test
```shell
ctest --preset default --output-on-failure
```

## Install
After build,
```shell
cmake --install build/debug # or build/release
```
