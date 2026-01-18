# miniDL

miniDL is a minimal deep learning framework written in modern C++,  
designed to explore **tensor systems, automatic differentiation, and training loops**
from first principles.

---
## Overview
**miniDL** implements the core components of a deep learning framework up to
a complete training loop:
- Tensor system with views, strides, and broadcasting
- Core operations (pointwise ops, matmul)
- Automatic differentiation engine
- Neural network modules and optimizers
- Data loading and simple training examples

The goal of this project is **learning and experimentation**, not production use.

---
## Build & Run

### Requirements

- C++17 compatible compiler
- CMake ≥ 3.20
- OpenMP-enabled toolchain

### Build

```bash
cmake -S . -B build -DMINIDL_BUILD_EXAMPLES=ON -DMINIDL_BUILD_BENCHMARKS=ON
cmake --build build -j
```

**Note (macOS / AppleClang)**
OpenMP is not enabled by default on AppleClang.
Install `libomp` and configure CMake with:
```bash
cmake -S . -B build \
  -DMINIDL_BUILD_EXAMPLES=ON \
  -DMINIDL_BUILD_BENCHMARKS=ON \
  -DCMAKE_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/opt/libomp/lib -lomp"
```

---
## Quick Start
### Tensor & Ops
```cpp
Tensor a = Tensor::rand_uniform({2, 3}, DType::f32);
Tensor b = a.transpose({1, 0});
Tensor c = ops::add(a, b);

std::cout << c << std::endl;
```

### Autograd
```cpp
Tensor x = Tensor::rand_uniform({4, 8}, DType::f32, nullptr, true);
Tensor w = Tensor::rand_uniform({8, 3}, DType::f32, nullptr, true);

Tensor y = ops::matmul(x, w);
Tensor loss = ops::sum(y);

loss.backward();

auto grad_w = w.grad();
```

---
## Benchmark

miniDL includes micro-benchmarks for evaluating the performance of its
matrix multiplication (GEMM) kernels.

The benchmarks compare:
- a native (naive) GEMM implementation
- a SIMD-optimized GEMM implementation

They are intended to study relative performance characteristics
(e.g. SIMD effects and multi-threading), rather than to compete with
highly optimized BLAS libraries.

Detailed build instructions, execution steps, and results are documented
in `benchmarks/README.md`.

---
## Examples
miniDL provides a single end-to-end training example based on MNIST.

### MNIST MLP

The example implements a simple multi-layer perceptron (MLP) for MNIST
classification using the full miniDL stack:

- Custom `Dataset` for loading raw MNIST IDX files
- `DataLoader` with batching and shuffling
- Two-layer MLP (`Linear → ReLU → Linear`)
- Cross-entropy loss
- SGD optimizer
- Training and evaluation loop

The model takes MNIST images of shape `[1, 28, 28]`, flattens them to
`[784]`, and outputs logits over 10 classes.

Source file:
```bash
examples/mnist_mlp.cpp
```

### Run

After building the project, run the example by providing the MNIST data
directory:
```bash
./build/examples/mnist_mlp ./data/mnist
```

The directory is expected to contain the standard MNIST files:
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

During training, the program prints loss and top-1 accuracy periodically
and evaluates on the test set after each epoch.

---
## License
This project is licensed under the MIT License.