# ZAGRA: Zig Approximate Graph Retrieval Algorithm
[CAGRA](https://arxiv.org/pdf/2308.15136) implementation on CPU, written in Zig.

## Why?

ZAGRA is an attempt to explore the upper limits of CPU performance in graph-based vector search, aiming to achieve competitive search recall compared to other ANNS algorithms, while trying to squeeze every bit of performance out of the CPU.

To achieve this, I chose Zig to leverages Zig's unique capabilities, such as **comptime**, first-class **SIMD** support, and **explicit memory management**, to implement fast, parallelized algorithms with mechanical sympathy.

## Roadmap

### Phase 1: Vectors & Datasets (Finished)
- [x] Basic vector operations
  - [x] Support `f32` and `f64` data types (float and half), with dimensionality of 128, 256, and 512
  - [x] SIMD-accelerated squared Euclidean distance of vectors
  - [x] `align(64)` memory layout of vector buffers for cache efficiency
- [x] Vector dataset loading and vector access
  - [x] Support for `.npy` file loading via **buffered I/O** (`std.io.Reader`) or via **memory-mapping** `std.posix.mmap` for large datasets

### Phase 2: NN-Descent (Ongoing)
- [x] Struct-of-Arrays (SoA) data layout for heaps in graph
- [x] NN-Descent algorithm implementation:
  1. [x] Random neighbor population for initial graph
  2. [x] Forward and reverse neighbor sampling
  3. [x] Graph update proposal generation
  4. [ ] Applying graph updates
  5. [ ] Iterative refinement until convergence
- [x] Lock-free, multi-threaded graph operations

### Phase 3: Graph Optimization (Planned)
- [ ] Implement graph optimization techniques from CAGRA paper
  - [ ] Rank-based edge reordering
  - [ ] Reversed graph construction
  - [ ] Reordered graph and reversed graph combination
- [ ] Graph serialization and de-serialization for persistence

### Phase 4: Search Algorithm (Planned)
- [ ] Implement search algorithm as described in CAGRA paper

### Phase 5: Benchmarking & Evaluation (Planned)
- [ ] Benchmark against HNSW

## Acknowledgements
- [CAGRA paper](https://arxiv.org/pdf/2308.15136)
- [PyNNDescent](https://github.com/lmcinnes/pynndescent)
- [Znpy: Npy file reading and writing library in Zig](https://github.com/hunglongtrangithub/znpy)
- [Zig](https://ziglang.org/)
