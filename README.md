# LBM: lattice Boltzmann methods

This repository contains the code for optimizing a numerical simulation of Computational Fluid Dynamics (CFD). The simulation is based on the Lattice Boltzmann Method (LBM), a relatively recent technique which consists in replacing the Navier-Stokes equations with a discretization of the Boltzmann equation in order to simulate the complex behavior of fluids using streaming and collision (relaxation) processes. 

![](./assets/visual.gif)

## Installation

The code is written in C and uses OpenMP for parallelization. The code can be compiled using the following command:

```bash
make all
```

## Benchmarking

The code is optimized using the following techniques:

1. OpenMP parallelization
2. SIMD vectorization (AVX2)
3. Memory alignment
4. Cache blocking
5. Software pipelining

The code is benchmarked using execution time as metric. The benchmarking is performed on a 2.2 GHz Intel Xeon E5-2698 v4 processor with 20 cores and 40 threads. The following table shows the execution time for different optimization techniques:

| Optimization            | Execution time (s) |
| ----------------------- | ------------------ |
| Baseline                | 119.5              |
| +OpenMP parallelization | 26.9               |
| +SIMD vectorization     | 18.6               |
| +Memory Alignment       | 14.8               |
| +Cache Blocking         | 10.2               |
| +Software Pipelining    | 5.8                |

## Contribution

This repository uses the implementation by [Mousany](github.com/Mousany) and [Clarivy](github.com/Clarivy). However, commits are modified to correspond to authors of the optimization techniques. The baseline code can be found in [toast lab](https://toast-lab.sist.shanghaitech.edu.cn/courses/CS110@ShanghaiTech/Spring-2023/project/3/3.html).
