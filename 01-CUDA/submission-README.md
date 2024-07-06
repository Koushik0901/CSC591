# Matrix Multiplication using Parallel Computing

This README provides an overview of the implementation of matrix multiplication using various techniques, including CPU, CUDA, atomic add, parallel reduction, warp level primitives, and tensor cores. All these programs are run and tested on Google Colab.

## ABSTRACT
This report is a compilation of the understanding and application of the fundamentals of CUDA GPU using C++. Firstly, a simple matrix multiplication program is written in C++ and optimized to incorporate parallel computing in CUDA. This program was then changed to use more threads for each corresponding output element of the resultant matrix; this caused race conditions, which were solved using atomic operations, parallel reduction, and warp-level primitives. The effectiveness of these different approaches to eliminate race conditions was then measured and stated. This report also briefly compares tensor and CUDA cores with a view to understanding their suitability in parallel computation. The study’s results, therefore, affirm the effectiveness of CUDA programming in enhancing computational operations, especially in machine learning.

## CPU Implementation
The CPU implementation of matrix multiplication involves using traditional nested loops to iterate over the matrices and perform the multiplication operation.

To compile and execute the CPU implementation of matrix multiplication, follow these steps:

1. Upload the `mat_mul_cpu.cpp` file into Google Colab.
2. On a new cell, use the following command to compile the program:
    ```
    g++ -o mat_mul_cpu mat_mul_cpu.cpp
    ```
4. Once the compilation is successful, execute the program using the following command:
    ```
    ./mat_mul_cpu
    ```

## CUDA Implementation
The CUDA implementation utilizes the power of GPU parallelism to accelerate matrix multiplication. It leverages CUDA kernels and device memory to efficiently perform the computation.

To compile and execute the CUDA implementation of matrix multiplication, follow these steps:

1. Upload the `mat_mul_cuda.cu` file into Google Colab.
2. On a new cell, use the following command to compile the program:
    ```
    nvcc -o mat_mul_cuda mat_mul_cuda.cu
    ```
3. Once the compilation is successful, execute the program using the following command:
    ```
    ./mat_mul_cuda
    ```

Please refer to the source code for detailed implementation and usage instructions.



## Atomic Add
The atomic add technique is used to handle concurrent write operations to the same memory location. It ensures that the addition operation is performed atomically, avoiding race conditions.

To compile and execute the CUDA implementation with Atomic Add of matrix multiplication, follow these steps:

1. Upload the `mat_mul_atomic.cu` file into Google Colab.
2. On a new cell, use the following command to compile the program:
    ```
    nvcc -o mat_mul_atomic mat_mul_atomic.cu
    ```
3. Once the compilation is successful, execute the program using the following command:
    ```
    ./mat_mul_atomic
    ```

Please refer to the source code for detailed implementation and usage instructions.

## Parallel Reduction
Parallel reduction is a technique that allows for efficient computation of sums across multiple threads. It reduces the number of operations required to compute the final result.

To compile and execute the CUDA implementation with Parallel Reduction of matrix multiplication, follow these steps:

1. Upload the `mat_mul_atomic.cu` file into Google Colab.
2. On a new cell, use the following command to compile the program:
    ```
    nvcc -o ./mat_mul_parallel_reduction mat_mul_parallel_reduction.cu
    ```
3. Once the compilation is successful, execute the program using the following command:
    ```
    ./mat_mul_parallel_reduction
    ```

Please refer to the source code for detailed implementation and usage instructions.

## Warp Level Primitives
Warp level primitives are low-level operations provided by CUDA that allow for efficient synchronization and communication between threads within a warp.

To compile and execute the CUDA implementation with Warp Level Primitives of matrix multiplication, follow these steps:

1. Upload the `mat_mul_warp.cu` file into Google Colab.
2. On a new cell, use the following command to compile the program:
    ```
    nvcc -o mat_mul_warp mat_mul_warp.cu
    ```
3. Once the compilation is successful, execute the program using the following command:
    ```
    ./mat_mul_warp
    ```

Please refer to the source code for detailed implementation and usage instructions.

## Tensor Cores
Tensor cores are specialized hardware units available in certain NVIDIA GPUs. They provide high-performance matrix multiplication operations specifically optimized for deep learning workloads.

To compile and execute the cuBLAS tensor version of matrix multiplication, follow these steps:

1. Upload the `mat_mul_tensor.cu` file into Google Colab.
2. On a new cell, use the following command to compile the program:
    ```
    nvcc -o mat_mul_tensor mat_mul_tensor.cu -lcublas
    ```
3. Once the compilation is successful, execute the program using the following command:
    ```
    ./mat_mul_tensor
    ```

Please refer to the source code for detailed implementation and usage instructions.
