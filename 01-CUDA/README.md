# Introduction to CUDA GPU Programming

_Fundamentals of CUDA programming and the single-instruction-multiple-thread (SIMT) threading model_

## Key Objectives

 1. To be able to describe solutions to problems using the SIMT model of parallelism popularised by Nvidia/CUDA
 2. To be able to implement basic CUDA programs that have applications to machine leaerning, such as matrix multiplication
 3. To understand how tensor cores differ from typical CUDA cores and how each of these differ from traditional CPU cores

## Key Deliverables

|Name  |Date        |Description|
|------|------------|-----------|
|Report|30 June 2024|A report describing a CUDA implementation of matrix multiplication with tensor cores|

The report should be in .pdf format and not longer than six pages. It is recommended, but not required, to use [the standard format of the Association of Computing Machinery (ACM)](https://www.acm.org/publications/proceedings-template), preferably in double-column layout.
The focus of the report should be on describing how you worked through these milestones. 
The grade will be the percentage of milestones achieved.

In preparing the report, you should work through the following milestones:

 - [ ] Implement a "hello world" CUDA program to add two vectors A and B into a third one C
 - [ ] Implement matrix multiplication in CUDA
 - [ ] Analyse the performance of your implementation using a tool like [NSight Compute](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
 - [ ] Apply tiling with shared memory and/or registers to improve the memory throughput of matrix multiplication
 - [ ] Leverage tensor cores in your matrix multiplication kernel to reduce instruction count

## Tools & Resources

The following tools and resources might help get the project off the ground faster:
  * [A plugin to write CUDA code from Python jupyter notebooks](https://github.com/andreinechaev/nvcc4jupyter)
  * [An introduction to Thrust, an STL-like library to simplify some aspects of CUDA programming](https://docs.nvidia.com/cuda/archive/9.0/pdf/Thrust_Quick_Start_Guide.pdf)
  * [Nvidia slide deck for an intro to CUDA](https://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf)
  * [An introduction to CUDA by Nvidia](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
  * ["What Every Programmer Should Know about Memory" by Ulrich Drepper; Section 6.2.1 talks about "tiling" for matrix multiplication](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)

The following resources might be useful for making the matrix multiplication a bit faster:
  * [UC Riverside slides on histograms](https://www.cs.ucr.edu/~mchow009/teaching/cs147/winter20/slides/11-Histogram.pdf): my usual example for teaching about data races and atomics in CUDA the first time
  * [More advanced histogram code from Nvidia](https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)
  * [Recent intermediate-level CUDA course by AMD](https://gpu-primitives-course.github.io/): look specifically for the part on "parallel reduction"
