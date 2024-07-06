
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

__global__ void matrix_multiply_with_reduction(float *A, float *B, float *C, int num_A_rows, int num_A_cols, int num_B_cols) {
    extern __shared__ float sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //int bx = blockIdx.x;
    //int by = blockIdx.y;

    //int row = by * blockDim.y + ty;
    //int col = bx * blockDim.x + tx;

    int thread_id = ty * blockDim.x + tx;
    int elem_id = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y / 16) + thread_id / 16;
    int part = thread_id % 16;

    if (elem_id < num_A_rows * num_B_cols) {
        int element_row = elem_id / num_B_cols;
        int element_col = elem_id % num_B_cols;

        float sum = 0.0f;
        int start = part * (num_A_cols / 16);
        int end = (part == 15) ? num_A_cols : (part + 1) * (num_A_cols / 16);

        for (int i = start; i < end; i++) {
            sum += A[element_row * num_A_cols + i] * B[i * num_B_cols + element_col];
        }

        sdata[thread_id] = sum;
    } else {
        sdata[thread_id] = 0.0f;
    }

    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = 8; stride > 0; stride /= 2) {
        if (part < stride) {
            sdata[thread_id] += sdata[thread_id + stride];
        }
        __syncthreads();
    }

    if (part == 0) {
        int element_row = elem_id / num_B_cols;
        int element_col = elem_id % num_B_cols;
        if (element_row < num_A_rows && element_col < num_B_cols) {
            C[element_row * num_B_cols + element_col] = sdata[thread_id];
        }
    }
}

void initializeMatrix(float* matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            matrix[i * numCols + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}

void runMatrixMultiplication(int num_A_rows, int num_A_cols, int num_B_cols) {
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((num_B_cols + threads_per_block.x - 1) / threads_per_block.x,
                    (num_A_rows + threads_per_block.y - 1) / threads_per_block.y);

    std::cout << "Matrix size: " << num_A_rows << "x" << num_B_cols << std::endl;
    std::cout << "Number of blocks: " << num_blocks.x << " " << num_blocks.y << std::endl;
    std::cout << "Number of threads per block: " << threads_per_block.x << " " << threads_per_block.y << std::endl;

    float *A, *B, *C;
    float *A_gpu, *B_gpu, *C_gpu;
    A = (float *) malloc(num_A_rows * num_A_cols * sizeof(float));
    B = (float *) malloc(num_A_cols * num_B_cols * sizeof(float));
    C = (float *) malloc(num_A_rows * num_B_cols * sizeof(float));

    std::srand(42);

    // Initialize matrices A and B with random values
    initializeMatrix(A, num_A_rows, num_A_cols);
    initializeMatrix(B, num_A_cols, num_B_cols);

    cudaMalloc((void **) &A_gpu, num_A_rows * num_A_cols * sizeof(float));
    cudaMalloc((void **) &B_gpu, num_A_cols * num_B_cols * sizeof(float));
    cudaMalloc((void **) &C_gpu, num_A_rows * num_B_cols * sizeof(float));

    cudaMemcpy(A_gpu, A, num_A_rows * num_A_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, num_A_cols * num_B_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize the output matrix on the GPU
    cudaMemset(C_gpu, 0, num_A_rows * num_B_cols * sizeof(float));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    size_t sharedMemSize = threads_per_block.x * threads_per_block.y * sizeof(float);
    matrix_multiply_with_reduction<<<num_blocks, threads_per_block, sharedMemSize>>>(A_gpu, B_gpu, C_gpu, num_A_rows, num_A_cols, num_B_cols);

    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C, C_gpu, num_A_rows * num_B_cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the resulting matrix C
    std::cout << "Matrix C (Result):" << std::endl;
    for (int i = 0; i < 10; i++) { // Print only the first 10 rows for brevity
        for (int j = 0; j < 10; j++) { // Print only the first 10 columns for brevity
            std::cout << C[i * num_B_cols + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Time taken for matrix multiplication: " << milliseconds / 1000 << " seconds" << std::endl;

    // Free device memory
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);

    // Free host memory
    free(A);
    free(B);
    free(C);
}

int main() {
    runMatrixMultiplication(1000, 1000, 1000);
    runMatrixMultiplication(2500, 2500, 2500);
    runMatrixMultiplication(5000, 5000, 5000);

    return 0;
}
