
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

// Kernel function to perform matrix multiplication on the GPU
__global__ void matrix_multiply(float *A, float *B, float *C, int num_A_rows, int num_A_cols, int num_B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_A_rows && col < num_B_cols) {
        float sum = 0.0f;
        for (int i = 0; i < num_A_cols; i++) {
            sum += A[row * num_A_cols + i] * B[i * num_B_cols + col];
        }
        C[row * num_B_cols + col] = sum;
    }
}

// Function to initialize a matrix with random values
void initializeMatrix(float* matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            matrix[i * numCols + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}

void runMatrixMultiplication(int num_A_rows, int num_A_cols, int num_B_cols) {
   
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((num_B_cols + threads_per_block.x - 1) / threads_per_block.x,
                    (num_A_rows + threads_per_block.y - 1) / threads_per_block.y);

    std::cout << "Number of blocks: " << num_blocks.x << " " << num_blocks.y << std::endl;
    std::cout << "Number of threads per block: " << threads_per_block.x << " " << threads_per_block.y << std::endl;

    float *A, *B, *C;
    float *A_gpu, *B_gpu, *C_gpu;
    A = (float *)malloc(num_A_rows * num_A_cols * sizeof(float));
    B = (float *)malloc(num_A_cols * num_B_cols * sizeof(float));
    C = (float *)malloc(num_A_rows * num_B_cols * sizeof(float));

    std::srand(42);

    // Initialize matrices A and B with random values
    initializeMatrix(A, num_A_rows, num_A_cols);
    initializeMatrix(B, num_A_cols, num_B_cols);

    cudaMalloc((void **)&A_gpu, num_A_rows * num_A_cols * sizeof(float));
    cudaMalloc((void **)&B_gpu, num_A_cols * num_B_cols * sizeof(float));
    cudaMalloc((void **)&C_gpu, num_A_rows * num_B_cols * sizeof(float));

    cudaMemcpy(A_gpu, A, num_A_rows * num_A_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, num_A_cols * num_B_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch the matrix multiplication kernel
    matrix_multiply<<<num_blocks, threads_per_block>>>(A_gpu, B_gpu, C_gpu, num_A_rows, num_A_cols, num_B_cols);
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
