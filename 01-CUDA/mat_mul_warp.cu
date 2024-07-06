
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

__global__ void matrix_multiply_with_warp(float *A, float *B, float *C, int num_A_rows, int num_A_cols, int num_B_cols) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global thread index in the grid
    int thread_id = (by * blockDim.y + ty) * (gridDim.x * blockDim.x) + (bx * blockDim.x + tx);
    
    // Calculate the element index each warp works on
    int elem_id = thread_id / 16;  // 16 threads work collaboratively per output element
    int lane = thread_id % 16;     // Lane index within the warp

    if (elem_id < num_A_rows * num_B_cols) {
        int element_row = elem_id / num_B_cols;
        int element_col = elem_id % num_B_cols;
        float sum = 0.0f;

        // Compute partial product
        for (int k = lane; k < num_A_cols; k += 16) {
            sum += A[element_row * num_A_cols + k] * B[k * num_B_cols + element_col];
        }

        // Warp-level reduction using __shfl_down_sync
        for (int offset = 8; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Thread 0 of each warp writes the result
        if (lane == 0) {
            atomicAdd(&C[element_row * num_B_cols + element_col], sum);
        }
    }
}

void initializeMatrix(float* matrix, int numRows, int numCols) {
    srand(42);
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

    std::cout << "Number of blocks: " << num_blocks.x << " " << num_blocks.y << std::endl;
    std::cout << "Number of threads per block: " << threads_per_block.x << " " << threads_per_block.y << std::endl;
    
    float *A, *B, *C;
    float *A_gpu, *B_gpu, *C_gpu;
    A = new float[num_A_rows * num_A_cols];
    B = new float[num_A_cols * num_B_cols];
    C = new float[num_A_rows * num_B_cols];

    initializeMatrix(A, num_A_rows, num_A_cols);
    initializeMatrix(B, num_A_cols, num_B_cols);

    cudaMalloc(&A_gpu, num_A_rows * num_A_cols * sizeof(float));
    cudaMalloc(&B_gpu, num_A_cols * num_B_cols * sizeof(float));
    cudaMalloc(&C_gpu, num_A_rows * num_B_cols * sizeof(float));

    cudaMemcpy(A_gpu, A, num_A_rows * num_A_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, num_A_cols * num_B_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(C_gpu, 0, num_A_rows * num_B_cols * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrix_multiply_with_warp<<<num_blocks, threads_per_block>>>(A_gpu, B_gpu, C_gpu, num_A_rows, num_A_cols, num_B_cols);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C, C_gpu, num_A_rows * num_B_cols * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << C[i * num_B_cols + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Time taken for matrix multiplication: " << milliseconds / 1000 << " seconds" << std::endl;

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    delete[] A;
    delete[] B;
    delete[] C;
}

int main() {
    runMatrixMultiplication(1000, 1000, 1000);
    runMatrixMultiplication(2500, 2500, 2500);
    runMatrixMultiplication(5000, 5000, 5000);
    return 0;
}
