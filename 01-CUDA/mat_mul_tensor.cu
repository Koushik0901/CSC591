#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for clock()

// Error handling macro
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \\
    printf("Error at %s:%d\\n",__FILE__,__LINE__); \\
    return EXIT_FAILURE;}} while(0)

#define CUBLAS_CALL(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \\
    printf("Error at %s:%d\\n",__FILE__,__LINE__); \\
    return EXIT_FAILURE;}} while(0)



void initializeMatrix(float* matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            matrix[i * numCols + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}

void runMatrixMultiplication(int num_A_rows, int num_A_cols, int num_B_cols) {
    // Initialize matrices with random values
    int size_A = num_A_rows * num_A_cols;
    int size_B = num_A_cols * num_B_cols;
    int size_C = num_A_rows * num_B_cols;

    // Host memory allocation
    float *h_A = (float*)malloc(size_A * sizeof(float));
    float *h_B = (float*)malloc(size_B * sizeof(float));
    float *h_C = (float*)malloc(size_C * sizeof(float));

    std::srand(42); // Set seed for reproducibility

    // Initialize matrices with random values
    initializeMatrix(h_A, num_A_rows, num_A_cols);
    initializeMatrix(h_B, num_A_cols, num_B_cols);

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    CUDA_CALL(cudaMalloc((void**)&d_A, size_A * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_B, size_B * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_C, size_C * sizeof(float)));

    // Copy matrices from host to device
    CUDA_CALL(cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS handle initialization
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));

    // Time matrix multiplication
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start, 0));

    // Perform matrix multiplication using Tensor Cores
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_B_cols, num_A_rows, num_A_cols, &alpha, d_B, num_B_cols, d_A, num_A_cols, &beta, d_C, num_B_cols));

    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Matrix multiplication took " << milliseconds / 1000 << " seconds." << std::endl;

    // Copy result from device to host
    CUDA_CALL(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the resulting matrix C
    std::cout << "Matrix C (Result):" << std::endl;
    for (int i = 0; i < 10; i++) { // Print only the first 10 rows for brevity
        for (int j = 0; j < 10; j++) { // Print only the first 10 columns for brevity
            std::cout << h_C[i * num_B_cols + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
int main() {
    runMatrixMultiplication(1000, 1000, 1000);
    runMatrixMultiplication(2500, 2500, 2500);
    runMatrixMultiplication(5000, 5000, 5000);
    return 0;
}