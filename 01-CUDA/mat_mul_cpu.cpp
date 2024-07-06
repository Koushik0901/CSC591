
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

// Function to initialize a matrix with random values
void initializeMatrix(float* matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            matrix[i * numCols + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}

// Function to multiply matrices on the CPU
void matrixMultiplyCPU(float* A, float* B, float* C, int num_A_rows, int num_A_cols, int num_B_cols) {
    for (int row = 0; row < num_A_rows; row++) {
        for (int col = 0; col < num_B_cols; col++) {
            float sum = 0.0f;
            for (int i = 0; i < num_A_cols; i++) {
                sum += A[row * num_A_cols + i] * B[i * num_B_cols + col];
            }
            C[row * num_B_cols + col] = sum;
        }
    }
}

int main() {
    int num_A_rows = 1000, num_A_cols = 1000, num_B_cols = 1000;
    int seed = 42; // Set manual seed

    // Allocate memory for matrices A, B, and C
    float *A = new float[num_A_rows * num_A_cols];
    float *B = new float[num_A_cols * num_B_cols];
    float *C = new float[num_A_rows * num_B_cols];

    // Initialize random seed
    std::srand(seed);

    // Initialize matrices A and B with random values
    initializeMatrix(A, num_A_rows, num_A_cols);
    initializeMatrix(B, num_A_cols, num_B_cols);

    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication on CPU
    matrixMultiplyCPU(A, B, C, num_A_rows, num_A_cols, num_B_cols);

    // Stop measuring time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Print the resulting matrix C
    std::cout << "Matrix C (Result):" << std::endl;
    for (int i = 0; i < 10; i++) { // Print only the first 10 rows for brevity
        for (int j = 0; j < 10; j++) { // Print only the first 10 columns for brevity
            std::cout << C[i * num_B_cols + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print the time taken for matrix multiplication
    std::cout << "Time taken for matrix multiplication: " << elapsed.count() << " seconds" << std::endl;

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
