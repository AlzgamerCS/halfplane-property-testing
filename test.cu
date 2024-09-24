#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

#define N 1024 // Size of the matrices (NxN)

// Kernel for matrix multiplication
__global__ void matrixMultiplyKernel(int *a, int *b, int *c, int n)
{
    // Get the row and column of the matrix element this thread should compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        int value = 0;
        // Compute the value of c[row][col]
        for (int k = 0; k < n; ++k)
        {
            value += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = value;
    }
}

void matrixMultiplyHost(const int *a, const int *b, int *c, int n)
{
    for (int row = 0; row < n; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            int value = 0;
            for (int k = 0; k < n; ++k)
            {
                value += a[row * n + k] * b[k * n + col];
            }
            c[row * n + col] = value;
        }
    }
}

int main()
{
    int size = N * N * sizeof(int); // Matrix size in bytes

    // Allocate host memory
    int *h_a = (int *)malloc(size);
    int *h_b = (int *)malloc(size);
    int *h_c = (int *)malloc(size);

    // Initialize host matrices with random values
    for (int i = 0; i < N * N; ++i)
    {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy matrices from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    // Wait for GPU to finish before accessing results
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    cout << duration.count() << endl;
    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Optionally, print part of the result matrix
    std::cout << "Resulting matrix C (showing first 5 elements):\n";
    for (int i = 0; i < 5; ++i)
    {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    start = std::chrono::high_resolution_clock::now();
    matrixMultiplyHost(h_a, h_b, h_c, N);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    cout << duration.count() << endl;

    std::cout << "Resulting matrix C (showing first 5 elements):\n";
    for (int i = 0; i < 5; ++i)
    {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}