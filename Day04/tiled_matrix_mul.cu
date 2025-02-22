#include <cstdio>
#include <iostream>
#include <chrono>

# define TILE_SIZE 32 // assuming block size is the same as TILE_SIZE

__global__ void matrixMul(float *mat1, float *mat2, float *out, int M, int K, int N) {
    // mat1 -> [M, K]; mat2 -> [K, N]; out -> [M, N]
    // find the row nad col of output matrix which this thread will caclulate
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // initialize the tile matrices for mat1 and mat2 in the shared memory for this block 
    __shared__ float tile_mat1[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_mat2[TILE_SIZE][TILE_SIZE];
    
    // iterate over all the TILE_SIZE submatrices
    float ans = 0.0f;
    for (int tile_idx = 0; tile_idx < (K + TILE_SIZE - 1)/ TILE_SIZE; tile_idx++) {
        // fill the tile matrices
        // bounds check
        if ((row < M) && (tile_idx * TILE_SIZE + threadIdx.x) < K) 
            tile_mat1[threadIdx.y][threadIdx.x] = mat1[(row * K) + (tile_idx * TILE_SIZE + threadIdx.x)];
        else
            tile_mat1[threadIdx.y][threadIdx.x] = 0.0f;
        if ((col < N) && (tile_idx * TILE_SIZE + threadIdx.y) < K)
            tile_mat2[threadIdx.y][threadIdx.x] = mat2[(tile_idx * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_mat2[threadIdx.y][threadIdx.x] = 0.0f;
        // wait till all the threads in the block fill the tiled matrices
        __syncthreads();

        // perform matrix multiplication bw the tiled matrices
        for (int k=0; k<TILE_SIZE; k++) 
            ans += tile_mat1[threadIdx.y][k] * tile_mat2[k][threadIdx.x];
        // wait till all threads have done the matrix multiplication from tiled matrices
        __syncthreads();
    }
    // store the answer in the output matrix
    if (row < M && col < N)
        out[row * N + col] = ans;
}

// CPU function for matrix multiplication (for validation)
void cpuMatrixMul(float *A, float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Function to compare CPU and GPU results
bool validateResults(float *cpuRes, float *gpuRes, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(cpuRes[i] - gpuRes[i]) > 1e-3) {  // Allow small floating point error
            return false;
        }
    }
    return true;
}

// Main function
int main() {
    int M = 128, K = 128, N = 128;  // Changeable matrix sizes

    // Allocate host memory
    float *h_A = (float*) malloc(M * K * sizeof(float));
    float *h_B = (float*) malloc(K * N * sizeof(float));
    float *h_C_cpu = (float*) malloc(M * N * sizeof(float));
    float *h_C_gpu = (float*) malloc(M * N * sizeof(float));

    // Initialize matrices with random values
    for (int i = 0; i < M * K; i++) h_A[i] = rand() % 10;
    for (int i = 0; i < K * N; i++) h_B[i] = rand() % 10;

    // Measure CPU time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpuMatrixMul(h_A, h_B, h_C_cpu, M, K, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "🖥 CPU Time: " << cpu_time << " ms" << std::endl;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    // Launch CUDA kernel
    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    // wait for finish
    cudaDeviceSynchronize();

    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
    std::cout << "🖥 GPU Time: " << gpu_time << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate results
    if (validateResults(h_C_cpu, h_C_gpu, M, N)) {
        std::cout << "✅ CUDA kernel output is CORRECT!" << std::endl;
    } else {
        std::cout << "❌ CUDA kernel output is INCORRECT!" << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
