#include <cstdio>
#include <iostream>
#include <chrono>

# define TILE_SIZE 32 // assuming block size is the same as TILE_SIZE

// perform matrix multiplication bw mat1 and mat2.T
__global__ void matrixMul(float *mat1, float *mat2, float *out, int size) {
   // calculate row and startCol for this thread
   int row = blockIdx.y * TILE_SIZE + threadIdx.y;
   int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // initialised tile matrix for mat1 and mat2
    __shared__ float tile_mat1[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_mat2[TILE_SIZE][TILE_SIZE];

    // result for this thread
    int ans = 0.0f;

    // loop across the matrices in tiles
    for (int tile_idx=0; tile_idx<(size+TILE_SIZE-1)/TILE_SIZE; tile_idx++) {
        // load mat1 value into the tile_mat1 (row major order)
        if (row < size && (tile_idx * TILE_SIZE + threadIdx.x) < size)
            tile_mat1[threadIdx.y][threadIdx.x] = mat1[row * size + tile_idx * TILE_SIZE + threadIdx.x];
        else
            tile_mat1[threadIdx.y][threadIdx.x] = 0.0f;

        // load mat2 matrix into tile_mat2 (col major order)
        if (col < size && (tile_idx * TILE_SIZE + threadIdx.y) < size)
            tile_mat2[threadIdx.x][threadIdx.y] = mat2[col * size + (tile_idx * TILE_SIZE + threadIdx.y)];
        else
            tile_mat2[threadIdx.x][threadIdx.y] = 0.0f;

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // copy values to the output matrix
        for (int k = 0; k < TILE_SIZE; k++)
            ans += tile_mat1[threadIdx.y][k] * tile_mat2[threadIdx.x][k];

        // Synchronize before loading the next tile
        __syncthreads();
    }
    if (row < size && col < size)
        out[row * size + col] = ans;
}

// CPU implementation for verification
void cpuMatrixMulATB(float *A, float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[j * size + k];  // Transpose B in-place
            }
            C[i * size + j] = sum;
        }
    }
}

// Validate GPU results with CPU results
bool validateResults(float *cpuRes, float *gpuRes, int size) {
    for (int i = 0; i < size * size; i++) {
        if (fabs(cpuRes[i] - gpuRes[i]) > 1e-3) {
            return false;
        }
    }
    return true;
}

int main() {
    int size = 64;  // Square matrices
    int numElements = size * size;
    int bytes = numElements * sizeof(float);

    // Allocate host memory
    float *h_A = (float*) malloc(bytes);
    float *h_B = (float*) malloc(bytes);
    float *h_C_cpu = (float*) malloc(bytes);
    float *h_C_gpu = (float*) malloc(bytes);

    // Initialize host matrices with random values (0 to 9)
    for (int i = 0; i < numElements; i++) {
        h_A[i] = rand() % 10;
        h_B[i] = rand() % 10;
    }

    // Run CPU matrix multiplication
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuMatrixMulATB(h_A, h_B, h_C_cpu, size);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((size + TILE_SIZE - 1) / TILE_SIZE, (size + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the CUDA kernel and measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;

    // Copy result from device to host
    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    // Validate results
    if (validateResults(h_C_cpu, h_C_gpu, size))
        std::cout << "✅ CUDA kernel output is CORRECT!" << std::endl;
    else
        std::cout << "❌ CUDA kernel output is INCORRECT!" << std::endl;

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
