#include <cstdio>
#include <iostream>
#include <chrono>

# define TILE_SIZE 32 // assuming block size is the same as TILE_SIZE
# define COARSE_SIZE 4 // how many blocks to coarse

__global__ void matrixMul(float *mat1, float *mat2, float *out, int size) {
   // calculate row and startCol for this thread
   int row = blockIdx.y * TILE_SIZE + threadIdx.y;
   int startCol = blockIdx.x * TILE_SIZE * COARSE_SIZE + threadIdx.x;

    // initialised tile matrix for mat1 and mat2
    __shared__ float tile_mat1[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_mat2[TILE_SIZE][TILE_SIZE];

    // this thread will calculate c partial output values 
    float out_partial[COARSE_SIZE];
    for (int i=0; i<COARSE_SIZE; i++)
        out_partial[i] = 0.0f;

    // loop across the matrices in tiles
    for (int tile_idx=0; tile_idx<(size+TILE_SIZE-1)/TILE_SIZE; tile_idx++) {
        // load mat1 value into the tile_mat1
        if (row < size && (tile_idx * TILE_SIZE + threadIdx.x) < size)
            tile_mat1[threadIdx.y][threadIdx.x] = mat1[row * size + tile_idx * TILE_SIZE + threadIdx.x];
        else
            tile_mat1[threadIdx.y][threadIdx.x] = 0.0f;

        // loop until COARSE_SIZE for mat2
        for (int coarse_idx=0; coarse_idx<COARSE_SIZE; coarse_idx++) {
            // col value will be different according to the current coarse window
            int col = startCol + coarse_idx * TILE_SIZE;
            // load mat2 value into the tile_mat2
            if (col < size && (tile_idx * TILE_SIZE + threadIdx.y) < size)
                tile_mat2[threadIdx.y][threadIdx.x] = mat2[(tile_idx * TILE_SIZE + threadIdx.y) * size + col];
            else
                tile_mat2[threadIdx.y][threadIdx.x] = 0.0f;
            // wait till the tile_mat2 is loaded by all threads in the block
            __syncthreads();
            // perform mat mul between tiled matrices
            for (int k=0; k<TILE_SIZE; k++) 
                out_partial[coarse_idx] += tile_mat1[threadIdx.y][k] * tile_mat2[k][threadIdx.x];
            // wait till all threads in block have read the tiled matrices values
            __syncthreads();
        }
    }
    // copy values to the output matrix
    if (row < size) {
        for (int c = 0; c < COARSE_SIZE; c++) {
            int col = startCol + c * TILE_SIZE;
            if (col < size)
                out[row * size + col] = out_partial[c];
        }
    }
}

// Simple CPU implementation of matrix multiplication (for validation)
// Computes out = mat1 * mat2 for a square matrix of dimension 'size'
void cpuMatrixMul(float *mat1, float *mat2, float *out, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < size; k++) {
                sum += mat1[i * size + k] * mat2[k * size + j];
            }
            out[i * size + j] = sum;
        }
    }
}

// Function to compare two matrices
bool validateResults(float *cpuRes, float *gpuRes, int size) {
    for (int i = 0; i < size * size; i++) {
        if (fabs(cpuRes[i] - gpuRes[i]) > 1e-3) {
            return false;
        }
    }
    return true;
}

int main() {
    // Set the matrix dimensions (square matrices)
    int size = 128; // e.g., 512x512 matrices
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

    // Measure CPU matrix multiplication time
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuMatrixMul(h_A, h_B, h_C_cpu, size);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy host data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions.
    // Each block has TILE_SIZE x TILE_SIZE threads.
    // Note: Each thread computes COARSE_SIZE consecutive output elements.
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    // The grid dimensions:
    // For rows: we need enough blocks to cover 'size' rows
    // For columns: each block covers TILE_SIZE * COARSE_SIZE columns.
    dim3 gridDim((size + (TILE_SIZE * COARSE_SIZE) - 1) / (TILE_SIZE * COARSE_SIZE),
                 (size + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the CUDA kernel and measure time using cudaEvent
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

    // Validate the results by comparing GPU and CPU outputs
    if (validateResults(h_C_cpu, h_C_gpu, size))
        std::cout << "✅ CUDA kernel output is CORRECT!" << std::endl;
    else
        std::cout << "❌ CUDA kernel output is INCORRECT!" << std::endl;

    // Clean up: free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
