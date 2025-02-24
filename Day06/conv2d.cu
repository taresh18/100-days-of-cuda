#include <cstdio>
#include <iostream>
#include <chrono>

# define IN_TILE_SIZE 32 // assuming block size is the same as IN_TILE_SIZE
# define FILTER_RADIUS 1
# define OUT_TILE_SIZE  (IN_TILE_SIZE - 2 * FILTER_RADIUS)
#define FILTER_DIM (2 * FILTER_RADIUS + 1) 

// initialise the filter matrix in constant memory
__constant__ float filter_c[FILTER_DIM][FILTER_DIM] = {
    {1.0f/9, 1.0f/9, 1.0f/9},
    {1.0f/9, 1.0f/9, 1.0f/9},
    {1.0f/9, 1.0f/9, 1.0f/9}
};

__global__ void conv2d(float *mat, float *out, int width, int height) {
    // row and col (of input / output tile) this thread is responsible for
    int row = blockIdx.y * OUT_TILE_SIZE + threadIdx.y - FILTER_RADIUS;
    int col = blockIdx.y * OUT_TILE_SIZE + threadIdx.x - FILTER_RADIUS;

    // initalize shared memory for the input tile and fill the corresponding element
    __shared__ float intput_tile_mat[IN_TILE_SIZE][IN_TILE_SIZE];

    if(row >= 0 && row < height && col >=0 && col < width) 
        intput_tile_mat[threadIdx.y][threadIdx.x] = mat[row * width + col];
    else
        intput_tile_mat[threadIdx.y][threadIdx.x] = 0.0f;

    // wait till all threads in block have loaded the elements into shared memory
    __syncthreads();

    // output tile index
    int out_tile_row = threadIdx.y - FILTER_RADIUS;
    int out_tile_col = threadIdx.x - FILTER_RADIUS;
    // bounds check
    if(row >= 0 && row < height && col >=0 && col < width) {
        if(out_tile_row >= 0 && out_tile_row < OUT_TILE_SIZE && out_tile_col >=0 && out_tile_col < OUT_TILE_SIZE) {
            float ans = 0.0f;
            for (int frow = 0; frow<2*FILTER_RADIUS+1; frow++) {
                for(int fcol=0; fcol<2*FILTER_RADIUS+1; fcol++) 
                    ans += intput_tile_mat[out_tile_row+frow][out_tile_col+fcol] * filter_c[frow][fcol];
            }
            out[row * width + col] = ans;
        }
    }
}


void cpuConv2D(const float *mat, float *out, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            // Iterate over filter window
            for (int fr = -FILTER_RADIUS; fr <= FILTER_RADIUS; fr++) {
                for (int fc = -FILTER_RADIUS; fc <= FILTER_RADIUS; fc++) {
                    int r = i + fr;
                    int c = j + fc;
                    float val = 0.0f;
                    if (r >= 0 && r < height && c >= 0 && c < width)
                        val = mat[r * width + c];
                    // For simplicity, assume filter is an averaging filter like constant memory
                    // (The constant memory filter used in GPU is defined as 1/9 each for a 3x3 filter)
                    float coef = 1.0f / 9.0f;
                    sum += val * coef;
                }
            }
            out[i * width + j] = sum;
        }
    }
}

// --------------------------------------------------------------------
// Main function: Allocate memory, run GPU and CPU implementations, and compare.
// --------------------------------------------------------------------
int main() {
    int width = 64;
    int height = 64;
    int numElements = width * height;
    int bytes = numElements * sizeof(float);

    // Allocate host memory
    float *h_mat = (float*) malloc(bytes);
    float *h_out_cpu = (float*) malloc(bytes);
    float *h_out_gpu = (float*) malloc(bytes);

    // Initialize the input matrix with some values (e.g., random values)
    for (int i = 0; i < numElements; i++) {
        h_mat[i] = rand() % 256 / 255.0f;  // normalized between 0 and 1
    }

    // Run CPU convolution and measure time
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuConv2D(h_mat, h_out_cpu, width, height);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU Convolution Time: " << cpu_time << " ms" << std::endl;

    // Allocate device memory
    float *d_mat, *d_out;
    cudaMalloc(&d_mat, bytes);
    cudaMalloc(&d_out, bytes);

    // Copy input matrix to device
    cudaMemcpy(d_mat, h_mat, bytes, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions.
    // Each block covers a tile of size IN_TILE_SIZE x IN_TILE_SIZE threads.
    dim3 blockDim(IN_TILE_SIZE, IN_TILE_SIZE);
    // The output "valid" region per block is OUT_TILE_SIZE x OUT_TILE_SIZE.
    dim3 gridDim((width + OUT_TILE_SIZE - 1) / OUT_TILE_SIZE,
                 (height + OUT_TILE_SIZE - 1) / OUT_TILE_SIZE);

    // Launch the convolution kernel and measure time using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    conv2d<<<gridDim, blockDim>>>(d_mat, d_out, width, height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    std::cout << "GPU Convolution Time: " << gpu_time << " ms" << std::endl;

    // Copy result from device to host
    cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost);

    // Compare CPU and GPU outputs
    int errors = 0;
    for (int i = 0; i < numElements; i++) {
        if (fabs(h_out_cpu[i] - h_out_gpu[i]) > 1e-3)
            errors++;
    }
    if (errors == 0)
        std::cout << "✅ CUDA convolution output is CORRECT!" << std::endl;
    else
        std::cout << "❌ CUDA convolution output is INCORRECT! " << errors << " mismatches." << std::endl;

    // Clean up
    cudaFree(d_mat);
    cudaFree(d_out);
    free(h_mat);
    free(h_out_cpu);
    free(h_out_gpu);

    return 0;
}
