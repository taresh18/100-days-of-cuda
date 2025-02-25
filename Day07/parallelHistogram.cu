#include <cstdio>
#include <iostream>
#include <chrono>

// each bin of histogram is of interval 4
# define NUM_BINS ((26 + 4 - 1) / 4)

// parallel histogram using shared memory and privatisation
__global__ void parallelHist(char *data, int *histo, int length) {
    // create a private copy of histogram in the shrared memory for this block
    __shared__ int bin_s[NUM_BINS];
    for (int i=threadIdx.x; i<NUM_BINS; i+=blockDim.x) 
        bin_s[i] = 0;
    // wait for shared memory initialisation
    __syncthreads();

    // update bin value in shared memory for input element corres to this thread
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < length) {
        int alpha_idx = data[idx] - 'A';
        if (alpha_idx >= 0 && alpha_idx < 26)
        atomicAdd(&bin_s[alpha_idx / 4], 1);
    }
    // wait till all threads have updated the shared memory bins
    __syncthreads();

    // write from shared memory to global
    for (int i=threadIdx.x; i<NUM_BINS; i+=blockDim.x) {
        int binValue = bin_s[i];
        if (binValue > 0)
            atomicAdd(&histo[i], binValue);
    }
}

# define COARSE_FACTOR 4 // each thread will process these many elements

// parallel histogram using shared memory and privatisation and thread coarsening
__global__ void parallelHist2(char *data, int *histo, int length) {
    // create a private copy of histogram in the shrared memory for this block
    __shared__ int bin_s[NUM_BINS];
    for (int i=threadIdx.x; i<NUM_BINS; i+=blockDim.x) 
        bin_s[i] = 0;
    // wait for shared memory initialisation
    __syncthreads();

    // update shared memory histogram upto coarse_value number of elements from input array
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int input_idx=thread_id*COARSE_FACTOR; input_idx<length; input_idx++) {
        if (input_idx-thread_id*COARSE_FACTOR < COARSE_FACTOR) {
            int alpha_idx = data[input_idx] - 'A';
            if (alpha_idx >= 0 && alpha_idx < 26)
            atomicAdd(&bin_s[alpha_idx / 4], 1);
        }
        else
            break;
    }
    // wait till all threads have updated the shared memory bins
    __syncthreads();

    // copy bin values from shared memory to global
    // write from shared memory to global
    for (int i=threadIdx.x; i<NUM_BINS; i+=blockDim.x) {
        int binValue = bin_s[i];
        if (binValue > 0)
            atomicAdd(&histo[i], binValue);
    }   
}

// CPU version of histogram computation.
void cpuHist(const char *data, int *histo, int length) {
    // Initialize histogram to zero.
    for (int i = 0; i < NUM_BINS; i++) {
        histo[i] = 0;
    }
    for (int i = 0; i < length; i++) {
        int alpha_idx = data[i] - 'A';
        if (alpha_idx >= 0 && alpha_idx < 26)
            histo[alpha_idx / 4]++;
    }
}

int main() {
    // For this example, we use an array of 1024 characters.
    int length = 64;
    char *h_data = new char[length];

    // Initialize h_data with random letters between 'A' and 'Z'
    for (int i = 0; i < length; i++) {
        h_data[i] = 'A' + rand() % 26;
    }

    // Allocate memory for CPU histogram result.
    int h_histo_cpu[NUM_BINS];
    int h_histo_gpu[NUM_BINS];
    memset(h_histo_cpu, 0, sizeof(h_histo_cpu));
    memset(h_histo_gpu, 0, sizeof(h_histo_gpu));

    // Compute histogram on CPU.
    cpuHist(h_data, h_histo_cpu, length);

    // Allocate device memory.
    char *d_data;
    int *d_histo;
    cudaMalloc((void**)&d_data, length * sizeof(char));
    cudaMalloc((void**)&d_histo, NUM_BINS * sizeof(int));

    // Copy input data to device.
    cudaMemcpy(d_data, h_data, length * sizeof(char), cudaMemcpyHostToDevice);
    // Initialize device histogram to zero.
    cudaMemset(d_histo, 0, NUM_BINS * sizeof(int));

    // Choose block and grid sizes.
    int blockSize = 256;  // Number of threads per block.
    int gridSize = (length + blockSize - 1) / blockSize;

    // Launch the histogram kernel.
    parallelHist<<<gridSize, blockSize>>>(d_data, d_histo, length);
    cudaDeviceSynchronize();

    // Copy the histogram result from device to host.
    cudaMemcpy(h_histo_gpu, d_histo, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Compare CPU and GPU histograms.
    std::cout << "Histogram bins (CPU vs GPU):" << std::endl;
    bool correct = true;
    for (int i = 0; i < NUM_BINS; i++) {
        std::cout << "Bin " << i << ": " << h_histo_cpu[i] << " vs " << h_histo_gpu[i] << std::endl;
        if (h_histo_cpu[i] != h_histo_gpu[i])
            correct = false;
    }
    if (correct)
        std::cout << "✅ Histogram is correct!" << std::endl;
    else
        std::cout << "❌ Histogram mismatch!" << std::endl;

    // Clean up.
    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_histo);

    return 0;
}
