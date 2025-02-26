#include <cstdio>
#include <iostream>
#include <chrono>

# define BLOCK_DIM 32
# define COARSE_FACTOR 2

// reduction kernel - opitmised control divergence
__global__ void reduction(float *mat, int lenght, float* output) {
    // each thread is responsible to adding two values of input array placed at blockDim.x distance apart
    int idx = threadIdx.x;
    for(int stride=blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride)
            mat[idx] += mat[idx + stride];
        // wait till all active threads update for this stride value
        __syncthreads();
    }
    // final output will be in the 0th index element
    if (idx == 0)
        *output = mat[idx];
}

// reduction kernel - using shread memory
__global__ void reduction2(float *mat, int lenght, float* output) {
    __shared__ float mat_s[BLOCK_DIM];
    // each thread will find the sum of element at index i and i + block dim.x away
    int idx = threadIdx.x;
    mat_s[idx] = mat[idx] + mat[idx + BLOCK_DIM];

    for(int stride=blockDim.x/2; stride >= 1; stride /= 2) {
        // wait for all threads to reach this pt
        __syncthreads();
        if (threadIdx.x < stride)
            // intermediate updates will be in the shared memory
            mat_s[idx] += mat_s[idx + stride];        
    }
    if (idx == 0)
        *output = mat_s[idx];
}

// reduction kernel - using heirarchial reduction
__global__ void reduction3(float *mat, int lenght, float* output) {
    __shared__ float mat_s[BLOCK_DIM];
    // each segment size if 2 * blockDim.x
    int segment = blockIdx.x * blockDim.x * 2;  // each segment starts with this index into input array
    int input_idx = segment + threadIdx.x;
    int thread_idx = threadIdx.x;

    mat_s[thread_idx] = mat[input_idx] + mat[input_idx + BLOCK_DIM];

    for(int stride=blockDim.x/2; stride >= 1; stride /= 2) {
        // wait for all threads to reach this pt
        __syncthreads();
        if (threadIdx.x < stride)
            // intermediate updates will be in the shared memory
            mat_s[thread_idx] += mat_s[thread_idx + stride];        
    }
    // add the output for this segment to the final output
    if (thread_idx == 0)
        atomicAdd(output, mat_s[thread_idx]);
}

// reduction kernel - using thread coarsening
__global__ void reduction4(float *mat, int lenght, float* output) {
    __shared__ float mat_s[BLOCK_DIM];
    // each segment size if 2 * blockDim.x * COARSE_FACTOR
    int segment = blockIdx.x * blockDim.x * 2 * COARSE_FACTOR ;// each segment starts with this index into input array
    int input_idx = segment + threadIdx.x;
    int thread_idx = threadIdx.x;

    // accumulate initial answer for this thread 
    float init_ans = mat[input_idx];
    for (int i=1; i<2*COARSE_FACTOR; i++) {
        init_ans += mat[input_idx + i * BLOCK_DIM];
    }

    // write initial answer to shared memory
    mat_s[thread_idx] = init_ans;

    for(int stride=blockDim.x/2; stride >= 1; stride /= 2) {
        // wait for all threads to reach this pt
        __syncthreads();
        if (thread_idx < stride)
            // intermediate updates will be in the shared memory
            mat_s[thread_idx] += mat_s[thread_idx + stride];        
    }
    // add the output for this segment to the final output
    if (thread_idx == 0)
        atomicAdd(output, mat_s[thread_idx]);
}

int main() {
    // Set the total number of elements; ensure it is a multiple of (BLOCK_DIM * 2 * COARSE_FACTOR)
    int numElements = 1024; // Example: 1M elements
    size_t size = numElements * sizeof(float);
    
    // Allocate host memory and initialize the input array
    float *h_mat = (float *)malloc(size);
    float h_output = 0.0f;
    
    for (int i = 0; i < numElements; i++) {
        h_mat[i] = 1.0f;  // Expected sum = numElements
    }
    
    // Allocate device memory
    float *d_mat, *d_output;
    cudaMalloc((void **)&d_mat, size);
    cudaMalloc((void **)&d_output, sizeof(float));
    
    // Copy the input array to the device and initialize the output on device
    cudaMemcpy(d_mat, h_mat, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute grid dimensions based on the total number of elements and segment size
    int threadsPerBlock = BLOCK_DIM;
    int segmentSize = BLOCK_DIM * 2 * COARSE_FACTOR;
    int numSegments = (numElements + segmentSize - 1) / segmentSize;
    
    // Launch the kernel
    reduction<<<numSegments, threadsPerBlock>>>(d_mat, numElements, d_output);
    
    // Copy the result back to host
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Sum is: %f (Expected: %d)\n", h_output, numElements);
    
    // Clean up
    cudaFree(d_mat);
    cudaFree(d_output);
    free(h_mat);
    
    return 0;
}
