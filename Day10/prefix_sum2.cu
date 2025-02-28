#include <cstdio>
#include <iostream>
#include <chrono>

# define SEGMENT_SIZE 32

// prefix sum - kogge stone algo
__global__ void prefixSum(float *mat, int length) {
    // create shared memory
    __shared__ float mat_s[SEGMENT_SIZE];
    // each thread is repsonsible for this index element into input array
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // intialise the shared mamory segment
    if (idx < length)
        mat_s[threadIdx.x] = mat[idx];
    else
        mat_s[threadIdx.x] = 0;

    for (int stride=1; stride<blockDim.x; stride*=2) {
        // wait for shared memory initialisation
        __syncthreads();
        float temp = 0.0f;
        // all threads with id < stride have already calculated their final value
        if(threadIdx.x >= stride)
            temp = mat_s[threadIdx.x] + mat_s[threadIdx.x - stride];
        // wait till all read operations are done into shared memory
        __syncthreads();
        // write the result back into shared memory
        if(threadIdx.x >= stride)
            mat_s[threadIdx.x] = temp;
    }
    // write result back into input array
    if (idx < length)
        mat[idx] = mat_s[threadIdx.x];
}

// prefix sum - kogge stone algo
__global__ void prefixSum2(float *mat, int length) {
    // create shared memory
    __shared__ float mat_s[SEGMENT_SIZE];
    // each thread is responsible for loading two elements into shared memory
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    // load the element at index idx and idx + blockDim.x
    if (idx < length)
        mat_s[threadIdx.x] = mat[idx];
    if (idx + blockDim.x < length)
        mat_s[threadIdx.x + blockDim.x] = mat[idx + blockDim.x];

    // reduction
    for (int stride=1; stride<blockDim.x; stride*=2) {
        // wait for shared memory initialisation
        __syncthreads();
        int input_idx = (threadIdx.x + 1) * 2 * stride - 1;
        if (input_idx < SEGMENT_SIZE)
            mat_s[input_idx] += mat_s[input_idx - stride];
    }

    // distribution
    for (int stride=SEGMENT_SIZE/4; stride>0; stride /= 2) {
        // wait for shared memory initialisation
        __syncthreads();
        int input_idx = (threadIdx.x + 1) * 2 * stride - 1;
        if (input_idx < SEGMENT_SIZE)
            mat_s[input_idx + stride] += mat_s[input_idx];
    }

    // write result back into input array
    if (idx < length)
        mat[idx] = mat_s[threadIdx.x];
    if (threadIdx.x + blockDim.x < length)
        mat[idx + blockDim.x] = mat_s[threadIdx.x + blockDim.x];
}




int main() {
    // Define an input array of eight elements.
    const int length = 8;
    const int size = length * sizeof(float);
    float h_data[length] = {1, 2, 3, 4, 5, 6, 7, 8};

    // Allocate device memory.
    float *d_data;
    cudaMalloc((void**)&d_data, size);

    // Copy input array from host to device.
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Launch kernel with one block. Even though SEGMENT_SIZE is 32, our input length is 8.
    int threadsPerBlock = SEGMENT_SIZE; // 32 threads per block
    int blocks = 1; // One block is enough because input length <= SEGMENT_SIZE.
    prefixSum2<<<blocks, threadsPerBlock>>>(d_data, length);

    // Wait for the kernel to complete.
    cudaDeviceSynchronize();

    // Copy result from device to host.
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_data);

    // Print the prefix sum result.
    std::cout << "Prefix sum result: ";
    for (int i = 0; i < length; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
