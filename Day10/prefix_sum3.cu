#include <iostream>

# define BLOCK_DIM 32
# define COARSE_FACTOR 4
# define SEGMENT_SIZE (BLOCK_DIM * COARSE_FACTOR)

// prefix sum - brent kung algo + thread coarsening
__global__ void prefixSum3(float *mat, int length) {
    // create shared memory
    __shared__ float mat_s[SEGMENT_SIZE];
    int base_idx = blockIdx.x * blockDim.x * COARSE_FACTOR;
    // each thread will load elements i, i + cf*blockDim.x into the shared memory to enable memory coalescing
    for (int i=0; i<COARSE_FACTOR; i++) {
        int input_idx = base_idx + threadIdx.x + i * blockDim.x;
        if (input_idx < length)
            mat_s[threadIdx.x + i * blockDim.x] = mat[input_idx];
        else
            mat_s[threadIdx.x + i * blockDim.x] = 0.0f;
    }
    // wait for all threads in block to load elements in memory
    __syncthreads();
    // perform sequential prefix sum on COARSE_FACTOR # of elements
    for (int i=1; i<COARSE_FACTOR; i++) {
        int input_idx = threadIdx.x * COARSE_FACTOR + i;
        mat_s[input_idx] += mat_s[input_idx-1];
    }
    // perform brent - kung algorithm on the selected elements in the resulting mat_s
    // elements -> threadIdx.x * COARSE_FACTOR + COARSE_FACTOR - 1
    __shared__ float in_buffer_s[BLOCK_DIM];
    __shared__ float out_buffer_s[BLOCK_DIM];
    in_buffer_s[threadIdx.x] = mat_s[threadIdx.x * COARSE_FACTOR + COARSE_FACTOR - 1];
    // wait for all threads to load elements into shared memory
    __syncthreads();

    for (int stride=1; stride<blockDim.x; stride*=2) {
        if (threadIdx.x >= stride)
            out_buffer_s[threadIdx.x] = in_buffer_s[threadIdx.x - stride];
        // if thread idex is smaller than stride, it already reached its final value
        else
            out_buffer_s[threadIdx.x] = in_buffer_s[threadIdx.x];
        // wait for all threads to write to out_buffer
        __syncthreads();
        // update the in_buffer for next iteration
        in_buffer_s[threadIdx.x] = out_buffer_s[threadIdx.x];
        __syncthreads();
    }
    // scan results are stored in in_buffer_s
    // For thread 0, no carry is needed; for thread t>0, the carry is the sum of all previous threads' chunks
    float carry = (threadIdx.x > 0) ? in_buffer_s[threadIdx.x - 1] : 0.0f;
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int input_idx = threadIdx.x * COARSE_FACTOR + i;
        mat_s[input_idx] += carry;
    }
    // STEP 5: Write the scanned (prefix sum) results back to global memory.
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int input_idx = base_idx + threadIdx.x * COARSE_FACTOR + i;
        int thread_idx = threadIdx.x * COARSE_FACTOR + i;
        if (input_idx < length)
            mat[input_idx] = mat_s[thread_idx];
    }
}


int main() {
    // Example: Compute inclusive prefix sum on an array of 8 elements.
    // For input: 1, 2, 3, 4, 5, 6, 7, 8,
    // the expected prefix sum is: 1, 3, 6, 10, 15, 21, 28, 36.
    const int length = 8;
    const int size = length * sizeof(float);
    float h_data[length] = {1, 2, 3, 4, 5, 6, 7, 8};

    float *d_data;
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Launch kernel with one block.
    // SEGMENT_SIZE = BLOCK_DIM * COARSE_FACTOR = 32 * 4 = 128.
    // Our array length is 8, so only the first 8 elements are used.
    int threadsPerBlock = BLOCK_DIM;
    int blocks = 1; // one block suffices for this small example
    prefixSum3<<<blocks, threadsPerBlock>>>(d_data, length);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    std::cout << "Prefix sum result: ";
    for (int i = 0; i < length; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
