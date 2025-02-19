#include <stdio.h>

__global__ 
void vectorAdd(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}



int main() {
    int n = 10;
    float a_h[n] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    float b_h[n] = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0};
    float c_h[n];

    float *a_d, *b_d, *c_d;
    cudaMalloc((void**)&a_d, n*sizeof(float));
    cudaMalloc((void**)&b_d, n*sizeof(float));
    cudaMalloc((void**)&c_d, n*sizeof(float));

    cudaMemcpy(a_d, a_h, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, n*sizeof(float), cudaMemcpyHostToDevice);

    vectorAdd<<<ceil(n/256.0), 256>>>(a_d, b_d, c_d, n);

    cudaMemcpy(c_h, c_d, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    for (int i=0; i<n; i++) {
        printf("%.4f ", c_h[i]);
    }
}
