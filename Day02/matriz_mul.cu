#include <cstdio>


__global__ void matrixMul(float *mat1, float *mat2, float *out, int size) {
    // 2d kernel
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // assuming square matrix
    if (row < size && col < size) {
        float temp = 0;
        for (int k=0; k<size; k++) 
            temp += mat1[row * size + k] * mat2[k * size + col];
        out[row * size + col] = temp;
    }
}

int main() {
    int size = 4;
    int size_bytes = size * size * sizeof(float);

    // initialise host matrices
    float mat1_h[16] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float mat2_h[16] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
        13, 14, 15, 16 
    };
    float out_h[16];

    // initialize device matrixes
    float *mat1_d, *mat2_d, *out_d;
    cudaMalloc((void**)&mat1_d, size_bytes);
    cudaMalloc((void**)&mat2_d, size_bytes);
    cudaMalloc((void**)&out_d, size_bytes);

    // memory copy: host -> device
    cudaMemcpy(mat1_d, mat1_h, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(mat2_d, mat2_h, size_bytes, cudaMemcpyHostToDevice);

    // kernel launch
    dim3 block_dim(16, 16);
    dim3 grid_dim(ceil(size / 16.0), ceil(size / 16.0));

    matrixMul<<<grid_dim, block_dim>>>(mat1_d, mat2_d, out_d, size);

    // memory copy: device -> host
    cudaMemcpy(out_h, out_d, size_bytes, cudaMemcpyDeviceToHost);

    // print the output image
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            printf("%.4f ", out_h[row * size + col]);
        }
        printf("\n");
    }

    // free device memory
    cudaFree(mat1_d);
    cudaFree(mat2_d);
    cudaFree(out_d);

}
