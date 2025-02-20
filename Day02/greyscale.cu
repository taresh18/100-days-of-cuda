#include <cstdio>


__global__ void color2greyscale(unsigned char* p_out, unsigned char* p_in, int channels, int width, int height) {
    // 2d kernel
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < height && col < width) {
        int output_image_idx = row * width + col;
        int input_image_idx = (row * width + col) * channels;
        // L = 0.21*r + 0.72*g + 0.07*b
        p_out[output_image_idx] = 0.21 * p_in[input_image_idx] +  0.72 * p_in[input_image_idx + 1] + 0.07 * p_in[input_image_idx + 2];
    }
}

int main() {
    int width = 4, height = 3, channels = 3;
    int num_pixels = width * height;
    int image_size = num_pixels * channels * sizeof(unsigned char);
    int grey_size = num_pixels * sizeof(unsigned char);

    // host memory
    unsigned char h_image[num_pixels * channels];
    unsigned char h_grey[num_pixels];

    // initialize image with random values
    for (int i = 0; i < num_pixels; i++) {
        h_image[i * channels] = (unsigned char)(10 * i); // R
        h_image[i * channels + 1] = (unsigned char)(20 * i); // G
        h_image[i * channels + 2] = (unsigned char)(30 * i); // B
    }

    // device memory
    unsigned char *d_image, *d_grey;
    cudaMalloc((void**)&d_image, image_size);
    cudaMalloc((void**)&d_grey, grey_size);

    // copy from host to device
    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);

    // kernel launch
    dim3 grid_dim(ceil(width / 16.0), ceil(height / 16.0));
    dim3 block_dim(16, 16);
    color2greyscale<<<grid_dim, block_dim>>>(d_grey, d_image, channels,width, height);

    // copy from device to host
    cudaMemcpy(h_grey, d_grey, grey_size, cudaMemcpyDeviceToHost);

    // check output
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            printf("%3d ", h_grey[row * width + col]);
        }
        printf("\n");
    }

    // free memory
    cudaFree(d_image);
    cudaFree(d_grey);

}
