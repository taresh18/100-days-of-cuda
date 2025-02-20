#include <cstdio>


const int FILTER_SIZE=1;

__global__ void imageBlur(unsigned char* p_out, unsigned char* p_in, int width, int height) {
    // 2d kernel
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < height && col < width) {
        float val = 0;
        int num_val = 0;
        // filter size will be 2 * FILTER_SIZE + 1 (-FILTER_SIZE to FILTER_SIZE)
        for (int i=-FILTER_SIZE; i<=FILTER_SIZE; i++) {
            for (int j=-FILTER_SIZE; j<=FILTER_SIZE; j++) {
                int r = row + i;
                int c = col + j;
                // check bounds
                if (r >= 0 && r < height && c >= 0 && c < width) {
                    val += p_in[r * width + c];
                    num_val++;
                }
            }
        }
        p_out[row * width + col] = (unsigned char)(val / num_val);
    }
}

int main() {
    int width = 4, height = 4;
    int num_pixels = width * height;
    int image_size = num_pixels * sizeof(unsigned char);

    // initialise host image and output
    unsigned char h_in[16] = {
        10,  20,  30,  40,
        50,  60,  70,  80,
        90, 100, 110, 120,
       130, 140, 150, 160
    };
    unsigned char h_out[16];

    // initialize device image and output
    unsigned char *d_in, *d_out;
    cudaMalloc((void**)&d_in, image_size);
    cudaMalloc((void**)&d_out, image_size);

    // memory copy: host -> device
    cudaMemcpy(d_in, h_in, image_size, cudaMemcpyHostToDevice);

    // kernel launch
    dim3 block_dim(16, 16);
    dim3 grid_dim(ceil(width / 16.0), ceil(height / 16.0));

    imageBlur<<<grid_dim, block_dim>>>(d_out, d_in, width, height);

    // memory copy: device -> host
    cudaMemcpy(h_out, d_out, image_size, cudaMemcpyDeviceToHost);

    // print the output image
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            printf("%3d ", h_out[row * width + col]);
        }
        printf("\n");
    }

    // free device memory
    cudaFree(d_in);
    cudaFree(d_out);

}
