#include <iostream>
#include <cuda_runtime.h>

__device__ void matrixMultiplication(float *a, float *b, float *c, int M, int N, int K) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("Thread (%d, %d) - Row: %d, Col: %d\n", threadIdx.x, threadIdx.y, row, col);
	if (row < M && col < K) {
		int sum = 0;
		for (int i = 0; i < N; ++i) {
			sum += a[row * N + i] * b[i * K + col];
		}
		c[row * K + col] = sum;
	}
}

__device__ void im2col( float* input, int channels, int height, int width,
	int kernel_size, int stride, int padding,
	int output_height, int output_width, float* col_matrix) {
	// Iterate over output matrix
	for (int ky = 0; ky < kernel_size; ++ky) {
		for (int kx = 0; kx < kernel_size; ++kx) {
			for (int c = 0; c < channels; ++c) {
				for (int oy = 0; oy < output_height; ++oy) {
					for (int ox = 0; ox < output_width; ++ox) {
						// Calculate input indices
						int iy = oy * stride - padding + ky;
						int ix = ox * stride - padding + kx;
						
						// Check if indices are within bounds
						bool valid = (iy >= 0 && iy < height && ix >= 0 && ix < width);
						
						// Calculate linear indices
						int input_index = c * height * width + iy * width + ix;
						int col_index = (ky * kernel_size + kx) * channels + c;
						
						// Assign value to col_matrix, use 0 if indices are out of bounds
						col_matrix[oy * output_width + ox + col_index * output_height * output_width] = 
						valid ? input[input_index] : 0.0f;
					}
				}
			}
		}
	}
}

__global__ void convolutionWithIm2Col( float* input,  float* kernel, float* output,
	int input_channels, int input_height, int input_width,
	int kernel_size, int stride, int padding,
	int output_height, int output_width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < output_height && col < output_width) {
		// Allocate space for im2col result
		float* col_matrix = new float[kernel_size * kernel_size * input_channels];
		
		// Use im2col function to convert input matrix to column matrix
		im2col(input, input_channels, input_height, input_width,
			kernel_size, stride, padding, output_height, output_width, col_matrix);
		
		// Call matrix multiplication kernel for convolution
		matrixMultiplication(col_matrix, kernel, &output[row * output_width + col],
			kernel_size * kernel_size * input_channels, 1, output_height * output_width);
		
		// Free memory
		delete[] col_matrix;
	}
}

int main(int argc, char** argv) {
	int input_size = atoi(argv[1]);
	int kernel_size = atoi(argv[2]);
	int stride = atoi(argv[3]);
	int channels = atoi(argv[4]);
	int padding = atoi(argv[5]);
	int block_size = atoi(argv[6]);
	
	int output_size = (input_size + 2 * padding - kernel_size) / stride + 1;
	
	// 分配和初始化输入数据和卷积核
	float *h_input = new float[input_size * input_size * channels];
	float *h_kernel = new float[kernel_size * kernel_size * channels];
	float *h_output = new float[output_size * output_size];
	
	
	// 设置种子
	std::srand(2333);
	// 初始化 h_input 和 h_kernel 为随机值
	for (int i = 0; i < input_size * input_size * channels; ++i) {
		h_input[i] = static_cast<float>(rand()) / RAND_MAX;
	}
	
	for (int i = 0; i < kernel_size * kernel_size * channels; ++i) {
		h_kernel[i] = static_cast<float>(rand()) / RAND_MAX;
	}
	
	// 分配设备内存
	float *d_input, *d_kernel, *d_output;
	cudaMalloc((void**)&d_input, input_size * input_size * channels * sizeof(float));
	cudaMalloc((void**)&d_kernel, kernel_size * kernel_size * channels * sizeof(float));
	cudaMalloc((void**)&d_output, output_size * output_size * sizeof(float));
	
	// 将数据从主机内存复制到设备内存
	cudaMemcpy(d_input, h_input, input_size * input_size * channels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * channels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, h_output, output_size * output_size * sizeof(float), cudaMemcpyHostToDevice); 
	
	dim3 blockSize(block_size, block_size);
	dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x, (output_size + blockSize.y - 1) / blockSize.y);
	
	// 记录开始时间
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	// 启动CUDA核函数
	convolutionWithIm2Col<<<gridSize, blockSize>>>(d_input, d_kernel, d_output,
		channels, input_size, input_size,
		kernel_size, stride, padding,
		output_size, output_size);
	
	cudaDeviceSynchronize();
	
	// 记录结束时间
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	// 将结果从设备内存复制到主机内存
	cudaMemcpy(h_output, d_output, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost); 
	
	// 计算时间差并输出
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Convolution time: " << milliseconds << " ms" << std::endl;
	
	// 释放内存
	delete[] h_input;
	delete[] h_kernel;
	delete[] h_output;
	
	cudaFree(d_input);
	cudaFree(d_kernel);
	cudaFree(d_output);
	
	return 0;
}
