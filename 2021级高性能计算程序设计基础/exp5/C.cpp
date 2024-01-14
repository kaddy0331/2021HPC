#include <iostream>
#include <cuda_runtime.h>

// Kernel函数，用于在GPU上执行卷积操作
__global__ void convolution(float *input, float *kernel, float *output, int input_size, int kernel_size, int stride, int padding, int channels, int output_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Calculate row and column indices
	int row = idx / output_size;
	int col = idx % output_size;
	
	// Iterate over channels
	for (int channel = 0; channel < channels; ++channel) {
		float sum = 0.0f;
		
		// Iterate over the kernel
		for (int i = 0; i < kernel_size; ++i) {
			for (int j = 0; j < kernel_size; ++j) {
				int input_row = row * stride + i - padding;
				int input_col = col * stride + j - padding;
				
				// Check if the input indices are within bounds
				if (input_row >= 0 && input_row < input_size && input_col >= 0 && input_col < input_size) {
					int input_index = (input_row * input_size + input_col) * 1 + channel;
					int kernel_index = (i * kernel_size + j) * 1 + channel;
					sum += input[input_index] * kernel[kernel_index];
				}
			}
		}
		
		// Store the result in the output array
		output[(row * output_size + col) * channels + channel] = sum;
	}
}

int main(int argc, char const *argv[]) {
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
	
	// 输出初始化的矩阵
	std::cout << "Input Matrix:" << std::endl;
	for (int i = 0; i < input_size; ++i) {
		for (int j = 0; j < input_size; ++j) {
			std::cout << h_input[(i * input_size + j) * channels] << " ";
		}
		std::cout << std::endl;
	}
	
	// 输出卷积核矩阵
	std::cout << "Kernel Matrix:" << std::endl;
	for (int i = 0; i < kernel_size; ++i) {
		for (int j = 0; j < kernel_size; ++j) {
			std::cout << h_kernel[(i * kernel_size + j) * channels] << " ";
		}
		std::cout << std::endl;
	}
	
	// 在GPU上分配内存
	float *d_input, *d_kernel, *d_output;
	cudaMalloc((void**)&d_input, input_size * input_size * channels * sizeof(float));
	cudaMalloc((void**)&d_kernel, kernel_size * kernel_size * channels * sizeof(float));
	cudaMalloc((void**)&d_output, output_size * output_size * sizeof(float));
	
	// 将输入数据和卷积核从主机内存复制到GPU内存
	cudaMemcpy(d_input, h_input, input_size * input_size * channels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * channels * sizeof(float), cudaMemcpyHostToDevice);
	
	// 调用CUDA内核函数执行卷积
	dim3 blockDim(block_size);  // 一维线程块
	dim3 gridDim((output_size * output_size + blockDim.x - 1) / blockDim.x);
	
	// 记录开始时间
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	convolution<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, input_size, kernel_size, stride, padding, channels, output_size);
	cudaDeviceSynchronize();
	
	// 记录结束时间
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	// 将结果从GPU内存复制回主机内存
	cudaMemcpy(h_output, d_output, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
	
	
	// 输出卷积后的矩阵
	std::cout << "Convolution Output Matrix:" << std::endl;
	for (int i = 0; i < output_size; ++i) {
		for (int j = 0; j < output_size; ++j) {
			std::cout << h_output[i * output_size + j] << " ";
		}
		std::cout << std::endl;
	}
	
	
	// 计算时间差并输出
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Convolution time: " << milliseconds << " ms" << std::endl;
	
	// 释放GPU内存
	cudaFree(d_input);
	cudaFree(d_kernel);
	cudaFree(d_output);
	
	// 释放主机内存
	delete[] h_input;
	delete[] h_kernel;
	delete[] h_output;
	
	return 0;
}
