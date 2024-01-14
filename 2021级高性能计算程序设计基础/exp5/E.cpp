#include <iostream>
#include <cstdlib>
#include <cudnn.h>

#define checkCUDNN(expression)                               \
{                                                          \
cudnnStatus_t status = (expression);                     \
if (status != CUDNN_STATUS_SUCCESS) {                    \
std::cerr << "Error on line " << __LINE__ << ": "      \
<< cudnnGetErrorString(status) << std::endl; \
std::exit(EXIT_FAILURE);                               \
}                                                        \
}



int main(int argc, char const *argv[]) {
	
	int input_height;
	std::cout << "Input Height: ";
	std::cin >> input_height;
	
	int input_width;
	std::cout << "Input Width: ";
	std::cin >> input_width;
	
	int kernel_height = 3;
	
	int kernel_width = 3;
	
	int vertical_stride;
	std::cout << "Vertical Stride: ";
	std::cin >> vertical_stride;
	
	int horizontal_stride;
	std::cout << "Horizontal Stride: ";
	std::cin >> horizontal_stride;
	
	int channels;
	std::cout << "Channels: ";
	std::cin >> channels;
	
	int padding_height = (kernel_height - 1) / 2;
	int padding_width = (kernel_width - 1) / 2;
	
	int output_height = (input_height - kernel_height + 2 * padding_height) / vertical_stride + 1;
	int output_width = (input_width - kernel_width + 2 * padding_width) / horizontal_stride + 1;
	
	float *h_input = new float[input_height * input_width * channels];
	float *h_kernel = new float[kernel_height * kernel_width * channels];
	
	int isprint;
	std::cout << "print?(1 or 0)";
	std::cin >> isprint;
	
	std::srand(2333);
	
	// 初始化 h_input 和 h_kernel 为随机值
	for (int i = 0; i < input_height * input_width * channels; ++i) {
		h_input[i] = static_cast<float>(rand()) / RAND_MAX;
	}
	
	for (int i = 0; i < kernel_height * kernel_width * channels; ++i) {
		h_kernel[i] = static_cast<float>(rand()) / RAND_MAX;
	}
	
	if (isprint == 1) {
		std::cout << "Input Matrix:" << std::endl;
		for (int i = 0; i < input_height; ++i) {
			for (int j = 0; j < input_width; ++j) {
				std::cout << h_input[(i * input_width + j) * channels] << " ";
			}
			std::cout << std::endl;
		}
		
		std::cout << "Kernel Matrix:" << std::endl;
		for (int i = 0; i < kernel_height; ++i) {
			for (int j = 0; j < kernel_width; ++j) {
				std::cout << h_kernel[(i * kernel_height + j)] << " ";
			}
			std::cout << std::endl;
		}
	}
	
	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);
	
	cudnnTensorDescriptor_t input_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/1,
		/*channels=*/channels,
		/*image_height=*/input_height,
		/*image_width=*/input_width));
	
	cudnnTensorDescriptor_t output_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/1,
		/*channels=*/1,
		/*image_height=*/output_height,
		/*image_width=*/output_width));            
	
	cudnnFilterDescriptor_t kernel_descriptor;
	checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*out_channels=*/1,
		/*in_channels=*/channels,
		/*kernel_height=*/kernel_height,
		/*kernel_width=*/kernel_width));
	
	cudnnConvolutionDescriptor_t convolution_descriptor;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
		/*pad_height=*/padding_height,
		/*pad_width=*/padding_width,
		/*vertical_stride=*/vertical_stride,
		/*horizontal_stride=*/horizontal_stride,
		/*dilation_height=*/1,
		/*dilation_width=*/1,
		/*mode=*/CUDNN_CROSS_CORRELATION,
		/*computeType=*/CUDNN_DATA_FLOAT));
	
	cudnnConvolutionFwdAlgo_t convolution_algorithm;
	checkCUDNN(
		cudnnGetConvolutionForwardAlgorithm(cudnn,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			/*memoryLimitInBytes=*/0,
			&convolution_algorithm));
	
	size_t workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		input_descriptor,
		kernel_descriptor,
		convolution_descriptor,
		output_descriptor,
		convolution_algorithm,
		&workspace_bytes));
	
	void* d_workspace{nullptr};
	cudaMalloc(&d_workspace, workspace_bytes);
	
	int image_bytes = channels * input_height * input_width * sizeof(float);
	
	float* d_input{nullptr};
	cudaMalloc(&d_input, image_bytes);
	cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice);
	
	float* d_output{nullptr};
	cudaMalloc(&d_output, output_height * output_width * sizeof(float));
	cudaMemset(d_output, 0, output_height * output_width * sizeof(float) );
	
	
	float* d_kernel{nullptr};
	cudaMalloc(&d_kernel, channels * kernel_height * kernel_width * sizeof(float));
	cudaMemcpy(d_kernel, h_kernel, channels * kernel_height * kernel_width * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	const float alpha = 1, beta = 0;
	checkCUDNN(cudnnConvolutionForward(cudnn,
		&alpha,
		input_descriptor,
		d_input,
		kernel_descriptor,
		d_kernel,
		convolution_descriptor,
		convolution_algorithm,
		d_workspace,
		workspace_bytes,
		&beta,
		output_descriptor,
		d_output));
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float* h_output = new float[output_height * output_width * sizeof(float)];
	cudaMemcpy(h_output, d_output, sizeof(float) * output_height * output_width, cudaMemcpyDeviceToHost);
	
	if (isprint == 1) {
		std::cout << "Convolution Output Matrix:" << std::endl;
		for (int i = 0; i < output_height; ++i) {
			for (int j = 0; j < output_width; ++j) {
				std::cout << h_output[i * output_width + j] << " ";
			}
			std::cout << std::endl;
		}
	}
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "Convolution time: " << milliseconds << " ms" << std::endl;
	
	delete[] h_output;
	cudaFree(d_kernel);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_workspace);
	
	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);
	
	cudnnDestroy(cudnn);
	
	return 0;
}

