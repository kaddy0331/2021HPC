#include <iostream>
#include <cuda_runtime.h>

int main() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	
	if (deviceCount == 0) {
		std::cerr << "No CUDA devices found." << std::endl;
		return 1;
	}
	
	std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
	
	for (int dev = 0; dev < deviceCount; ++dev) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		
		std::cout << "\nDevice " << dev << ": " << deviceProp.name << std::endl;
		std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
		std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
		std::cout << "  Maximum threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
		std::cout << "  Maximum dimensions of block: " << deviceProp.maxThreadsDim[0] << " x "
		<< deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << std::endl;
		std::cout << "  Maximum dimensions of grid: " << deviceProp.maxGridSize[0] << " x "
		<< deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << std::endl;
	}
	
	return 0;
}

