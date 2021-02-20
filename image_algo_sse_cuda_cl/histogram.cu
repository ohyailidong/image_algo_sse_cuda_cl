#include "common_define_gpu.h"

#define HIST_SZ 256

__global__ void calhist_kernel_v1(unsigned char* devimage, float* devhist,
	const int length)
{
	__shared__ unsigned int temp[HIST_SZ];
	temp[threadIdx.x] = 0;
	__syncthreads();
	int i = threadIdx.x + blockIdx.x* blockDim.x;
	int offset = blockDim.x* gridDim.x;
	while (i < length)
	{
		atomicAdd(&temp[devimage[i]], 1);
		i += offset;
	}
	__syncthreads();
	atomicAdd(&devhist[threadIdx.x], temp[threadIdx.x]);
}
__global__ void calhist_kernel(unsigned char* devimage, float* devhist,
	const int height, const int width)
{
	//int x = blockIdx.x* blockDim.x + threadIdx.x;
	//int y = blockIdx.y* blockDim.y + threadIdx.y;
	//unsigned int value = 0;
	//if (x < width&& y < height)
	//{
	//	value = devimage[y*width + x];
	//	atomicAdd(&(devhist[value]), 1);
	//}
	//__syncthreads();


	///////////共享内存直接实现，竟然和上面的方法时间相似/////////////////////////////////
	__shared__ float devhist_shared[HIST_SZ];
	int x = blockIdx.x* blockDim.x + threadIdx.x;
	int y = blockIdx.y* blockDim.y + threadIdx.y;
	int inner = threadIdx.y* blockDim.x + threadIdx.x;

	devhist_shared[inner] = 0;//shared memory 必须得初始化
	__syncthreads();

	unsigned int value = 0;
	if (x < width&& y < height)
	{
		value = devimage[y*width + x];
		atomicAdd(&(devhist_shared[value]),1);
	}
	__syncthreads();
	atomicAdd(&(devhist[inner]), devhist_shared[inner]);
}

__global__ void calhist_ratio_kernel(float*devhist, int total, float* devhistratio)
{
	int id = threadIdx.x;
	devhistratio[id] = devhist[id] / total * (HIST_SZ - 1);//将概率归一到0-255
}

__global__ void sum_ratio_kernel(float* devhistratio, float *devsumratio)
{
	for (size_t i = 0; i < HIST_SZ; i++)
	{
		if (0 == i) continue;
		devsumratio[i] += devhistratio[i - 1];
	}
}
__global__ void sum_ratio_kernel_v1(float* devhistratio, float *devsumratio)
{
	int index = threadIdx.x;
	for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2) {
		float in1 = devhistratio[threadIdx.x - stride];
		__syncthreads();
		devsumratio[threadIdx.x] += in1;
		__syncthreads();
	}
}

__global__ void cal_lut_kernel(float* devsumratio, unsigned int* devlut)
{
	int id = threadIdx.x;
	devlut[id] = int(devsumratio[id] + 0.5f);
}
__global__ void cal_map_image(unsigned char* devsrc, unsigned int * devlut, unsigned char* devdst, int height, int width)
{
	int x = blockIdx.x* blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y * width + x;
	if (x < width&& y < height)
	{
		int temp = devsrc[index];
		devdst[index] = devlut[temp];
	}
}

extern "C" void equalizeHistGPU(unsigned char* devimage, float* devhist, unsigned int* devlut,
	const int height, const int width, unsigned char* devdst)
{
#define THREAD 16
	dim3 dimblock(THREAD, THREAD);//block的维度
	dim3 dimgrid((width + THREAD - 1) / THREAD, (height + THREAD - 1) / THREAD);//grid的维度
	
	//calhist_kernel << < dimgrid, dimblock >> > (devimage, devhist, height, width);//calulate histogram
	calhist_kernel_v1 << < 32, HIST_SZ >> > (devimage, devhist, height* width);

	float* devhistratio = devhist;
	calhist_ratio_kernel << <1, HIST_SZ >> > (devhist, height * width, devhistratio);
	//sum_ratio_kernel << <1, 1 >> > (devhistratio, devhistratio);
	sum_ratio_kernel_v1 << <1, HIST_SZ >> > (devhistratio, devhistratio);

	cal_lut_kernel << <1, HIST_SZ >> > (devhistratio, devlut);
	cal_map_image << <dimgrid, dimblock >> > (devimage, devlut, devdst, height, width);
}