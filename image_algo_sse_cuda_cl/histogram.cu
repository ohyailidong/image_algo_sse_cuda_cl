#include "common_define.h"

#define HIST_SZ 256

__global__ void calhist_kernel(unsigned char* devimage, float* devhist,
	const int height, const int width)
{
	int x = blockIdx.x* blockDim.x + threadIdx.x;
	int y = blockIdx.y* blockDim.y + threadIdx.y;
	unsigned int value = 0;
	if (x < width&& y < height)
	{
		value = devimage[y*width + x];
		atomicAdd(&(devhist[value]), 1);
	}
	__syncthreads();


	///////////共享内存直接实现，竟然比上面的方法慢（？？？？？？？）/////////////////////////////////
	//__shared__ float devhist_shared[HIST_SZ];
	//int x = blockIdx.x* blockDim.x + threadIdx.x;
	//int y = blockIdx.y* blockDim.y + threadIdx.y;
	//int inner = threadIdx.y* blockDim.x + threadIdx.x;

	//if (inner < HIST_SZ)//shared memory 必须得初始化
	//{
	//	devhist_shared[inner] = 0;
	//}
	//__syncthreads();

	//unsigned int value = 0;
	//if (x < width&& y < height)
	//{
	//	value = devimage[y*width + x];
	//	atomicAdd(&(devhist_shared[value]),1);
	//}
	//__syncthreads();
	//if (inner < HIST_SZ)
	//{
	//	atomicAdd(&(devhist[inner]), devhist_shared[inner]);
	//}
}

__global__ void calhist_ratio_kernel(float*devhist, int total, float* devhistratio)
{
	int id = threadIdx.x;
	devhistratio[id] = devhist[id] / total * (HIST_SZ - 1);//将概率归一到0-255
}

__global__ void sum_ratio_kernel(float* devhistratio, float *devsumratio)
{
	//该方式比cpu循环慢，先用cpu代替
	if (threadIdx.x == 0)
	{
		for (size_t i = 0; i < HIST_SZ; i++)
		{
			if (0 == i) continue;
			devsumratio[i] += devhistratio[i - 1];
		}
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
	//calulate histogram
	calhist_kernel << < dimgrid, dimblock >> > (devimage, devhist, height, width);
	float* devhistratio = devhist;
	int total = height * width;
	calhist_ratio_kernel << <1, HIST_SZ >> > (devhist, total, devhistratio);
	//sum_ratio_kernel << <1, HIST_SZ >> > (devhistratio, devhistratio);
	//将概率累加和在cpu上进行
	float* hostSumRatio = new float[HIST_SZ];
	cudaMemcpy(hostSumRatio, devhistratio, HIST_SZ * sizeof(float), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < HIST_SZ; i++)
	{
		if (0 == i) continue;
		hostSumRatio[i] += hostSumRatio[i - 1];
	}
	float* devSumRatio = devhistratio;
	cudaMemcpy(devSumRatio, hostSumRatio, HIST_SZ * sizeof(float), cudaMemcpyHostToDevice);
	cal_lut_kernel << <1, HIST_SZ >> > (devSumRatio, devlut);
	cal_map_image << <dimgrid, dimblock >> > (devimage, devlut, devdst, height, width);

	delete[]hostSumRatio;
}