#include "common_define.h"

#define THREAD 64
//注意：核函数传参不能用int& 等类型，只能用int.
__global__ void d_box_filter_x(unsigned char* devsrc, const int srcheight, const int srcwidth, const int channel,
	const int ksize, int* devdst, const int dstheight, const int dstwidth)
{
	int y = blockIdx.x*blockDim.x + threadIdx.x;

	if (y < dstheight)
	{

		int iLocSrcy = y * srcwidth*channel;
		int iLocDsty = y * dstwidth*channel;
		for (int x = 0; x < channel*dstwidth; x++)
		{
			int sum = 0;
			for (int k = 0; k < ksize; k++)
			{
				sum += devsrc[iLocSrcy + x + channel * k];
			}
			devdst[iLocDsty + x] = sum;
		}
	}
}
__global__ void d_box_filter_y(int* devsrc, const int height, const int width, const int channel,
	const int ksize, int* devdst)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (x < width)
	{
		for (int y = 0; y < height; y++)
		{
			int sum = 0;
			for (int k = 0; k < ksize; k++)
			{
				sum += devsrc[y*width + x + k * width];
			}
			devdst[y*width + x] = sum;
		}
	}
}
__global__ void devide_kernel(int* src, float scale, int height, int width, int channel, unsigned char* dst)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z;
	int loc = y * width*channel + x * channel + z;
	if (x < width && y < height)
		dst[loc] = int(src[loc] * scale);
}

__global__ void d_box_fliter_global_x(unsigned char* devsrc, const int srcheight, const int srcwidth, const int channel,
	const int ksize, int* devdst, const int dstheight, const int dstwidth)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z;
	int srcloc = y * srcwidth*channel + x * channel + z;
	int dstloc = y * dstwidth*channel + x * channel + z;
	if (x < dstwidth&& y < dstheight)
	{
		int sum = 0;
#pragma unroll
		for (int k = 0; k < ksize; k++)
		{
			sum += devsrc[srcloc + k * channel];
		}
		devdst[dstloc] = sum;
	}
}
__global__ void d_box_fliter_global_y(int* devsrc, const int srcheight, const int srcwidth, const int channel,
	const int ksize, int* devdst, const int dstheight, const int dstwidth)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z;
	int srcloc = y * srcwidth*channel + x * channel + z;
	int dstloc = y * dstwidth*channel + x * channel + z;
	if (x < dstwidth&& y < dstheight)
	{
		int sum = 0;
#pragma unroll
		for (int k = 0; k < ksize; k++)
		{
			sum += devsrc[srcloc + k * srcwidth*channel];
		}
		devdst[dstloc] = sum;
	}
}
__global__ void d_box_fliter_global_x_char(unsigned char* devsrc, const int srcheight, const int srcwidth, const int channel,
	const int ksize, unsigned char* devdst, const int dstheight, const int dstwidth)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z;
	int srcloc = y * srcwidth*channel + x * channel + z;
	int dstloc = y * dstwidth*channel + x * channel + z;
	float scale = 1.0f / ksize;

	if (x < dstwidth&& y < dstheight)
	{
		int sum = 0;
#pragma unroll
		for (int k = 0; k < ksize; k++)
		{
			sum += devsrc[srcloc + k * channel];
		}
		devdst[dstloc] = int(sum*scale);
	}
}
__global__ void d_box_fliter_global_y_char(unsigned char* devsrc, const int srcheight, const int srcwidth, const int channel,
	const int ksize, unsigned char* devdst, const int dstheight, const int dstwidth)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z;
	int srcloc = y * srcwidth*channel + x * channel + z;
	int dstloc = y * dstwidth*channel + x * channel + z;
	float scale = 1.0f / ksize;

	if (x < dstwidth&& y < dstheight)
	{
		int sum = 0;
#pragma unroll
		for (int k = 0; k < ksize; k++)
		{
			sum += devsrc[srcloc + k * srcwidth*channel];
		}
		devdst[dstloc] = int(sum*scale);
	}
}

extern "C" void boxfilterGPU(unsigned char* devsrc, const int& srcheight, const int& srcwidth, const int& channel,
	const int& ksize, unsigned char* devdst, const int& dstheight, const int& dstwidth)
{
	//调试代码
	//cv::Mat temp(srcheight, dstwidth, CV_8UC1); //代表int类型
	//cv::Mat tempy(dstheight, dstwidth, CV_32SC1);
	//cv::Mat tempdst(dstheight, dstwidth, CV_8UC1);
	//cudaMemcpy(tempx.data, devtempx, srcheight*dstwidth * sizeof(int), cudaMemcpyDeviceToHost);

	//method1,method2 时间主要消耗在开辟内存和析构，占时80%
//	{//method 1
//		int* devtempx, *devtempy;
//		cudaMalloc(&devtempx, srcheight*dstwidth*channel * sizeof(int));
//		cudaMalloc(&devtempy, dstheight*dstwidth*channel * sizeof(int));
//		d_box_filter_x << <(srcheight + THREAD - 1) / THREAD, THREAD >> > (devsrc, srcheight, srcwidth, channel, ksize, devtempx, srcheight, dstwidth);
//		d_box_filter_y << <(dstwidth + THREAD - 1) / THREAD, THREAD >> > (devtempx, dstheight, dstwidth, channel, ksize, devtempy);
//		float scale = 1.0f / (ksize*ksize);
//#define DEVIDE 32
//		dim3 dimblock(DEVIDE, DEVIDE);
//		dim3 dimgrid((srcwidth + DEVIDE - 1) / DEVIDE, (srcheight + DEVIDE - 1) / DEVIDE, channel);//grid的维度
//		devide_kernel << < dimgrid, dimblock >> > (devtempy, scale, dstheight, dstwidth, channel, devdst);
//		cudaFree(devtempx);
//		cudaFree(devtempy);
//	}

//	{//method 2
//		int* devtempx, *devtempy;
//		cudaMalloc(&devtempx, srcheight*dstwidth*channel * sizeof(int));
//		cudaMalloc(&devtempy, dstheight*dstwidth*channel * sizeof(int));
//		float scale = 1.0f / (ksize*ksize);
//#define DEVIDE 32
//		dim3 dimblock(DEVIDE, DEVIDE);
//		dim3 dimgrid((srcwidth + DEVIDE - 1) / DEVIDE, (srcheight + DEVIDE - 1) / DEVIDE, channel);//grid的维度
//
//		d_box_fliter_global_x << < dimgrid, dimblock >> >(devsrc, srcheight, srcwidth, channel, ksize, devtempx, srcheight, dstwidth);
//		d_box_fliter_global_y << < dimgrid, dimblock >> >(devtempx, dstheight, dstwidth, channel, ksize, devtempy, dstheight, dstwidth);
//		devide_kernel << < dimgrid, dimblock >> > (devtempy, scale, dstheight, dstwidth, channel, devdst);
//		cudaFree(devtempx);
//		cudaFree(devtempy);
//	}

	{//method 3
		unsigned char* devtemp;
		cudaMalloc(&devtemp, srcheight*dstwidth*channel * sizeof(unsigned char));
#define DEVIDE 32
		dim3 dimblock(DEVIDE, DEVIDE);
		dim3 dimgrid((srcwidth + DEVIDE - 1) / DEVIDE, (srcheight + DEVIDE - 1) / DEVIDE, channel);//grid的维度

		d_box_fliter_global_x_char << < dimgrid, dimblock >> > (devsrc, srcheight, srcwidth, channel, ksize, devtemp, srcheight, dstwidth);
		d_box_fliter_global_y_char << < dimgrid, dimblock >> > (devtemp, dstheight, dstwidth, channel, ksize, devdst, dstheight, dstwidth);
		cudaFree(devtemp);
	}
}