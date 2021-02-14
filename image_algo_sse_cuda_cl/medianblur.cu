#include "common_define_gpu.h"

#define THREAD 16

__device__ inline void op(int &a, int &b)
{
	int t = (a > b)*(b - a);
	b -= t;
	a += t;
}
__global__ void medianBlurGPU3x3(unsigned char* devsrc, const int srcheight, const int srcwidth, const int channel,
	const int ksize, unsigned char* devdst, const int dstheight, const int dstwidth)
{
	int x = blockIdx.x* blockDim.x + threadIdx.x;
	int y = blockIdx.y* blockDim.y + threadIdx.y;
	int index = y * dstwidth*channel + x * channel;
	__shared__ int srcstep;
	srcstep = srcwidth * channel;

	if (x < dstwidth && y < dstheight)
	{
		unsigned char* row0 = devsrc + ((y - 1) > 0)*(y - 1)* srcstep;
		unsigned char* row1 = devsrc + y * srcstep;
		unsigned char* row2 = devsrc + ((y + 1) < (srcheight - 1) ? y + 1 : srcheight - 1)* srcstep;
#pragma unroll 
		for (int c = 0; c < channel; c++)
		{
			int j0 = x * channel + c;
			int j1 = j0 + channel;
			int j2 = j1 + channel;
			int p0 = row0[j0], p1 = row0[j1], p2 = row0[j2];
			int p3 = row1[j0], p4 = row1[j1], p5 = row1[j2];
			int p6 = row2[j0], p7 = row2[j1], p8 = row2[j2];

			op(p1, p2); op(p4, p5); op(p7, p8); op(p0, p1);
			op(p3, p4); op(p6, p7); op(p1, p2); op(p4, p5);
			op(p7, p8); op(p0, p3); op(p5, p8); op(p4, p7);
			op(p3, p6); op(p1, p4); op(p2, p5); op(p4, p7);
			op(p4, p2); op(p6, p4); op(p4, p2);
			devdst[index + c] = p4;
		}
	}
}

__global__ void medianBlurGPU5x5()
{

}

__device__ int findMedian(int* hist, int ksize)
{
	int i, sum = 0;
	int medsize = ksize * ksize / 2;
	for (i = 0; i < 255; i++)
	{
		sum += hist[i];
		if (sum > medsize)
			break;
	}
	return i;
}

__global__ void initHist(int* hist)
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	int tid = y * blockDim.x + x;

	hist[tid] = 0;
	//__syncthreads();

}

__global__ void cal_hist(int* hist, int count)
{
	hist[0] += count;
}
__global__ void _medianBlurGPU1(unsigned char* devsrc, const int srcheight, const int srcwidth, const int channel,
	const int ksize, unsigned char* devdst, const int dstheight, const int dstwidth) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int index = y * dstwidth*channel + x * channel;

	int srcstep = srcwidth * channel;
	int *hist_tab;
	cudaMalloc(&hist_tab, sizeof(int) * 256 * 3);//max channel is three

	initHist << <3, 256 >> > (hist_tab);

	//cudaMemset(hist_tab, 0, sizeof(int)*channel * 256);//核函数中不能使用cudaMemset函数

	const unsigned char* top, *bottom;
	int top_time = 1;
	int bottom_time = 1;
	if (y < ksize / 2)
	{
		top = devsrc;
		top_time = ksize / 2 - y + 1;
		bottom = top + (ksize - top_time)*srcstep;
		bottom_time = 1;
	}
	else if (y + ksize / 2 > (srcheight - 1))
	{
		top = devsrc + srcstep * (y - ksize / 2);
		top_time = 1;
		bottom = devsrc + srcstep * (srcheight - 1);
		bottom_time = y + 1 + ksize / 2 - srcheight + 1;
	}
	else
	{
		top = devsrc + srcstep * (y - ksize / 2);
		top_time = 1;
		bottom = top + (ksize - 1)* srcstep;
		bottom_time = 1;
	}

	int i, j, c;
	for (c = 0; c < channel; c++)
	{
		const unsigned char* cur_top = top;
		//for (i = 0; i < ksize; i++)
		{
			int temp = cur_top[x*channel + c + i * channel];
			hist_tab[c * 256 + temp] += top_time;
			cal_hist << <1, ksize >> > (&hist_tab[c * 256 + temp], top_time);
		}
		{
			int temp = bottom[x*channel + c + i * channel];
			hist_tab[c * 256 + temp] += bottom_time;
			cal_hist << <1, ksize >> > (&hist_tab[c * 256 + temp], bottom_time);
		}
		for (j = 0; j < (ksize - top_time - bottom_time); j++)
		{
			cur_top += srcstep;
			{
				int temp = cur_top[x*channel + c + i * channel];
				hist_tab[c * 256 + temp] ++;
				cal_hist << <1, ksize >> > (&hist_tab[c * 256 + temp], 1);
			}
		}

		devdst[index + c] = findMedian(&hist_tab[c * 256], ksize);
	}
	cudaFree(hist_tab);

}

__global__ void _medianBlurGPU(unsigned char* devsrc, const int srcheight, const int srcwidth, const int channel,
	const int ksize, unsigned char* devdst, const int dstheight, const int dstwidth) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y * dstwidth*channel + x * channel;
	int srcstep = srcwidth * channel;
	int hist_tab[3][256];//max channel is 3
	for (int c = 0; c < channel; c++) {
		for (size_t k = 0; k < 256; k++)
		{
			hist_tab[c][k] = 0;
		}
	}

	const unsigned char* top, *bottom;
	int top_time = 1;
	int bottom_time = 1;
	if (y < ksize / 2)
	{
		top = devsrc;
		top_time = ksize / 2 - y + 1;
		bottom = top + (ksize - top_time)*srcstep;
		bottom_time = 1;
	}
	else if (y + ksize / 2 > (srcheight - 1))
	{
		top = devsrc + srcstep * (y - ksize / 2);
		top_time = 1;
		bottom = devsrc + srcstep * (srcheight - 1);
		bottom_time = y + 1 + ksize / 2 - srcheight + 1;
	}
	else
	{
		top = devsrc + srcstep * (y - ksize / 2);
		top_time = 1;
		bottom = top + (ksize - 1)* srcstep;
		bottom_time = 1;
	}

	int i, j;
	for (int c = 0; c < channel; c++)
	{
		const unsigned char* cur_top = top;
		for (i = 0; i < ksize; i++)
		{
			int temp = cur_top[x*channel + c + i * channel];
			hist_tab[c][temp] += top_time;
		}
		for (i = 0; i < ksize; i++)
		{
			int temp = bottom[x*channel + c + i * channel];
			hist_tab[c][temp] += bottom_time;

		}
		for (j = 0; j < (ksize - top_time - bottom_time); j++)
		{
			cur_top += srcstep;
			for (i = 0; i < ksize; i++)
			{
				int temp = cur_top[x*channel + c + i * channel];
				hist_tab[c][temp] ++;

			}
		}

		devdst[index + c] = findMedian(hist_tab[c], ksize);
	}
}

extern "C" void medianBlurGPU(unsigned char* devsrc, const int& srcheight, const int& srcwidth, const int& channel,
	const int& ksize, unsigned char* devdst, const int& dstheight, const int& dstwidth)
{
	dim3 dimblock(THREAD, THREAD);
	dim3 dimgrid((dstwidth + THREAD - 1) / THREAD, (dstheight + THREAD - 1) / THREAD);//grid的维度
	if (ksize == 3)
		medianBlurGPU3x3 << <dimgrid, dimblock >> > (devsrc, srcheight, srcwidth, channel, ksize, devdst, dstheight, dstwidth);
	//else if(ksize==5)
	//	medianBlurGPU5x5 << <dimgrid, dimblock >> >(/*devsrc, srcheight, srcwidth, channel, ksize, devdst, dstheight, dstwidth*/);
	else
		_medianBlurGPU << <dimgrid, dimblock >> > (devsrc, srcheight, srcwidth, channel, ksize, devdst, dstheight, dstwidth);
	//核函数分离，核函数这个方法还是有问题
	//_medianBlurGPU1 << <8, dimblock >> > (devsrc, srcheight, srcwidth, channel, ksize, devdst, dstheight, dstwidth);
}