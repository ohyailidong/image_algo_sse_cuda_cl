#include "boxfilter.h"

CVLIB_NAMESPACE_BEGIN

void boxFilterRow(void* src, int height, int width, const int& channel,
	const int& ksize, void* dst)
{
	const unsigned char*S = (unsigned char*)src;
	float* D = (float*)dst;
	int i = 0, j, r = ksize / 2;
	int begin(0), end(0);
	if (channel == 1)
	{
		for (j = 0; j < height; j++)
		{
			S = (unsigned char*)src + j * width*channel;
			D = (float*)dst + j * width*channel;
			float s = 0;
			for (i = -r; i <= r; i++)
			{
				begin = cpu::borderInterpolateDefault(i, width);
				s += S[begin];
			}
			D[0] = s;
			for (i = 0; i < width - 1; i++)
			{
				begin = cpu::borderInterpolateDefault(i - r, width);
				end = cpu::borderInterpolateDefault(i + r + 1, width);
				s += S[end] - S[begin];
				D[i + 1] = s;
			}
		}
	}
	else if (channel == 3)
	{
		for (j = 0; j < height; j++)
		{
			S = (unsigned char*)src + j * width*channel;
			D = (float*)dst + j * width*channel;
			float s0 = 0, s1 = 0, s2 = 0;
			for (i = -r; i <= r; i++)
			{
				begin = cpu::borderInterpolateDefault(i, width);
				begin *= channel;
				s0 += S[begin];
				s1 += S[begin + 1];
				s2 += S[begin + 2];
			}
			D[0] = s0;
			D[1] = s1;
			D[2] = s2;
			for (i = 0; i < width - 1; i ++)
			{
				begin = cpu::borderInterpolateDefault(i - r, width);
				end = cpu::borderInterpolateDefault(i + r + 1, width);
				begin *= channel, end *= channel;
				s0 += S[end] - S[begin];
				s1 += S[end + 1] - S[begin + 1];
				s2 += S[end + 2] - S[begin + 2];
				D[i * channel + 3] = s0;
				D[i * channel + 4] = s1;
				D[i * channel + 5] = s2;
			}
		}
	}
	else
	{
		assert(channel == 1 || channel == 3);
	}
}
void boxFilterColumn(void* src, int height, int width, const int& channel,
	const int& ksize, void* dst)
{
	float* S = (float*)src;
	unsigned char* D = (unsigned char*)dst;
	float scale = 1. / (ksize*ksize);
	int i, j, k, begin(0), end(0), r=ksize/2;
	width = width * channel;
	int step = width ;
	std::vector<float>sum(width, 0);
	for (i = 0; i < width; i++)
	{
		for (k = -r; k <= r; k++)
		{
			begin = cpu::borderInterpolateDefault(k, height);
			sum[i] += S[i + begin * step];
		}
	}
	for (i = 0; i < width; i++)
		D[i] = sum[i] * scale;
	//update column sum
	for (j = 0; j < height - 1; j++)
	{
		D += width;
		begin = cpu::borderInterpolateDefault(j - r, height);
		end = cpu::borderInterpolateDefault(j + r + 1, height);
		for (i = 0; i < width; i++)
		{
			sum[i] += S[i + end * step] - S[i+ begin*step];
			D[i] = sum[i] * scale;
		}
	}
}
void boxFilter(unsigned char* src, int height, int width, const int& channel,
	const int& ksize, unsigned char* dst)
{
	float* rowresult = new float[width* height*channel];
	boxFilterRow(src, height, width, channel, ksize, rowresult);
	boxFilterColumn(rowresult, height, width, channel, ksize, dst);
	delete[]rowresult;
}

MULTI_THREAD_NAMESPACE_BEGIN
void boxFilter(unsigned char* src, int srcheight, int srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, int dstheight, int dstwidth)
{
	//////github.com/komrad36/BoxBlur/blob/master/BoxBlur.h;
	//const int threadnum = std::thread::hardware_concurrency();
	//if (threadnum > 1)
	{

	}
	//else
	//	boxFilter(src,srchei,s)
}
MULTI_THREAD_NAMESPACE_END

CUDA_NAMESPACE_BEGIN
void boxFilter(unsigned char* src, int height, int width, const int &channel,
	const int&ksize, unsigned char* dst)
{
	unsigned char* devsrc, *devdst;
	int imagesize = height * width*channel * sizeof(unsigned char);
	cudaMalloc(&devsrc, imagesize);
	cudaMalloc(&devdst, imagesize);
	cudaMemcpy(devsrc, src, imagesize, cudaMemcpyHostToDevice);
	boxfilterGPU(devsrc, height, width, channel, ksize, devdst);
	cudaMemcpy(dst, devdst, imagesize, cudaMemcpyDeviceToHost);
	cudaFree(devsrc);
	cudaFree(devdst);
}
CUDA_NAMESPACE_END

CVLIB_NAMESPACE_END