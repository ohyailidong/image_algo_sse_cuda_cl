#include "boxfilter.h"

CVLIB_NAMESPACE_BEGIN

void boxFilterRow(void* src, int srcheight, int srcwidth, const int& channel,
	const int& ksize, void* dst, int dstheight, int dstwidth)
{
	const unsigned char*S = (unsigned char*)src;
	float* D = (float*)dst;
	int i = 0, j, ksz_cn = channel * ksize;

	if (ksize == 3)
	{
		for (j = 0; j < dstheight; j++)
		{
			S = (unsigned char*)src + j * srcwidth*channel;
			D = (float*)dst + j * dstwidth*channel;
			for (i = 0; i < dstwidth*channel; i++)
				D[i] = S[i] + S[i + channel] + S[i + 2 * channel];
		}

	}
	else if (ksize == 5)
	{
		for (j = 0; j < dstheight; j++)
		{
			S = (unsigned char*)src + j * srcwidth*channel;
			D = (float*)dst + j * dstwidth*channel;
			for (i = 0; i < dstwidth*channel; i++)
				D[i] = S[i] + S[i + channel] + S[i + 2 * channel] + S[i + 3 * channel] + S[i + 4 * channel];
		}
	}
	else if (channel == 1)
	{
		for (j = 0; j < dstheight; j++)
		{
			S = (unsigned char*)src + j * srcwidth*channel;
			D = (float*)dst + j * dstwidth*channel;
			float s = 0;
			for (i = 0; i < ksz_cn; i++)
				s += S[i];
			D[0] = s;
			for (i = 0; i < dstwidth - 1; i++)
			{
				s += S[i + ksz_cn] - S[i];
				D[i + 1] = s;
			}
		}
	}
	else if (channel == 3)
	{
		for (j = 0; j < dstheight; j++)
		{
			S = (unsigned char*)src + j * srcwidth*channel;
			D = (float*)dst + j * dstwidth*channel;
			float s0 = 0, s1 = 0, s2 = 0;
			for (i = 0; i < ksz_cn; i += 3)
			{
				s0 += S[i];
				s1 += S[i + 1];
				s2 += S[i + 2];
			}
			D[0] = s0;
			D[1] = s1;
			D[2] = s2;
			for (i = 0; i < (dstwidth - 1)*channel; i += 3)
			{
				s0 += S[i + ksz_cn] - S[i];
				s1 += S[i + ksz_cn + 1] - S[i + 1];
				s2 += S[i + ksz_cn + 2] - S[i + 2];
				D[i + 3] = s0;
				D[i + 4] = s1;
				D[i + 5] = s2;
			}
		}
	}
	else
	{
		assert(channel == 1 || channel == 3);
	}
}
void boxFilterColumn(void* src, int srcheight, int srcwidth, const int& channel,
	const int& ksize, void* dst, int dstheight, int dstwidth)
{
	float* S = (float*)src;
	unsigned char* D = (unsigned char*)dst;
	float scale = 1. / (ksize*ksize);
	int i, j, k;
	dstwidth = dstwidth * channel;
	int srcstep = srcwidth * channel;
	std::vector<float>sum(dstwidth, 0);
	for (i = 0; i < dstwidth; i++)
	{
		for (k = 0; k < ksize; k++)
		{
			sum[i] += S[i + k * srcstep];
		}
	}
	for (i = 0; i < dstwidth; i++)
		D[i] = sum[i] * scale;
	//update column sum
	for (j = 0; j < dstheight - 1; j++)
	{
		D += dstwidth;
		for (i = 0; i < dstwidth; i++)
		{
			sum[i] += S[i + ksize * srcstep] - S[i];
			D[i] = sum[i] * scale;
		}
		S += srcstep;
	}

}
void boxFilter(unsigned char* src, int srcheight, int srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, int dstheight, int dstwidth)
{
	float* rowresult = new float[dstwidth* srcheight*channel];
	boxFilterRow(src, srcheight, srcwidth, channel, ksize, rowresult, srcheight, dstwidth);
	boxFilterColumn(rowresult, srcheight, dstwidth, channel, ksize, dst, dstheight, dstwidth);
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
void boxFilter(unsigned char* src, int srcheight, int srcwidth, const int &channel,
	const int&ksize, unsigned char* dst, int dstheight, int dstwidth)
{
	unsigned char* devsrc, *devdst;
	int srcimagesize = srcheight * srcwidth*channel * sizeof(unsigned char);
	int dstimagesize = dstheight * dstwidth*channel * sizeof(unsigned char);
	cudaMalloc(&devsrc, srcimagesize);
	cudaMalloc(&devdst, dstimagesize);
	cudaMemcpy(devsrc, src, srcimagesize, cudaMemcpyHostToDevice);
	boxfilterGPU(devsrc, srcheight, srcwidth, channel, ksize, devdst, dstheight, dstwidth);
	cudaMemcpy(dst, devdst, dstimagesize, cudaMemcpyDeviceToHost);
	cudaFree(devsrc);
	cudaFree(devdst);
}
CUDA_NAMESPACE_END

CVLIB_NAMESPACE_END