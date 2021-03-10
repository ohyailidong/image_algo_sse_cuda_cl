#include "copymakeborder.h"
#include "common_data_define.h"

CVLIB_NAMESPACE_BEGIN
int borderInterpolate(int p, int len, int borderType)
{
	int q = p;
	if ((unsigned)q < (unsigned)len);
	else if (BORDER_REPLICATE == borderType)
	{
		q = q < 0 ? 0 : len - 1;
	}
	else if (BORDER_REFLECT == borderType || BORDER_REFLECT_101 == borderType)
	{
		int delta = borderType == BORDER_REFLECT_101;
		if (1 == len)
			return 0;
		do
		{
			if (q < 0)
				q = -q - 1 + delta;
			else
				q = len - 1 - (q - len) - delta;
		} while ((unsigned)q >= (unsigned)len);
	}
	else if (borderType == BORDER_WRAP)
	{
		if (q < 0)
			q -= ((q - len + 1) / len)*len;
		if (q >= len)
			q %= len;
	}
	else if (borderType == BORDER_CONSTANT)
		q = -1;
	else
	{
		;
	}
	return q;
}

void copyMakeBorder_8u(const unsigned char*srcptr, size_t srcstep, int* srcdim,
	unsigned char* dstptr, size_t dststep, int* dstdim,
	int top, int left, int channel, int bordertype)
{
	int i, j, k;
	int* tab = new int[(dstdim[1] - srcdim[1])*channel];
	int right = dstdim[1] - srcdim[1] - left;
	int bottom = dstdim[2] - srcdim[2] - top;
	//建立查找表索引
	for (i = 0; i < left; i++)
	{
		j = borderInterpolate(i - left, srcdim[1], bordertype);
		j = borderInterpolate(i - left, srcdim[1], bordertype)*channel;
		for (k = 0; k < channel; k++)
			tab[i*channel + k] = j + k;
	}

	for (i = 0; i < right; i++)
	{
		j = borderInterpolate(srcdim[1] + i, srcdim[1], bordertype)*channel;
		for (k = 0; k < channel; k++)
			tab[(i + left)*channel + k] = j + k;
	}

	left *= channel;
	right *= channel;
	unsigned char* dstInner = dstptr + top * dststep + left;
	for (i = 0; i < srcdim[2]; i++, dstInner += dststep, srcptr += srcstep)
	{
		if (srcptr != dstInner)
			memcpy(dstInner, srcptr, srcstep);
		for (j = 0; j < left; j++)
			dstInner[j - left] = srcptr[tab[j]];
		for (j = 0; j < right; j++)
			dstInner[j + srcstep] = srcptr[tab[j + left]];
	}
	//memcpy top and bottom data
	dstptr += top * dststep;
	for (i = 0; i < top; i++)
	{
		j = borderInterpolate(i - top, srcdim[2], bordertype);
		memcpy(dstptr + (i - top)*dststep, dstptr + j * dststep, dststep);
	}
	for (i = 0; i < bottom; i++)
	{
		j = borderInterpolate(i + srcdim[2], srcdim[2], bordertype);
		memcpy(dstptr + (i + srcdim[2])*dststep, dstptr + j * dststep, dststep);
	}

	delete[]tab;
}

void copyMakeConstBorder_8u(const unsigned char*srcptr, size_t srcstep, int* srcdim,
	unsigned char* dstptr, size_t dststep, int* dstdim,
	int top, int left, int channel, std::vector<unsigned char>&value)
{
	int i, j;
	//构造dst width的常数向量
	unsigned char* constBuf = new unsigned char[dststep];
	int right = dstdim[1] - srcdim[1] - left;
	int bottom = dstdim[2] - srcdim[2] - top;
	for (i = 0; i < dstdim[1]; i++)
	{
		for (j = 0; j < channel; j++)
		{
			constBuf[i*channel + j] = value[j];
		}
	}

	left *= channel;
	right *= channel;
	unsigned char* dstInner = dstptr + top * dststep + left;
	for (i = 0; i < srcdim[2]; i++, dstInner += dststep, srcptr += srcstep)
	{
		if (dstInner != srcptr)
			memcpy(dstInner, srcptr, srcstep);
		memcpy(dstInner - left, constBuf, left);
		memcpy(dstInner + srcstep, constBuf, right);
	}
	dstptr += dststep * top;

	for (i = 0; i < top; i++)
		memcpy(dstptr + (i - top)*dststep, constBuf, dststep);

	for (i = 0; i < bottom; i++)
		memcpy(dstptr + (i + srcdim[2])*dststep, constBuf, dststep);

	delete[]constBuf;
}


void copyMakeborder(void* srcImage, void* dstImage, int top, int bottom, int left, int right, int bordertype)
{
	bordertype &= ~BORDER_ISOLATED;
	Image* srcPtr = (Image*)srcImage;
	Image* dstPtr = (Image*)dstImage;
	int channel = srcPtr->channel;
	int srcDim[3] = { channel,srcPtr->width,srcPtr->height };
	int dstDim[3] = { channel,dstPtr->width,dstPtr->height };
	int srcstep = srcPtr->width * channel, dststep = dstPtr->width * channel;
	const unsigned char* ptrSrc = (const unsigned char*)srcPtr->data;
	unsigned char* ptrDst = (unsigned char*)dstPtr->data;
	std::vector<unsigned char>value = { 0,0,0,0 };
	if (bordertype != 0)
	{
		copyMakeBorder_8u(ptrSrc, srcstep, srcDim, ptrDst, dststep, dstDim, top, left, channel, bordertype);
	}
	else
	{
		copyMakeConstBorder_8u(ptrSrc, srcstep, srcDim, ptrDst, dststep, dstDim, top, left, channel, value);
	}
}

CUDA_NAMESPACE_BEGIN
void copyMakeborder(void* srcImage, void* dstImage,	int top, int bottom, int left, int right, int bordertype)
{
	Image* srcPtr = (Image*)srcImage;
	Image* dstPtr = (Image*)dstImage;
	int channel = srcPtr->channel;
	unsigned char* devsrc, *devdst;
	int srcimagesize = srcPtr->height * srcPtr->width*channel * sizeof(unsigned char);
	int dstimagesize = dstPtr->height * dstPtr->width*channel * sizeof(unsigned char);
	cudaMalloc(&devsrc, srcimagesize);
	cudaMalloc(&devdst, dstimagesize);
	cudaMemcpy(devsrc, srcPtr->data, srcimagesize, cudaMemcpyHostToDevice);
	CopyMakeborderGPU(devsrc, srcPtr->height, srcPtr->width, channel,
		devdst, dstPtr->height, dstPtr->width,
		top, bottom, left, right, bordertype);
	cudaMemcpy(dstPtr->data, devdst, dstimagesize, cudaMemcpyDeviceToHost);
	cudaFree(devsrc);
	cudaFree(devdst);
}
CUDA_NAMESPACE_END
CVLIB_NAMESPACE_END