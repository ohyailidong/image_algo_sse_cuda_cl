#include "common_define_gpu.h"
#include "common_extern_c.h"
#include "common_data_define.h"
__device__ int device_borderInterpolate(int p, int len, int borderType)
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

__global__ void device_copyMakeBorder_8u(const unsigned char * srcptr, size_t srcstep, int * srcdim,
	unsigned char * dstptr, size_t dststep, int * dstdim,
	int top, int left, int channel, int bordertype)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z;
	int srcwidth(srcdim[1]), srcheight(srcdim[2]);
	int dstwidth(dstdim[1]), dstheight(dstdim[2]);

	if (x >= dstwidth || y >= dstheight)return;
	int src_x = device_borderInterpolate(x - left, srcdim[1], bordertype);
	int src_y = device_borderInterpolate(y - top, srcdim[2], bordertype);
	int srcloc = src_y * srcstep + src_x * channel + z;
	int dstloc = y * dststep + x * channel + z;
	dstptr[dstloc] = srcptr[srcloc];

}
void CopyMakeborderGPU(unsigned char* devsrc, int srcheight, int srcwidth, int channel,
	unsigned char* devdst, int dstheight, int dstwidth,
	int top, int bottom, int left, int right, int bordertype)
{
	bordertype &= ~BORDER_ISOLATED;

	int srcDim[3] = { channel,srcwidth,srcheight };
	int dstDim[3] = { channel,dstwidth,dstheight };
	int srcstep = srcwidth * channel, dststep = dstwidth * channel;
	int *srcDimPtr, *dstDimPtr;
	int dimSize = sizeof(srcDim);
	cudaMalloc((&srcDimPtr), sizeof(srcDim));
	cudaMalloc(&dstDimPtr, dimSize);
	cudaMemcpy(srcDimPtr, srcDim, dimSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dstDimPtr, dstDim, dimSize, cudaMemcpyHostToDevice);

#define DEVIDE 32
	dim3 dimblock(DEVIDE, DEVIDE);
	dim3 dimgrid((dstwidth + DEVIDE - 1) / DEVIDE, (dstheight + DEVIDE - 1) / DEVIDE, channel);//gridµÄÎ¬¶È

	//std::vector<uchar>value = { 0,0,0,0 };
	if (bordertype != 0)
	{
		device_copyMakeBorder_8u << < dimgrid, dimblock >> > (devsrc, srcstep, srcDimPtr,
			devdst, dststep, dstDimPtr, top, left, channel, bordertype);

	}
	else
	{
		//copyMakeConstBorder_8u(devsrc, srcstep, srcDimPtr, devdst, dststep, dstDimPtr, top, left, channel, value);
	}

	cudaFree(srcDimPtr);
	cudaFree(dstDimPtr);
}