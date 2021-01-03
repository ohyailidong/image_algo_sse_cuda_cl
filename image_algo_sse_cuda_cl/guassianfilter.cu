#include "common_extern_c.h"
#include "common_define.h"
#define THREAD 32

__device__ int borderInterpolateDefault(int p, int len)
{

	int q = p;
	if ((unsigned)q < (unsigned)len);

	{
		int delta = 1;
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

	return q;
}

__global__ void RowFilter_Kernel(unsigned char* devsrc, const int height, const int width, const int channel,
	const int ksize, const float* kx, unsigned char* devdst)
{
	int x = blockIdx.x* blockDim.x + threadIdx.x;
	int y = blockIdx.y* blockDim.y + threadIdx.y;
	int z = blockIdx.z;
	if (x >= width || y >= height)return;
	int k, r = ksize / 2;
	int step = width * channel;
	float sum = 0;
	for (k = -r; k <= r; k++)
	{
		int loc = borderInterpolateDefault(x + k, width);
		sum += devsrc[y* step + loc * channel + z] * kx[k + r];
	}
	devdst[y* step + x * channel + z] = (int)sum;
}

__global__ void ColomnFilter_Kernel(unsigned char* devsrc, const int height, const int width, const int channel,
	const int ksize, const float* ky, unsigned char* devdst)
{
	int x = blockIdx.x* blockDim.x + threadIdx.x;
	int y = blockIdx.y* blockDim.y + threadIdx.y;
	int z = blockIdx.z;
	if (x >= width || y >= height)return;
	int k, r = ksize / 2;
	int step = width * channel;
	float sum = 0;
	for (k = -r; k <= r; k++)
	{
		int loc = borderInterpolateDefault(y + k, height);
		sum += devsrc[loc*step + x * channel + z] * ky[k + r];
	}
	devdst[y* step + x * channel + z] = (int)sum;

}

extern "C" void GaussianBlurGPU(unsigned char* devsrc, const int height, const int width, const int channel,
	const int ksize, const float* kx, const float* ky, unsigned char* devdst)
{
	unsigned char* tempFilter = devdst + height * width* channel * sizeof(unsigned char);
	dim3 block(THREAD, THREAD);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, channel);
	RowFilter_Kernel << <grid, block >> > (devsrc, height, width, channel, ksize, kx, tempFilter);
	ColomnFilter_Kernel << <grid, block >> > (tempFilter, height, width, channel, ksize, ky, devdst);
}