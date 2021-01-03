#include "common_define.h"

#define RED_SCALE 4898
#define GREEN_SCALE 9618
#define BLUE_SCALE 1868

__global__ void rgb2gray_kernel(unsigned char* devsrc, int height, int width, unsigned char* devdst)
{
	int x = blockIdx.x* blockDim.x + threadIdx.x;
	int y = blockIdx.y* blockDim.y + threadIdx.y;
	int srcLoc = y * width * 3 + x * 3;
	int dstLoc = y * width + x;
	if (x >= width || y >= height)
		return;
	int gray = (devsrc[srcLoc] * BLUE_SCALE + devsrc[srcLoc + 1] * GREEN_SCALE + devsrc[srcLoc + 2] * RED_SCALE) >> 14;
	devdst[dstLoc] = gray > 255 ? 255 : gray;
}

extern "C" void RGB2GRAY_GPU(unsigned char* devsrc, const int height, const int width, unsigned char* devdst)
{
#define THREAD 32
	dim3 block(THREAD, THREAD);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	rgb2gray_kernel << <grid, block >> > (devsrc, height, width, devdst);
}
