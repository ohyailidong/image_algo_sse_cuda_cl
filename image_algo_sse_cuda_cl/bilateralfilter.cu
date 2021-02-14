#include "common_define_gpu.h"
#include "common_extern_c.h"

#define THREAD 32
#define MAXCN 3
#define MAXKERNEL 64

__constant__ float dcolorweight[255 * MAXCN];
__constant__ float dspaceweight[MAXKERNEL * MAXKERNEL];
__constant__ int dspaceoffset[MAXKERNEL * MAXKERNEL];

__global__ void filter_kernel_cn1(unsigned char* devsrc, const int srcheight, const int srcwidth, const int channel,
	const int r, const int maxk, unsigned char* devdst, const int dstheight, const int dstwidth)
{
	int x = blockIdx.x* blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dst_loc = y * dstwidth + x;
	int src_center = (y + r) *srcwidth + (x + r);
	if (x < dstwidth&& y < dstheight)
	{
		float sum = 0, wsum = 0;
		int val_center = devsrc[src_center];
		for (int k = 0; k < maxk; k++)
		{
			int val = devsrc[src_center + dspaceoffset[k]];
			float w = dspaceweight[k] * dcolorweight[std::abs(val - val_center)];
			sum += val * w;
			wsum += w;
		}
		devdst[dst_loc] = int(sum / wsum);
	}
}

__global__ void filter_kernel_cn3(unsigned char* devsrc, const int srcheight, const int srcwidth, const int channel,
	const int r, const int maxk, unsigned char* devdst, const int dstheight, const int dstwidth)
{
	int x = blockIdx.x* blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z;
	int dst_loc = y * dstwidth*channel + x * channel + z;
	int src_center = (y + r) *srcwidth*channel + (x + r)*channel;
	if (x < dstwidth&& y < dstheight)
	{
		float sum = 0, wsum = 0;
		int val_center0 = devsrc[src_center];
		int val_center1 = devsrc[src_center + 1];
		int val_center2 = devsrc[src_center + 2];

		for (int k = 0; k < maxk; k++)
		{
			int b = devsrc[src_center + dspaceoffset[k]];
			int g = devsrc[src_center + dspaceoffset[k] + 1];
			int r = devsrc[src_center + dspaceoffset[k] + 2];

			int val = devsrc[src_center + dspaceoffset[k] + z];
			float w = dspaceweight[k] * dcolorweight[std::abs(b - val_center0) +
				std::abs(g - val_center1) + std::abs(r - val_center2)];
			sum += val * w;
			wsum += w;
		}
		devdst[dst_loc] = int(sum / wsum);
	}
}

extern "C" void bilateralFilterGPU(unsigned char* devsrc, const int srcheight, const int srcwidth, const int channel,
	const int d, double sigmacolor, double sigmaspace,
	unsigned char* devdst, const int dstheight, const int dstwidth)
{
	float guass_color_coff = -0.5 / (sigmacolor*sigmacolor);
	float guass_space_coff = -0.5 / (sigmaspace*sigmaspace);
	std::vector<float> colorweight(channel * 255);
	std::vector<float> spaceweight(d*d, 0);
	std::vector<int> spaceoffset(d*d, 0);

	//initialize color bilateral filter coff
	int i, j, maxk, radius = d / 2;
	for (i = 0; i < channel * 255; i++)
		colorweight[i] = std::exp(i*i* guass_color_coff);
	//initialize space bilatera filter coff
	for (i = -radius, maxk = 0; i <= radius; i++)
	{
		for (j = -radius; j <= radius; j++)
		{
			double r = std::sqrt(double(i*i + j * j));
			if (r > radius)
				continue;
			spaceweight[maxk] = (float)std::exp(r*r* guass_space_coff);
			spaceoffset[maxk++] = (int)i* srcwidth*channel + j * channel;
		}
	}
	cudaMemcpyToSymbol(dcolorweight, &colorweight[0], sizeof(float)*channel * 255, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dspaceweight, &spaceweight[0], sizeof(float)*maxk, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dspaceoffset, &spaceoffset[0], sizeof(int)*maxk, 0, cudaMemcpyHostToDevice);

	dim3 block(THREAD, THREAD);
	dim3 grid((dstwidth + block.x - 1) / block.x, (dstheight + block.y - 1) / block.y, channel);
	if (channel == 1)
		filter_kernel_cn1 << <grid, block >> > (devsrc, srcheight, srcwidth, channel, radius, maxk, devdst, dstheight, dstwidth);
	else if (channel == 3)
		filter_kernel_cn3 << <grid, block >> > (devsrc, srcheight, srcwidth, channel, radius, maxk, devdst, dstheight, dstwidth);
	else
		assert(false);
}