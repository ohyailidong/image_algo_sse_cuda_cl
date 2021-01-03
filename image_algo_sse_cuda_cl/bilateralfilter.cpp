#include "bilateralfilter.h"

CVLIB_NAMESPACE_BEGIN
void bilateralFilter(unsigned char* src, const int srcheight, const int srcwidth, const int channel,
	const int d, double sigmacolor, double sigmaspace,
	unsigned char* dst, const int dstheight, const int dstwidth)
{
	double guass_color_coff = -0.5 / (sigmacolor*sigmacolor);
	double guass_space_coff = -0.5 / (sigmaspace*sigmaspace);
	std::vector<float> colorweight(channel * 255);
	std::vector<float> spaceweight(d*d, 0);
	std::vector<int> spaceoffset(d*d, 0);
	float* pcolorweight = &colorweight[0];
	float* pspaceweight = &spaceweight[0];
	//initialize color bilateral filter coff
	int i, j, maxk, radius = d / 2;
	for (i = 0; i < channel * 255; i++)
		pcolorweight[i] = std::exp(i*i* guass_color_coff);
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
	//filter
	for (j = 0; j < dstheight; j++)
	{
		unsigned char* srcLine = src + srcwidth * channel * j;
		unsigned char* dstLine = dst + dstwidth * channel * j;
		for (i = 0; i < dstwidth; i++)
		{
			unsigned char* src_center = srcLine + srcwidth * channel* radius + (i + radius) * channel;
			if (channel == 1)
			{
				float sum = 0, wsum = 0;
				int val_center = src_center[0];
				for (int k = 0; k < maxk; k++)
				{
					int val = src_center[spaceoffset[k]];
					float w = spaceweight[k] * colorweight[std::abs(val - val_center)];
					sum += val * w;
					wsum += w;
				}
				dstLine[i] = int(sum / wsum);
			}
			else if (channel == 3)
			{
				int val_center0 = src_center[0];
				int val_center1 = src_center[1];
				int val_center2 = src_center[2];
				for (int cn = 0; cn < channel; cn++)
				{
					float sum = 0, wsum = 0;

					for (int k = 0; k < maxk; k++)
					{
						int b = src_center[spaceoffset[k]];
						int g = src_center[spaceoffset[k] + 1];
						int r = src_center[spaceoffset[k] + 2];

						int val = src_center[spaceoffset[k] + cn];
						float w = spaceweight[k] * colorweight[std::abs(b - val_center0) + std::abs(g - val_center1) + std::abs(r - val_center2)];
						sum += val * w;
						wsum += w;

					}
					dstLine[i*channel + cn] = int(sum / wsum);
				}
			}
			else
			{
				assert(false);
			}
		}
	}
}

CUDA_NAMESPACE_BEGIN
void bilateralFilter(unsigned char* src, const int srcheight, const int srcwidth, const int channel,
	const int d, double sigmacolor, double sigmaspace,
	unsigned char* dst, const int dstheight, const int dstwidth)
{
	unsigned char* devsrc, *devdst;
	int srcimagebyte = srcheight * srcwidth* channel * sizeof(unsigned char);
	int dstimagebyte = dstheight * dstwidth*channel * sizeof(unsigned char);
	cudaMalloc(&devsrc, srcimagebyte);
	cudaMalloc(&devdst, dstimagebyte);
	cudaMemcpy(devsrc, src, srcimagebyte, cudaMemcpyHostToDevice);
	cudaMemcpy(devdst, dst, dstimagebyte, cudaMemcpyHostToDevice);
	bilateralFilterGPU(devsrc, srcheight, srcwidth, channel, d, sigmacolor, sigmaspace, devdst, dstheight, dstwidth);
	cudaMemcpy(dst, devdst, dstimagebyte, cudaMemcpyDeviceToHost);
	cudaFree(devsrc);
	cudaFree(devdst);
}

CUDA_NAMESPACE_END
CVLIB_NAMESPACE_END