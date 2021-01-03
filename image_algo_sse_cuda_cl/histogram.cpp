#include "histogram.h"

CVLIB_NAMESPACE_BEGIN
//Source image with CV_8UC1 type.
void equalizeHist(unsigned char* src, unsigned char*dst, int &height, int &width)
{
	const int hist_sz = 256;
	int hist[hist_sz] = { 0, };
	int lut[hist_sz];
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
			hist[src[i*width + j]]++;
	}

	int i = 0;
	while (!hist[i])
		i++;

	int total = width * height;
	if (hist[i] == total)
	{
		//原始图像只有一个灰度级
		std::for_each(dst, dst + total, [i](unsigned char& value) { value = i; });
		return;
	}
	float scale = (hist_sz - 1.f) / (total - hist[i]);
	int sum = 0;
	//将第一个灰度级置为0
	for (lut[i++] = 0; i < hist_sz; i++)
	{
		sum += hist[i];
		lut[i] = static_cast<unsigned char>(sum * scale + 0.5f);
	}
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			dst[y * width + x] = static_cast<unsigned char>(lut[src[y * width + x]]);
		}
	}
}

CUDA_NAMESPACE_BEGIN
void equalizeHist(unsigned char* src, unsigned char*dst, int &height, int &width)
{
#define HIST_SZ 256

	unsigned char *devimage, *devdst;
	float *devhist;
	unsigned int* devlut;
	const int bytecharsize = sizeof(unsigned char);
	const int byteintsize = sizeof(unsigned int);
	const int sizecount = height * width * bytecharsize;
	cudaMalloc((void**)&devimage, sizecount);
	cudaMemcpy(devimage, src, sizecount, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&devdst, sizecount);
	cudaMalloc((void**)&devhist, HIST_SZ * sizeof(float));
	cudaMemset(devhist, 0, HIST_SZ * sizeof(float));
	cudaMalloc((void**)&devlut, HIST_SZ * byteintsize);
	cudaMemset(devlut, 0, HIST_SZ*byteintsize);

	equalizeHistGPU(devimage, devhist, devlut, height, width, devdst);
	cudaMemcpy(dst, (void*)devdst, sizecount, cudaMemcpyDeviceToHost);

	cudaFree((void*)devimage);
	cudaFree((void*)devhist);
	cudaFree((void*)devlut);
	cudaFree((void*)devdst);
}
CUDA_NAMESPACE_END

CVLIB_NAMESPACE_END