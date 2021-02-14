#include "guassianfilter.h"

CVLIB_NAMESPACE_BEGIN
//int borderInterpolateDefault(int p, int len)
//{
//	int q = p;
//	if ((unsigned)q < (unsigned)len);
//
//	{
//		int delta = 1;
//		if (1 == len)
//			return 0;
//		do
//		{
//			if (q < 0)
//				q = -q - 1 + delta;
//			else
//				q = len - 1 - (q - len) - delta;
//		} while ((unsigned)q >= (unsigned)len);
//	}
//
//	return q;
//}

void getGaussianKernel(const int ksize, const float &sigma, float* kernel)
{
	const int SMALL_GAUSSIAN_SIZE = 7;
	static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
	{
		{ 1.f },
		{ 0.25f, 0.5f, 0.25f },
		{ 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f },
		{ 0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f }
	};
	const float* fixed_kernel = ksize % 2 == 1 && ksize <= SMALL_GAUSSIAN_SIZE && sigma <= 0 ? small_gaussian_tab[ksize >> 1] : 0;

	float sigmaX = sigma > 0 ? sigma : ((ksize - 1)*0.5 - 1)*0.3 + 0.8;
	float scale2X = -0.5 / (sigmaX*sigmaX);
	float sum = 0;
	int i;
	for (i = 0; i < ksize; i++)
	{
		float x = i - (ksize - 1)*0.5;
		kernel[i] = fixed_kernel ? fixed_kernel[i] : (float)std::exp(scale2X* x* x);
		sum += kernel[i];
	}
	sum = 1.f / sum;
	for (i = 0; i < ksize; i++)
		kernel[i] = sum * kernel[i];
}

void createGaussianKernels(float* kx, float* ky, const int ksize, const float& sigmaX, const float& sigmaY)
{
	assert(sigmaX >= 0 && sigmaY >= 0 && ksize > 0);
	getGaussianKernel(ksize, sigmaX, kx);
	if (sigmaX == sigmaY)
		memcpy(ky, kx, sizeof(float)* ksize);
	else
		getGaussianKernel(ksize, sigmaY, ky);
}

void RowFilter(unsigned char* src, const int height, const int width, const int channel,
	const int ksize, float *kx, unsigned char* dst)
{
	int i, j, n, k, r = ksize / 2;
	int step = width * channel;
	for (j = 0; j < height; ++j, src += step, dst += step)
	{
		for (i = 0; i < width; ++i)
		{
			for (n = 0; n < channel; ++n)
			{
				float sum = 0;
				for (k = -r; k <= r; k++)
				{
					int loc = cpu::borderInterpolateDefault(i + k, width);
					int a = src[loc*channel + n];
					float b = kx[k + r];
					sum += src[loc*channel + n] * kx[k + r];
				}
				dst[i*channel + n] = (int)sum;
			}
		}
	}
}

void ColomnFilter(unsigned char* src, const int height, const int width, const int channel,
	const int ksize, float *ky, unsigned char* dst)
{
	int i, j, n, k, r = ksize / 2;
	int step = width * channel;
	for (j = 0; j < height; ++j, dst += step)
	{
		for (i = 0; i < width; ++i)
		{
			for (n = 0; n < channel; ++n)
			{
				float sum = 0;
				for (k = -r; k <= r; k++)
				{
					int loc = cpu::borderInterpolateDefault(j + k, height);
					int a = src[loc*step + i * channel + n];
					float b = ky[k + r];
					sum += src[loc*step + i * channel + n] * ky[k + r];
				}
				dst[i*channel + n] = (int)sum;
			}
		}
	}
}

void SepFilter2D(unsigned char* src, const int height, const int width, const int channel,
	const int ksize, float *kx, float*ky, unsigned char* dst)
{
	unsigned char* tempFilter = new unsigned char[width* height* channel];
	RowFilter(src, height, width, channel, ksize, kx, tempFilter);
	ColomnFilter(tempFilter, height, width, channel, ksize, ky, dst);
	delete[]tempFilter;
}

void GaussianBlur(unsigned char* src, const int height, const int width, const int channel,
	const int ksize, const float &sigmaX, const float&sigmaY, unsigned char* dst)
{
	float *kx = new float[ksize];
	float *ky = new float[ksize];
	createGaussianKernels(kx, ky, ksize, sigmaX, sigmaY);
	SepFilter2D(src, height, width, channel, ksize, kx, ky, dst);
	delete[]kx;
	delete[]ky;
}

CUDA_NAMESPACE_BEGIN
void GaussianBlur(unsigned char* src, const int height, const int width, const int channel,
	const int ksize, const float &sigmaX, const float&sigmaY, unsigned char* dst)
{
	unsigned char* mallocPtr, *devsrc, *devdst;
	int imagesize = height * width*channel * sizeof(unsigned char);
	int totalsize = imagesize * 3;
	cudaMalloc(&mallocPtr, totalsize);
	devsrc = mallocPtr;
	devdst = devsrc + imagesize;
	cudaMemcpy(devsrc, src, imagesize, cudaMemcpyHostToDevice);
	float *kernelPtr, *kx, *ky;
	int kernelsize = ksize * 2 * sizeof(float);
	cudaMalloc(&kernelPtr, kernelsize);
	kx = kernelPtr, ky = kx + ksize;
	//calculate kernel 
	float *kxHost = new float[ksize];
	float *kyHost = new float[ksize];
	createGaussianKernels(kxHost, kyHost, ksize, sigmaX, sigmaY);
	cudaMemcpy(kx, kxHost, kernelsize / 2, cudaMemcpyHostToDevice);
	cudaMemcpy(ky, kyHost, kernelsize / 2, cudaMemcpyHostToDevice);
	delete[]kxHost;
	delete[]kyHost;
	GaussianBlurGPU(devsrc, height, width, channel, ksize, kx, ky, devdst);
	cudaMemcpy(dst, devdst, imagesize, cudaMemcpyDeviceToHost);
	cudaFree(mallocPtr);
	cudaFree(kernelPtr);
}
CUDA_NAMESPACE_END

CVLIB_NAMESPACE_END

