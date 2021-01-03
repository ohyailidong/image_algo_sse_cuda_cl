#ifndef GUASSIAN_FILTER_H
#define	GUASSIAN_FILTER_H
#include "common_define.h"
#include "common_extern_c.h"

CVLIB_NAMESPACE_BEGIN

void getGaussianKernel(const int ksize, const float &sigma, float* kernel);

void createGaussianKernels(float* kx, float* ky, const int ksize, const float& sigmaX, const float& sigmaY);

void RowFilter(unsigned char* src, const int height, const int width, const int channel,
	const int ksize, float *kx, unsigned char* dst);

void ColomnFilter(unsigned char* src, const int height, const int width, const int channel,
	const int ksize, float *ky, unsigned char* dst);

void SepFilter2D(unsigned char* src, const int height, const int width, const int channel,
	const int ksize, float *kx, float*ky, unsigned char* dst);

EXPORT_IMAGE_ALGO_DLL void GaussianBlur(unsigned char* src, const int height, const int width, const int channel,
	const int ksize, const float &sigmaX, const float&sigmaY, unsigned char* dst);

CUDA_NAMESPACE_BEGIN
EXPORT_IMAGE_ALGO_DLL void GaussianBlur(unsigned char* src, const int height, const int width, const int channel,
	const int ksize, const float &sigmaX, const float&sigmaY, unsigned char* dst);
CUDA_NAMESPACE_END

CVLIB_NAMESPACE_END

#endif


