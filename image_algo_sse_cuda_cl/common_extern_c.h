#ifndef COMMON_EXTERN_C_H
#define COMMON_EXTERN_C_H

extern "C" void GaussianBlurGPU(unsigned char* devsrc, const int height, const int width, const int channel,
	const int ksize, const float* kx, const float* ky, unsigned char* devdst);

extern "C" void equalizeHistGPU(unsigned char* devimage, float* devhist, unsigned int* devlut,
	const int height, const int width, unsigned char* devdst);

extern "C" void medianBlurGPU(unsigned char* devsrc, const int& srcheight, const int& srcwidth, const int& channel,
	const int& ksize, unsigned char* devdst, const int& dstheight, const int& dstwidth);

extern "C" void boxfilterGPU(unsigned char* devsrc, const int& heigth, const int& width, const int& channel,
	const int& ksize, unsigned char* devdst);

extern "C" void bilateralFilterGPU(unsigned char* devsrc, const int height, const int width, const int channel,
	const int d, double sigmacolor, double sigmaspace, unsigned char* devdst);

extern "C" void RGB2GRAY_GPU(unsigned char* devsrc, const int height, const int width, unsigned char* devdst);

extern "C" void CopyMakeborderGPU(unsigned char* devsrc, int srcheight, int srcwidth, int channel,
	unsigned char* devdst, int dstheight, int dstwidth,
	int top, int bottom, int left, int right, int bordertype);
#endif
