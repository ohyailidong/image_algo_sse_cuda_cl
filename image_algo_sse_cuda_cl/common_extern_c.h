#ifndef COMMON_EXTERN_C_H
#define COMMON_EXTERN_C_H

extern "C" void GaussianBlurGPU(unsigned char* devsrc, const int height, const int width, const int channel,
	const int ksize, const float* kx, const float* ky, unsigned char* devdst);

#endif
