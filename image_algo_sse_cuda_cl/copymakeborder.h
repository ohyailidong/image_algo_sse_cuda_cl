#ifndef COPY_MAKE_BORDER_H_
#define COPY_MAKE_BORDER_H_

#include "common_define_cpu.h"
#include "common_extern_c.h"
CVLIB_NAMESPACE_BEGIN

void copyMakeBorder_8u(const unsigned char*srcptr, size_t srcstep, int* srcdim,
	unsigned char* dstptr, size_t dststep, int* dstdim,
	int top, int left, int channel, int bordertype);

void copyMakeConstBorder_8u(const unsigned char*ptrsrc, size_t srcstep, int* srcdim,
	unsigned char* dstptr, size_t dststep, int* dstdim,
	int top, int left, int channel, std::vector<unsigned char>&value);

int borderInterpolate(int p, int len, int borderType);

EXPORT_IMAGE_ALGO_DLL void copyMakeborder(void* srcImage, void* dstImage, 
	int top, int bottom, int left, int right, int bordertype);

CUDA_NAMESPACE_BEGIN
EXPORT_IMAGE_ALGO_DLL void copyMakeborder(void* srcImage, void* dstImage,
	int top, int bottom, int left, int right, int bordertype);
CUDA_NAMESPACE_END

CVLIB_NAMESPACE_END
#endif