#ifndef MEDIANBLUR_H
#define MEDIANBLUR_H

#include "common_define_cpu.h"
#include "common_extern_c.h"

CVLIB_NAMESPACE_BEGIN

//const copy make border and only left and rignt
EXPORT_IMAGE_ALGO_DLL void medianBlur(unsigned char* src, const int& srcheight, const int& srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, const int& dstheight, const int& dstwidth);

void _medianBlur(unsigned char* src, const int& srcheight, const int& srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, const int& dstheight, const int& dstwidth);

void _medianBlur_Sort(unsigned char* src, const int& srcheight, const int& srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, const int& dstheight, const int& dstwidth);

CUDA_NAMESPACE_BEGIN
EXPORT_IMAGE_ALGO_DLL void medianBlur(unsigned char* src, const int& srcheight, const int& srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, const int& dstheight, const int& dstwidth);
CUDA_NAMESPACE_END

CVLIB_NAMESPACE_END


#endif
