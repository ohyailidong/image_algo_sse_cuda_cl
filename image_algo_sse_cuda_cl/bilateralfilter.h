#ifndef BILATERAL_FILTER_H_
#define BILATERAL_FILTER_H_

#include "common_define.h"
#include "common_extern_c.h"
CVLIB_NAMESPACE_BEGIN

EXPORT_IMAGE_ALGO_DLL void bilateralFilter(unsigned char* src, const int srcheight, const int srcwidth, const int channel,
	const int d, double sigmacolor, double sigmaspace,
	unsigned char* dst, const int dstheight, const int dstwidth);

CUDA_NAMESPACE_BEGIN
EXPORT_IMAGE_ALGO_DLL void bilateralFilter(unsigned char* src, const int srcheight, const int srcwidth, const int channel,
	const int d, double sigmacolor, double sigmaspace,
	unsigned char* dst, const int dstheight, const int dstwidth);

CUDA_NAMESPACE_END
CVLIB_NAMESPACE_END

#endif
