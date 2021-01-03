#ifndef BOX_FILTER_H
#define BOX_FILTER_H
#include "common_define.h"
#include "common_extern_c.h"

CVLIB_NAMESPACE_BEGIN

void boxFilterRow(void* src, int srcheight, int srcwidth, const int& channel,
	const int& ksize, void* dst, int dstheight, int dstwidth);
void boxFilterColumn(void* src, int srcheight, int srcwidth, const int& channel,
	const int& ksize, void* dst, int dstheight, int dstwidth);
EXPORT_IMAGE_ALGO_DLL void boxFilter(unsigned char* src, int srcheight, int srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, int dstheight, int dstwidth);

MULTI_THREAD_NAMESPACE_BEGIN
EXPORT_IMAGE_ALGO_DLL void boxFilter(unsigned char* src, int srcheight, int srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, int dstheight, int dstwidth);
MULTI_THREAD_NAMESPACE_END

CUDA_NAMESPACE_BEGIN
EXPORT_IMAGE_ALGO_DLL void boxFilter(unsigned char* src, int srcheight, int srcwidth, const int &channel,
	const int&ksize, unsigned char* dst, int dstheight, int dstwidth);
CUDA_NAMESPACE_END

CVLIB_NAMESPACE_END
#endif
