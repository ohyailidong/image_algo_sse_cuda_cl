#ifndef BOX_FILTER_H
#define BOX_FILTER_H
#include "common_define_cpu.h"
#include "common_extern_c.h"

CVLIB_NAMESPACE_BEGIN

void boxFilterRow(void* src, int height, int width, const int& channel,
	const int& ksize, void* dst);
void boxFilterColumn(void* src, int height, int width, const int& channel,
	const int& ksize, void* dst);
EXPORT_IMAGE_ALGO_DLL void boxFilter(unsigned char* src, int height, int width, const int& channel,
	const int& ksize, unsigned char* dst);

MULTI_THREAD_NAMESPACE_BEGIN
EXPORT_IMAGE_ALGO_DLL void boxFilter(unsigned char* src, int srcheight, int srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, int dstheight, int dstwidth);
MULTI_THREAD_NAMESPACE_END

CUDA_NAMESPACE_BEGIN
EXPORT_IMAGE_ALGO_DLL void boxFilter(unsigned char* src, int height, int width, const int &channel,
	const int&ksize, unsigned char* dst);
CUDA_NAMESPACE_END

CVLIB_NAMESPACE_END
#endif
