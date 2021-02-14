#ifndef HISTOGRAM_HIST_H
#define HISTOGRAM_HIST_H

#include "common_define_cpu.h"
#include "common_extern_c.h"

CVLIB_NAMESPACE_BEGIN
//Source image with CV_8UC1 type.
EXPORT_IMAGE_ALGO_DLL void equalizeHist(unsigned char* src, unsigned char*dst, int &height, int &width);

CUDA_NAMESPACE_BEGIN
EXPORT_IMAGE_ALGO_DLL void equalizeHist(unsigned char* src, unsigned char*dst, int &height, int &width);
CUDA_NAMESPACE_END

CVLIB_NAMESPACE_END
#endif
