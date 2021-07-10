#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H

#include "common_define_cpu.h"
#include "common_data_define.h"
CVLIB_NAMESPACE_BEGIN

EXPORT_IMAGE_ALGO_DLL void morphologyEx(Image& src, Image& dst, int op, Image& kernel, int iter, int bordertype);

EXPORT_IMAGE_ALGO_DLL void erode(Image& src, Image& dst, Image& kernel,	int iterations, int borderType);

EXPORT_IMAGE_ALGO_DLL void dilate(Image& src, Image& dst, Image& kernel,int iterations, int borderType);

EXPORT_IMAGE_ALGO_DLL void getStructuringElement(int shape, Image& dstEle);

void morphApply(Image& src, Image& dst, Image& kernel, int iterations, void*pfun);


CVLIB_NAMESPACE_END

#endif