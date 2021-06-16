#ifndef MATCH_TEMPLATE_H
#define MATCH_TEMPLATE_H

#include "common_define_cpu.h"
#include "common_data_define.h"
CVLIB_NAMESPACE_BEGIN

EXPORT_IMAGE_ALGO_DLL void matchTemplate(Image& src, Image& matchtemplate, Image& dst,
	int method, Image _mask = Image());

void crossCorr(const Image& img, const Image& templ, Image& corr);

void common_matchTemplate(Image& img, Image& templ, Image& result, int method, int cn);

EXPORT_IMAGE_ALGO_DLL void integral(Image img, Image sum, Image sqsum=Image());

CVLIB_NAMESPACE_END

#endif