#ifndef RGB_2_YUV_H
#define RGB_2_YUV_H

#include "common_define.h"
namespace cvLib {

EXPORT_IMAGE_ALGO_DLL void RGB2YUV(unsigned char *RGB, unsigned char *dst, int Width, int Height, int channel);

}

#endif
