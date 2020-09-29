#include "rgb2yuv.h"

namespace cvLib {

void RGB2YUV(unsigned char *RGB, unsigned char *dst, int Width, int Height, int channel)
{
	for (int YY = 0; YY < Height; YY++) {
		unsigned char *LinePS = RGB + YY * Width*channel;
		unsigned char *LinePtrDst = dst + YY * Width*channel;

		for (int XX = 0; XX < Width; XX++, LinePS += 3, LinePtrDst += 3)
		{
			int Blue = LinePS[0], Green = LinePS[1], Red = LinePS[2];
			LinePtrDst[0] = int(0.299*Red + 0.587*Green + 0.144*Blue);
			LinePtrDst[1] = int(-0.147*Red - 0.289*Green + 0.436*Blue);
			LinePtrDst[2] = int(0.615*Red - 0.515*Green - 0.100*Blue);
		}
	}
}

}


