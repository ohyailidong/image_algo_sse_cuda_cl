#include "common_define_cpu.h"
#include "common_data_define.h"
CPU_FUN_NMAESPACE_BEGIN
int borderInterpolateDefault(int p, int len)
{
	int q = p;
	if ((unsigned)q < (unsigned)len);

	{
		int delta = 1;
		if (1 == len)
			return 0;
		do
		{
			if (q < 0)
				q = -q - 1 + delta;
			else
				q = len - 1 - (q - len) - delta;
		} while ((unsigned)q >= (unsigned)len);
	}

	return q;
}
void ImageSub(Image& src1, Image& src2, Image& dst) {
#define uchar unsigned char
	uchar* src1Ptr = (uchar*)src1.data;
	uchar* src2Ptr = (uchar*)src2.data;
	uchar* dstPtr  = (uchar*)dst.data;
	for (int i = 0; i < src1.GetElementNum(); ++i)
		dstPtr[i] = src1Ptr[i] - src2Ptr[i];

}

CPU_FUN_NMAESPACE_END

