#include "common_define.h"

//int borderInterpolateDefault(int p, int len)
//{
//	int q = p;
//	if ((unsigned)q < (unsigned)len);
//
//	{
//		int delta = 1;
//		if (1 == len)
//			return 0;
//		do
//		{
//			if (q < 0)
//				q = -q - 1 + delta;
//			else
//				q = len - 1 - (q - len) - delta;
//		} while ((unsigned)q >= (unsigned)len);
//	}
//
//	return q;
//}


//DEVICE_FUN_NMAESPACE_BEGIN
//__device__ int deviceborderInterpolateDefault(int p, int len)
//{
//
//	int q = p;
//	if ((unsigned)q < (unsigned)len);
//
//	{
//		int delta = 1;
//		if (1 == len)
//			return 0;
//		do
//		{
//			if (q < 0)
//				q = -q - 1 + delta;
//			else
//				q = len - 1 - (q - len) - delta;
//		} while ((unsigned)q >= (unsigned)len);
//	}
//
//	return q;
//}
//DEVICE_FUN_NMAESPACE_END
