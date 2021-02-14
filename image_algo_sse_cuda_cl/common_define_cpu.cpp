#include "common_define_cpu.h"

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

CPU_FUN_NMAESPACE_END

