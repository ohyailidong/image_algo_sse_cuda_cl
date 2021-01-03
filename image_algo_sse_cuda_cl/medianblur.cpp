#include "medianblur.h"

CVLIB_NAMESPACE_BEGIN

void _medianBlur(unsigned char* src, const int& srcheight, const int& srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, const int& dstheight, const int& dstwidth);
void _medianBlur_Sort(unsigned char* src, const int& srcheight, const int& srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, const int& dstheight, const int& dstwidth);
//const copy make border and only left and rignt
void medianBlur(unsigned char* src, const int& srcheight, const int& srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, const int& dstheight, const int& dstwidth)
{
	bool useSort = ksize == 3;
	if (useSort)
	{
		_medianBlur_Sort(src, srcheight, srcwidth, channel, ksize, dst, dstheight, dstwidth);
	}
	else
		_medianBlur(src, srcheight, srcwidth, channel, ksize, dst, dstheight, dstwidth);
}
//opencv 源码 中序遍历顺序，偶数列从 上-下，奇数列从 下-上
//1   8	  9
//2	  7	  10
//3	  6	  11
//4   5	  12

//本文实现的所有列都是从 上-下
void _medianBlur(unsigned char* src, const int& srcheight, const int& srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, const int& dstheight, const int& dstwidth)
{
#define N 16
#define UPDATE(pix,cn,op)		\
	{							\
		int p = (pix);			\
		zone1[cn][pix] op;		\
		zone0[cn][pix >> 4] op; \
	}

	int zone0[4][N];//粗分灰度等级，方便粗定位中值位置
	int zone1[4][N*N];//细分灰度等级，精确查找中值位置，最大支持4通道
	assert(channel > 0 && channel <= 4);
	int srcstep = srcwidth * channel;
	int dststep = dstwidth * channel;
	const unsigned char*src_max = src + srcstep * srcheight;
	int x, y;
	int n2 = ksize * ksize / 2;//中值位置

	//按照列优先的顺序计算ksize内的直方图
	for (x = 0; x < dstwidth; x++, src += channel, dst += channel)
	{
		unsigned char* curdst = dst;
		const unsigned char* srctop = src;
		const unsigned char* srcbottom = src;
		int c, k;
		memset(zone0, 0, sizeof(zone0[0])* channel);
		memset(zone1, 0, sizeof(zone1[0])* channel);
		//计算每列的第一个直方图
		for (y = 0; y <= ksize / 2; y++)
		{
			for (c = 0; c < channel; c++)
			{
				if (y > 0)
				{
					for (k = 0; k < channel* ksize; k += channel)
						UPDATE(srcbottom[k + c], c, ++);
				}
				else
				{
					//第0行进行const 边界扩充，有(ksize/2+1)个值一样
					for (k = 0; k < channel* ksize; k += channel)
						UPDATE(srcbottom[k + c], c, +=ksize / 2 + 1);
				}
			}
			if (y < dstheight - 1)
				srcbottom += srcstep;
		}
		//计算中值，并更新直方图
		for (y = 0; y < dstheight; y++, curdst += dststep)
		{
			for (c = 0; c < channel; c++)
			{
				int s = 0;
				//k代表灰度等级
				for (k = 0;; k++)
				{
					int t = s + zone0[c][k];
					if (t > n2)break;
					s = t;
				}
				for (k = k * N;; k++)
				{
					s += zone1[c][k];
					if (s > n2) break;
				}
				curdst[c] = k;
			}
			if (y + 1 == dstheight)
				break;
			//更新直方图
			if (channel == 1)
			{
				for (k = 0; k < ksize; k++)
				{
					int p = srctop[k];
					int q = srcbottom[k];
					zone1[0][p]--;
					zone0[0][p >> 4]--;
					zone1[0][q]++;
					zone0[0][q >> 4]++;
				}
			}
			else if (channel == 3)
			{
				for (k = 0; k < ksize * 3; k += 3)
				{
					UPDATE(srctop[k], 0, --);
					UPDATE(srctop[k + 1], 1, --);
					UPDATE(srctop[k + 2], 2, --);

					UPDATE(srcbottom[k], 0, ++);
					UPDATE(srcbottom[k + 1], 1, ++);
					UPDATE(srcbottom[k + 2], 2, ++);
				}
			}
			else
			{
				assert(channel == 4);
				for (k = 0; k < ksize * 4; k += 4)
				{
					UPDATE(srctop[k], 0, --);
					UPDATE(srctop[k + 1], 1, --);
					UPDATE(srctop[k + 2], 2, --);
					UPDATE(srctop[k + 3], 3, --);

					UPDATE(srcbottom[k], 0, ++);
					UPDATE(srcbottom[k + 1], 1, ++);
					UPDATE(srcbottom[k + 2], 2, ++);
					UPDATE(srcbottom[k + 3], 3, ++);
				}
			}
			//update srctop and srcbottom
			if (srcbottom + srcstep < src_max)
				srcbottom += srcstep;
			if (y >= ksize / 2)
				srctop += srcstep;
		}
	}
}

//inline void op(int& a, int &b)
//{
//	int t = a;				
//	a = std::min(a, b);		
//	b = std::max(b, t);
//}
void _medianBlur_Sort(unsigned char* src, const int& srcheight, const int& srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, const int& dstheight, const int& dstwidth)
{
	//#define op(min,max)				\
	//	{							\
	//		if(min>max)				\
	//			std::swap(min,max);	\
	//	}
		//比swap计算快
#define op(a,b)					\
	{							\
        int t = a;				\
		a = std::min(a, b);		\
		b = std::max(b, t);		\
	}

	assert(ksize == 3 || ksize == 5);
	if (ksize == 3)
	{
		int dststep = dstwidth * channel;
		int srcstep = srcwidth * channel;
		int x, y;
		for (y = 0; y < dstheight; y++, dst += dststep)
		{
			const unsigned char* row0 = src + std::max(y - 1, 0) * srcstep;
			const unsigned char* row1 = src + y * srcstep;
			const unsigned char* row2 = src + std::min(y + 1, srcheight - 1)*srcstep;
			for (x = 0; x < dstwidth*channel; x++)
			{
				int j0 = x;
				int j1 = x + channel;
				int j2 = x + 2 * channel;
				int p0 = row0[j0], p1 = row0[j1], p2 = row0[j2];
				int p3 = row1[j0], p4 = row1[j1], p5 = row1[j2];
				int p6 = row2[j0], p7 = row2[j1], p8 = row2[j2];

				op(p1, p2); op(p4, p5); op(p7, p8); op(p0, p1);
				op(p3, p4); op(p6, p7); op(p1, p2); op(p4, p5);
				op(p7, p8); op(p0, p3); op(p5, p8); op(p4, p7);
				op(p3, p6); op(p1, p4); op(p2, p5); op(p4, p7);
				op(p4, p2); op(p6, p4); op(p4, p2);
				dst[x] = p4;
			}
		}

	}
	else
	{

	}
}

CUDA_NAMESPACE_BEGIN
void medianBlur(unsigned char* src, const int& srcheight, const int& srcwidth, const int& channel,
	const int& ksize, unsigned char* dst, const int& dstheight, const int& dstwidth)
{
	unsigned char* devsrc, *devdst;
	int srcimagesize = srcwidth * channel*srcheight;
	int dstimagesize = dstwidth * channel* dstheight;
	cudaMalloc(&devsrc, srcimagesize * sizeof(unsigned char));
	cudaMalloc(&devdst, dstimagesize * sizeof(unsigned char));
	cudaMemcpy(devsrc, src, srcimagesize * sizeof(unsigned char), cudaMemcpyHostToDevice);
	medianBlurGPU(devsrc, srcheight, srcwidth, channel, ksize, devdst, dstheight, dstwidth);
	cudaMemcpy(dst, devdst, dstimagesize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(devsrc);
	cudaFree(devdst);
}
CUDA_NAMESPACE_END

CVLIB_NAMESPACE_END