//#ifndef RGB_2_GRAY_H
//#define RGB_2_GRAY_H
//
//#include "common_define.h"
//#include "tbb/task_scheduler_init.h"
//#include "tbb/blocked_range.h"
//#include "tbb/blocked_range2d.h"
//#include "tbb/parallel_for.h"
//
////Gray = (R * 1 + G * 2 + B * 1) >> 2
////Gray = (R * 2 + G * 5 + B * 1) >> 3
////Gray = (R * 4 + G * 10 + B * 2) >> 4
////Gray = (R * 9 + G * 19 + B * 4) >> 5
////Gray = (R * 19 + G * 37 + B * 8) >> 6
////Gray = (R * 38 + G * 75 + B * 15) >> 7
////Gray = (R * 76 + G * 150 + B * 30) >> 8
////Gray = (R * 153 + G * 300 + B * 59) >> 9
////Gray = (R * 306 + G * 601 + B * 117) >> 10
////Gray = (R * 612 + G * 1202 + B * 234) >> 11
////Gray = (R * 1224 + G * 2405 + B * 467) >> 12
////Gray = (R * 2449 + G * 4809 + B * 934) >> 13
////Gray = (R * 4898 + G * 9618 + B * 1868) >> 14
////Gray = (R * 9797 + G * 19235 + B * 3736) >> 15
////Gray = (R * 19595 + G * 38469 + B * 7472) >> 16
////Gray = (R * 39190 + G * 76939 + B * 14943) >> 17
////Gray = (R * 78381 + G * 153878 + B * 29885) >> 18
////Gray = (R * 156762 + G * 307757 + B * 59769) >> 19
////Gray = (R * 313524 + G * 615514 + B * 119538) >> 2
//
//#define RED 0.2989
//#define GREEN 0.587
//#define BLUE 0.114
//#define RED_SCALE 4898
//#define GREEN_SCALE 9618
//#define BLUE_SCALE 1868
//
//CVLIB_NAMESPACE_BEGIN
//
//void RGB2GRAY(unsigned char* src, const int height, const int width, unsigned char* dst)
//{
//	for (int i = 0; i < height*width; i++, src += 3)
//	{
//		int gray = src[0] * BLUE + src[1] * GREEN + src[2] * RED;
//		dst[i] = gray > 255 ? 255 : gray;
//	}
//}
//
//void RGB2GRAY_SCALE(unsigned char* src, const int height, const int width, unsigned char* dst)
//{
//	for (int i = 0; i < height*width; i++, src += 3)
//	{
//		int gray = (src[0] * BLUE_SCALE + src[1] * GREEN_SCALE + src[2] * RED_SCALE) >> 14;
//		dst[i] = gray > 255 ? 255 : gray;
//	}
//}
//
//void RGB2GRAY_TBB(unsigned char* src, const int height, const int width, unsigned char* dst)
//{
//	using namespace tbb;
//	//parallel_for(blocked_range<int>(0, height*width),
//	//	[=]( blocked_range<int> &r) 
//	//{
//	//	unsigned char* localsrc=src+ r.begin() * 3;
//	//	unsigned char* localdst = dst+ r.begin();
//	//	for (auto iter = r.begin(); iter != r.end(); iter++, localsrc +=3, localdst++)
//	//	{
//	//		int gray = (localsrc[0] * BLUE_SCALE + localsrc[1] * GREEN_SCALE + localsrc[2] * RED_SCALE) >> 14;
//	//		localdst[0] = gray > 255 ? 255 : gray;
//	//	}
//	//}
//	//);
//
//	parallel_for(blocked_range2d<int, int>(0, height, 0, width),
//		[=](const blocked_range2d<int, int>& r) {
//		int StartX = r.cols().begin();
//		int StopX = r.cols().end();
//		int StartY = r.rows().begin();
//		int StopY = r.rows().end();
//
//		for (int y = StartY; y != StopY; y++)
//		{
//			unsigned char *LocalSourceImagePtr = (unsigned char *)src + (StartX + y * width) * 3;
//			unsigned char *LocalYImagePtr = (unsigned char *)dst + (StartX + y * width);
//			for (int x = StartX; x != StopX; x++)
//			{
//				int gray = (LocalSourceImagePtr[0] * BLUE_SCALE) +
//					(LocalSourceImagePtr[1] * GREEN_SCALE) +
//					(LocalSourceImagePtr[2] * RED_SCALE) >> 14;
//				LocalSourceImagePtr += 3;
//				LocalYImagePtr[0] = gray > 255 ? 255 : gray;
//				LocalYImagePtr++;
//			}
//		}
//	}
//	);
//}
//
//void RGB2GRAY_MTHREAD(unsigned char* src, const int height, const int width, unsigned char* dst)
//{
//#define THREAD_NUM 4
//
//}
//void RGB2GRAY_SIMD(unsigned char* src, const int height, const int width, unsigned char* dst)
//{
//	//sse 一次处理12个
//	const int B_WT = int(0.114 * 256 + 0.5);
//	const int G_WT = int(0.587 * 256 + 0.5);
//	const int R_WT = 256 - B_WT - G_WT; // int(0.299 * 256 + 0.5)
//
//	for (int Y = 0; Y < height; Y++)
//	{
//		unsigned char *LinePS = src + Y * width * 3;
//		unsigned char *LinePD = dst + Y * width;
//		int X = 0;
//		for (; X < width - 12; X += 12, LinePS += 36)
//		{
//			__m128i p1aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 0))),
//				_mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT)); //1
//			__m128i p2aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 1))),
//				_mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT)); //2
//			__m128i p3aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 2))),
//				_mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT)); //3
//
//			__m128i p1aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 8))),
//				_mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));//4
//			__m128i p2aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 9))),
//				_mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));//5
//			__m128i p3aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 10))),
//				_mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));//6
//
//			__m128i p1bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 18))),
//				_mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));//7
//			__m128i p2bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 19))),
//				_mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));//8
//			__m128i p3bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 20))),
//				_mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));//9
//
//			__m128i p1bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 26))),
//				_mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));//10
//			__m128i p2bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 27))),
//				_mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));//11
//			__m128i p3bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 28))),
//				_mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));//12
//
//			__m128i sumaL = _mm_add_epi16(p3aL, _mm_add_epi16(p1aL, p2aL));//13
//			__m128i sumaH = _mm_add_epi16(p3aH, _mm_add_epi16(p1aH, p2aH));//14
//			__m128i sumbL = _mm_add_epi16(p3bL, _mm_add_epi16(p1bL, p2bL));//15
//			__m128i sumbH = _mm_add_epi16(p3bH, _mm_add_epi16(p1bH, p2bH));//16
//			__m128i sclaL = _mm_srli_epi16(sumaL, 8);//17
//			__m128i sclaH = _mm_srli_epi16(sumaH, 8);//18
//			__m128i sclbL = _mm_srli_epi16(sumbL, 8);//19
//			__m128i sclbH = _mm_srli_epi16(sumbH, 8);//20
//			__m128i shftaL = _mm_shuffle_epi8(sclaL, _mm_setr_epi8(0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));//21
//			__m128i shftaH = _mm_shuffle_epi8(sclaH, _mm_setr_epi8(-1, -1, -1, 18, 24, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));//22
//			__m128i shftbL = _mm_shuffle_epi8(sclbL, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 6, 12, -1, -1, -1, -1, -1, -1, -1));//23
//			__m128i shftbH = _mm_shuffle_epi8(sclbH, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 24, 30, -1, -1, -1, -1));//24
//			__m128i accumL = _mm_or_si128(shftaL, shftbL);//25
//			__m128i accumH = _mm_or_si128(shftaH, shftbH);//26
//			__m128i h3 = _mm_or_si128(accumL, accumH);//27
//			_mm_storeu_si128((__m128i *)(LinePD + X), h3);
//		}
//		for (; X < width; X++, LinePS += 3)
//		{
//			LinePD[X] = (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8;
//		}
//	}
//}
//
//CUDA_NAMESPACE_BEGIN
//void RGB2GRAY(unsigned char* src, const int height, const int width, unsigned char* dst)
//{
//	unsigned char* devsrc, *devdst;
//	int dstsize = sizeof(unsigned char) *height* width;
//	int srcsize = dstsize * 3;
//	Time obj;
//	obj.start();
//	cudaMalloc(&devsrc, srcsize);
//	cudaMalloc(&devdst, dstsize);
//	obj.end();
//	obj.start();
//	cudaMemcpy(devsrc, src, srcsize, cudaMemcpyHostToDevice);
//	cudaMemcpy(devdst, dst, dstsize, cudaMemcpyHostToDevice);
//	obj.end();
//	obj.start();
//	RGB2GRAY_GPU(devsrc, height, width, devdst);
//	obj.end();
//	obj.start();
//	cudaMemcpy(dst, devdst, dstsize, cudaMemcpyDeviceToHost);
//	cudaFree(devsrc);
//	cudaFree(devdst);
//	obj.end();
//
//	//zero copy,竟然很慢
//	//unsigned char* devsrc, *devdst;
//	//int dstsize = sizeof(unsigned char)* height*width;
//	//int srcsize = dstsize * 3;
//	//cudaMalloc(&devdst, dstsize);
//	//cudaError er;
//	//er = cudaHostRegister(src, srcsize, cudaHostRegisterMapped);
//	//er = cudaHostGetDevicePointer((void **)&devsrc, src, 0);
//	//RGB2GRAY_GPU(devsrc, height, width, devdst);
//	//er= cudaHostUnregister(src);
//	//cudaMemcpy(dst, devdst, dstsize, cudaMemcpyDeviceToHost);
//	//cudaFree(devdst);
//}
//CUDA_NAMESPACE_END
//
//CVLIB_NAMESPACE_END
//#endif
