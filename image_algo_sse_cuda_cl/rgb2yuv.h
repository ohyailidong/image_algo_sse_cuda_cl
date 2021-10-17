#ifndef RGB_2_YUV_H
#define RGB_2_YUV_H
//#include "common_define.h"
//namespace cvLib {
//
//EXPORT_IMAGE_ALGO_DLL void RGB2YUV(unsigned char *RGB, unsigned char *dst, int Width, int Height, int channel);
//
//}

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <algorithm>
#include <iostream>
#include "E:\opencv_4.5_sln\Source_Code\opencv-4.5.0\modules\imgproc\include\opencv2\imgproc\types_c.h"

/////////// YUV_NV12 -> BGR ////////////////////////////
//R = 1.164(Y - 16) + 1.596(V - 128)
//G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
//B = 1.164(Y - 16)                  + 2.018(U - 128)

//R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
//G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
//B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20

struct Image
{
	int width;
	int height;
	int channel;
	void* data;
	Image() :width(0), height(0), channel(0), data(nullptr) {}
	Image(int iwidth, int iheight, int ichannel, void* p) :width(iwidth),
		height(iheight),
		channel(ichannel),
		data(p) {}
	~Image() {}
	int GetElementNum() { return width * height* channel; }
};

static inline uchar clamp(int v)
{
	return (uchar)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0);
}

void YUV420ToNV12(cv::Mat &yuv)
{
	int height = yuv.rows;
	int width = yuv.cols;
	int src_height = height / 1.5;
	int uv_height = height - src_height;
	unsigned char *yuv_data = yuv.data;
	unsigned char *uv_data = (unsigned char*)malloc(uv_height * width);
	unsigned char *uv_data_begin = uv_data;
	unsigned char *u_data = yuv_data + src_height * width;
	int usize = uv_height / 2 * width;
	unsigned char *v_data = u_data + usize;

	for (int i = 0; i < usize; ++i)
	{
		uv_data[0] = u_data[0];
		uv_data[1] = v_data[0];
		u_data++, v_data++, uv_data += 2;
	}
	memcpy(yuv_data + src_height * width, uv_data_begin, usize * 2);
	free(uv_data_begin);
}

static const int ITUR_BT_601_SHIFT = 20;
static const int ITUR_BT_601_CRY = 269484;
static const int ITUR_BT_601_CGY = 528482;
static const int ITUR_BT_601_CBY = 102760;
static const int ITUR_BT_601_CRU = -155188;
static const int ITUR_BT_601_CGU = -305135;
static const int ITUR_BT_601_CBU = 460324;
static const int ITUR_BT_601_CGV = -385875;
static const int ITUR_BT_601_CBV = -74448;

static const int ITUR_BT_601_CY = 1220542;
static const int ITUR_BT_601_CUB = 2116026;
static const int ITUR_BT_601_CUG = -409993;
static const int ITUR_BT_601_CVG = -852492;
static const int ITUR_BT_601_CVR = 1673527;

static inline uchar rgbToY42x(uchar r, uchar g, uchar b)
{
	const int shifted16 = (16 << ITUR_BT_601_SHIFT);
	const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
	int yy = ITUR_BT_601_CRY * r + ITUR_BT_601_CGY * g + ITUR_BT_601_CBY * b + halfShift + shifted16;

	return clamp(yy >> ITUR_BT_601_SHIFT);
}

static inline void rgbToUV42x(uchar r, uchar g, uchar b, uchar& u, uchar& v)
{
	const int halfShift = (1 << (ITUR_BT_601_SHIFT - 1));
	const int shifted128 = (128 << ITUR_BT_601_SHIFT);
	int uu = ITUR_BT_601_CRU * r + ITUR_BT_601_CGU * g + ITUR_BT_601_CBU * b + halfShift + shifted128;
	int vv = ITUR_BT_601_CBU * r + ITUR_BT_601_CGV * g + ITUR_BT_601_CBV * b + halfShift + shifted128;

	u = clamp(uu >> ITUR_BT_601_SHIFT);
	v = clamp(vv >> ITUR_BT_601_SHIFT);
}

static inline void uvToRGBuv(const uchar u, const uchar v, int *ruv, int *guv, int *buv)
{
	int uu, vv;
	uu = int(u) - 128;
	vv = int(v) - 128;

	*ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * vv;
	*guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * vv + ITUR_BT_601_CUG * uu;
	*buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * uu;
}

static inline void yRGBuvToRGBA(const uchar vy, const int ruv, const int guv, const int buv,
	uchar* r, uchar* g, uchar* b)
{
	int yy = int(vy);
	int y = std::max(0, yy - 16) * ITUR_BT_601_CY;
	*r = clamp((y + ruv) >> ITUR_BT_601_SHIFT);
	*g = clamp((y + guv) >> ITUR_BT_601_SHIFT);
	*b = clamp((y + buv) >> ITUR_BT_601_SHIFT);
}

void BGR8ToNV12(Image *bgr, Image *yuv)
{
	int bgr_height = bgr->height;
	int width = bgr->width;
	int cn = bgr->channel;
	int bgr_step = cn * width;
	unsigned char *bgr_data = (unsigned char*)bgr->data;

	int yuv_height = yuv->height;
	unsigned char *y_data = (unsigned char*)yuv->data;
	unsigned char *uv_data = y_data + bgr_height * width;
	int half_width = width / 2;
	for (int i = 0; i < bgr_height; i++)
	{
		int is_even_row = (i % 2) == 0;
		for (int j = 0; j < half_width; j++)
		{
			uchar b0, g0, r0;
			uchar b1, g1, r1;
			b0 = bgr_data[(2 * j + 0) * cn + 0];
			g0 = bgr_data[(2 * j + 0) * cn + 1];
			r0 = bgr_data[(2 * j + 0) * cn + 2];
			b1 = bgr_data[(2 * j + 1) * cn + 0];
			g1 = bgr_data[(2 * j + 1) * cn + 1];
			r1 = bgr_data[(2 * j + 1) * cn + 2];
			uchar y0 = rgbToY42x(r0, g0, b0);
			uchar y1 = rgbToY42x(r1, g1, b1);

			y_data[2 * j + 0] = y0;
			y_data[2 * j + 1] = y1;

			if (is_even_row)
			{
				uchar uu, vv;
				rgbToUV42x(r0, g0, b0, uu, vv);
				uv_data[2 * j + 0] = uu;
				uv_data[2 * j + 1] = vv;
			}
		}
		bgr_data += bgr_step;
		y_data += width;
		if (is_even_row)
		{
			uv_data += width;
		}
	}
}

void NV12ToBGR8(Image *yuv, Image *bgr)
{
	if (3 != bgr->channel)
	{
		return;
	}
	int bgr_width = bgr->width;
	int bgr_height = bgr->height;
	int bgr_step = bgr_width * 3;
	int yuv_width = yuv->width;
	unsigned char *bgr_data = (unsigned char*)bgr->data;
	unsigned char *y_data = (unsigned char*)yuv->data;
	unsigned char *uv_data = y_data + bgr_height * bgr_width;

	unsigned char *bgr_row1 = bgr_data;
	unsigned char *bgr_row2 = bgr_row1 + bgr_step;
	unsigned char *y_row1 = y_data;
	unsigned char *y_row2 = y_row1 + yuv_width;
	for (int i = 0; i < bgr_height; i += 2)
	{
		for (int j = 0; j < bgr_width; j += 2)
		{
			uchar u = uv_data[j];
			uchar v = uv_data[j + 1];
			uchar y11 = y_row1[j];
			uchar y12 = y_row1[j + 1];
			uchar y21 = y_row2[j];
			uchar y22 = y_row2[j + 1];

			int ruv, guv, buv;
			uvToRGBuv(u, v, &ruv, &guv, &buv);
			uchar r, g, b;
			yRGBuvToRGBA(y11, ruv, guv, buv, &r, &g, &b);
			bgr_row1[j * 3] = b;
			bgr_row1[j * 3 + 1] = g;
			bgr_row1[j * 3 + 2] = r;
			yRGBuvToRGBA(y12, ruv, guv, buv, &r, &g, &b);
			bgr_row1[j * 3 + 3] = b;
			bgr_row1[j * 3 + 4] = g;
			bgr_row1[j * 3 + 5] = r;

			yRGBuvToRGBA(y21, ruv, guv, buv, &r, &g, &b);
			bgr_row2[j * 3] = b;
			bgr_row2[j * 3 + 1] = g;
			bgr_row2[j * 3 + 2] = r;
			yRGBuvToRGBA(y22, ruv, guv, buv, &r, &g, &b);
			bgr_row2[j * 3 + 3] = b;
			bgr_row2[j * 3 + 4] = g;
			bgr_row2[j * 3 + 5] = r;
		}
		uv_data += yuv_width;
		y_row1 += yuv_width * 2;
		y_row2 += yuv_width * 2;
		bgr_row1 += bgr_step * 2;
		bgr_row2 += bgr_step * 2;
	}
}

int main()
{
	// opencv res
	cv::Mat image = cv::imread("E:/image_algo_sse_cuda_cl/image/lena.jpg", 1);
	cv::resize(image, image, cv::Size(4800, 2600));
	cv::Mat mat_yuv, mat_yuv_bgr;
	cvtColor(image, mat_yuv, CV_BGR2YUV_I420);
	YUV420ToNV12(mat_yuv);
	cv::cvtColor(mat_yuv, mat_yuv_bgr, CV_YUV2BGR_NV12);
	// our res
	Image src_bgr(image.cols, image.rows, 3, image.data);
	int brg_uv_height = image.rows * 1.5;
	unsigned char *bgr_nv_data = (unsigned char*)malloc(brg_uv_height * image.cols);
	Image bgr_nv(image.cols, brg_uv_height, 1, bgr_nv_data);
	BGR8ToNV12(&src_bgr, &bgr_nv);
	unsigned char *nv_to_bgr_data = (unsigned char*)malloc(image.rows * image.cols * image.channels());
	Image nv_bgr(image.cols, image.rows, 3, nv_to_bgr_data);
	NV12ToBGR8(&bgr_nv, &nv_bgr);
	//show res
	cv::Mat my_bgr_nv(bgr_nv.height, bgr_nv.width, CV_8UC1, bgr_nv.data);
	cv::Mat my_nv_bgr(nv_bgr.height, nv_bgr.width, CV_8UC3, nv_bgr.data);
	//check error
	cv::Mat error = cv::abs(mat_yuv_bgr - my_nv_bgr);
	error.convertTo(error, CV_32FC1);
	free(bgr_nv_data);
	free(nv_to_bgr_data);
	return 0;
}

#endif
