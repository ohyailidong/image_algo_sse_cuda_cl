#include "matchtemplate.h"

#include <opencv2/world.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

CVLIB_NAMESPACE_BEGIN
void matchTemplate(Image& src, Image& matchtemplate, Image& dst,
	int method, Image _mask)
{
	assert(src.channel == matchtemplate.channel);
	int srcclos = src.width, templcols = matchtemplate.width;
	int srcrows = src.height, templrows = matchtemplate.height;
	unsigned int corrcols = srcclos - templcols + 1;
	unsigned int corrrows = srcrows - templrows + 1;
	crossCorr(src, matchtemplate, dst, 0);//相关匹配法
	common_matchTemplate(src, matchtemplate, dst, method, src.channel);
}

void crossCorr(const Image& img, const Image& templ, Image& corr, int borderType)
{
	//根据相关公式计算
	int srccn = img.channel, templcn = templ.channel;
	int srccols = img.width, templcols = templ.width, templrows = templ.height;
	int cols = corr.width, rows = corr.height;
	unsigned char* imgptr = (unsigned char*)img.data;
	unsigned char* templptr = (unsigned char*)templ.data;
	float* corrptr = (float*)corr.data;
	for (size_t i = 0; i < rows; i++)
	{
		unsigned char* imglineptr = imgptr + i * srccn*srccols;
		unsigned char* templlineptr = templptr;
		float* corrlineptr = corrptr + i * cols;
		for (size_t j = 0; j < cols; j++)
		{
			float sum = 0;
			for (int templrow = 0; templrow < templrows; templrow++)
			{
				unsigned char* ptr1 = imglineptr + j * srccn + templrow * srccn*srccols;
				unsigned char* ptr2 = templlineptr + templrow * templcn*templcols;
				for (int templcol = 0; templcol < templcols*templcn; templcol++)
					sum += ptr1[templcol] * ptr2[templcol];
			}
			corrlineptr[j] = sum;
		}
	}
}

void common_matchTemplate(Image& img, Image& templ, Image& result, int method, int cn)
{
	if (method == TM_CCORR)
		return;
	int numType = method == TM_CCORR || method == TM_CCORR_NORMED ? 0 :
		method == TM_CCOEFF || method == TM_CCOEFF_NORMED ? 1 : 2;
	bool isNormed = method == TM_CCORR_NORMED ||
		method == TM_SQDIFF_NORMED ||
		method == TM_CCOEFF_NORMED;

	float invArea = 1. / ((float)templ.height * templ.width);
	int imgChannel = img.channel, imgCols = img.width, imgRows = img.height;
	float *pSum = new float[imgChannel*(imgCols + 1)*(imgRows + 1)];
	Image sum(imgCols + 1, imgRows + 1, imgChannel, pSum);
	float *pSqSum = new float[imgChannel*(imgCols + 1)*(imgRows + 1)];
	Image sqsum(imgCols + 1, imgRows + 1, imgChannel, pSqSum);
	std::vector<float> templMean(4, 0), templSdv(4, 0);
	float*q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;
	float templNorm = 0, templSum2 = 0;

	if (method == TM_CCOEFF)
	{
		integral(img, sum);
		//templMean = mean(templ);
		//meanStdDev(templ, templMean, templSdv, stdDevMask);
		cv::Scalar cvtemplMean, cvtemplSdv;
		cv::Mat cvTempl(templ.height, templ.width, CV_8UC3, templ.data);
		cv::meanStdDev(cvTempl, cvtemplMean, cvtemplSdv);
		for (size_t i = 0; i < 4; i++)
		{
			templMean[i] = ((double*)&cvtemplMean)[i];
			templSdv[i] = ((double*)&cvtemplSdv)[i];
		}
	}
	else
	{
		integral(img, sum, sqsum);//积分图计算像素的和与平方和
		//meanStdDev(templ, templMean, templSdv, stdDevMask);
		cv::Scalar cvtemplMean, cvtemplSdv;
		cv::Mat cvTempl(templ.height, templ.width, CV_8UC3, templ.data);
		cv::meanStdDev(cvTempl, cvtemplMean, cvtemplSdv);
		for (size_t i = 0; i < 4; i++)
		{
			templMean[i] = ((double*)&cvtemplMean)[i];
			templSdv[i] = ((double*)&cvtemplSdv)[i];
		}
		//templNorm代表整个图像的方差
		templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];
		if (templNorm < DBL_EPSILON && method == TM_CCOEFF_NORMED)
			return;
		//templSum2代表所有像素平方和*invArea
		templSum2 = templNorm + templMean[0] * templMean[0] + templMean[1] * templMean[1] +
			templMean[2] * templMean[2] + templMean[3] * templMean[3];
		if (numType != 1)
		{
			templMean = std::vector<float>(4, 0);
			templNorm = templSum2;
		}
		templSum2 /= invArea;
		templNorm = std::sqrt(templNorm);
		templNorm /= std::sqrt(invArea); // care of accuracy here

		q0 = (float*)sqsum.data;
		q1 = q0 + templ.width*cn;
		q2 = (float*)((float*)sqsum.data + templ.height*sqsum.channel*sqsum.width);
		q3 = q2 + templ.width*cn;
	}

	float* p0 = (float*)sum.data;
	float* p1 = p0 + templ.width*cn;
	float* p2 = (float*)((float*)sum.data + templ.height*sum.channel*sum.width);
	float* p3 = p2 + templ.width*cn;

	int sumstep = sum.channel*sum.width;
	int sqstep = sqsum.channel*sqsum.width;
	int resultstep = result.channel*result.width;
	int i, j, k;

	for (i = 0; i < result.height; i++)
	{
		float* rrow = (float*)result.data + i * resultstep;
		int idx = i * sumstep;
		int idx2 = i * sqstep;

		for (j = 0; j < result.width; j++, idx += cn, idx2 += cn)
		{
			float num = rrow[j], t;
			float wndMean2 = 0, wndSum2 = 0;

			if (numType == 1)
			{
				for (k = 0; k < cn; k++)
				{
					t = p0[idx + k] - p1[idx + k] - p2[idx + k] + p3[idx + k];
					wndMean2 += t * t;
					num -= t * templMean[k];
				}

				wndMean2 *= invArea;
			}

			if (isNormed || numType == 2)
			{
				for (k = 0; k < cn; k++)
				{
					t = q0[idx2 + k] - q1[idx2 + k] - q2[idx2 + k] + q3[idx2 + k];
					wndSum2 += t;
				}

				if (numType == 2)
				{
					num = wndSum2 - 2 * num + templSum2;
					num = MAX(num, 0.);
				}
			}

			if (isNormed)
			{
				float diff2 = MAX(wndSum2 - wndMean2, 0);
				if (diff2 <= std::min(0.5f, 10 * FLT_EPSILON * wndSum2))
					t = 0; // avoid rounding errors
				else
					t = std::sqrt(diff2)*templNorm;

				if (fabs(num) < t)
					num /= t;
				else if (fabs(num) < t*1.125)
					num = num > 0 ? 1 : -1;
				else
					num = method != TM_SQDIFF_NORMED ? 0 : 1;
			}

			rrow[j] = (float)num;
		}
	}
	delete[]pSum;
	delete[]pSqSum;
}

void integral(Image img, Image sum, Image sqsum)
{
	int cn = img.channel, width = img.width, height = img.height;
	int imgstep, sumstep, sqsumstep;
	imgstep = cn * width;
	sumstep = sum.width *sum.channel;
	bool sqsumEmpty = sqsum.width == 0;
	if (!sqsumEmpty)
		sqsumstep = sqsum.width*sqsum.channel;
	width = img.width*img.channel;
	unsigned char* imgPtr = (unsigned char*)img.data;
	float* sumPtr = (float*)sum.data;
	float* sqsumPtr = (float*)sqsum.data;

	sumPtr += sumstep + cn;
	if (!sqsumEmpty)
		sqsumPtr += sqsumstep + cn;

	int x, y, k;
	if (sqsumEmpty)//only calculate sum
	{
		for (y = 0; y < height; y++, imgPtr += imgstep - cn, sumPtr += sumstep - cn)
		{
			for (k = 0; k < cn; k++, imgPtr++, sumPtr++)
			{
				float s = sumPtr[-cn] = 0;
				for (x = 0; x < width; x += cn)
				{
					s += imgPtr[x];
					sumPtr[x] = s + sumPtr[x - sumstep];
				}
			}
		}
	}
	else//calculate sum and square sum
	{
		for (y = 0; y < height; y++, imgPtr += imgstep - cn,
			sumPtr += sumstep - cn, sqsumPtr += sqsumstep - cn)
		{
			for (k = 0; k < cn; k++, imgPtr++, sumPtr++, sqsumPtr++)
			{
				float s = sumPtr[-cn] = 0;
				float sq = sqsumPtr[-cn] = 0;
				for (x = 0; x < width; x += cn)
				{
					unsigned char it = imgPtr[x];
					s += it;
					sq += (float)it*it;
					float t = sumPtr[x - sumstep] + s;
					float temp = sqsumPtr[x - sqsumstep];
					float tq = sqsumPtr[x - sqsumstep] + sq;
					sumPtr[x] = t;
					sqsumPtr[x] = tq;
				}
			}
		}
	}
}
CVLIB_NAMESPACE_END