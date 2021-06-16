//#include "morphology.h"
//
//CVLIB_NAMESPACE_BEGIN
//void morphologyEx(Image& src, Image& dst, int op, Image& kernel,
//	int iter, int bordertype, std::vector<int> bordervalue)
//{
//	if (kernel.GetElementNum() == 0) assert(false);
//	Image temp;
//	switch (op)
//	{
//	case MORPH_ERODE:
//		erode(src, dst, kernel, iter, bordertype, bordervalue);
//		break;
//	case MORPH_DILATE:
//		dilate(src, dst, kernel, iter, bordertype, bordervalue);
//		break;
//	case MORPH_OPEN:
//		erode(src, temp, kernel, iter, bordertype, bordervalue);
//		dilate(temp, dst, kernel, iter, bordertype, bordervalue);
//		break;
//	case MORPH_CLOSE:
//		dilate(src, temp, kernel, iter, bordertype, bordervalue);
//		erode(temp, dst, kernel, iter, bordertype, bordervalue);
//		break;
//	case MORPH_GRADIENT:		   
//		erode(src, temp, kernel, iter, bordertype, bordervalue);
//		dilate(src, dst, kernel, iter, bordertype, bordervalue);
//		//pudi::tensilica::Sub(dst, dst, temp);
//		break;
//	case MORPH_TOPHAT:
//		erode(src, temp, kernel, iter, bordertype, bordervalue);
//		dilate(temp, dst, kernel, iter, bordertype, bordervalue);
//		//pudi::tensilica::Sub(dst, src, dst);
//		break;
//	case MORPH_BLACKHAT:
//		dilate(src, temp, kernel, iter, bordertype, bordervalue);
//		erode(temp, dst, kernel, iter, bordertype, bordervalue);
//		//pudi::tensilica::Sub(dst, dst, src);
//		break;
//	case MORPH_HITMISS:
//		break;
//	default:
//		break;
//	}
//}
//
//void erode(Image& src, Image& dst, Image& kernel, int iterations, int borderType, std::vector<int> scalar)
//{
//	if (src.GetElementNum() != dst.GetElementNum())
//		assert(false);
//	int kerWidth = kernel.width, kerHeight = kernel.height;
//	if (kernel.GetElementNum() == 0)
//	{
//		kernel = getStructuringElement(MORPH_RECT, { 1 + iterations * 2,1 + iterations * 2 });
//		anchor = Point(iterations, iterations);
//		iterations = 1;
//	}
//	//如果核是矩形，将核变为迭代的倍数
//	else if (iterations > 1 && countNonZero(kernel) == kerWidth * kerHeight)
//	{
//		anchor = Point(anchor.x*iterations, anchor.y*iterations);
//		kernel = getStructuringElement(MORPH_RECT,
//			{ kerWidth + (iterations - 1)*(kerWidth - 1),	kerHeight + (iterations - 1)*(kerHeight - 1) });
//		iterations = 1;
//	}
//	//边界扩充
//	pudi::Tensor srcborder;
//	copyMakeBorder(src, srcborder, anchor.y, kerHeight - anchor.y - 1, anchor.x, kerWidth - anchor.x - 1,
//		(cvlib::BorderTypes)borderType, { 255,255,255,255 });
//	cv::Mat bb = Tensor2Mat(srcborder);
//	morphApply(srcborder, dst, kernel, anchor, iterations, Min);
//}
//
//void dilate(Image& src, Image& dst, Image& kernel, int iterations, int borderType, std::vector<int> scalar)
//{
//	if (src.GetElementNum() != dst.GetElementNum())
//		dst = pudi::Zeros(src.GetDataType(), { src.GetDim(0),src.GetDim(1),src.GetDim(2) });
//	int kerWidth = kernel.GetDim(1), kerHeight = kernel.GetDim(2);
//	if (kernel.GetElementNum() == 0)
//	{
//		kernel = getStructuringElement(MORPH_RECT, { 1 + iterations * 2,1 + iterations * 2 });
//		anchor = Point(iterations, iterations);
//		iterations = 1;
//	}
//	//如果核是矩形，将核变为迭代的倍数
//	else if (iterations > 1 && countNonZero(kernel) == kerWidth * kerHeight)
//	{
//		anchor = Point(anchor.x*iterations, anchor.y*iterations);
//		kernel = getStructuringElement(MORPH_RECT,
//			{ kerWidth + (iterations - 1)*(kerWidth - 1),	kerHeight + (iterations - 1)*(kerHeight - 1) });
//		iterations = 1;
//	}
//	//边界扩充
//	pudi::Tensor srcborder;
//	copyMakeBorder(src, srcborder, anchor.y, kerHeight - anchor.y - 1, anchor.x, kerWidth - anchor.x - 1,
//		(cvlib::BorderTypes)borderType, { 0,0,0,0 });
//	morphApply(srcborder, dst, kernel, anchor, iterations, Max);
//
//}
//
//void preprocess2DKernel(Image& kernel, std::vector<Point>&coords)
//{
//	assert(kernel.channel == 1);
//	int kerwidth = kernel.width, kerheight = kernel.height;
//	int i, j;
//	uchar* kerPtr = (uchar*)kernel.data;
//	for (i = 0; i < kerheight; i++)
//	{
//		uchar* kerPtrLine = kerPtr + kerwidth * i;
//		for (j = 0; j < kerwidth; j++)
//		{
//			if (kerPtrLine[j] == 0)
//				continue;
//			coords.push_back(Point(j, i));
//		}
//	}
//}
//void morphApply(Image& src, Image& dst, Image& kernel, int iterations, void*pfun)
//{
//	std::vector<Point> coords;
//	preprocess2DKernel(kernel, coords);
//	typedef int(*opobj)(int, int);
//	opobj op = (opobj)pfun;
//	int i, j, k, nz = (int)coords.size();
//	std::vector<uchar*>ptrs(nz, nullptr);
//	//src is after makeborder
//	uchar* srcPtr = (uchar*)src.data;
//	uchar* dstPtr = (uchar*)dst.data;
//	int cn = dst.channel, dstwidth = dst.width, dstheight = dst.height;
//	dstwidth = cn * dstwidth;
//	int srcwidth = cn * src.width;
//	for (i = 0; i < dstheight; i++)
//	{
//		uchar* sLine = srcPtr + srcwidth * i;
//		uchar* dLine = dstPtr + dstwidth * i;
//		for (k = 0; k < nz; k++)
//			ptrs[k] = sLine + coords[k].y*srcwidth + coords[k].x*cn;
//		for (j = 0; j < dstwidth; j += 4)
//		{
//			const uchar*sptr = ptrs[0] + j;
//			int s0 = sptr[0], s1 = sptr[1], s2 = sptr[2], s3 = sptr[3];
//			for (k = 1; k < nz; k++)
//			{
//				sptr = ptrs[k] + j;
//				s0 = op(s0, sptr[0]); s1 = op(s1, sptr[1]);
//				s2 = op(s2, sptr[2]); s3 = op(s3, sptr[3]);
//			}
//			dLine[j] = s0; dLine[j + 1] = s1;
//			dLine[j + 2] = s2; dLine[j + 3] = s3;
//		}
//		for (; j < dstwidth; j++)
//		{
//			int s0 = *(ptrs[0] + j);
//			for (k = 1; k < nz; k++)
//				s0 = op(s0, *(ptrs[k] + j));
//			dLine[j] = s0;
//		}
//	}
//}
//
////size[0]width/col ,size[1] height/row
//void getStructuringElement(int shape, Image& dstEle)
//{
//	int eleWidth = dstEle.width, eleHeight = dstEle.height;
//	assert(eleWidth> 0 && eleHeight> 0);
//	int i, j;
//	int r = 0, c = 0;
//	double inv_r2 = 0;
//	assert(shape == MORPH_RECT || shape == MORPH_CROSS || shape == MORPH_ELLIPSE);
//	if (eleWidth == 1 && eleHeight == 1)
//		shape = MORPH_RECT;
//	if (shape == MORPH_ELLIPSE)
//	{
//		r = eleHeight / 2;
//		c = eleWidth / 2;
//		inv_r2 = r ? 1. / ((double)r*r) : 0;
//	}
//	for (i = 0; i < eleHeight; i++)
//	{
//		unsigned char* ptr = (unsigned char*)dstEle.data + i * eleWidth;
//		int j1 = 0, j2 = 0;
//
//		if (shape == MORPH_RECT || (shape == MORPH_CROSS && i == eleHeight / 2))
//			j2 = eleWidth;
//		else if (shape == MORPH_CROSS)
//			j1 = eleWidth / 2, j2 = j1 + 1;
//		else
//		{
//			int dy = i - r;
//			if (std::abs(dy) <= r)
//			{
//				int dx = int(c*std::sqrt((r*r - dy * dy)*inv_r2) + 0.5);
//				j1 = std::max(c - dx, 0);
//				j2 = std::min(c + dx + 1, eleWidth);
//			}
//		}
//
//		for (j = 0; j < j1; j++)
//			ptr[j] = 0;
//		for (; j < j2; j++)
//			ptr[j] = 1;
//		for (; j < eleWidth; j++)
//			ptr[j] = 0;
//	}
//}
//CVLIB_NAMESPACE_END