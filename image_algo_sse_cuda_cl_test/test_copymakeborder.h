#pragma once
#include "copymakeborder.h"
#include "common_data_define.h"
#include "test_common_define.h"

class TEST_COPYMAKEBORDER
{
public:
	static void Run()
	{
		std::cout << "TEST_COPYMAKEBORDER: \n ";
		cv::Mat image = cv::imread("../image/bilateral.png", 1);
		std::cout << "height= " << image.rows << "  ,width= " << image.cols << "  ,channel= " << image.channels() << "\n";
		int ksize = 33;
		cv::Mat border;//×óÓÒ±ß½çÀ©³ä£¬ÉÏÏÂ²»À©³ä
		cv::copyMakeBorder(image, border, ksize / 2, ksize / 2, ksize / 2, ksize / 2, BORDER_REPLICATE | BORDER_ISOLATED);
		cv::Mat myborder(border.size(), border.type());

		int channel = image.channels();
		Image srcImage(image.cols, image.rows, channel, image.data);
		Image borderImage(myborder.cols, myborder.rows, channel, myborder.data);
		cvlib::copyMakeborder(&srcImage, &borderImage, ksize / 2, ksize / 2, ksize / 2, ksize / 2, BORDER_REPLICATE | BORDER_ISOLATED);

		Check(myborder ,border);
		cvlib::cuda::copyMakeborder(&srcImage, &borderImage, ksize / 2, ksize / 2, ksize / 2, ksize / 2, BORDER_REPLICATE | BORDER_ISOLATED);

		Check(myborder, border);

	}
};


