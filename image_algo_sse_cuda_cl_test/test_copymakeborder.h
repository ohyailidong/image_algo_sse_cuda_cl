#pragma once
#include "copymakeborder.h"
#include "common_data_define.h"
#include "test_common_define.h"

class TEST_COPYMAKEBORDER
{
public:
	static void Run()
	{
		SetSetConsoleTextColor(GREEN);
		std::cout << "****************************TEST_COPYMAKEBORDER:**************************** \n ";
		SetSetConsoleTextColor(DEFAULT);

		Time time;
		cv::Mat image = cv::imread("../image/bilateral.png", 1);
		DescriptionImage(image);
		int ksize = 33;
		cv::Mat border;
		time.start(platform_cv_time);
		cv::copyMakeBorder(image, border, ksize / 2, ksize / 2, ksize / 2, ksize / 2, BORDER_REPLICATE | BORDER_ISOLATED);
		time.end();
		cv::Mat matCpuDst(border.size(), border.type());
		cv::Mat matGpuDst(border.size(), border.type());

		int channel = image.channels();
		Image srcImage(image.cols, image.rows, channel, image.data);
		Image cpuDst(matCpuDst.cols, matCpuDst.rows, channel, matCpuDst.data);
		time.start(platform_cpu_time);
		cvlib::copyMakeborder(&srcImage, &cpuDst, ksize / 2, ksize / 2, ksize / 2, ksize / 2, BORDER_REPLICATE | BORDER_ISOLATED);
		time.end();

		Image gpuDst(matGpuDst.cols, matGpuDst.rows, channel, matGpuDst.data);
		time.start(platform_gpu_time);
		cvlib::cuda::copyMakeborder(&srcImage, &gpuDst, ksize / 2, ksize / 2, ksize / 2, ksize / 2, BORDER_REPLICATE | BORDER_ISOLATED);
		time.end();

		Check(matCpuDst, border, platform_cpu);
		Check(matGpuDst, border, platform_gpu);
	}
};


