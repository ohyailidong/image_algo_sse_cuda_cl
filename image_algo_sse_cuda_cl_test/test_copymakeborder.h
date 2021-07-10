#pragma once
#include "copymakeborder.h"
#include "morphology.h"
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

class TEST_MORPHOLOGY
{
public:
	static void Run()
	{
		SetSetConsoleTextColor(GREEN);
		std::cout << "****************************TEST_MORPHOLOGY:**************************** \n ";
		SetSetConsoleTextColor(DEFAULT);

		Time time;
		cv::Mat image = cv::imread("../image/bilateral.png", 1);
		DescriptionImage(image);
		cv::Mat cvEle = cv::getStructuringElement(2, cv::Size(11, 11));
		cv::Mat matCpuEle(cvEle.size(), cvEle.type());
		Image cpuEle(matCpuEle.cols, matCpuEle.rows, 1, matCpuEle.data);
		cvlib::getStructuringElement(2, cpuEle);
		int channel = image.channels();
		Image srcImage(image.cols, image.rows, channel, image.data);

		for (int i = 0; i < MORPH_HITMISS; ++i) {
			cv::Mat cvdst;
			time.start(platform_cv_time);
			cv::morphologyEx(image, cvdst, i, cvEle);
			time.end();
			cv::Mat matCpuDst(cvdst.size(), cvdst.type());

			Image cpuDst(matCpuDst.cols, matCpuDst.rows, channel, matCpuDst.data);
			time.start(platform_cpu_time);
			cvlib::morphologyEx(srcImage, cpuDst, i, cpuEle, 1, 0);
			time.end();
			Check(matCpuDst, cvdst, platform_cpu);
		}


		//cv::Mat matGpuDst(cvdst.size(), cvdst.type());

		//Image gpuDst(matGpuDst.cols, matGpuDst.rows, channel, matGpuDst.data);
		//time.start(platform_gpu_time);
		//cvlib::cuda::dilate(&srcImage, &gpuDst, ksize / 2, ksize / 2, ksize / 2, ksize / 2, BORDER_REPLICATE | BORDER_ISOLATED);
		//time.end();

		//Check(matGpuDst, border, platform_gpu);
	}
};


