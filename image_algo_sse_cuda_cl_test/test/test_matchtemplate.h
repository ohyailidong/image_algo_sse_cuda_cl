#pragma once
#include "test_common_define.h"
#include "matchtemplate.h"

class TEST_INTEGRAL 
{
public:
	static void Run()
	{
		SetSetConsoleTextColor(GREEN);
		std::cout << "****************************TEST_INTEGRAL:**************************** \n ";
		SetSetConsoleTextColor(DEFAULT);
		cv::Mat image = cv::imread("../image/lena.jpg", 1);
		std::cout << "height= " << image.rows << "  ,width= " << image.cols << "  ,channel= "
			<< image.channels() << "\n";
		Time time;
		cv::Mat cvdstSum, cvdstSq;
		time.start(platform_cv_time);
		cv::integral(image, cvdstSum, cvdstSq);
		time.end();
		cv::Mat mysum(cvdstSum.size(), cvdstSum.type());
		cv::Mat mysq(cvdstSq.size(), cvdstSq.type());
		Image src(image.cols, image.rows, image.channels(), image.data);
		Image imgsum(mysum.cols, mysum.rows, mysum.channels(), mysum.data);
		Image imgsq(mysq.cols, mysq.rows, mysq.channels(), mysq.data);
		time.start(platform_cpu_time);
		cvlib::integral(src, imgsum, imgsq);
		time.end();
		Check(mysum, cvdstSum, platform_cpu);
		Check(mysq, cvdstSq, platform_cpu);
	}
};
class TEST_MATCH_TEMPLATE
{
public:
	static void Run()
	{
		SetSetConsoleTextColor(GREEN);
		std::cout << "****************************TEST_MATCH_TEMPLATE:**************************** \n ";
		SetSetConsoleTextColor(DEFAULT);

		cv::Mat image = cv::imread("../image/lena.jpg", 1);
		std::cout << "height= " << image.rows << "  ,width= " << image.cols	<< "  ,channel= " 
			<< image.channels() << "\n";
		cv::Mat templ = cv::imread("../image/lena_template.png", 1);
		Time time;
		cv::Mat cvdst;
		for (int imethod = 0; imethod < 6; ++imethod) {
			time.start("opencv计算总时间");
			int METHOD = imethod;
			cv::matchTemplate(image, templ, cvdst, METHOD);
			time.end();
			cv::Mat cpudst(cvdst.size(), cvdst.type());
			Image src(image.cols, image.rows, image.channels(), image.data);
			Image Templ(templ.cols, templ.rows, templ.channels(), templ.data);
			Image dst(cpudst.cols, cpudst.rows, cpudst.channels(), cpudst.data);
			time.start(std::string("cpu计算总时间"));
			cvlib::matchTemplate(src, Templ, dst, METHOD);
			time.end();
			Check(cpudst, cvdst, platform_cpu);
		}

	}
};