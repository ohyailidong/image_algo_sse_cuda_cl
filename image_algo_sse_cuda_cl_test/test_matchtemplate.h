#pragma once
#include "test_common_define.h"
#include "matchtemplate.h"

class TEST_MATCH_TEMPLATE
{
public:
	static void Run()
	{
		std::cout << "TEST_MATCH_TEMPLATE: \n ";
		cv::Mat image = cv::imread("../image/lena.jpg", 1);
		std::cout << "height= " << image.rows << "  ,width= " << image.cols	<< "  ,channel= " 
			<< image.channels() << "\n";
		cv::Mat templ = cv::imread("../image/lena_template.png", 1);
		Time time;
		cv::Mat cvdst;
		time.start("opencv计算总时间");
		cv::matchTemplate(image, templ, cvdst, 3);
		time.end();
		cv::Mat cpudst(cvdst.size(), cvdst.type());
		Image src(image.cols, image.rows, image.channels(), image.data);
		Image Templ(templ.cols, templ.rows, templ.channels(), templ.data);
		Image dst(cpudst.cols, cpudst.rows, cpudst.channels(), cpudst.data);
		time.start(std::string("cpu计算总时间"));
		cvlib::matchTemplate(src, Templ, dst, 3);
		time.end();
		Check(cpudst, cvdst);
	}
};