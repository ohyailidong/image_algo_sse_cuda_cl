#pragma once

#include "rgb2gray.h"


//void main()
//{
//	int*initcuda;
//	cudaMalloc((void**)&initcuda, 1);//ÿ���߳��״�ʹ��cudaʱ�����г�ʼ��һ�Σ����Խ�Լ���������ڴ�ռ�ʱ��
//	cudaDeviceProp prop;
//	cudaGetDeviceProperties(&prop, 0);
//
//	Time time;
//	cv::Mat image = cv::imread("lena.jpg", 1);
//	cv::Mat image0 = cv::imread("lena.jpg", 0);
//	cv::Mat dst(image.size(), CV_8UC1);
//	auto imagesize = image.size();
//	std::cout << "height= " << imagesize.height << "  ,width= " << imagesize.width << "\n";
//	time.start(std::string("RGB2GRAY ������ʱ��"));
//	cvlib::cuda::RGB2GRAY(image.data, image.size().height, image.size().width, dst.data);
//	time.end();
//	time.start(std::string("RGB2GRAY_SCALE ������ʱ��"));
//	cvlib::RGB2GRAY_SCALE(image.data, image.size().height, image.size().width, dst.data);
//	time.end();
//	time.start(std::string("RGB2GRAY_TBB ������ʱ��"));
//	cvlib::RGB2GRAY_TBB(image.data, image.size().height, image.size().width, dst.data);
//	time.end();
//	time.start(std::string("RGB2GRAY_SIMD ������ʱ��"));
//	cvlib::RGB2GRAY_SIMD(image.data, image.size().height, image.size().width, dst.data);
//	time.end();
//
//	cv::Mat cvdst;
//	time.start(std::string("opencv������ʱ��"));
//	//LOOP_100
//	cv::cvtColor(image, cvdst, cv::COLOR_RGB2GRAY);
//	time.end();
//	cv::Mat error = abs(image0 - dst);
//	cv::threshold(error, error, 2, 255, cv::THRESH_TOZERO);// ������һ�����ص����
//	error.convertTo(error, CV_32F);
//
//	double minval, maxval;
//	if (error.channels() == 1)
//	{
//		cv::Point minid, maxid;
//		cv::minMaxLoc(error, &minval, &maxval, &minid, &maxid);
//	}
//
//	double err = sum(error.mul(error))[0];
//	std::cout << "���ƽ���� error��" << err << "\n";
//	cudaFree(initcuda);
//}

