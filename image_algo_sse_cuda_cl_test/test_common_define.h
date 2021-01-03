#ifndef TEST_COMMON_DEFINE_H
#define TEST_COMMON_DEFINE_H
#include <iostream>
#include <string>

#include <opencv2/world.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class Time {
public:
	Time() {}
	~Time() {}
	void start(std::string des = "")
	{
		m_des = des;
		m_start = cv::getTickCount();
	}
	void end()
	{
		m_time = double(cv::getTickCount() - m_start)/cv::getTickFrequency()*1000;
		std::cout << m_des << ", time :" << m_time << "ms.\n";
	}
private:
	double m_time;
	double m_start;
	std::string m_des;
};

#define LOOP_100 for(int loop=0;loop<100;loop++)

#endif
