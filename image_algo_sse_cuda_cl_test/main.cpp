#include "test_guassianfilter.h"

int main()
{
	int*initcuda;
	cudaMalloc((void**)&initcuda, 1);//每个线程首次使用cuda时，进行初始化一次，可以节约后续开辟内存空间时间
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	TEST_GUASSIAN_FILTER::Run();

	cudaFree(initcuda);
	system("pause");
	return 0;
}