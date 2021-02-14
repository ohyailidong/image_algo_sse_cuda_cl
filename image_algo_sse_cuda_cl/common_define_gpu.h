#ifndef COMMON_DEFINE_CU_H
#define COMMON_DEFINE_CU_H
#include <iostream>
#include <algorithm>
#include <assert.h>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define DEVICE_FUN_NMAESPACE_BEGIN namespace device{
#define DEVICE_FUN_NMAESPACE_END }

DEVICE_FUN_NMAESPACE_BEGIN
__device__ int borderInterpolateDefault(int p, int len);
DEVICE_FUN_NMAESPACE_END

#endif // !COMMON_DEFINE_CU_H
