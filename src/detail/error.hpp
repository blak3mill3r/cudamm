#ifndef CUDA_DETAIL_ERROR_HPP
#define CUDA_DETAIL_ERROR_HPP

#include <cuda.h>

namespace cuda
{
	namespace detail
	{
		void error_check(CUresult result, const char *msg = 0);
		void error_warn(CUresult result, const char *msg = 0);
	}
}

#endif

