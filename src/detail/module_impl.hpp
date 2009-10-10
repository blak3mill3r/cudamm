#ifndef CUDA_DETAIL_MODULE_IMPL_HPP
#define CUDA_DETAIL_MODULE_IMPL_HPP

#include <cuda.h>

#include <cudamm/module.hpp>

namespace cuda
{
	struct Module::impl_t
	{
		CUmodule mod;
	};
}

#endif

