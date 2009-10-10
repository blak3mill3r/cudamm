#ifndef CUDA_DETAIL_ARRAY_IMPL_HPP
#define CUDA_DETAIL_ARRAY_IMPL_HPP

#include <cudamm/array.hpp>

namespace cuda
{
	struct Array::impl_t
	{
		CUarray array;
	};
}

#endif

