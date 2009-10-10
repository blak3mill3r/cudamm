#ifndef CUDA_DETAIL_STREAM_IMPL_HPP
#define CUDA_DETAIL_STREAM_IMPL_HPP

#include <cuda.h>

#include <cudamm/stream.hpp>

namespace cuda
{
	struct Stream::impl_t
	{
		CUstream stream;
	};
}

#endif

