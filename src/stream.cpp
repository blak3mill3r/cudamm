#include <cuda.h>

#include <cudamm/stream.hpp>

#include <detail/error.hpp>
#include <detail/stream_impl.hpp>

namespace cuda
{
	Stream::Stream()
		: impl(new impl_t)
	{
		detail::error_check(cuStreamCreate(&impl->stream, 0),
			"Can't create Cuda stream");
	}
	
	Stream::~Stream()
	{
		detail::error_warn(cuStreamDestroy(impl->stream),
			"Can't destroy Cuda stream");
	}
	
	void Stream::synchronize() const
	{
		detail::error_check(cuStreamSynchronize(impl->stream),
			"Can't synchronize Cuda stream");
	}
	
	bool Stream::query() const
	{
		CUresult result = cuStreamQuery(impl->stream);
		if(result == CUDA_ERROR_NOT_READY) return false;
		detail::error_check(result, "Can't query Cuda stream state");
		return true;
	}
}

