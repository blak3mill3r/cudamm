#include <cuda.h>

#include <detail/error.hpp>
#include <detail/stream_impl.hpp>

#include <cudamm/event.hpp>

namespace cuda
{
	struct Event::impl_t
	{
		CUevent event;
	};

	Event::Event()
		: impl(new impl_t)
	{
		detail::error_check(cuEventCreate(&impl->event, 0),
			"Can't create Cuda event");
	}
	
	Event::~Event()
	{
		detail::error_warn(cuEventDestroy(impl->event),
			"Can't destroy Cuda event");
	}
	
	void Event::record() const
	{
		detail::error_check(cuEventRecord(impl->event, 0),
			"Can't record Cuda event");
	}

	void Event::record(const Stream &stream) const
	{
		detail::error_check(cuEventRecord(impl->event, stream.impl->stream),
			"Can't record Cuda stream event");
	}
	
	void Event::synchronize() const
	{
		detail::error_check(cuEventSynchronize(impl->event),
			"Can't synchronize Cuda event");
	}
	
	bool Event::query() const
	{
		CUresult result = cuEventQuery(impl->event);
		if(result == CUDA_ERROR_NOT_READY) return false;
		detail::error_check(result, "Can't query Cuda event state");
		return true;
	}
	
	float operator-(const Event &end, const Event &start)
	{
		float timer;
		detail::error_check(cuEventElapsedTime(&timer, start.impl->event, end.impl->event),
			"Can't get Cuda event elapsed time");
		return timer;
	}
}

