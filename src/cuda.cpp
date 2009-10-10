#include <cudamm/cuda.hpp>

#include <detail/error.hpp>

namespace cuda
{
	struct Cuda::impl_t
	{
		CUcontext ctx;
	};

	Cuda::Cuda(int numDevice)
		: impl(new impl_t)
	{
		detail::error_check(cuInit(0));
	
		CUdevice dev;
		detail::error_check(cuDeviceGet(&dev, numDevice), "Can't get Cuda device");
		detail::error_check(cuCtxCreate(&impl->ctx, 0, dev), "Can't create Cuda context");
	}

	Cuda::~Cuda()
	{
		detail::error_warn(cuCtxDetach(impl->ctx), "Can't detach Cuda context");
	}
}
