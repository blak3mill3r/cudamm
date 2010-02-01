#include <cudamm/cuda.hpp>

#include <detail/error.hpp>

namespace cuda
{
	struct Cuda::impl_t
	{
		CUcontext ctx;
	};

	Cuda::Cuda(int numDevice, bool gl)
		: impl(new impl_t)
	{
		detail::error_check(cuInit(0));
	
		CUdevice dev;
		detail::error_check(cuDeviceGet(&dev, numDevice), "Can't get Cuda device");
    //if(gl) detail::error_check(cuGLCtxCreate(&impl->ctx, 0, dev), "Can't create (OpenGL interoperable) Cuda context");
    //else   detail::error_check(cuCtxCreate(&impl->ctx, 0, dev), "Can't create Cuda context");
    detail::error_check(cuCtxCreate(&impl->ctx, 0, dev), "Can't create Cuda context");
	}

	Cuda::~Cuda()
	{
		detail::error_warn(cuCtxDetach(impl->ctx), "Can't detach Cuda context");
	}
}
