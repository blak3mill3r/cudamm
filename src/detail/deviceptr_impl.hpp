#ifndef CUDA_DETAIL_DEVICEPTR_IMPL_HPP
#define CUDA_DETAIL_DEVICEPTR_IMPL_HPP

#include <cuda.h>
#include <cudamm/deviceptr.hpp>

namespace cuda
{
	struct DevicePtr::impl_t
	{
		impl_t()
			: devicePtr(static_cast<CUdeviceptr>(0))
		{
		}

		CUdeviceptr devicePtr;
	};
}

#endif

