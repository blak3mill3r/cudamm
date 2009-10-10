#include <cuda.h>

#include <cudamm/memcpy2d.hpp>
#include <cudamm/deviceptr.hpp>
#include <cudamm/array.hpp>
#include <cudamm/stream.hpp>

#include <detail/error.hpp>
#include <detail/stream_impl.hpp>
#include <detail/deviceptr_impl.hpp>
#include <detail/array_impl.hpp>

namespace cuda
{
	struct Memcpy2D::impl_t
	{
		CUDA_MEMCPY2D_st memcpy2d;
	};
	
	Memcpy2D::Memcpy2D(unsigned int widthBytes, unsigned int height)
		: impl(new impl_t)
	{
		impl->memcpy2d.srcXInBytes = 0;
		impl->memcpy2d.srcY = 0;
		impl->memcpy2d.dstXInBytes = 0;
		impl->memcpy2d.dstY = 0;
		
		size(widthBytes, height);
	}
	
	Memcpy2D::Memcpy2D(const Memcpy2D &copy)
		: impl(new impl_t(*copy.impl))
	{
	}
	
	Memcpy2D::~Memcpy2D()
	{
	}

	Memcpy2D& Memcpy2D::source(const void *src, unsigned int pitch)
	{
		impl->memcpy2d.srcMemoryType = CU_MEMORYTYPE_HOST;
		impl->memcpy2d.srcHost = src;
		impl->memcpy2d.srcPitch = pitch;
		return *this;
	}
	
	Memcpy2D& Memcpy2D::source(const DevicePtr& src, unsigned int pitch)
	{
		impl->memcpy2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		impl->memcpy2d.srcDevice = src.impl->devicePtr;
		impl->memcpy2d.srcPitch = pitch;
		return *this;
	}
	
	Memcpy2D& Memcpy2D::source(const Array& src)
	{
		impl->memcpy2d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
		impl->memcpy2d.srcArray = src.impl->array;
		return *this;
	}

	Memcpy2D& Memcpy2D::destination(void *dest, unsigned int pitch)
	{
		impl->memcpy2d.dstMemoryType = CU_MEMORYTYPE_HOST;
		impl->memcpy2d.dstHost = dest;
		impl->memcpy2d.dstPitch = pitch;
		return *this;
	}
	
	Memcpy2D& Memcpy2D::destination(const DevicePtr& dest, unsigned int pitch)
	{
		impl->memcpy2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		impl->memcpy2d.dstDevice = dest.impl->devicePtr;
		impl->memcpy2d.dstPitch = pitch;
		return *this;
	}
		
	Memcpy2D& Memcpy2D::destination(const Array& dest)
	{
		impl->memcpy2d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		impl->memcpy2d.dstArray = dest.impl->array;
		return *this;
	}	
	
	Memcpy2D& Memcpy2D::sourcePos(unsigned int xBytes, unsigned int y)
	{
		impl->memcpy2d.srcXInBytes = xBytes;
		impl->memcpy2d.srcY = y;
		return *this;
	}
	
	Memcpy2D& Memcpy2D::destinationPos(unsigned int xBytes, unsigned int y)
	{
		impl->memcpy2d.dstXInBytes = xBytes;
		impl->memcpy2d.dstY = y;
		return *this;
	}
	
	Memcpy2D& Memcpy2D::size(unsigned int widthBytes, unsigned int height)
	{
		impl->memcpy2d.WidthInBytes = widthBytes;
		impl->memcpy2d.Height = height;
		return *this;
	}
	
	void Memcpy2D::copy() const
	{
		detail::error_check(cuMemcpy2D(&impl->memcpy2d),
			"Can't execute Cuda 2D memcpy");
	}
	
	void Memcpy2D::copy(const Stream &stream) const
	{
		detail::error_check(cuMemcpy2DAsync(&impl->memcpy2d, stream.impl->stream),
			"Can't execute Cuda 2D memcpy asynchronously");
	}
	
	void Memcpy2D::copyUnaligned() const
	{
		detail::error_check(cuMemcpy2DUnaligned(&impl->memcpy2d),
			"Can't execute Cuda 2D unaligned memcpy");
	}
}

