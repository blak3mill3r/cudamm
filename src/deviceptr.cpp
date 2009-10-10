#include <cuda.h>

#include <cudamm/deviceptr.hpp>
#include <cudamm/stream.hpp>

#include <detail/error.hpp>
#include <detail/stream_impl.hpp>
#include <detail/deviceptr_impl.hpp>

namespace cuda
{
	DevicePtr::DevicePtr()
		: impl(new impl_t)
	{
	}
	
	DevicePtr::DevicePtr(const DevicePtr& copy)
		: impl(new impl_t(*copy.impl))
	{		
	}

	DevicePtr::~DevicePtr()
	{
	}
	
	void memcpy(const DevicePtr &dest, const void *src, unsigned int len)
	{
		detail::error_check(cuMemcpyHtoD(dest.impl->devicePtr, src, len),
			"Can't memcpy from host memory to device memory");
	}
	
	void memcpy(void *dest, const DevicePtr& src, unsigned int len)
	{	
		detail::error_check(cuMemcpyDtoH(dest, src.impl->devicePtr, len),
			"Can't memcpy from device memory to host memory");
	}

	void memcpy(const DevicePtr& dest, const DevicePtr& src, unsigned int len)
	{
		detail::error_check(cuMemcpyDtoD(dest.impl->devicePtr, src.impl->devicePtr, len),
			"Can't memcpy from device memory to device memory");
	}

	void memcpy(const DevicePtr &dest, const void *src, unsigned int len, const Stream &stream)
	{
		detail::error_check(cuMemcpyHtoDAsync(dest.impl->devicePtr, src, len, stream.impl->stream),
			"Can't memcpy from host memory to device memory asynchronously");
	}
	
	void memcpy(void *dest, const DevicePtr& src, unsigned int len, const Stream &stream)
	{
		detail::error_check(cuMemcpyDtoHAsync(dest, src.impl->devicePtr, len, stream.impl->stream),
			"Can't memcpy from device memory to host memory asynchronously");
	}
	
	void memset8(const DevicePtr &ptr, unsigned char value, unsigned int count)
	{
		detail::error_check(cuMemsetD8(ptr.impl->devicePtr, value, count),
			"Can't memset device memory (unsigned char)");
	}
	
	void memset16(const DevicePtr &ptr, unsigned short value, unsigned int count)
	{
		detail::error_check(cuMemsetD16(ptr.impl->devicePtr, value, count),
			"Can't memset device memory (unsigned short)");
	}
	
	void memset32(const DevicePtr &ptr, unsigned int value, unsigned int count)
	{
		detail::error_check(cuMemsetD32(ptr.impl->devicePtr, value, count),
			"Can't memset device memory (unsigned int)");
	}

	DevicePtr malloc(unsigned int size)
	{
		CUdeviceptr devPtr;
		detail::error_check(cuMemAlloc(&devPtr, size),
			"Can't allocate device memory");

		// NOTE: it is assumed that creating a DevicePtr never fails
			
		DevicePtr ptr;
		ptr.impl->devicePtr = devPtr;
		return ptr;
	}

	DevicePtr malloc2D(unsigned int &pitch, unsigned int widthBytes, unsigned int height, unsigned int elementSize)
	{
		CUdeviceptr devPtr;
		unsigned int p;
		detail::error_check(cuMemAllocPitch(&devPtr, &p, widthBytes, height, elementSize),
			"Can't allocate device memory with pitch");
		
		// NOTE: it is assumed that creating a DevicePtr never fails
		
		DevicePtr ptr;
		ptr.impl->devicePtr = devPtr;

		pitch = p;
		return ptr;
	}
	
	void free(const DevicePtr &ptr)
	{
		detail::error_check(cuMemFree(ptr.impl->devicePtr),
			"Can't deallocate device memory");
	}
}

