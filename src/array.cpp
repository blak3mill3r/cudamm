#include <map>

#include <cuda.h>

#include <cudamm/exception.hpp>
#include <cudamm/array.hpp>
#include <cudamm/stream.hpp>
#include <cudamm/deviceptr.hpp>

#include <detail/error.hpp>
#include <detail/array_impl.hpp>
#include <detail/stream_impl.hpp>
#include <detail/deviceptr_impl.hpp>

namespace
{
	size_t getFormatSize(cuda::Array::Format format)
	{
		static const struct temp_t
		{
			temp_t()
			{
				sizes[cuda::Array::UNSIGNED_INT_8] = 1;
				sizes[cuda::Array::UNSIGNED_INT_16] = 2;
				sizes[cuda::Array::UNSIGNED_INT_32] = 4;
				sizes[cuda::Array::SIGNED_INT_8] = 1;
				sizes[cuda::Array::SIGNED_INT_16] = 2;
				sizes[cuda::Array::SIGNED_INT_32] = 4;
				sizes[cuda::Array::HALF] = 2;
				sizes[cuda::Array::FLOAT] = 4;
			}
				
			typedef std::map<cuda::Array::Format, size_t> map_t;
			map_t sizes;
		} temp;
		
		temp_t::map_t::const_iterator it = temp.sizes.find(format);
		if(it == temp.sizes.end()) throw cuda::Exception("Unknown array format");
		return it->second;
	}
	
	template <bool>
	CUarray_format cudaArrayFormatImpl(cuda::Array::Format format)
	{
		return static_cast<CUarray_format>(format);
	}
	
	template <>
	CUarray_format cudaArrayFormatImpl<false>(cuda::Array::Format format)
	{
		static const struct temp_t
		{
			temp_t()
			{
				formats[cuda::Array::UNSIGNED_INT_8] = CU_AD_FORMAT_UNSIGNED_INT8;
				formats[cuda::Array::UNSIGNED_INT_16] = CU_AD_FORMAT_UNSIGNED_INT16;
				formats[cuda::Array::UNSIGNED_INT_32] = CU_AD_FORMAT_UNSIGNED_INT32;
				formats[cuda::Array::SIGNED_INT_8] = CU_AD_FORMAT_SIGNED_INT8;
				formats[cuda::Array::SIGNED_INT_16] = CU_AD_FORMAT_SIGNED_INT16;
				formats[cuda::Array::SIGNED_INT_32] = CU_AD_FORMAT_SIGNED_INT32;
				formats[cuda::Array::HALF] = CU_AD_FORMAT_HALF;
				formats[cuda::Array::FLOAT] = CU_AD_FORMAT_FLOAT;
			}
		
			typedef std::map<cuda::Array::Format, CUarray_format> map_t;
			map_t formats;
		} temp;
		
		temp_t::map_t::const_iterator it = temp.formats.find(format);
		if(it == temp.formats.end()) throw ::cuda::Exception("Unknown array format");
		return it->second;
	}
	
	inline CUarray_format cudaArrayFormat(cuda::Array::Format format)
	{
		typedef unsigned int temp_t;

		return cudaArrayFormatImpl<
			static_cast<temp_t>(cuda::Array::UNSIGNED_INT_8) == static_cast<temp_t>(CU_AD_FORMAT_UNSIGNED_INT8) &&
			static_cast<temp_t>(cuda::Array::UNSIGNED_INT_16) == static_cast<temp_t>(CU_AD_FORMAT_UNSIGNED_INT16) && 
			static_cast<temp_t>(cuda::Array::UNSIGNED_INT_32) == static_cast<temp_t>(CU_AD_FORMAT_UNSIGNED_INT32) &&
			static_cast<temp_t>(cuda::Array::SIGNED_INT_8) == static_cast<temp_t>(CU_AD_FORMAT_SIGNED_INT8) && 
			static_cast<temp_t>(cuda::Array::SIGNED_INT_16) == static_cast<temp_t>(CU_AD_FORMAT_SIGNED_INT16) &&
			static_cast<temp_t>(cuda::Array::SIGNED_INT_32) == static_cast<temp_t>(CU_AD_FORMAT_SIGNED_INT32) &&
			static_cast<temp_t>(cuda::Array::HALF) == static_cast<temp_t>(CU_AD_FORMAT_HALF) &&
			static_cast<temp_t>(cuda::Array::FLOAT) == static_cast<temp_t>(CU_AD_FORMAT_FLOAT)
			>(format);
	}
}

namespace cuda
{
	Array::Array(unsigned int width, unsigned int height, Format format, unsigned int channels)
		: impl(new impl_t)
		, width_(width)
		, height_(height)
		, channels_(channels)
		, format_(format)
		, elementSize_(getFormatSize(format))
	{
		CUDA_ARRAY_DESCRIPTOR desc;
		desc.Width = width;
		desc.Height = height;
		desc.Format = cudaArrayFormat(format);
		desc.NumChannels = channels;
		
		detail::error_check(cuArrayCreate(&impl->array, &desc),
			"Can't create Cuda array");
	}
	
	Array::~Array()
	{
		detail::error_warn(cuArrayDestroy(impl->array),
			"Can't destroy Cuda array");
	}
	
	void memcpy(const Array& dest, unsigned int destIndex, const Array &src, unsigned int srcIndex, unsigned int len)
	{
		detail::error_check(cuMemcpyAtoA(dest.impl->array, destIndex, src.impl->array, srcIndex, len),
			"Can't memcpy from device array to device array");
	}
	
	void memcpy(const Array &dest, unsigned int destIndex, const DevicePtr &src, unsigned int len)
	{
		detail::error_check(cuMemcpyDtoA(dest.impl->array, destIndex, src.impl->devicePtr, len),
			"Can't memcpy from device memory to device array");
	}
	
	void memcpy(const DevicePtr &dest, const Array &src, unsigned int srcIndex, unsigned int len)
	{
		detail::error_check(cuMemcpyAtoD(dest.impl->devicePtr, src.impl->array, srcIndex, len),
			"Can't memcpy from device array to device memory");
	}
	
	void memcpy(void *dest, const Array &src, unsigned int srcIndex, unsigned int len)
	{
		detail::error_check(cuMemcpyAtoH(dest, src.impl->array, srcIndex, len),
			"Can't memcpy from device array to host memory");
	}
	
	void memcpy(const Array& dest, unsigned int destIndex, const void *src, unsigned int len)
	{
		detail::error_check(cuMemcpyHtoA(dest.impl->array, destIndex, src, len),
			"Can't memcpy from host memory to device array");
	}

	void memcpy(void *dest, const Array &src, unsigned int srcIndex, unsigned int len, const Stream &stream)
	{
		detail::error_check(cuMemcpyAtoHAsync(dest, src.impl->array, srcIndex, len, stream.impl->stream),
			"Can't memcpy from device array to host memory asynchronously");
	}
	
	void memcpy(const Array& dest, unsigned int destIndex, const void *src, unsigned int len, const Stream &stream)
	{
		detail::error_check(cuMemcpyHtoAAsync(dest.impl->array, destIndex, src, len, stream.impl->stream),
			"Can't memcpy from host memory to device array asynchronously");
	}
}

