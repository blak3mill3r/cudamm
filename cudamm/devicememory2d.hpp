#ifndef CUDA_DEVICEMEMORY2D_HPP
#define CUDA_DEVICEMEMORY2D_HPP

#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>

#include <cudamm/stream.hpp>
#include <cudamm/exception.hpp>
#include <cudamm/deviceptr.hpp>
#include <cudamm/memcpy2d.hpp>

namespace cuda
{
	class DeviceMemory2D : boost::noncopyable
	{
		public:
			DeviceMemory2D(unsigned int widthBytes, unsigned int height, unsigned int elementSize)
				: ptr_(cuda::malloc2D(pitch_, widthBytes, height, elementSize))
				, width_(widthBytes)
				, height_(height)
				, elementSize_(elementSize)
			{
			}
			
			~DeviceMemory2D()
			{
				free(ptr());
			}

			const DevicePtr& ptr() const { return ptr_; }						
			unsigned int width() const { return width_; }
			unsigned int height() const { return height_; }
			unsigned int elementSize() const { return elementSize_; }
			unsigned int pitch() const { return pitch_; }
			
			void upload2D(const void *src, unsigned int srcPitch, Stream &stream) const
			{
				Memcpy2D(width(), height())
					.source(src, srcPitch)
					.destination(ptr(), pitch())
					.copy(stream);
			}

			void upload2D(const void *src, unsigned int srcPitch) const
			{
				Memcpy2D(width(), height())
					.source(src, srcPitch)
					.destination(ptr(), pitch())
					.copy();
			}
			
			void upload2D(const void *src) const { upload2D(src, width()); }
			void upload2D(const void *src, Stream &stream) const { upload2D(src, width(), stream); }
			
			void download2D(void *dest, unsigned int destPitch, Stream &stream) const
			{
				Memcpy2D(width(), height())
					.source(ptr(), pitch())
					.destination(dest, destPitch)
					.copy(stream);
			}

			void download2D(void *dest, unsigned int destPitch) const
			{
				Memcpy2D(width(), height())
					.source(ptr(), pitch())
					.destination(dest, destPitch)
					.copy();
			}

			void download2D(void *src) const { download2D(src, width()); }
			void download2D(void *src, Stream &stream) const { download2D(src, width(), stream); }
			
		private:
			DevicePtr ptr_;			
			unsigned int width_, height_, elementSize_, pitch_;
	};
}

#endif


