#ifndef CUDA_ARRAY_HPP
#define CUDA_ARRAY_HPP

#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>

#include <cudamm/stream.hpp>
#include <cudamm/memcpy2d.hpp>

namespace cuda
{
	class DevicePtr;
	class Stream;

	/// CUDA device arrays
	/**
		Noncopyable.
	*/
	class Array : boost::noncopyable
	{
		public:
			/// Device array format
			enum Format {
				UNSIGNED_INT_8 = 0x01,
				UNSIGNED_INT_16 = 0x02,
				UNSIGNED_INT_32 = 0x03,
				SIGNED_INT_8 = 0x08,
				SIGNED_INT_16 = 0x09,
				SIGNED_INT_32 = 0x0a,
				HALF = 0x10,
				FLOAT = 0x20 };
			
			/// Create new array
			/**
				@param width the width of the array (in texels)
				@param height the height of the array
				@param format the format of the array
				@param channels the number of channels in one texel
			*/
			Array(unsigned int width, unsigned int height, Format format, unsigned int channels);
			
			/// Destroy array
			~Array();
			
			
			/// Get the width of the array (in texels)
			/**
				@return the width of the array (in texels)
			*/			
			unsigned int width() const { return width_; }

			/// Get the height of the array
			/**
				@return the height of the array
			*/			
			unsigned int height() const { return height_; }

			/// Get the number of channels in one texel of the array
			/**
				@return the number of channels in one texel of the array (in elements)
			*/			
			unsigned int channels() const { return channels_; }

			/// Get the format of the array
			/**
				@return the format of the array
			*/
			Format format() const { return format_; }
			
			/// Get the size of one element
			/**
				@return the size of one element
			*/
			size_t elementSize() const { return elementSize_; }
			
			/// Get the pitch of the array
			/**
				The pitch is equal to width * channels * elementSize.
			
				@return the size of one row in the array (in bytes)
			*/
			size_t pitch() const { return width() * channels() * elementSize();  }
			
			/// Get the size of the array (in bytes)
			/**
				The size is equal to height * pitch.
				
				@return the size of the array
			*/
			size_t size() const { return  pitch() * height(); }
			
			void upload2D(const void *src, unsigned int srcPitch, Stream &stream) const
			{
				size_t widthBytes = width() * elementSize() * channels();

				Memcpy2D(widthBytes, height())
					.source(src, srcPitch)
					.destination(*this)
					.copy(stream);
			}

			void upload2D(const void *src, unsigned int srcPitch) const
			{
				size_t widthBytes = width() * elementSize() * channels();

				Memcpy2D(widthBytes, height())
					.source(src, srcPitch)
					.destination(*this)
					.copy();
			}
			
			void upload2D(const void *src) const
			{
				size_t widthBytes = width() * elementSize() * channels();
				upload2D(src, widthBytes);
			}

			void upload2D(const void *src, Stream &stream) const
			{
				size_t widthBytes = width() * elementSize() * channels();
				upload2D(src, widthBytes, stream);
			}
			
			void upload(const void *src, unsigned int destIndex = 0) const
			{
				memcpy(*this, destIndex, src, width());
			}
			
			void download(void *dest, unsigned int srcIndex = 0) const
			{
				memcpy(dest, *this, srcIndex, width());
			}
			
			void upload(const void *src, unsigned int destIndex, Stream &stream) const
			{
				memcpy(*this, destIndex, src, width(), stream);
			}

			void download(void *dest, unsigned int srcIndex, Stream &stream) const
			{
				memcpy(dest, *this, srcIndex, width(), stream);
			}
			
			void upload(const void *src, Stream &stream) const
			{
				upload(src, 0, stream);
			}
			
			void download(void *dest, Stream &stream) const
			{
				download(dest, 0, stream);
			}
			
		private:
			struct impl_t;
			boost::scoped_ptr<impl_t> impl;
			
			unsigned int width_, height_, channels_;
			Format format_;
			size_t elementSize_;
			
			friend class TextureReference;
			friend class Memcpy2D;

			friend void memcpy(const Array& dest, unsigned int destIndex, const Array &src, unsigned int srcIndex, unsigned int len);

			friend void memcpy(const Array &dest, unsigned int destIndex, const DevicePtr &src, unsigned int len);
			friend void memcpy(const DevicePtr &dest, const Array &src, unsigned int srcIndex, unsigned int len);
	
			friend void memcpy(void *dest, const Array &src, unsigned int srcIndex, unsigned int len);
			friend void memcpy(const Array& dest, unsigned int destIndex, const void *src, unsigned int len);

			friend void memcpy(void *dest, const Array &src, unsigned int srcIndex, unsigned int len, const Stream &stream);
			friend void memcpy(const Array& dest, unsigned int destIndex, const void *src, unsigned int len, const Stream &stream);
	};

	/// Copy from device array to device array
	/**
		@param dest the destination array
		@param destIndex the destination index in the destination array
		@param src the source array
		@param srcIndex the source index in the source array
		@param len the number of bytes to copy
	*/
	void memcpy(const Array& dest, unsigned int destIndex, const Array &src, unsigned int srcIndex, unsigned int len);

	/// Copy from device memory to device array
	/**
		@param dest the destination array
		@param destIndex the destination index in the destination array
		@param src the source device pointer
		@param len the number of bytes to copy
	*/
	void memcpy(const Array &dest, unsigned int destIndex, const DevicePtr &src, unsigned int len);

	/// Copy from device array to device memory
	/**
		@param dest the destination device pointer
		@param src the source array
		@param srcIndex the source index in the source array
		@param len the number of bytes to copy
	*/
	void memcpy(const DevicePtr &dest, const Array &src, unsigned int srcIndex, unsigned int len);

	/// Copy from device array to host memory
	/**
		@param dest the destination pointer
		@param src the source array
		@param srcIndex the source index in the source array
		@param len the number of bytes to copy
	*/
	void memcpy(void *dest, const Array &src, unsigned int srcIndex, unsigned int len);

	/// Copy from host memory to device array
	/**
		@param dest the destination array
		@param destIndex the destination index in the destination array
		@param src the source pointer
		@param len the number of bytes to copy
	*/
	void memcpy(const Array& dest, unsigned int destIndex, const void *src, unsigned int len);

	/// Copy from device array to host memory asynchronously
	/**
		@param dest the destination pointer
		@param src the source array
		@param srcIndex the source index in the source array
		@param len the number of bytes to copy
		@param stream the stream to associate this copy operation with
	*/
	void memcpy(void *dest, const Array &src, unsigned int srcIndex, unsigned int len, const Stream &stream);

	/// Copy from host memory to device array asynchronously
	/**
		@param dest the destination array
		@param destIndex the destination index in the destination array
		@param src the source pointer
		@param len the number of bytes to copy
		@param stream the stream to associate this copy operation with
	*/
	void memcpy(const Array& dest, unsigned int destIndex, const void *src, unsigned int len, const Stream &stream);

}

#endif

