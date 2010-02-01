#ifndef CUDA_DEVICEPTR_HPP
#define CUDA_DEVICEPTR_HPP

#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>

namespace cuda
{
	class Stream;
	class Array;

	/// CUDA device pointer
	/**
		
	
		Not default constructible.
	*/
	class DevicePtr
	{
		public:
			/// Copy constructor
			/**
				@param copy the device pointer to copy
			*/
			DevicePtr(const DevicePtr& copy);
			
			/// Destructor
			~DevicePtr();
			
			/// Assignment
			/**
				Strong exception safety.
				
				@param copy a device pointer to assign to *this
			*/
			DevicePtr& operator=(const DevicePtr &copy)
			{
				if(&copy == this) return *this;
				DevicePtr temp(copy);
				swap(*this, temp);
				return *this;
			}

			DevicePtr operator+(int bytes) const;
	
		private:
			struct impl_t;
			boost::scoped_ptr<impl_t> impl;
			
			DevicePtr();
			
			/// Swap two device pointers
			/**
				Never throws.
		
				@param a the device pointer to swap with b
				@param b the device pointer to swap with a
			*/	
			friend void swap(DevicePtr& a, DevicePtr& b);

			friend void memset8(const DevicePtr &ptr, unsigned char value, unsigned int count);
			friend void memset16(const DevicePtr &ptr, unsigned short value, unsigned int count);
			friend void memset32(const DevicePtr &ptr, unsigned int value, unsigned int count);

			friend void memcpy(const DevicePtr &dest, const void *src, unsigned int len);
			friend void memcpy(void *dest, const DevicePtr& src, unsigned int len);
			friend void memcpy(const DevicePtr& dest, const DevicePtr& src, unsigned int len);

			friend void memcpy(const DevicePtr &dest, const void *src, unsigned int len, const Stream &stream);
			friend void memcpy(void *dest, const DevicePtr& src, unsigned int len, const Stream &stream);

			friend void memcpy(const Array &dest, unsigned int destIndex, const DevicePtr &src, unsigned int len);
			friend void memcpy(const DevicePtr &dest, const Array &src, unsigned int srcIndex, unsigned int len);
			
			friend DevicePtr malloc(unsigned int size);
			
			friend DevicePtr malloc2D(
				unsigned int &pitch,
				unsigned int widthBytes,
				unsigned int height,
				unsigned int elementSize);

			friend void free(const DevicePtr &ptr);

			friend class Function;
			friend class TextureReference;
			friend class Memcpy2D;
	};
	
	inline void swap(DevicePtr& a, DevicePtr& b)
	{
		swap(a.impl, b.impl);
	}

	/// Allocate device memory
	/**
		The allocated memory is is suitably aligned for any kind of variable.
		The memory is not cleared.
	
		@param size the number of bytes to allocate
		@return a device pointer to the newly allocated memory
	*/
	DevicePtr malloc(unsigned int size);

	/// Allocate device memory with pitch
	/**
		Allocate at least widthBytes * height bytes of linear memory
		on the device. The allocation may be padded to meet the alignment
		requirements for coalescing as the address is updated
		from row to row.
	
		elementSize specifies the size of the largest reads and writes that
		will be performed on the memory range. elementSize may be 4, 8
		or 16 (since coalesced memory transactions are not possible on other
		data sizes). If elementSizeBytes is smaller than the actual read/write
		size of a kernel, the kernel will run correctly, but possibly at reduced
		speed.
	
		The pitch returned is guaranteed to work with 2D copies under all
		circumstances.
	
		@param pitch a reference where to store the pitch
		@param widthBytes the width of the allocation, in bytes
		@param height the height of the allocation
		@param elementSize the largest read/write size to/from this memory (4, 8 or 16 bytes).
		@return a device pointer to the newly allocated memory
	*/
	DevicePtr malloc2D(unsigned int &pitch, unsigned int widthBytes, unsigned int height, unsigned int elementSize);

	/// Free device memory
	/**
		@param ptr a device pointer returned by a previous call to malloc or malloc2D.
	*/
	void free(const DevicePtr &ptr);

	/// Set a device memory range to a value
	/**
		@param ptr the device pointer
		@param value the value to write
		@param count the number of values to write
	*/
	void memset8(const DevicePtr &ptr, unsigned char value, unsigned int count);

	/// Set a device memory range to a value
	/**
		@param ptr the device pointer
		@param value the value to write
		@param count the number of values to write
	*/
	void memset16(const DevicePtr &ptr, unsigned short value, unsigned int count);

	/// Set a device memory range to a value
	/**
		@param ptr the device pointer
		@param value the value to write
		@param count the number of values to write
	*/
	void memset32(const DevicePtr &ptr, unsigned int value, unsigned int count);

	/// Copy from host memory to device memory
	/**
		@param dest the destination memory pointer
		@param src the source memory pointer
		@param len the number of bytes to copy
	*/
	void memcpy(const DevicePtr &dest, const void *src, unsigned int len);

	/// Copy from device memory to host memory
	/**
		@param dest the destination memory pointer
		@param src the source memory pointer
		@param len the number of bytes to copy
	*/
	void memcpy(void *dest, const DevicePtr& src, unsigned int len);
	
	/// Copy from device memory to device memory
	/**
		@param dest the destination memory pointer
		@param src the source memory pointer
		@param len the number of bytes to copy
	*/
	void memcpy(const DevicePtr& dest, const DevicePtr& src, unsigned int len);

	/// Copy from host memory to device memory asynchronously
	/**
		Works only with page locked host memory.
	
		@param dest the destination memory pointer
		@param src the source memory pointer
		@param len the number of bytes to copy
		@param stream the stream to associate the operation with
	*/
	void memcpy(const DevicePtr &dest, const void *src, unsigned int len, const Stream &stream);

	/// Copy from device memory to host memory asynchronously
	/**
		Works only with page locked host memory.
	
		@param dest the destination memory pointer
		@param src the source memory pointer
		@param len the number of bytes to copy
		@param stream the stream to associate the operation with
	*/
	void memcpy(void *dest, const DevicePtr& src, unsigned int len, const Stream &stream);

};

#endif

