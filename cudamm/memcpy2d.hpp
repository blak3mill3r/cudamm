#ifndef CUDA_MEMCPY2D_HPP
#define CUDA_MEMCPY2D_HPP

#include <boost/scoped_ptr.hpp>

namespace cuda
{
	class Stream;
	class DevicePtr;
	class Array;

	/// 2D copy descriptor
	/**
	*/
	class Memcpy2D
	{
		public:
			/// Create a new copy descriptor
			/**
				Assign source and destination position to 0, 0.
				
				@param widthBytes the width of the memory region to copy (in bytes)
				@param height the height of the memory region to copy
			*/
			explicit Memcpy2D(unsigned int widthBytes = 0, unsigned int height = 0);
			
			/// Copy constructor
			/**
				@param copy the Memcpy2D to copy
			*/
			Memcpy2D(const Memcpy2D &copy);
			
			/// Destructor
			~Memcpy2D();

			/// Assignment
			/**	
				@param copy the Memcpy2D to assign to *this
			*/
			Memcpy2D& operator=(const Memcpy2D &copy)
			{
				if(&copy == this) return *this;
				Memcpy2D temp(copy);
				swap(*this, temp);
				return *this;
			}

			/// Set the source of the copy to host memory
			/**
				@param src the pointer to the source memory
				@param pitch the pitch of the source memory
			*/
			Memcpy2D& source(const void *src, unsigned int pitch);
			
			/// Set the source of the copy to device memory
			/**
				@param src the device pointer to the source memory
				@param pitch the pitch of the source memory
			*/
			Memcpy2D& source(const DevicePtr& src, unsigned int pitch);
			
			/// Set the source of the copy to device array
			/**
				@param src the source array
			*/
			Memcpy2D& source(const Array& src);

			/// Set the destination of the copy to host memory
			/**
				@param dest the pointer to the destination memory
				@param pitch the pitch of the destination memory
			*/
			Memcpy2D& destination(void *dest, unsigned int pitch);
			
			/// Set the destination of the copy to device memory
			/**
				@param dest the device pointer to the destination memory
				@param pitch the pitch of the destination memory
			*/
			Memcpy2D& destination(const DevicePtr& dest, unsigned int pitch);

			/// Set the destination of the copy to device array
			/**
				@param dest the destination array
			*/			
			Memcpy2D& destination(const Array& dest);
			
			/// Set the source position of the copy
			/**
				@param xBytes the X position in bytes
				@param y the Y position
			*/			
			Memcpy2D& sourcePos(unsigned int xBytes, unsigned int y);
			
			/// Set the destination position of the copy
			/**
				@param xBytes the X position in bytes
				@param y the Y position
			*/			
			Memcpy2D& destinationPos(unsigned int xBytes, unsigned int y);
			
			/// Set the size of the copy
			/**
				@param widthBytes the width of the memory region to copy (in bytes)
				@param height the height of the memory region to copy
			*/
			Memcpy2D& size(unsigned int widthBytes, unsigned int height);
			
			/// Execute copy
			/**
				Device-to-device copies may fail for pitches not given
				by malloc2D. Use copyUnaligned for a (maybe) slow workaround.
			*/
			void copy() const;
			
			/// Execute copy asynchronously
			/**
				Device-to-device copies may fail for pitches not given
				by malloc2D. Use copyUnaligned for a (maybe) slow workaround.

				Works only with page locked host memory.
			
				@param stream the stream to associate the copy operation with
			*/
			void copy(const Stream &stream) const;
			
			/// Execute copy unaligned
			/**
				Unaligned copies are not restricted to aligned pitches from
				malloc2D. Unaligned copies may run significantly slower where
				aligned copies fail.
			*/
			void copyUnaligned() const;
		
		private:
			struct impl_t;
			boost::scoped_ptr<impl_t> impl;
			
			/// Swap two Memcpy2D's
			/**
				@param a the Memcpy2D to swap with b
				@param b the Memcpy2D to swap with a
			*/
			friend void swap(Memcpy2D &a, Memcpy2D &b);
	};
	
	inline void swap(Memcpy2D &a, Memcpy2D &b)
	{
		swap(a.impl, b.impl);
	}
}

#endif

