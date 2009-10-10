#ifndef CUDA_STREAM_HPP
#define CUDA_STREAM_HPP

#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>

namespace cuda
{
	class Array;
	class DevicePtr;

	/// CUDA streams
	/**
		CUDA streams are used to keep track of asynchronous operations.
		
		Noncopyable.
	*/
	class Stream : boost::noncopyable
	{
		public:
			/// Create stream
			Stream();
			
			/// Destroy stream
			~Stream();
			
			/// Block until device has completed all operations in stream 
			void synchronize() const;
			
			/// Query if all operations in the stream have completed
			/**
				@return true if complete
			*/
			bool query() const;
		private:
			struct impl_t;
			boost::scoped_ptr<impl_t> impl;
			
			friend class Event;
			friend class Function;
			friend class Memcpy2D;

			friend void memcpy(const DevicePtr &dest, const void *src, unsigned int len, const Stream &stream);
			friend void memcpy(void *dest, const DevicePtr& src, unsigned int len, const Stream &stream);

			friend void memcpy(void *dest, const Array &src, unsigned int srcIndex, unsigned int len, const Stream &stream);
			friend void memcpy(const Array& dest, unsigned int destIndex, const void *src, unsigned int len, const Stream &stream);
	};
}

#endif

