#ifndef CUDA_DEVICEMEMORY_HPP
#define CUDA_DEVICEMEMORY_HPP

#include <iostream>

#include <boost/utility.hpp>

#include <cudamm/exception.hpp>
#include <cudamm/deviceptr.hpp>

namespace cuda
{
	class DeviceMemory : boost::noncopyable
	{
		public:
			explicit DeviceMemory(unsigned int size)
				: ptr_(cuda::malloc(size)), size_(size)
			{	
			}

			~DeviceMemory()
			{
				try
				{
					free(ptr());
				} catch(cuda::Exception const &e)
				{
					std::cerr << e.what() << std::endl;
				}
			}
			
			DevicePtr const &ptr() const { return ptr_; }
			unsigned int size() const { return size_; }
			
			void set8(unsigned char value) const
			{
				memset8(ptr(), value, size());
			}
			
			void set16(unsigned short value) const
			{
				memset16(ptr(), value, size() >> 1);
			}
			
			void set32(unsigned int value) const
			{
				memset32(ptr(), value, size() >> 2);
			}
			
			void upload(const void *src) const
			{
				memcpy(ptr(), src, size());
			}
			
			void download(void *dest) const
			{
				memcpy(dest, ptr(), size());
			}
			
		private:
			DevicePtr ptr_;			
			unsigned int size_;
	};
}

#endif

