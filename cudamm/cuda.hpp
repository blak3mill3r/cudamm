#ifndef CUDA_HPP
#define CUDA_HPP

#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>

#include <cudamm/array.hpp>
#include <cudamm/devicememory.hpp>
#include <cudamm/devicememory2d.hpp>
#include <cudamm/deviceptr.hpp>
#include <cudamm/event.hpp>
#include <cudamm/exception.hpp>
#include <cudamm/function.hpp>
#include <cudamm/memcpy2d.hpp>
#include <cudamm/module.hpp>
#include <cudamm/stream.hpp>
#include <cudamm/texturereference.hpp>

/// CUDAmm namespace
namespace cuda
{
	/// Quick and dirty RAII CUDA setup and cleanup
	/**
		Noncopyable.
	*/
	class Cuda : boost::noncopyable
	{
		public:
			/// Initialize CUDA and create context for a device
			/**
				@param numDevice the number of the CUDA device to use
			*/
			explicit Cuda(int numDevice = 0, bool gl = false);
			
			/// Detach CUDA context
			~Cuda();
		private:
			struct impl_t;
			boost::scoped_ptr<impl_t> impl;
	};
}

#endif

