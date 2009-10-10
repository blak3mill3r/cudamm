#ifndef CUDA_EXCEPTION_HPP
#define CUDA_EXCEPTION_HPP

#include <stdexcept>

namespace cuda
{
	/// CUDAmm exception
	struct Exception : public std::runtime_error
	{
		/// Create exception
		/**
			@param msg Error message
		*/
		explicit Exception(const char *msg)
			: std::runtime_error(msg)
		{
		}
	};
}

#endif

