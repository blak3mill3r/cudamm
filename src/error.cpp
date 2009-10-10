#include <iostream>
#include <string>
#include <map>

#include <cuda.h>

#include <cudamm/exception.hpp>

namespace
{
	const char *getErrorString(CUresult result)
	{
		static const struct temp_t
		{
			temp_t()
			{
				errors[CUDA_SUCCESS] = "success";
				errors[CUDA_ERROR_INVALID_VALUE] = "invalid value";
				errors[CUDA_ERROR_OUT_OF_MEMORY] = "out of memory";
				errors[CUDA_ERROR_NOT_INITIALIZED] = "not initialized";
				errors[CUDA_ERROR_DEINITIALIZED] = "deinitialized";
				errors[CUDA_ERROR_NO_DEVICE] = "no device";
				errors[CUDA_ERROR_INVALID_DEVICE] = "invalid device";
				errors[CUDA_ERROR_INVALID_IMAGE] = "invalid image";
				errors[CUDA_ERROR_INVALID_CONTEXT] = "invalid context";
				errors[CUDA_ERROR_CONTEXT_ALREADY_CURRENT] = "context already current";
				errors[CUDA_ERROR_MAP_FAILED] = "map failed";
				errors[CUDA_ERROR_UNMAP_FAILED] = "unmap failed";
				errors[CUDA_ERROR_ARRAY_IS_MAPPED] = "array is mapped";
				errors[CUDA_ERROR_ALREADY_MAPPED] = "already mapped";
				errors[CUDA_ERROR_NO_BINARY_FOR_GPU] = "no binary for gpu";
				errors[CUDA_ERROR_ALREADY_ACQUIRED] = "already acquired";
				errors[CUDA_ERROR_NOT_MAPPED] = "not mapped";
				errors[CUDA_ERROR_INVALID_SOURCE] = "invalid source";
				errors[CUDA_ERROR_FILE_NOT_FOUND] = "file not found";
				errors[CUDA_ERROR_INVALID_HANDLE] = "invalid handle";
				errors[CUDA_ERROR_NOT_FOUND] = "not found";
				errors[CUDA_ERROR_NOT_READY] = "not ready";
				errors[CUDA_ERROR_LAUNCH_FAILED] = "launch failed";
				errors[CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES] = "launch out of resources";
				errors[CUDA_ERROR_LAUNCH_TIMEOUT] = "launch timeout";
				errors[CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING] = "launch incompatible texturing";
				errors[CUDA_ERROR_UNKNOWN] = "unknown";
			}
		
			typedef std::map<CUresult, const char *> map_t;
			map_t errors;
		} temp;
		
		temp_t::map_t::const_iterator it = temp.errors.find(result);
		if(it == temp.errors.end()) return "unknown";
		return it->second;
	 }
}

namespace cuda
{
	namespace detail
	{
		void error_check(CUresult result, const char *msg)
		{
			if(result == CUDA_SUCCESS) return;
			std::string str;
			if(msg) str += std::string(msg) + ": ";
			str += getErrorString(result);
			throw ::cuda::Exception(str.c_str());
		}
		
		void error_warn(CUresult result, const char *msg)
		{
			if(result == CUDA_SUCCESS) return;
			if(msg) std::cerr << msg << ": ";
			std::cerr << getErrorString(result) << std::endl;
		}
	}
}

