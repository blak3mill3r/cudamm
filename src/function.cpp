#include <cuda.h>

#include <cudamm/function.hpp>
#include <cudamm/module.hpp>
#include <cudamm/texturereference.hpp>
#include <cudamm/stream.hpp>

#include <detail/error.hpp>
#include <detail/module_impl.hpp> 
#include <detail/texturereference_impl.hpp>
#include <detail/stream_impl.hpp>
#include <detail/deviceptr_impl.hpp>

namespace cuda
{
	struct Function::impl_t
	{
		CUfunction func;
	};

	Function::Function(Module &module, const char *name)
		: impl(new impl_t)
	{
		detail::error_check(cuModuleGetFunction(&impl->func, module.impl->mod, name),
			"Can't get Cuda function");
	}

	Function::~Function()
	{
	}
	
	void Function::setBlockShape(int x, int y, int z) const
	{
		detail::error_check(cuFuncSetBlockShape(impl->func, x, y, z),
			"Can't set Cuda function block shape");
	}
	
	void Function::setSharedSize(unsigned int bytes) const
	{
		detail::error_check(cuFuncSetSharedSize(impl->func, bytes),
			"Can't set Cuda function shared memory size");
	}
	
	void Function::setParameterSize(unsigned int bytes) const
	{
		detail::error_check(cuParamSetSize(impl->func, bytes),
			"Can't set Cuda function parameter size");
	}
	
	void Function::setParameter(int offset, int value) const
	{
		detail::error_check(cuParamSeti(impl->func, offset, value),
			"Can't set Cuda function parameter (int)");
	}
	
	void Function::setParameter(int offset, float value) const
	{
		detail::error_check(cuParamSetf(impl->func, offset, value),
			"Can't set Cuda function parameter (float)");
	}
	
	void Function::setParameter(int offset, void *data, unsigned int len) const
	{
		detail::error_check(cuParamSetv(impl->func, offset, data, len),
			"Can't set Cuda function parameter");
	}

	void Function::setParameter(int offset, const DevicePtr &ptr) const
	{
		setParameter(offset, static_cast<int>(ptr.impl->devicePtr));
	}	
	
	void Function::launch() const
	{
		detail::error_check(cuLaunch(impl->func),
			"Can't launch Cuda function");
	}
	
	void Function::launch(int gridWidth, int gridHeight) const
	{
		detail::error_check(cuLaunchGrid(impl->func, gridWidth, gridHeight),
			"Can't launch Cuda function grid");
	}
	
	void Function::launch(int gridWidth, int gridHeight, const Stream &stream) const
	{
		detail::error_check(
			cuLaunchGridAsync(impl->func, gridWidth, gridHeight, stream.impl->stream),
			"Can't launch asynchronous Cuda function grid");
	}
		
	void Function::useTexture(const TextureReference &texref) const
	{	
		detail::error_check(cuParamSetTexRef(impl->func, CU_PARAM_TR_DEFAULT, texref.impl->texref),
			"Can't use Cuda texture reference in function");
	}
}



