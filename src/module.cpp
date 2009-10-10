#include <cuda.h>

#include <cudamm/module.hpp>

#include <detail/error.hpp>
#include <detail/module_impl.hpp>

namespace cuda
{
	Module::Module(const char *filename)
		: impl(new impl_t)
	{
		detail::error_check(cuModuleLoad(&impl->mod, filename), "Can't load Cuda module");
	}

	Module::~Module()
	{
		detail::error_warn(cuModuleUnload(impl->mod), "Can't unload Cuda module");
	}
}

