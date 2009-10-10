#include <cuda.h>

#include <cudamm/module.hpp>
#include <cudamm/array.hpp>
#include <cudamm/texturereference.hpp>

#include <detail/error.hpp>
#include <detail/module_impl.hpp>
#include <detail/deviceptr_impl.hpp>
#include <detail/array_impl.hpp>
#include <detail/texturereference_impl.hpp>

namespace cuda
{
	TextureReference::TextureReference(Module &mod, const char *name)
		: impl(new impl_t)
	{
		cuda::detail::error_check(cuModuleGetTexRef(&impl->texref, mod.impl->mod, name),
			"Can't get Cuda texture reference from module");
	}

	
	TextureReference::~TextureReference()
	{
		detail::error_warn(cuTexRefDestroy(impl->texref),
			"Can't destroy Cuda texture reference");
	}

	unsigned int TextureReference::bind(const DevicePtr &ptr, int size) const
	{
		unsigned int offset;
		detail::error_check(cuTexRefSetAddress(&offset, impl->texref, ptr.impl->devicePtr, size),
			"Can't bind Cuda texture to device memory");
		return offset;
	}

	void TextureReference::bind(const Array &array) const
	{
		detail::error_check(cuTexRefSetArray(impl->texref, array.impl->array, CU_TRSA_OVERRIDE_FORMAT),
			"Can't bind Cuda texture reference to device array");
	}

}

