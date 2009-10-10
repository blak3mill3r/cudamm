#ifndef CUDA_DETAIL_TEXTUREREFERENCE_IMPL_HPP
#define CUDA_DETAIL_TEXTUREREFERENCE_IMPL_HPP

#include <cudamm/texturereference.hpp>

namespace cuda
{
	struct TextureReference::impl_t
	{
		CUtexref texref;
	};
}

#endif

