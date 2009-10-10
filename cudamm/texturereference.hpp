#ifndef CUDA_TEXTUREREFERENCE_HPP
#define CUDA_TEXTUREREFERENCE_HPP

#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>

namespace cuda
{
	class Module;
	class DevicePtr;
	class Array;
	
	/// CUDA texture references
	/**
		Noncopyable.
	*/
	class TextureReference : boost::noncopyable
	{
		public:
			/// Load a texture reference from a CUDA module
			/**
				@param mod the module to load from
				@param name the name of the texture reference to load
			*/
			TextureReference(Module &mod, const char *name);
			
			/// Destroy texture reference
			~TextureReference();
			
			/// Bind a linear address range to the texture reference
			/**
				Any previously bound memory will be unbound.
				
				Since the hardware enforces an alignment requirement to
				texture base addresses,  this function returns a byte offset
				that must be applied to texture fetches. This offset must be
				divided by the texel size and passed to the kernels that read
				from the texture so that they can be applied to the
				texture1Dfetch() function. If the device memory pointer was
				returned from cuda::malloc, the offset is guaranteed to be 0.
				
				@param ptr the device pointer to the texture memory
				@param size the size of the texture
				@return the byte offset for texture fetches.
			*/
			unsigned int bind(const DevicePtr &ptr, int size) const;


			/// Bind a device array to the texture reference
			/**
				Any previously bound memory will be unbound.
				
				@param array the array to bind
			*/
			void bind(const Array &array) const;
			
		private:
			struct impl_t;
			boost::scoped_ptr<impl_t> impl;
			
			friend class Function;
	};
}

#endif

