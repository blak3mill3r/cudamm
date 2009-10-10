#ifndef CUDA_FUNCTION_HPP
#define CUDA_FUNCTION_HPP

#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>

namespace cuda
{
	class Module;
	class TextureReference;
	class Stream;
	class DevicePtr;

	/// CUDA device function
	/**
		Noncopyable.
	*/
	class Function : boost::noncopyable
	{
		public:
			/// Get a function from a CUDA Module
			/**
				@param module the module to load from
				@param name the name of the function
				@note nvcc may use name mangling extern "C" { } is used in CUDA source.
			*/
			Function(Module &module, const char *name);
			
			/// Destroy function
			~Function();
			
			/// Specify the dimensions of the thread blocks
			/**
				@param x the X dimension of the thread blocks
				@param y the Y dimension of the thread blocks
				@param z the Z dimension of the thread blocks
			*/
			void setBlockShape(int x, int y, int z) const;
			
			/// Set the amount of shared memory for the thread blocks
			/**
				@param bytes the number of bytes that will be available to each thread block
			*/
			void setSharedSize(unsigned int bytes) const;
			
			/// Set the size of function parameters
			/**
				@param bytes the total size in bytes needed by the function parameters
			*/
			void setParameterSize(unsigned int bytes) const;
			
			/// Sets an integer parameter
			/**
				@param offset byte offset in parameter space of the kernel
				@param value the value of the parameter
			*/
			void setParameter(int offset, int value) const;

			/// Sets a float parameter
			/**
				@param offset byte offset in parameter space of the kernel
				@param value the value of the parameter
			*/
			void setParameter(int offset, float value) const;
			
			/// Copies an arbitrary amount of data into the parameter space
			/**
				@param offset byte offset in parameter space of the kernel
				@param data the data
				@param len size of the data in bytes
			*/
			void setParameter(int offset, void *data, unsigned int len) const;

			/// Sets a device pointer parameter
			/**
				@param offset byte offset in parameter space of the kernel
				@param ptr the device pointer value
			*/
			void setParameter(int offset, const DevicePtr &ptr) const;

			/// Invoke the kernel
			/**
				Invoke the kernel on a 1x1 grid of blocks.
			*/
			void launch() const;
			
			/// Invoke the kernel on a grid
			/**
				@param gridWidth the width of the grid
				@param gridHeight the height of the grid
			*/
			void launch(int gridWidth, int gridHeight) const;
			
			/// Invoke the kernel on a grid asynchronously
			/**
				@param gridWidth the width of the grid
				@param gridHeight the height of the grid
				@param stream the stream to associate the kernel with
			*/
			void launch(int gridWidth, int gridHeight, const Stream &stream) const;
			
			/// Use a texture
			/**
				makes the CUDA array or linear memory bound to the given texture reference
				available to a device program as a texture.
				
				@param texref the texture reference
			*/
			void useTexture(const TextureReference &texref) const;
		private:
			struct impl_t;
			boost::scoped_ptr<impl_t> impl;
	};
}

#endif

