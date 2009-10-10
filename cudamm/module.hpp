#ifndef CUDA_MODULE_HPP
#define CUDA_MODULE_HPP

#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>

namespace cuda
{
	/// CUDA modules
	/**
		Noncopyable.
	*/
	class Module : boost::noncopyable
	{
		public:
			/// Load a module from file
			/**
				@param filename 
			*/
			explicit Module(const char *filename);
			
			/// Unload module
			~Module();
		private:
			struct impl_t;
			boost::scoped_ptr<impl_t> impl;
			
			friend class Function;
			friend class TextureReference;
	};
}

#endif
