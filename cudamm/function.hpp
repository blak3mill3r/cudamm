#ifndef CUDA_FUNCTION_HPP
#define CUDA_FUNCTION_HPP

#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/placeholders.hpp>
#include <iostream>
#include <typeinfo>

using namespace boost;

#define ALIGN_UP(offset, alignment) \
  (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

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

      // MPL magic for kernel invocation syntactic sugar

      struct not_specified {};
      void foo(int a ) {}

      struct set_parameters
      {
        set_parameters( const Function *function )
        {
          this->function = function;
          offset = 0;
        }
        template <class T>
        inline void operator()(T p)
        {
          function->setParameter( offset, p );
          ALIGN_UP( offset, __alignof(p) );
          offset += sizeof( p );
        }

        inline void finalize()
        {
          std::cout << "finalize at " << offset << " bytes " << std::endl;
          function->setParameterSize( offset );
        }

        protected:
        const Function *function;
        int offset;
      };

      template <class A,
                class B,
                class C,
                class D,
                class E,
                class F,
                class G,
                class H,
                class I,
                class J>
      void go_impl( int width, int height, const Stream &stream, A a, B b, C c, D d, E e, F f, G g, H h, I i, J j )
      {
        set_parameters sp = set_parameters(this);
        sp( a );
        sp( b );
        sp( c );
        sp( d );
        sp( e );
        sp( f );
        sp( g );
        sp( h );
        sp( i );
        sp( j );
        sp.finalize();
        launch( width, height, stream );
      }

			/// Kernel launch
			/**
				@param width grid width
				@param height grid height
        @param stream cuda Stream reference
        @param everything else, params for the kernel
			*/
      template <class A>
      void go( int width, int height, const Stream& stream, A a )
      {
        go_impl( width, height, stream,
                 a,
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified()
               );
      }

      // 2 argument wrapper
      template <class A, class B>
      void go( int width, int height, const Stream& stream, A a, B b )
      {
        go_impl( width, height, stream,
                 a,
                 b,
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified()
               );
      }

      // 3 argument wrapper
      template <class A, class B, class C>
      void go( int width, int height, const Stream& stream, A a, B b, C c )
      {
        go_impl( width, height, stream,
                 a,
                 b,
                 c,
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified()
               );
      }

      // 4 argument wrapper
      template <class A, class B, class C, class D>
      void go( int width, int height, const Stream& stream, A a, B b, C c, D d )
      {
        go_impl( width, height, stream,
                 a,
                 b,
                 c,
                 d,
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified()
               );
      }

      // 5 argument wrapper
      template <class A, class B, class C, class D, class E>
      void go( int width, int height, const Stream& stream, A a, B b, C c, D d, E e )
      {
        go_impl( width, height, stream,
                 a,
                 b,
                 c,
                 d,
                 e,
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified()
               );
      }

      // 6 argument wrapper
      template <class A, class B, class C, class D, class E, class F>
      void go( int width, int height, const Stream& stream, A a, B b, C c, D d, E e, F f )
      {
        go_impl( width, height, stream,
                 a,
                 b,
                 c,
                 d,
                 e,
                 f,
                 not_specified(),
                 not_specified(),
                 not_specified(),
                 not_specified()
               );
      }

      // 7 argument wrapper
      template <class A, class B, class C, class D, class E, class F, class G>
      void go( int width, int height, const Stream& stream, A a, B b, C c, D d, E e, F f, G g )
      {
        go_impl( width, height, stream,
                 a,
                 b,
                 c,
                 d,
                 e,
                 f,
                 g,
                 not_specified(),
                 not_specified(),
                 not_specified()
               );
      }

      // 8 argument wrapper
      template <class A, class B, class C, class D, class E, class F, class G, class H>
      void go( int width, int height, const Stream& stream, A a, B b, C c, D d, E e, F f, G g, H h )
      {
        go_impl( width, height, stream,
                 a,
                 b,
                 c,
                 d,
                 e,
                 f,
                 g,
                 h,
                 not_specified(),
                 not_specified()
               );
      }

      // 9 argument wrapper
      template <class A, class B, class C, class D, class E, class F, class G, class H, class I>
      void go( int width, int height, const Stream& stream, A a, B b, C c, D d, E e, F f, G g, H h, I i )
      {
        go_impl( width, height, stream,
                 a,
                 b,
                 c,
                 d,
                 e,
                 f,
                 g,
                 h,
                 i,
                 not_specified()
               );
      }

      // 10 argument wrapper
      template <class A, class B, class C, class D, class E, class F, class G, class H, class I, class J>
      void go( int width, int height, const Stream& stream, A a, B b, C c, D d, E e, F f, G g, H h, I i, J j )
      {
        go_impl( width, height, stream,
                 a,
                 b,
                 c,
                 d,
                 e,
                 f,
                 g,
                 h,
                 i,
                 j
               );
      }

		private:
			struct impl_t;
			boost::scoped_ptr<impl_t> impl;
	};

  template <>
  inline void Function::set_parameters::operator()(Function::not_specified)
  {
  }
}


#endif

