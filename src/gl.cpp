#include <cuda.h>

#include <cudamm/deviceptr.hpp>

#include <detail/error.hpp>
#include <detail/deviceptr_impl.hpp>

#include <GL/gl.h>

namespace cuda
{
  void RegisterBufferObject( GLuint buffer_object )
  {
    cuGLRegisterBufferObject( buffer_object );
  }

  void UnregisterBufferObject( GLuint buffer_object )
  {
    cuGLUnregisterBufferObject( buffer_object );
  }

  DevicePtr MapBufferObject(unsigned int &size, GLuint buffer_object)
  {
    unsigned int size;
    CUdeviceptr devPtr;

    cuGLMapBufferObject( &devPtr, size, buffer_object );

    DevicePtr ptr;
    ptr.impl->devicePtr = devPtr;
    return ptr;
  }

  void UnmapBufferObject( GLuint buffer_object )
  {
    cuGLUnmapBufferObject( buffer_object );
  }

}
