#include <boost/python.hpp>

#include <numarray/arrayobject.h>	// TODO: support numpy!

#include <cudamm/cuda.hpp>

namespace
{
	namespace num = boost::python::numeric;
	namespace py = boost::python;
	
	void checkArray(const num::array &arr)
	{
		if(!PyArray_Check(arr.ptr()))
		{
			PyErr_SetString(PyExc_ValueError, "excpected a PyArrayObject");
			py::throw_error_already_set();
		}
		
		if(!arr.iscontiguous())
		{
			PyErr_SetString(PyExc_ValueError, "given array not contiguous");
			py::throw_error_already_set();
		}
		
		if(!arr.is_c_array())
		{
			PyErr_SetString(PyExc_ValueError, "given array not a C array");
			py::throw_error_already_set();
		}

		if(arr.isbyteswapped())
		{
			PyErr_SetString(PyExc_ValueError, "given array is byteswapped");
			py::throw_error_already_set();
		}			
	}
	
	void checkArraySize(const num::array &arr, unsigned int len)
	{
		if(len > arr.itemsize() * arr.nelements())
		{
			PyErr_SetString(PyExc_ValueError, "given array too small");
			py::throw_error_already_set();
		}	
	}

	void *data(num::array &arr)
	{
		return reinterpret_cast<PyArrayObject*>(arr.ptr())->data;
	}
	
	const void *data(const num::array &arr)
	{
		return reinterpret_cast<PyArrayObject*>(arr.ptr())->data;
	}

	void memcpyAtoH(num::array &dest, const cuda::Array &src, unsigned int srcIndex, unsigned int len)
	{
		checkArray(dest);
		checkArraySize(dest, len);
		cuda::memcpy(data(dest), src, srcIndex, len);
	}

	void memcpyAtoHAsync(num::array &dest, const cuda::Array &src, unsigned int srcIndex, unsigned int len, const cuda::Stream &stream)
	{
		checkArray(dest);
		checkArraySize(dest, len);
		cuda::memcpy(data(dest), src, srcIndex, len, stream);
	}

	void memcpyHtoA(const cuda::Array &dest, unsigned int destIndex, const num::array &src, unsigned int len)
	{
		checkArray(src);
		checkArraySize(src, len);
		cuda::memcpy(dest, destIndex, data(src), len);
	}

	void memcpyHtoAAsync(const cuda::Array &dest, unsigned int destIndex, const num::array &src, unsigned int len, const cuda::Stream &stream)
	{
		checkArray(src);
		checkArraySize(src, len);
		cuda::memcpy(dest, destIndex, data(src), len, stream);
	}
	
	void memcpyHtoD(const cuda::DevicePtr &dest, const num::array &src, unsigned int len)
	{
		checkArray(src);
		checkArraySize(src, len);
		cuda::memcpy(dest, data(src), len);
	}
	
	void memcpyDtoH(num::array &dest, const cuda::DevicePtr &src, unsigned int len)
	{
		checkArray(dest);
		checkArraySize(dest, len);
		cuda::memcpy(data(dest), src, len);
	}

	void memcpyHtoDAsync(const cuda::DevicePtr &dest, const num::array &src, unsigned int len, const cuda::Stream &stream)
	{
		checkArray(src);
		checkArraySize(src, len);
		cuda::memcpy(dest, data(src), len, stream);
	}
	
	void memcpyDtoHAsync(num::array &dest, const cuda::DevicePtr &src, unsigned int len, const cuda::Stream &stream)
	{
		checkArray(dest);
		checkArraySize(dest, len);
		cuda::memcpy(data(dest), src, len, stream);
	}
	
	cuda::Memcpy2D sourceHost(cuda::Memcpy2D &memcpy, const num::array &src, unsigned int pitch)
	{
		checkArray(src);
		return memcpy.source(data(src), pitch);
	}
	
	cuda::Memcpy2D destinationHost(cuda::Memcpy2D &memcpy, num::array &dest, unsigned int pitch)
	{
		checkArray(dest);
		return memcpy.destination(data(dest), pitch);
	}
}

BOOST_PYTHON_MODULE(pycudamm)
{
	import_libnumeric(); // TODO: support numpy!

	namespace py = boost::python;
	
	py::class_<cuda::Cuda, boost::noncopyable>("Cuda");
	
	py::class_<cuda::Module, boost::noncopyable>("Module", py::init<const char*>());
	
	py::class_<cuda::Function, boost::noncopyable>("Function", py::init<cuda::Module&, const char*>())
		.def("launch", static_cast<void (cuda::Function::*)() const>(&cuda::Function::launch))
		.def("launchGrid", static_cast<void (cuda::Function::*)(int, int) const>(&cuda::Function::launch))
		.def("launchGridAsync", static_cast<void (cuda::Function::*)(int, int, const cuda::Stream&) const>(&cuda::Function::launch))
		.def("setBlockShape", &cuda::Function::setBlockShape)
		.def("setSharedSize", &cuda::Function::setSharedSize)
		.def("setParameterSize", &cuda::Function::setParameterSize)
		.def("setParameteri", static_cast<void (cuda::Function::*)(int, int) const>(&cuda::Function::setParameter))
		.def("setParameterf", static_cast<void (cuda::Function::*)(int, float) const>(&cuda::Function::setParameter))
		.def("setParameterp", static_cast<void (cuda::Function::*)(int, const cuda::DevicePtr&) const>(&cuda::Function::setParameter))
		.def("useTexture", &cuda::Function::useTexture);

	py::class_<cuda::Stream, boost::noncopyable>("Stream")
		.def("synchronize", &cuda::Stream::synchronize)
		.def("query", &cuda::Stream::query);
		
	py::class_<cuda::Event, boost::noncopyable>("Event")
		.def("record", static_cast<void (cuda::Event::*)() const>(&cuda::Event::record))
		.def("recordStream", static_cast<void (cuda::Event::*)(const cuda::Stream&) const>(&cuda::Event::record))
		.def("synchronize", &cuda::Event::synchronize)
		.def("query", &cuda::Event::query);
		
	py::enum_<cuda::Array::Format>("ArrayFormat")
		.value("UNSIGNED_INT_8", cuda::Array::UNSIGNED_INT_8)
		.value("UNSIGNED_INT_16", cuda::Array::UNSIGNED_INT_16)
		.value("UNSIGNED_INT_32", cuda::Array::UNSIGNED_INT_32)
		.value("SIGNED_INT_8", cuda::Array::SIGNED_INT_8)
		.value("SIGNED_INT_16", cuda::Array::SIGNED_INT_16)
		.value("SIGNED_INT_32", cuda::Array::SIGNED_INT_32)
		.value("HALF", cuda::Array::HALF)
		.value("FLOAT", cuda::Array::FLOAT);
		
	py::class_<cuda::Array, boost::noncopyable>("Array", py::init<unsigned int, unsigned int, cuda::Array::Format, unsigned int>())
		.add_property("width", &cuda::Array::width)
		.add_property("height", &cuda::Array::height)
		.add_property("format", &cuda::Array::format)
		.add_property("channels", &cuda::Array::channels)
		.add_property("elementSize", &cuda::Array::elementSize)
		.add_property("pitch", &cuda::Array::pitch)
		.add_property("size", &cuda::Array::size);

	py::class_<cuda::DevicePtr>("DevicePtr", py::no_init);
	
	py::def("malloc", &cuda::malloc);
	py::def("malloc2D", &cuda::malloc2D);
	py::def("free", &cuda::free);
	
	py::def("memcpyAtoA",
		static_cast<void (*)(const cuda::Array&, unsigned int, const cuda::Array&, unsigned int, unsigned int)>(&cuda::memcpy));
		
	py::def("memcpyDtoA",
		static_cast<void (*)(const cuda::Array &, unsigned int , const cuda::DevicePtr &, unsigned int)>(&cuda::memcpy));

	py::def("memcpyAtoD",
		static_cast<void (*)(const cuda::DevicePtr &, const cuda::Array &, unsigned int , unsigned int )>(&cuda::memcpy));
		
	py::def("memcpyDtoD",
		static_cast<void (*)(const cuda::DevicePtr& , const cuda::DevicePtr& , unsigned int )>(&cuda::memcpy));

	py::def("memcpyAtoH", &memcpyAtoH);
	py::def("memcpyAtoHAsync", &memcpyAtoHAsync);
	py::def("memcpyHtoA", &memcpyHtoA);
	py::def("memcpyHtoAAsync", &memcpyHtoAAsync);
	py::def("memcpyHtoD", &memcpyHtoD);
	py::def("memcpyHtoDAsync", &memcpyHtoDAsync);
	py::def("memcpyDtoH", &memcpyDtoH);
	py::def("memcpyDtoHAsync", &memcpyDtoHAsync);
	
	py::class_<cuda::Memcpy2D>("Memcpy2D", py::init<>())
		.def(py::init<unsigned int, unsigned int>())
		.def("sourceH", sourceHost)
		.def("sourceD",
			static_cast<cuda::Memcpy2D& (cuda::Memcpy2D::*)(const cuda::DevicePtr&, unsigned int)>(&cuda::Memcpy2D::source),
			py::return_internal_reference<>())
		.def("sourceA",
			static_cast<cuda::Memcpy2D& (cuda::Memcpy2D::*)(const cuda::Array&)>(&cuda::Memcpy2D::source),
			py::return_internal_reference<>())
		.def("destinationH", destinationHost)
		.def("destinationD",
			static_cast<cuda::Memcpy2D& (cuda::Memcpy2D::*)(const cuda::DevicePtr&, unsigned int)>(&cuda::Memcpy2D::destination),
			py::return_internal_reference<>())
		.def("destinationA",
			static_cast<cuda::Memcpy2D& (cuda::Memcpy2D::*)(const cuda::Array&)>(&cuda::Memcpy2D::destination),
			py::return_internal_reference<>())
		.def("sourcePos", &cuda::Memcpy2D::sourcePos,
			py::return_internal_reference<>())
		.def("destinationPos", &cuda::Memcpy2D::destinationPos,
			py::return_internal_reference<>())
		.def("size", &cuda::Memcpy2D::size,
			py::return_internal_reference<>())
		.def("copy",
			static_cast<void (cuda::Memcpy2D::*)() const>(&cuda::Memcpy2D::copy))
		.def("copyAsync",
			static_cast<void (cuda::Memcpy2D::*)(const cuda::Stream&) const>(&cuda::Memcpy2D::copy))
		.def("copyUnaligned", &cuda::Memcpy2D::copyUnaligned);

	py::class_<cuda::TextureReference, boost::noncopyable>("TextureReference", py::init<cuda::Module&, const char *>())
		.def("bindD",
			static_cast<unsigned int (cuda::TextureReference::*)(const cuda::DevicePtr&, int) const>(&cuda::TextureReference::bind))
		.def("bindA",
			static_cast<void (cuda::TextureReference::*)(const cuda::Array&) const>(&cuda::TextureReference::bind));

	py::def("memset8", &cuda::memset8);
	py::def("memset16", &cuda::memset16);
	py::def("memset32", &cuda::memset32);
}



