INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
ADD_LIBRARY(cudamm STATIC
	array.cpp
	cuda.cpp
	error.cpp
	function.cpp
	module.cpp
	texturereference.cpp
	event.cpp
	stream.cpp
	deviceptr.cpp
	memcpy2d.cpp)
