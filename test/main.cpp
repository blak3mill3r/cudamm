#include <iostream>
#include <iterator>
#include <algorithm>

#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>

#include <cudamm/cuda.hpp>

namespace
{
	void printArray(float const *arr, size_t w, size_t h)
	{
		while(w--)
		{
			for(size_t i = 0; i < h; ++i) std::cout << *arr++ << ' ';
			std::cout << std::endl;
		}
	}
}

int main()
{
	const size_t width = 10, height = 10;
	const size_t inputSize = width * height * sizeof(float);
	const size_t inputPitch = width * sizeof(float);
	const float inputData[width * height] = {
		0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
		0, 0, 1, 0, 1, 1, 0, 1, 0, 0,
		0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
		0, 0, 1, 0, 1, 1, 0, 1, 0, 0,
		0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 1, 0, 0, 0, 0 };
	
	std::cout.precision(1);
	fixed(std::cout);
	
	std::cout << "Input:\n";
	printArray(inputData, width, height);
	std::cout << std::endl;
	
	float outputData[width * height];

	try
	{
		cuda::Cuda cudaCtx(0);
		cuda::Module mod("test.cubin");
		
		cuda::DeviceMemory2D omem(inputPitch, height, sizeof(float));
		cuda::Array array(width, height, cuda::Array::FLOAT, 1);

		array.upload2D(inputData);
		
		cuda::TextureReference texRef(mod, "inputTexture");
		texRef.bind(array);
		
		cuda::Stream stream;
		
		{
			cuda::Function kernel(mod, "box_filter");
		
			kernel.setParameter(0, omem.ptr());
			kernel.setParameter(4, static_cast<int>(width));
			kernel.setParameter(8, static_cast<int>(height));
			kernel.setParameter(12, static_cast<int>(omem.pitch() / sizeof(float)));
			kernel.setParameterSize(16);
		
			kernel.useTexture(texRef);
			kernel.setBlockShape(16, 16, 1);
		
			kernel.launch(1, 1, stream);
		}
		
		cuda::Array array2(width, height, cuda::Array::FLOAT, 1);
		cuda::Memcpy2D(inputPitch, height)
			.source(omem.ptr(), omem.pitch())
			.destination(array2)
			.copy(stream);

		cuda::TextureReference texRef2(mod, "inputTexture2");
		texRef2.bind(array2);
		
		{
			cuda::Function kernel(mod, "difference");
			
			kernel.setParameter(0, omem.ptr());
			kernel.setParameter(4, static_cast<int>(width));
			kernel.setParameter(8, static_cast<int>(height));
			kernel.setParameter(12, static_cast<int>(omem.pitch() / sizeof(float)));
			kernel.setParameterSize(16);
		
			kernel.useTexture(texRef);
			kernel.useTexture(texRef2);
			kernel.setBlockShape(16, 16, 1);
		
			kernel.launch(1, 1, stream);
		}
		
		if(!stream.query())
		{
			std::cout << "Waiting for kernel to finish" << std::endl;
			stream.synchronize();
		}
		
		omem.download2D(outputData);
		
		std::cout << "Output:\n";
		printArray(outputData, width, height);
		std::cout << std::endl;
			
	} catch(cuda::Exception const &e)
	{
		std::cerr << "Cuda exception: " << e.what() << std::endl;
	}
}


