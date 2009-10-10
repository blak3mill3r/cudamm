extern "C"
{

texture<float, 2, cudaReadModeElementType> inputTexture;
texture<float, 2, cudaReadModeElementType> inputTexture2;

__global__ void box_filter(float *out, int width, int height, int pitch)
{
	if(threadIdx.x >= width || threadIdx.y >= height)
	{
		return;
	}

	float val = 0.0;
	for(int i = -1; i <= 1; ++i)
	{
		for(int j = -1; j <= 1; ++j) val += tex2D(inputTexture, threadIdx.x + i, threadIdx.y + j);
	}

	out[threadIdx.y * pitch + threadIdx.x] = val / 9.0;
}

__global__ void difference(float *out, int width, int height, int pitch)
{
	if(threadIdx.x >= width || threadIdx.y >= height)
	{
		return;
	}

	out[threadIdx.y * pitch + threadIdx.x] = 
		max(tex2D(inputTexture2, threadIdx.x, threadIdx.y) - tex2D(inputTexture, threadIdx.x, threadIdx.y)
			, 0.0);
}

}


