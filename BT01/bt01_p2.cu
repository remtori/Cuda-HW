#include <stdio.h>
#include <stdint.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define CHECK(call)                                                \
	{                                                              \
		const cudaError_t error = call;                            \
		if (error != cudaSuccess)                                  \
		{                                                          \
			fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
			fprintf(stderr, "code: %d, reason: %s\n", error,       \
					cudaGetErrorString(error));                    \
			exit(EXIT_FAILURE);                                    \
		}                                                          \
	}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

void readPnm(char *fileName,
			 int &width, int &height, uchar3 *&pixels)
{
	FILE *f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);

	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 *pixels, int width, int height,
			  char *fileName)
{
	FILE *f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "P3\n%i\n%i\n255\n", width, height);

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);

	fclose(f);
}

__global__ void blurImgKernel(uchar3 *inPixels, int width, int height,
							  float *filter, int filterWidth,
							  uchar3 *outPixels)
{
	// TODO	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	float3 sum = make_float3(0, 0, 0);
	for (int i = 0; i < filterWidth; i++) 
	{
		for (int j = 0; j < filterWidth; j++) 
		{
			int aX = x + (i - filterWidth / 2);
			int aY = y + (j - filterWidth / 2);
			if (aX < 0) aX = 0;
			if (aX >= width) aX = width - 1;
			if (aY < 0) aY = 0;
			if (aY >= height) aY = height - 1;

			sum.x += inPixels[aX + aY * width].x * filter[i];
			sum.y += inPixels[aX + aY * width].y * filter[i];
			sum.z += inPixels[aX + aY * width].z * filter[i];
		}
	}

	outPixels[x + y * width].x = sum.x;
	outPixels[x + y * width].y = sum.y;
	outPixels[x + y * width].z = sum.z;
}

void blurImg(uchar3 *inPixels, int width, int height, float *filter, int filterWidth,
			 uchar3 *outPixels,
			 bool useDevice = false, dim3 blockSize = dim3(1, 1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
		// TODO
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				float3 sum = make_float3(0, 0, 0);
				for (int i = 0; i < filterWidth; i++) 
				{
					for (int j = 0; j < filterWidth; j++) 
					{
						int aX = x + (i - filterWidth / 2);
						int aY = y + (j - filterWidth / 2);
						if (aX < 0) aX = 0;
						if (aX >= width) aX = width - 1;
						if (aY < 0) aY = 0;
						if (aY >= height) aY = height - 1;

						sum.x += inPixels[aX + aY * width].x * filter[i];
						sum.y += inPixels[aX + aY * width].y * filter[i];
						sum.z += inPixels[aX + aY * width].z * filter[i];
					}
				}

				outPixels[x + y * width].x = sum.x;
				outPixels[x + y * width].y = sum.y;
				outPixels[x + y * width].z = sum.z;
			}
		}
	}
	else // Use device
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

		// TODO
		float *d_filter;
		uchar3 *d_in, *d_out;
		CHECK(cudaMalloc(&d_in, width * height * sizeof(uchar3)));
		CHECK(cudaMalloc(&d_out, width * height * sizeof(uchar3)));
		CHECK(cudaMalloc(&d_filter, filterWidth * sizeof(float)));

		// TODO: Copy data to device memories
		CHECK(cudaMemcpy(d_in, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_filter, filter, filterWidth * sizeof(float), cudaMemcpyHostToDevice));

		// TODO: Set grid size and call kernel (remember to check kernel error)
		dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
		blurImgKernel<<<gridSize, blockSize>>>(d_in, width, height, d_filter, filterWidth, d_out);

		// TODO: Copy result from device memories
		CHECK(cudaMemcpy(outPixels, d_out, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

		// TODO: Free device memories
		CHECK(cudaFree(d_in));
		CHECK(cudaFree(d_out));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n",
		   useDevice == true ? "use device" : "use host", time);
}

float computeError(uchar3 *a1, uchar3 *a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].x - (int)a2[i].x);
		err += abs((int)a1[i].y - (int)a2[i].y);
		err += abs((int)a1[i].z - (int)a2[i].z);
	}
	err /= (n * 3);
	return err;
}

char *concatStr(const char *s1, const char *s2)
{
	char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

int main(int argc, char **argv)
{
	if (argc != 4 && argc != 6)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	// Read input image file
	int width, height;
	uchar3 *inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// Read correct output image file
	int correctWidth, correctHeight;
	uchar3 *correctOutPixels;
	readPnm(argv[3], correctWidth, correctHeight, correctOutPixels);
	if (correctWidth != width || correctHeight != height)
	{
		printf("The shape of the correct output image is invalid\n");
		return EXIT_FAILURE;
	}

	// Set up a simple filter with blurring effect
	int filterWidth = 9;
	float *filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
	for (int filterR = 0; filterR < filterWidth; filterR++)
	{
		for (int filterC = 0; filterC < filterWidth; filterC++)
		{
			filter[filterR * filterWidth + filterC] = 1. / (filterWidth * filterWidth);
		}
	}

	// Blur input image using host
	uchar3 *hostOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	blurImg(inPixels, width, height, filter, filterWidth, hostOutPixels);

	// Compute mean absolute error between host result and correct result
	float hostErr = computeError(hostOutPixels, correctOutPixels, width * height);
	printf("Error: %f\n\n", hostErr);

	// Blur input image using device
	uchar3 *deviceOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	dim3 blockSize(32, 32); // Default
	if (argc == 6)
	{
		blockSize.x = atoi(argv[4]);
		blockSize.y = atoi(argv[5]);
	}
	blurImg(inPixels, width, height, filter, filterWidth, deviceOutPixels, true, blockSize);

	// Compute mean absolute error between device result and correct result
	float deviceErr = computeError(deviceOutPixels, correctOutPixels, width * height);
	printf("Error: %f\n\n", deviceErr);

	// Write results to files
	char *outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(hostOutPixels, width, height, concatStr(outFileNameBase, "_host.pnm"));
	writePnm(deviceOutPixels, width, height, concatStr(outFileNameBase, "_device.pnm"));

	// Free memories
	free(inPixels);
	free(correctOutPixels);
	free(hostOutPixels);
	free(deviceOutPixels);
	free(filter);
}
