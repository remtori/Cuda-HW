// Last update: 24/12/2020
#include <stdio.h>
#include <stdint.h>

#ifdef NZ
    #define PRINT_ARRAY printArray
#else
    #define PRINT_ARRAY(...)
#endif

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
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
        cudaEventSynchronize(start);
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

// Sequential Radix Sort
// "const uint32_t * in" means: the memory region pointed by "in" is read-only
void sortByHost(const uint32_t * in, int n,
                uint32_t * out)
{
    int * bits = (int *)malloc(n * sizeof(int));
    int * nOnesBefore = (int *)malloc(n * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        for (int i = 0; i < n; i++)
            bits[i] = (src[i] >> bitIdx) & 1;

        // Compute nOnesBefore
        nOnesBefore[0] = 0;
        for (int i = 1; i < n; i++)
            nOnesBefore[i] = nOnesBefore[i-1] + bits[i-1];

        // Compute rank and write to dst
        int nZeros = n - nOnesBefore[n-1] - bits[n-1];

#ifdef NZ
        printf("bitIdx = %d, nZeros = %d\n", bitIdx, nZeros);
#endif

        for (int i = 0; i < n; i++)
        {
            int rank;
            if (bits[i] == 0)
                rank = i - nOnesBefore[i];
            else
                rank = nZeros + nOnesBefore[i];
            dst[rank] = src[i];
        }

        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Does out array contain results?
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memory
    free(originalSrc);
    free(bits);
    free(nOnesBefore);
}

__device__ int bCount = 0;
volatile __device__ int bCount1 = 0;

__global__ void deviceRadixExtractBits(const uint32_t* in, int n, uint32_t* out, int bitIdx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = (in[i] >> bitIdx) & 1;
}

__global__ void deviceRadixScan(const uint32_t* in, int n, uint32_t* out, volatile uint32_t* bSums)
{    
    // Get blockIdx
    __shared__ int bi;
    if (threadIdx.x == 0)
        bi = atomicAdd(&bCount, 1);        
    __syncthreads();

    extern __shared__ int s_nOnesBefore[];
    
    int i1 = bi * 2 * blockDim.x + threadIdx.x;
    int i2 = i1 + blockDim.x;

    if (i1 >= n && i2 >= n)
        return;
    
    // Copy data GMEM -> SMEM
    if (0 < i1 && i1 < n)
        s_nOnesBefore[threadIdx.x] = in[i1 - 1];
    else 
        s_nOnesBefore[threadIdx.x] = 0;

    if (0 < i2 && i2 < n)    
        s_nOnesBefore[threadIdx.x + blockDim.x] = in[i2 - 1];
    else 
        s_nOnesBefore[threadIdx.x + blockDim.x] = 0;

    __syncthreads();

    // Local scan
    for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
	{
		int idx = (threadIdx.x + 1) * 2 * stride - 1;
		if (idx < 2 * blockDim.x)
			s_nOnesBefore[idx] += s_nOnesBefore[idx - stride];
		__syncthreads();
    }
    
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		int idx = (threadIdx.x + 1) * 2 * stride - 1 + stride;
		if (idx < 2 * blockDim.x)
			s_nOnesBefore[idx] += s_nOnesBefore[idx - stride];
		__syncthreads();
	}
    
    // Wait for prev block
    if (threadIdx.x == 0)
    {        
        bSums[bi] = s_nOnesBefore[2 * blockDim.x - 1];

        if (bi > 0)
        {
            while (bCount1 < bi) {}
            bSums[bi] += bSums[bi - 1];
            __threadfence();
        }
        bCount1 += 1;
    }
    __syncthreads();

    if (i1 < n)
        out[i1] = s_nOnesBefore[threadIdx.x];

    if (i2 < n)
        out[i2] = s_nOnesBefore[threadIdx.x + blockDim.x];
    
    if (bi > 0)
    {
        if (i1 < n) out[i1] += bSums[bi - 1];
        if (i2 < n) out[i2] += bSums[bi - 1];
    }
}

__global__ void deviceRadixRank(const uint32_t * in, const uint32_t* bits, int n, uint32_t* out, uint32_t* nOnesBefore, int bitIdx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nZeros = n - nOnesBefore[n - 1] - bits[n - 1];

    if (i >= n)
        return;

    int rank;
    if (bits[i] == 0)
        rank = i - nOnesBefore[i];
    else
        rank = nZeros + nOnesBefore[i];

    out[rank] = in[i];
}

// Parallel Radix Sort
void radixSortDevice(const uint32_t * in, int n, uint32_t * out, int blockSize)
{    
    uint32_t * d_in, * d_out, * d_bits, * d_bSums, * d_nOnesBefore;

    int blockDataSize = blockSize * 2;
    int scanGridSize = (n - 1) / blockDataSize + 1;
    int commonGridSize = (n - 1) / blockSize + 1;
    size_t sMem = blockDataSize * sizeof(int);
    size_t nBytes = n * sizeof(uint32_t);

    CHECK(cudaMalloc(&d_in, nBytes));
    CHECK(cudaMalloc(&d_out, nBytes));
    CHECK(cudaMalloc(&d_bits, nBytes));
    CHECK(cudaMalloc(&d_bSums, scanGridSize * sizeof(int)));
    CHECK(cudaMalloc(&d_nOnesBefore, nBytes));

    CHECK(cudaMemcpy(d_in, in, nBytes, cudaMemcpyHostToDevice));    

    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Reset bCount, bCount1
        int zero = 0;
        CHECK(cudaMemcpyToSymbol(bCount, &zero, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(bCount1, &zero, sizeof(int)));

        deviceRadixExtractBits<<<commonGridSize, blockSize>>>(d_in, n, d_bits, bitIdx);
        cudaDeviceSynchronize();        
        CHECK(cudaGetLastError());

        deviceRadixScan<<<scanGridSize, blockSize, sMem>>>(d_bits, n, d_nOnesBefore, d_bSums);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

#ifdef NZ
        uint32_t nOnesBefore;
        CHECK(cudaMemcpy(&nOnesBefore, &d_nOnesBefore[n - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t nZeros = n - nOnesBefore - ((in[n - 1] >> bitIdx) & 1);
        printf("bitIdx = %d, nZeros = %d\n", bitIdx, nZeros);
#endif

        deviceRadixRank<<<commonGridSize, blockSize>>>(d_in, d_bits, n, d_out, d_nOnesBefore, bitIdx);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        uint32_t* tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    CHECK(cudaMemcpy(out, d_out, nBytes, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_bits));
    CHECK(cudaFree(d_bSums));
    CHECK(cudaFree(d_nOnesBefore));
}

// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else // use device
    {
    	printf("\nRadix Sort by device\n");
        radixSortDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
#ifdef NZ
    int n = 10; // For test by eye
#else
    int n = (1 << 24) + 1;
#endif
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        in[i] = rand();
    }
    PRINT_ARRAY(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sort(in, n, correctOut);
    PRINT_ARRAY(correctOut, n); // For test by eye
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    PRINT_ARRAY(out, n); // For test by eye
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
