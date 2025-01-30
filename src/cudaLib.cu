
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < size) {
		y[idx] = scale * x[idx] + y[idx];
	}
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	int size = vectorSize * sizeof(float);
	float * a_h, * b_h, * c_h;
	float * a_d, * c_d;

	a_h = (float *) malloc(size);
	b_h = (float *) malloc(size);
	c_h = (float *) malloc(size);

	if (a_h == NULL || b_h == NULL || c_h == NULL) {
		std::cout << "Unable to malloc memory ... Exiting!";
		return -1;
	}
	
	vectorInit(a_h, vectorSize);
	vectorInit(b_h, vectorSize);
	
	//	C = B
	std::memcpy(c_h, b_h, size);
	float scale = (float)(rand() % 100);
	
	#ifndef DEBUG_PRINT_DISABLE 
		std::cout << "\n Adding vectors : \n";
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", a_h[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", b_h[i]);
		}
		printf(" ... }\n");
	#endif
	
	//Part 1: Allocate device memory for a, b and c
	//Copy a and b to device memory
	cudaMalloc((void **) &a_d, size);
	cudaMalloc((void **) &c_d, size);
	
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice);
	
	//Part 2: Call Kernel - to launch a grid of threads
	//to perform the actual vector addition
	saxpy_gpu <<< ceil(vectorSize/256.0), 256>>> (a_d, c_d, scale, vectorSize);
	
	//Part 3: Copy C from the device memory
	//Free device vectors
	cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
	
	cudaFree(a_d);
	cudaFree(c_d);
	
	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", c_h[i]);
		}
		printf(" ... }\n");
	#endif
	
	int errorCount = verifyVector(a_h, b_h, c_h, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";
	
	free(a_h);
	free(b_h);
	free(c_h);
	
	//std::cout << "Lazy, you are!\n";
	//std::cout << "Write code, you must\n";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < pSumSize){
		// Setup state  
		curandState_t state;
		curand_init(12345ULL, idx, 0, &state);

        uint64_t localHits = 0;
        
        //	Main CPU Monte-Carlo Code
		for (uint64_t i = 0; i < sampleSize; ++i) { 
			float x = curand_uniform(&state);
            float y = curand_uniform(&state);
			
			if ( int(x * x + y * y) == 0 ) {
				++ localHits;
			}
		}
			
		// Write the local count to the global memory 
		pSums[idx] = localHits;
            
	}
	
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < reduceSize){
		uint64_t sum = 0;
		uint64_t start = idx * int(pSumSize/reduceSize);
		uint64_t end = start + int(pSumSize/reduceSize);
		
		for (uint64_t i = start; i < end; i++) { 
			sum += pSums[i]; 
		}
			
		totals[idx] = sum;
		
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	float approxPi = 0;

	//      Insert code here
	int sizeK1 = generateThreadCount*sizeof(uint64_t);
	int sizeK2 = reduceThreadCount*sizeof(uint64_t);
	
	//Part 1: Allocate Memory on the device
	uint64_t *pSums_d;
	cudaMalloc((void **) &pSums_d, sizeK1);
	
	uint64_t *totals_d;
    cudaMalloc((void **) &totals_d, sizeK2);
	
	//Part 2: Call Kernel - to launch a grid of threads
	generatePoints <<< ceil(generateThreadCount/256.0), 256>>> (pSums_d, generateThreadCount, sampleSize);
	cudaDeviceSynchronize();
	
	reduceCounts <<< ceil(reduceThreadCount/256.0), 256>>> (pSums_d, totals_d, generateThreadCount, reduceThreadCount);
	cudaDeviceSynchronize();
	
	//Part 3: Copy C from the device memory
	uint64_t * totals_h;
	totals_h = (uint64_t *) malloc(sizeK2);


	if (totals_h == NULL) {
		std::cout << "Unable to malloc memory ... Exiting!";
		return -1;
	}
	
	cudaMemcpy(totals_h, totals_d, sizeK2, cudaMemcpyDeviceToHost);
	
	// Final reduction on the host
	
	uint64_t totalHits = 0;     
	
	for (uint64_t i = 0; i < reduceThreadCount; i++) {         
		totalHits += totals_h[i];
	}     
	
	// Free device and host memory
	cudaFree(pSums_d); 
	cudaFree(totals_d);     
	
	// Calculate pi
	uint64_t totalPoints = generateThreadCount * sampleSize;     
	approxPi = 4.0f * (double)totalHits / totalPoints; 
	
	free(totals_h);
	
	//std::cout << "Sneaky, you are ...\n";
	//std::cout << "Compute pi, you must!\n";
	return approxPi;
}
