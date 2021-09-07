#ifndef SOS_RANDKERNEL_H_
#define SOS_RANDKERNEL_H_

#include <cinttypes>

#include "base.h"
#include "common.h"

#if defined(__CUDA__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h> 
#include <curand_kernel.h>
#endif 



__global__ void setup_kernel(curandStatePhilox4_32_10_t* state, int n, long clock_for_rand);

__global__ void generate_kernel(curandStatePhilox4_32_10_t* state, int n, int m, uint64_t* result);

namespace common {

	class RandEngine {
	public:
        RandEngine(int32_t n);
        ~RandEngine();
        void rand(int32_t n, uint64_t* result);

    private:
        int32_t kernel_num;
        curandStatePhilox4_32_10_t* devPHILOXStates;

	};

};



#endif