#include "randkernel.cuh"

common::RandEngine::RandEngine(int32_t n) : kernel_num(n) {
    cuda_handler(cudaMalloc((void**)&devPHILOXStates, sizeof(curandStatePhilox4_32_10_t) * kernel_num));

    long clock_for_rand = clock();
    setup_kernel <<<kernel_num, 1>>> (devPHILOXStates, kernel_num, clock_for_rand);
}


common::RandEngine::~RandEngine() {
    cuda_handler(cudaFree(devPHILOXStates));
}


void common::RandEngine::rand(int32_t n, uint64_t* result) {
    if (n < kernel_num) {
        generate_kernel <<<n, 1 >>> (devPHILOXStates, n, 1, result);
    }
    else {
        int32_t m = (n - 1) / kernel_num + 1;
        generate_kernel <<<kernel_num, 1 >>> (devPHILOXStates, n, m, result);
    }

}


__global__ void setup_kernel(curandStatePhilox4_32_10_t* state, int n, long clock_for_rand)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id<0 || id>n)
    {
        return;
    }
    curand_init(clock_for_rand, id, 0, &state[id]);
}



__global__ void generate_kernel(curandStatePhilox4_32_10_t* state, int n, int m, uint64_t* result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int count = 0;
    unsigned int x;
    curandStatePhilox4_32_10_t localState = state[id];
    for (int i = 0; i < m; ++i) {
        int idx = id * m + i;
        if (idx < n) {
            uint64_t x0 = curand(&localState); // &0x1;
            uint64_t x1 = curand(&localState); // &0x1;
            result[idx] = ((x1 << 32) | x0); // &0xffffffff;
        }
    }
    state[id] = localState;
}
