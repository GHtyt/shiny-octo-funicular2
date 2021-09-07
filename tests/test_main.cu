#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h> 
#include <curand_kernel.h>

#include <stdio.h>
#include <time.h>

#include "host_device_vector.h"


__global__ void setup_kernel2(curandStatePhilox4_32_10_t* state, int height, long clock_for_rand)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id<0 || id>height)
    {
        return;
    }
    curand_init(clock_for_rand, id, 0, &state[id]);
}

__global__ void generate_kernel2(curandStatePhilox4_32_10_t* state, int w, int h, unsigned int* result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int count = 0;
    unsigned int x;
    curandStatePhilox4_32_10_t localState = state[id];
    for (int i = 0; i < w; i++) {
        x = curand(&localState);
        result[id * w + i] = id+1;
    }
    state[id] = localState;
}


int k()
{
    const int width = 5;
    const int height = 10;
    unsigned int random_array[2 * width * height];
    for (int i = 0; i < 2 * width * height; i++)
    {
        random_array[i] = 0;
    }

    //error status
    cudaError_t cuda_status;

    //only chose one GPU
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-Capable GPU installed?");
        return 0;
    }

    unsigned int* dev_random_array;
    curandState* dev_states;

    curandStatePhilox4_32_10_t* devPHILOXStates;
    


    //allocate memory on the GPU
    cuda_status = cudaMalloc((void**)&dev_random_array, 2 * sizeof(float) * width * height);
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "dev_random_array cudaMalloc Failed");
        exit(EXIT_FAILURE);
    }
    //cuda_status = cudaMalloc((void**)&dev_states, sizeof(curandState) * array_size_width * array_size_height);
    cudaMalloc((void**)&devPHILOXStates, sizeof(curandStatePhilox4_32_10_t) * width);
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "dev_states cudaMalloc Failed");
        exit(EXIT_FAILURE);
    }

    long clock_for_rand = clock();




    dim3 threads(16, 1);
    dim3 grid((width + threads.x - 1) / threads.x, 1);

    setup_kernel2 << <height, 1 >>> (devPHILOXStates, height, clock_for_rand);

    printf("The first time \n");
    {
        generate_kernel2 <<<height, 1 >>> (devPHILOXStates, width, height, dev_random_array);

        //copy out the result
        cuda_status = cudaMemcpy(random_array, dev_random_array, 2 * sizeof(unsigned int) * width * height, cudaMemcpyDeviceToHost);//dev_depthMap

        for (int i = 0; i < 2 * width * height; i++)
        {
            printf("%d\n", random_array[i]);
        }
    }

    //free
    cudaFree(dev_random_array);
    //cudaFree(dev_states);
    cudaFree(devPHILOXStates);
    return 0;
}
