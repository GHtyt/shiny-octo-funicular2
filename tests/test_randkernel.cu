#include <gtest/gtest.h>
#include <iostream>

#include "randkernel.cuh"
#include "common.h"
#include "host_device_vector.h"



namespace common {

    TEST(PRNG, RANDN)
    {
        const int32_t length = 33;
        RandEngine randengine = RandEngine(7);
        uint64_t random_array[length];
        for (int i = 0; i < length; i++)
        {
            random_array[i] = 0;
        }

        cuda_handler(cudaSetDevice(0));

        uint64_t* dev_random_array;
        cuda_handler(cudaMalloc((void**)&dev_random_array, length * sizeof(uint64_t)));

        randengine.rand(length, dev_random_array);

        cuda_handler(cudaMemcpy(random_array, dev_random_array, length * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        for (int i = 0; i < length; i++)
        {
            //std::cout << random_array[i] << std::endl;
        }
    }

    TEST(PRNG, RANDVector)
    {
        const int32_t length = 33;
        RandEngine randengine = RandEngine(7);

        HostDeviceVector<uint64_t> v;
        v.Resize(length);

        randengine.rand(length, v.DevicePointer());

        auto h_v = v.HostVector();

        for (std::vector<uint64_t>::iterator i = h_v.begin(); i < h_v.end(); ++i)
        {
            //std::cout << *i << std::endl;
        }

    }
}
