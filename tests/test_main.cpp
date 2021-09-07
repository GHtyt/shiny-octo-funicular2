#include <gtest/gtest.h>
#include <omp.h>


void omp_test() {
const int sample_size = 1000000;
    unsigned int a[sample_size], b[sample_size];

    for (int i = 0; i < sample_size; ++i) {
        a[i] = i;
    }


    double start = omp_get_wtime();

    omp_set_num_threads(1024);



#pragma omp parallel for
    for (int i = 0; i < sample_size; ++i) {
        b[i] = ((a[i] >> 5) * (a[i] & 0x1f)) & 0x10;
    }

    double end = omp_get_wtime();

    double t1 = end - start;
    std::cout << t1 << std::endl;

    for (int i = 0; i < sample_size; ++i) {
        b[i] = ((a[i] >> 5) * (a[i] & 0x1f)) & 0x10;
    }

    double t2 = omp_get_wtime()-end;
    std::cout << t2 << std::endl;

    std::cout << t2 / t1 << std::endl;
    
}

int q(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int i = 0, j = 0;
    std::cout << i++ << ++j << std::endl;
    return RUN_ALL_TESTS();
    
    

    //system("pause");
}