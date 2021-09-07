#include "span.h"
#include "common.h"
#include "gtest/gtest.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>


#define SPAN_ASSERT_TRUE(cond, status)          \
  if (!(cond)) {                                \
    *(status) = -1;                             \
  }

#define SPAN_ASSERT_FALSE(cond, status)         \
  if ((cond)) {                                 \
    *(status) = -1;                             \
  }


namespace common {


    __global__ void TestFromOtherKernel(Span<float> span) {
        // don't get optimized out
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx >= span.size()) {
            return;
        }
    }
    // Test converting different T
    __global__ void TestFromOtherKernelConst(Span<float const, 16> span) {
        // don't get optimized out
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx >= span.size()) {
            return;
        }
    }
    
    TEST(GPUSpan, FromOther) {
        thrust::host_vector<float> h_vec(16);
        std::iota(h_vec.begin(), h_vec.end(), 0);

        thrust::device_vector<float> d_vec(h_vec.size());
        thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());
        // dynamic extent
        {
            Span<float> span(d_vec.data().get(), d_vec.size());
            TestFromOtherKernel << <1, 16 >> > (span);
        }
        {
            Span<float> span(d_vec.data().get(), d_vec.size());
            TestFromOtherKernelConst << <1, 16 >> > (span);
        }
        // static extent
        {
            Span<float, 16> span(d_vec.data().get(), d_vec.data().get() + 16);
            TestFromOtherKernel << <1, 16 >> > (span);
        }
        {
            Span<float, 16> span(d_vec.data().get(), d_vec.data().get() + 16);
            TestFromOtherKernelConst << <1, 16 >> > (span);
        }
    }
    /**/
    struct TestStatus {
    private:
        int* status_;

    public:
        TestStatus() {
            cuda_handler(cudaMalloc(&status_, sizeof(int)));
            int h_status = 1;
            cuda_handler(cudaMemcpy(status_, &h_status,
                sizeof(int), cudaMemcpyHostToDevice));
        }
        ~TestStatus() {
            cuda_handler(cudaFree(status_));
        }

        int Get() {
            int h_status;
            cuda_handler(cudaMemcpy(&h_status, status_,
                sizeof(int), cudaMemcpyDeviceToHost));
            return h_status;
        }

        int* Data() {
            return status_;
        }
    };



    struct TestTestStatus {
        int* status_;

        TestTestStatus(int* _status) : status_(_status) {}

        XGBOOST_DEVICE void operator()() {
            this->operator()(0);
        }
        XGBOOST_DEVICE void operator()(int _idx) {
            SPAN_ASSERT_TRUE(false, status_);
        }
    };

    struct TestAssignment {
        int* status_;

        TestAssignment(int* _status) : status_(_status) {}

        XGBOOST_DEVICE void operator()() {
            this->operator()(0);
        }
        XGBOOST_DEVICE void operator()(int _idx) {
            Span<float> s1;

            float arr[] = { 3, 4, 5 };

            Span<const float> s2 = arr;
            SPAN_ASSERT_TRUE(s2.size() == 3, status_);
            SPAN_ASSERT_TRUE(s2.data() == &arr[0], status_);

            s2 = s1;
            SPAN_ASSERT_TRUE(s2.empty(), status_);
        }
    };

    TEST(GPUSpan, Assignment) {
        cuda_handler(cudaSetDevice(0));
        TestStatus status;
        LaunchN(16, TestAssignment{ status.Data() });
        ASSERT_EQ(status.Get(), 1);
    }

    TEST(GPUSpan, TestStatus) {
        cuda_handler(cudaSetDevice(0));
        TestStatus status;
        LaunchN(16, TestTestStatus{ status.Data() });
        ASSERT_EQ(status.Get(), -1);
    }

    template <typename Iter>
    XGBOOST_DEVICE void InitializeRange(Iter _begin, Iter _end) {
        float j = 0;
        for (Iter i = _begin; i != _end; ++i, ++j) {
            *i = j;
        }
    }


    __global__ void TestFrontKernel(Span<float> _span) {
        _span.front();
    }

    __global__ void TestBackKernel(Span<float> _span) {
        _span.back();
    }

    TEST(GPUSpan, FrontBack) {
        cuda_handler(cudaSetDevice(0));

        Span<float> s;
        auto lambda_test_front = [=]() {
            // make sure the termination happens inside this test.
            try {
                TestFrontKernel << <1, 1 >> > (s);
                cuda_handler(cudaDeviceSynchronize());
                cuda_handler(cudaGetLastError());
            }
            catch (std::runtime_error const& e) {
                std::terminate();
            }
        };
        EXPECT_DEATH(lambda_test_front(), "");

        auto lambda_test_back = [=]() {
            try {
                TestBackKernel << <1, 1 >> > (s);
                cuda_handler(cudaDeviceSynchronize());
                cuda_handler(cudaGetLastError());
            }
            catch (std::runtime_error const& e) {
                std::terminate();
            }
        };
        EXPECT_DEATH(lambda_test_back(), "");
    }


    __global__ void TestSubspanDynamicKernel(Span<float> _span) {
        _span.subspan(16, 0);
    }
    __global__ void TestSubspanStaticKernel(Span<float> _span) {
        _span.subspan<16>();
    }
    TEST(GPUSpan, Subspan) {
        auto lambda_subspan_dynamic = []() {
            thrust::host_vector<float> h_vec(4);
            InitializeRange(h_vec.begin(), h_vec.end());

            thrust::device_vector<float> d_vec(h_vec.size());
            thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

            Span<float> span(d_vec.data().get(), d_vec.size());
            TestSubspanDynamicKernel << <1, 1 >> > (span);
        };
        testing::internal::CaptureStdout();
        EXPECT_DEATH(lambda_subspan_dynamic(), "");
        std::string output = testing::internal::GetCapturedStdout();

        auto lambda_subspan_static = []() {
            thrust::host_vector<float> h_vec(4);
            InitializeRange(h_vec.begin(), h_vec.end());

            thrust::device_vector<float> d_vec(h_vec.size());
            thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

            Span<float> span(d_vec.data().get(), d_vec.size());
            TestSubspanStaticKernel << <1, 1 >> > (span);
        };
        testing::internal::CaptureStdout();
        EXPECT_DEATH(lambda_subspan_static(), "");
        output = testing::internal::GetCapturedStdout();
    }
    

}

