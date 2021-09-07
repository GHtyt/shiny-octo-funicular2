#ifndef SOS_COMMON_H
#define SOS_COMMON_H

#include "base.h"
#include "log.h"

#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#include <device_launch_parameters.h>

#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>

#define cuda_handler(ans) ThrowOnCudaError((ans), __FILE__, __LINE__)


inline cudaError_t ThrowOnCudaError(cudaError_t code, const char *file,
                                    int line) {
    if (code != cudaSuccess) {
        LOG(SILENT) << thrust::system_error(code, thrust::cuda_category(),
                                          std::string{file} + ": " +  // NOLINT
                                          std::to_string(line)).what();
    }
    return code;
}

namespace common {

    template<typename T>
    inline std::string ToString(const T& data) {
        std::ostringstream os;
        os << data;
        return os.str();
    }

    template <typename T1, typename T2>
    XGBOOST_DEVICE T1 DivRoundUp(const T1 a, const T2 b) {
        return static_cast<T1>(std::ceil(static_cast<double>(a) / b));
    }


    class Range {
    public:
        using DifferenceType = int64_t;

        class Iterator {
            friend class Range;

        public:
            XGBOOST_DEVICE DifferenceType operator*() const { return i_; }
            XGBOOST_DEVICE const Iterator& operator++() {
                i_ += step_;
                return *this;
            }
            XGBOOST_DEVICE Iterator operator++(int) {
                Iterator res{ *this };
                i_ += step_;
                return res;
            }

            XGBOOST_DEVICE bool operator==(const Iterator& other) const {
                return i_ >= other.i_;
            }
            XGBOOST_DEVICE bool operator!=(const Iterator& other) const {
                return i_ < other.i_;
            }

            XGBOOST_DEVICE void Step(DifferenceType s) { step_ = s; }

        protected:
            XGBOOST_DEVICE explicit Iterator(DifferenceType start) : i_(start) {}
            XGBOOST_DEVICE explicit Iterator(DifferenceType start, DifferenceType step) :
                i_{ start }, step_{ step } {}

        private:
            int64_t i_;
            DifferenceType step_ = 1;
        };

        XGBOOST_DEVICE Iterator begin() const { return begin_; }  // NOLINT
        XGBOOST_DEVICE Iterator end() const { return end_; }      // NOLINT

        XGBOOST_DEVICE Range(DifferenceType begin, DifferenceType end)
            : begin_(begin), end_(end) {}
        XGBOOST_DEVICE Range(DifferenceType begin, DifferenceType end,
            DifferenceType step)
            : begin_(begin, step), end_(end) {}

        XGBOOST_DEVICE bool operator==(const Range& other) const {
            return *begin_ == *other.begin_ && *end_ == *other.end_;
        }
        XGBOOST_DEVICE bool operator!=(const Range& other) const {
            return !(*this == other);
        }

        XGBOOST_DEVICE void Step(DifferenceType s) { begin_.Step(s); }

    private:
        Iterator begin_;
        Iterator end_;
    };

    /*
    XGBOOST_DEV_INLINE void AtomicOrByte(unsigned int* __restrict__ buffer,
        size_t ibyte, unsigned char b) {
        atomicOr(&buffer[ibyte / sizeof(unsigned int)],
            static_cast<unsigned int>(b)
            << (ibyte % (sizeof(unsigned int)) * 8));
    }
    */

    template <typename T>
    __device__ Range GridStrideRange(T begin, T end) {
        begin += blockDim.x * blockIdx.x + threadIdx.x;
        Range r(begin, end);
        r.Step(gridDim.x * blockDim.x);
        return r;
    }

    template <typename T>
    __device__ Range BlockStrideRange(T begin, T end) {
        begin += threadIdx.x;
        Range r(begin, end);
        r.Step(blockDim.x);
        return r;
    }



    template <typename IterT, typename ValueT>
    __device__ void BlockFill(IterT begin, size_t n, ValueT value) {
        for (auto i : BlockStrideRange(static_cast<size_t>(0), n)) {
            begin[i] = value;
        }
    }



    template <typename L>
    __global__ void LaunchNKernel(size_t begin, size_t end, L lambda) {
        for (auto i : GridStrideRange(begin, end)) {
            lambda(i);
        }
    }
    template <typename L>
    __global__ void LaunchNKernel(int device_idx, size_t begin, size_t end,
        L lambda) {
        for (auto i : GridStrideRange(begin, end)) {
            lambda(i, device_idx);
        }
    }
    /**/
    class LaunchKernel {
        size_t shmem_size_;
        cudaStream_t stream_;

        dim3 grids_;
        dim3 blocks_;

    public:
        LaunchKernel(uint32_t _grids, uint32_t _blk, size_t _shmem = 0, cudaStream_t _s = nullptr) :
            grids_{ _grids, 1, 1 }, blocks_{ _blk, 1, 1 }, shmem_size_{ _shmem }, stream_{ _s } {}
        LaunchKernel(dim3 _grids, dim3 _blk, size_t _shmem = 0, cudaStream_t _s = nullptr) :
            grids_{ _grids }, blocks_{ _blk }, shmem_size_{ _shmem }, stream_{ _s } {}

        template <typename K, typename... Args>
        void operator()(K kernel, Args... args) {
            /*
            if (XGBOOST_EXPECT(grids_.x * grids_.y * grids_.z == 0, false)) {
                LOG(DEBUG) << "Skipping empty CUDA kernel.";
                return;
            }*/
            kernel << <grids_, blocks_, shmem_size_, stream_ >> > (args...);  // NOLINT
        }
    };

    template <int ITEMS_PER_THREAD = 8, int BLOCK_THREADS = 256, typename L>
    inline void LaunchN(size_t n, cudaStream_t stream, L lambda) {
        if (n == 0) {
            return;
        }
        const int GRID_SIZE =
            static_cast<int>(DivRoundUp(n, ITEMS_PER_THREAD * BLOCK_THREADS));
        LaunchNKernel << <GRID_SIZE, BLOCK_THREADS, 0, stream >> > (  // NOLINT
            static_cast<size_t>(0), n, lambda);
    }

    // Default stream version
    template <int ITEMS_PER_THREAD = 8, int BLOCK_THREADS = 256, typename L>
    inline void LaunchN(size_t n, L lambda) {
        LaunchN<ITEMS_PER_THREAD, BLOCK_THREADS>(n, nullptr, lambda);
    }
    
    template <typename Container>
    void Iota(Container array) {
        LaunchN(array.size(), [=] __device__(size_t i) { array[i] = i; });
    }
    

    template <size_t size>
    struct AtomicDispatcher;

    template <>
    struct AtomicDispatcher<sizeof(uint32_t)> {
        using Type = unsigned int;  // NOLINT
        static_assert(sizeof(Type) == sizeof(uint32_t), "Unsigned should be of size 32 bits.");
    };

    template <>
    struct AtomicDispatcher<sizeof(uint64_t)> {
        using Type = unsigned long long;  // NOLINT
        static_assert(sizeof(Type) == sizeof(uint64_t), "Unsigned long long should be of size 64 bits.");
    };



}

#endif