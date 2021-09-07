#include "bitfield.h"
#include "common.h"
#include <gtest/gtest.h>
#include <iostream>


template <typename T>
using device_vector = thrust::device_vector<T>;

__global__ void TestSetKernel(LBitField64 bits) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < bits.Size()) {
        bits.gpuSet(tid);
    }
}

TEST(BitField, GPUSet) {
    device_vector<LBitField64::value_type> storage;
    uint32_t constexpr kBits = 128;
    storage.resize(128);
    auto bits = LBitField64(common::ToSpan(storage));
    TestSetKernel << <1, kBits >> > (bits);

    std::vector<LBitField64::value_type> h_storage(storage.size());
    thrust::copy(storage.begin(), storage.end(), h_storage.begin());

    LBitField64 outputs{
      common::Span<LBitField64::value_type>{h_storage.data(),
                                         h_storage.data() + h_storage.size()} };
    for (size_t i = 0; i < kBits; ++i) {
        ASSERT_TRUE(outputs.Check(i));
    }
}

__global__ void TestOrKernel(LBitField64 lhs, LBitField64 rhs) {
    lhs |= rhs;
}

TEST(BitField, GPUAnd) {
    uint32_t constexpr kBits = 128;
    device_vector<LBitField64::value_type> lhs_storage(kBits);
    device_vector<LBitField64::value_type> rhs_storage(kBits);
    auto lhs = LBitField64(common::ToSpan(lhs_storage));
    auto rhs = LBitField64(common::ToSpan(rhs_storage));
    thrust::fill(lhs_storage.begin(), lhs_storage.end(), 0UL);
    thrust::fill(rhs_storage.begin(), rhs_storage.end(), ~static_cast<LBitField64::value_type>(0UL));
    TestOrKernel << <1, kBits >> > (lhs, rhs);

    std::vector<LBitField64::value_type> h_storage(lhs_storage.size());
    thrust::copy(lhs_storage.begin(), lhs_storage.end(), h_storage.begin());
    LBitField64 outputs{ {h_storage.data(), h_storage.data() + h_storage.size()} };
    for (size_t i = 0; i < kBits; ++i) {
        ASSERT_TRUE(outputs.Check(i));
    }
}


template <typename BitFieldT, typename VT = typename BitFieldT::value_type>
void TestBitFieldSet(typename BitFieldT::value_type res, size_t index, size_t true_bit) {
    using IndexT = typename common::Span<VT>::index_type;
    std::vector<VT> storage(4, 0);
    auto bits = BitFieldT({ storage.data(), static_cast<IndexT>(storage.size()) });

    //std::cout << storage[0] << storage[1] << std::endl;

    bits.cpuSet(true_bit);

    for (size_t i = 0; i < true_bit; ++i) {
        ASSERT_FALSE(bits.Check(i));
    }

    ASSERT_TRUE(bits.Check(true_bit));

    for (size_t i = true_bit + 1; i < storage.size() * BitFieldT::kValueSize; ++i) {
        ASSERT_FALSE(bits.Check(i));
    }
    ASSERT_EQ(storage[index], (1ULL << (62)));    // - res);
}

TEST(BitField, Set) {
    {
        TestBitFieldSet<LBitField64>(2, 2, 190);
    }
    {
        //TestBitFieldSet<RBitField8>(1 << 3, 2, 19);
    }
}

template <typename BitFieldT, typename VT = typename BitFieldT::value_type>
void TestBitFieldClear(size_t clear_bit) {
    using IndexT = typename common::Span<VT>::index_type;
    std::vector<VT> storage(4, 0);
    auto bits = BitFieldT({ storage.data(), static_cast<IndexT>(storage.size()) });

    bits.cpuSet(clear_bit);
    bits.cpuClear(clear_bit);

    ASSERT_FALSE(bits.Check(clear_bit));
}

TEST(BitField, Clear) {
    {
        TestBitFieldClear<LBitField64>(190);
    }
    {
        //TestBitFieldClear<RBitField8>(19);
    }
}

TEST(BitField, Cmp) {
    std::vector<uint64_t> v0{ 0, 1, 2 };
    std::vector<uint64_t> v1{ 0, 1, 3 };
    auto bits0 = LBitField64({ v0.data(), v0.size() });
    auto bits1 = LBitField64({ v1.data(), v1.size() });
    ASSERT_TRUE(bits0 < bits1);
    ASSERT_FALSE(bits0 > bits1);

}