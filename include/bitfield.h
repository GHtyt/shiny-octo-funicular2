#ifndef SOS_BITFIELD_H_
#define SOS_BITFIELD_H_


#include "base.h"
#include "common.h"
#include "span.h"
#include "host_device_vector.h"
#include "log.h"

#include <bitset>

#if defined(__CUDA__)
#include <cuda_runtime.h>
#endif  


#if defined(__CUDA__)

using BitFieldAtomicType = unsigned long long; 

__forceinline__ __device__ BitFieldAtomicType AtomicOr(BitFieldAtomicType* address, BitFieldAtomicType val);

__forceinline__ __device__ BitFieldAtomicType AtomicAnd(BitFieldAtomicType* address, BitFieldAtomicType val);

#endif




template <typename VT, typename Direction, bool IsConst = false>
struct BitFieldContainer {
    using value_type = std::conditional_t<IsConst, VT const, VT>;  // NOLINT
    using pointer = value_type*;  // NOLINT

    static value_type constexpr kValueSize = sizeof(value_type) * 8;
    static value_type constexpr kOne = 1;  // force correct type.

    struct Pos {
        std::remove_const_t<value_type> int_pos{ 0 };
        std::remove_const_t<value_type> bit_pos{ 0 };
    };

private:
    common::Span<value_type> bits_;
    static_assert(!std::is_signed<VT>::value, "Must use unsiged type as underlying storage.");

public:
    XGBOOST_DEVICE static Pos ToBitPos(value_type pos) {
        Pos pos_v;
        if (pos == 0) {
            return pos_v;
        }
        pos_v.int_pos = pos / kValueSize;
        pos_v.bit_pos = pos % kValueSize;
        return pos_v;
    }

public:
    BitFieldContainer() = default;
    XGBOOST_DEVICE explicit BitFieldContainer(common::Span<value_type> bits) : bits_{ bits } {}
    XGBOOST_DEVICE BitFieldContainer(BitFieldContainer const& other) : bits_{ other.bits_ } {}
    BitFieldContainer& operator=(BitFieldContainer const& that) = default;
    BitFieldContainer& operator=(BitFieldContainer&& that) = default;

    XGBOOST_DEVICE common::Span<value_type>       Bits() { return bits_; }
    XGBOOST_DEVICE common::Span<value_type const> Bits() const { return bits_; }

    /*\brief Compute the size of needed memory allocation.  The returned value is in terms
     *       of number of elements with `BitFieldContainer::value_type'.
     */
    XGBOOST_DEVICE static size_t ComputeStorageSize(size_t size) {
        return common::DivRoundUp(size, kValueSize);
    }
    __device__ BitFieldContainer& operator|=(BitFieldContainer const& rhs) {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        size_t min_size = min(bits_.size(), rhs.bits_.size());
        if (tid < min_size) {
            bits_[tid] |= rhs.bits_[tid];
        }
        return *this;
    }

    __device__ BitFieldContainer& operator&=(BitFieldContainer const& rhs) {
        size_t min_size = min(bits_.size(), rhs.bits_.size());
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < min_size) {
            bits_[tid] &= rhs.bits_[tid];
        }
        return *this;
    }

    __device__ auto gpuSet(value_type pos) {
        Pos pos_v = Direction::Shift(ToBitPos(pos));
        value_type& value = bits_[pos_v.int_pos];
        value_type set_bit = kOne << pos_v.bit_pos;
        //std::cout << "here0" << std::endl;
        using Type = typename common::AtomicDispatcher<sizeof(value_type)>::Type;
        //std::cout << "here1" << std::endl;
        atomicOr(reinterpret_cast<Type*>(&value), set_bit);
        //std::cout << "here2" << std::endl;
    }
    __device__ void gpuClear(value_type pos) {
        Pos pos_v = Direction::Shift(ToBitPos(pos));
        value_type& value = bits_[pos_v.int_pos];
        value_type clear_bit = ~(kOne << pos_v.bit_pos);
        using Type = typename common::AtomicDispatcher<sizeof(value_type)>::Type;
        atomicAnd(reinterpret_cast<Type*>(&value), clear_bit);
    }

    void cpuSet(value_type pos) {
        Pos pos_v = Direction::Shift(ToBitPos(pos));
        value_type& value = bits_[pos_v.int_pos];
        value_type set_bit = kOne << pos_v.bit_pos;
        value |= set_bit;
    }
    void cpuClear(value_type pos) {
        Pos pos_v = Direction::Shift(ToBitPos(pos));
        value_type& value = bits_[pos_v.int_pos];
        value_type clear_bit = ~(kOne << pos_v.bit_pos);
        value &= clear_bit;
    }

    XGBOOST_DEVICE bool Check(Pos pos_v) const {
        pos_v = Direction::Shift(pos_v);
        //SPAN_LT(pos_v.int_pos, bits_.size());
        value_type const value = bits_[pos_v.int_pos];
        value_type const test_bit = kOne << pos_v.bit_pos;
        value_type result = test_bit & value;
        return static_cast<bool>(result);
    }
    XGBOOST_DEVICE bool Check(value_type pos) const {
        Pos pos_v = ToBitPos(pos);
        return Check(pos_v);
    }

    XGBOOST_DEVICE size_t Size() const { return kValueSize * bits_.size(); }
    XGBOOST_DEVICE size_t size() const { return bits_.size(); }

    XGBOOST_DEVICE pointer Data() const { return bits_.data(); }


    bool operator==(const BitFieldContainer<VT, Direction, IsConst> b) const {
        for (int i = 0; i < bits_.size(); ++i)
            if (bits_[i] != b.Data()[i])
                return 0;
        return 1;
    }

    bool operator<(const BitFieldContainer<VT, Direction, IsConst> b) const {
        for (int i = 0; i < bits_.size(); ++i)
            if (bits_[i] < b.Data()[i])
                return 1;
            else if (bits_[i] > b.Data()[i])
                return 0;
        return 0;
    }


    bool operator>(const BitFieldContainer<VT, Direction, IsConst> b) const {
        for (int i = 0; i < bits_.size(); ++i)
            if (bits_[i] < b.Data()[i])
                return 0;
            else if (bits_[i] > b.Data()[i])
                return 1;
        return 0;
    }

    inline friend std::ostream&
        operator<<(std::ostream& os, BitFieldContainer<VT, Direction, IsConst> field) {
        os << "Bits " << "storage size: " << field.bits_.size() << "\n";
        for (typename common::Span<value_type>::index_type i = 0; i < field.bits_.size(); ++i) {
            std::bitset<BitFieldContainer<VT, Direction, IsConst>::kValueSize> bset(field.bits_[i]);
            os << bset << "\n";
        }
        return os;
    }
};

template <typename VT, bool IsConst = false>
struct LBitsPolicy : public BitFieldContainer<VT, LBitsPolicy<VT, IsConst>, IsConst> {
    using Container = BitFieldContainer<VT, LBitsPolicy<VT, IsConst>, IsConst>;
    using Pos = typename Container::Pos;
    using value_type = typename Container::value_type;

    XGBOOST_DEVICE static Pos Shift(Pos pos) {
        //LOG(INFO) << "kval: " << Container::kValueSize << " " << Container::kValueSize;
        pos.bit_pos = Container::kValueSize - pos.bit_pos - Container::kOne;
        return pos;
    }
};
template <typename VT>
struct RBitsPolicy : public BitFieldContainer<VT, RBitsPolicy<VT>> {
    using Container = BitFieldContainer<VT, RBitsPolicy<VT>>;
    using Pos = typename Container::Pos;
    using value_type = typename Container::value_type;  // NOLINT

    XGBOOST_DEVICE static Pos Shift(Pos pos) {
        return pos;
    }
};

using LBitField64 = BitFieldContainer<uint64_t, RBitsPolicy<uint64_t>>;

//using RBitField8 = BitFieldContainer<uint8_t, RBitsPolicy<uint32_t>>;
/*
template <typename VT, typename Direction>
BitFieldContainer<VT, Direction> ToHostBit(HostDeviceVector<VT> vec) {
    auto sp = vec.ConstHostSpan();
    auto bits = BitFieldContainer<VT, Direction>({ sp.data(), sp.size() });
    return bits;
}*/




#endif