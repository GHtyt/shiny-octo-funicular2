#include "bitfield.h"


__forceinline__ __device__ BitFieldAtomicType AtomicOr(BitFieldAtomicType* address,
    BitFieldAtomicType val) {
    BitFieldAtomicType old = *address, assumed;  // NOLINT
    do {
        assumed = old;
        old = atomicCAS(address, assumed, val | assumed);
    } while (assumed != old);

    return old;
}

__forceinline__ __device__ BitFieldAtomicType AtomicAnd(BitFieldAtomicType* address,
    BitFieldAtomicType val) {
    BitFieldAtomicType old = *address, assumed;  // NOLINT
    do {
        assumed = old;
        old = atomicCAS(address, assumed, val & assumed);
    } while (assumed != old);

    return old;
}