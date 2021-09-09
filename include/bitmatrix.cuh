#ifndef SOS_BITMATRIX_H_
#define SOS_BITMATRIX_H_

#include "bitfield.h"
#include "host_device_vector.h"
#include "randkernel.cuh"


#include <memory>

#include <omp.h>

//#include <cmath>


#define RAND_KERNEL_SIZE 1024
#define CAL_KERNEL_SIZE 1024

__global__ void cal_kernel(LBitField64 label, int size, int LineSize, common::Span<uint64_t> data_v);

__global__ void mask_kernel(common::Span<uint64_t> data_v, int size, int LineSize, common::Span<uint64_t> mask0, common::Span<uint64_t> mask1);

__global__ void construct(uint64_t* d, int size);

template<int32_t LineSize, typename Connection>
class BitMatrix {
public:
	constexpr BitMatrix() noexcept = default;
	explicit BitMatrix(int _size);
	explicit BitMatrix(HostDeviceVector<uint64_t>* _data);

	explicit BitMatrix(BitMatrix* bm);
	explicit BitMatrix(std::shared_ptr<BitMatrix> bm);
	//explicit BitMatrix (const BitMatrix& bm);

	~BitMatrix();


	void Resize(int32_t s);

	void sampling();
	void gpuCalLabel();
	void cpuCalLabel();
	void gpuMask(LBitField64 mask0, LBitField64 mask1);
	void cpuMask(LBitField64 mask0, LBitField64 mask1);

	common::Span<uint64_t> HostLabelSpan();
	common::Span<uint64_t> HostdataSpan();
	HostDeviceVector<uint64_t>* Data();
	HostDeviceVector<uint64_t>* Label_v();
	LBitField64 DeviceLabel();
	LBitField64 Label();
	int32_t Size();

private:
	HostDeviceVector<uint64_t>* data;
	HostDeviceVector<uint64_t>* label_v;
	LBitField64 label;
	int32_t size;

public:

	common::RandEngine* randengine;

};

struct MultipyPolicy : public BitMatrix<1, MultipyPolicy> {
	XGBOOST_DEVICE static uint64_t get(uint64_t data, int i) {
		return (data >> i) & 0x1;
	}

	XGBOOST_DEVICE static int32_t Cal(common::Span<uint64_t> span) {
		uint64_t d = (*span.begin());
		//return ((d >> 32) * (d & 0xffffffff)) & 0x1;
		//return get(d, 0) ^ get(d, 1) ^ get(d, 2) ^ get(d, 3);
		//return get(((d >> 32) & 0xfffffffff) * (d & 0xfffffffff), 31);
		return get(((d >> 4) & 0xf) + (d & 0xf), 4);
		//return 1;

	}
};


struct RiscVPolicy : public BitMatrix<1, RiscVPolicy> {
	XGBOOST_DEVICE static uint64_t get(uint64_t data, int i) {
		return (data >> i) & 0x1;
	}

	XGBOOST_DEVICE static int32_t Cal(common::Span<uint64_t> span) {
		uint64_t d0 = (*span.begin());
		uint64_t d1 = (*(span.begin() + 1));
		//return ((d >> 32) * (d & 0xffffffff)) & 0x1;
		//return get(d, 0) ^ get(d, 1) ^ get(d, 2) ^ get(d, 3);
		//return get(((d >> 32) & 0xfffffffff) * (d & 0xfffffffff), 31);
		//return get(((d >> 32) & 0xfffffffff) * (d & 0xfffffffff), 31);
		return get(((d0 >> 0) & 0xfffffffff) + (d1 & 0xfffffffff), 32);

	}
};


template class BitMatrix<1, MultipyPolicy>;
template class BitMatrix<1, RiscVPolicy>;
template class BitMatrix<2, RiscVPolicy>;
//using TestMatrix  = BitMatrix<1, MultipyPolicy>;
using TestMatrix = BitMatrix<2, RiscVPolicy>;


#endif