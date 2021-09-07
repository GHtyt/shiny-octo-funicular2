#include "bitmatrix.cuh"

#include "log.h"

#include <vector>

#include <gtest/gtest.h>


#if defined (__CUDA__)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

TEST(BITMATRIX, BASICCONSTRUCTOR)
{
	TestMatrix tm(3);
	//std::vector<uint64_t> v = { 1,2,3 };
	HostDeviceVector<uint64_t>* d = tm.Data();
	//d->Copy(v);
	//d->Resize(3);

	/*
	common::Span<uint64_t> d = tm.Data()->DeviceSpan();*/
	tm.sampling();

	ASSERT_TRUE(tm.Size(), 3);
	/*
	auto sp = d->DeviceSpan();
	for (auto i = 0; i < sp.size(); ++i) {
		//sp[i] = 5-i;
	}*/


	auto sp = d->ConstHostSpan();
	for (auto i = 0; i < sp.size(); ++i) {
		LOG(INFO) << sp[i];
		ASSERT_GE(sp[i], 0);
		//ASSERT_EQ(sp[i], 7 - i);
	}




}

TEST(BITMATRIX, CPUCal) {
	
	std::vector<uint64_t> v{0, 1, 5, 2, 6, 5, 4};
	HostDeviceVector<uint64_t> v1(v);
	TestMatrix tm(&v1);

	ASSERT_TRUE(tm.Size(), 7);/**/
	tm.cpuCalLabel();

	//LOG(DEBUG) << tm.Label().Bits()[0];
	ASSERT_TRUE(tm.Label().Bits()[0], 36ULL);
	/*
	HostDeviceVector<uint64_t>* res = tm.Label_v();
	

	auto sp = res->ConstHostSpan();
	for (auto i = 0; i < sp.size(); ++i) {
		std::cout << sp[i] << std::endl;
	}*/
}


TEST(BITMATRIX, GPUCal) {

	std::vector<uint64_t> v{ 0, 1, 5, 2, 6, 5, 4 };
	HostDeviceVector<uint64_t> v1(v);
	TestMatrix tm(&v1);

	ASSERT_TRUE(tm.Size(), 7);/**/
	tm.gpuCalLabel();

	ASSERT_TRUE(tm.Label().Bits()[0], 36ULL);
}


TEST(BITMATRIX, CPUMask) {

	std::vector<uint64_t> v{ 0, 1, 3, 2, 6, 3, 4 };
	HostDeviceVector<uint64_t> v1(v);
	TestMatrix tm(&v1);

	std::vector<uint64_t> mask0_storage(1, 0);
	std::vector<uint64_t> mask1_storage(1, 0);
	auto mask0 = LBitField64({ mask0_storage.data(), mask0_storage.size() });
	auto mask1 = LBitField64({ mask1_storage.data(), mask1_storage.size() });
	std::fill(mask0_storage.begin(), mask0_storage.end(), 1023UL);
	std::fill(mask1_storage.begin(), mask1_storage.end(), 2UL);

	ASSERT_TRUE(tm.Size(), 7);/**/
	tm.cpuMask(mask0, mask1);

	ASSERT_TRUE(tm.Data()->ConstHostSpan()[0], 2ULL);
}

TEST(BITMATRIX, GPUMask) {

	std::vector<uint64_t> v{ 0, 1, 3, 2, 6, 3, 4 };
	HostDeviceVector<uint64_t> v1(v);
	TestMatrix tm(&v1);


	thrust::device_vector<LBitField64::value_type> mask0_storage(1);
	thrust::device_vector<LBitField64::value_type> mask1_storage(1);
	auto mask0 = LBitField64(common::ToSpan(mask0_storage));
	auto mask1 = LBitField64(common::ToSpan(mask1_storage));
	thrust::fill(mask0_storage.begin(), mask0_storage.end(), 1023UL);
	thrust::fill(mask1_storage.begin(), mask1_storage.end(), 2UL);

	ASSERT_TRUE(tm.Size(), 7);/**/
	tm.gpuMask(mask0, mask1);

	ASSERT_TRUE(tm.Data()->ConstHostSpan()[0], 2ULL);
}

TEST(BITMATRIX, equal) {
	TestMatrix* tmbase = new TestMatrix(1);
	tmbase->sampling();

	auto hv = tmbase->Data()->ConstHostVector();


		TestMatrix* tm0 = new TestMatrix(tmbase);
		//std::vector<uint64_t> m0 = nodes[j]->mask1.ConstHostVector();
		//std::vector<uint64_t> m1 = nodes[j]->mask0.ConstHostVector();
		//auto m0 = nodes[j]->mask1.ConstDeviceSpan();
		//auto m1 = nodes[j]->mask0.ConstDeviceSpan();
		auto ma00 = LBitField64(m0);
		auto ma01 = LBitField64(m1);
		tm1->cpuMask(ma0, ma1);
		tm1->cpuCalLabel();
		//tm->gpuMask(ma0, ma1);
		//tm->gpuCalLabel();


		LOG(INFO) << tm0->Label().Bits()[0];


		TestMatrix* tm1 = new TestMatrix(tmbase);
		thrust::device_vector<LBitField64::value_type> m11(1);
		thrust::device_vector<LBitField64::value_type> m10(1);
		auto ma10 = LBitField64(m0);
		auto ma11 = LBitField64(m1);
		tm1->cpuMask(ma0, ma1);
		tm1->cpuCalLabel();
		//tm->gpuMask(ma0, ma1);


		LOG(INFO) << tm1->Label().Bits()[0];
}