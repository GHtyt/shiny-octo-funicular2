#include "bitmatrix.cuh"
#include "log.h"


//template<int32_t LineSize, typename Connection>
//const common::RandEngine* BitMatrix<LineSize, Connection>::randengine = new common::RandEngine(RAND_KERNEL_SIZE);


template<typename Connection>
__global__ void cal_kernel(LBitField64 label, int size, int LineSize, common::Span<uint64_t> data_v) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < size) {
		int32_t res = Connection::Cal(data_v.subspan(id * LineSize, LineSize));
		if (res == 1) {
			label.gpuSet(id);
		}
		else if (res == 0) {
			label.gpuClear(id);
		}

	}

}

__global__ void mask_kernel(common::Span<uint64_t> data_v, int size, int LineSize, common::Span<uint64_t> mask0, common::Span<uint64_t> mask1) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < size) {
		for (int i = 0; i < LineSize; ++i) {
			data_v[i + id * LineSize] &= mask0[i];
			data_v[i + id * LineSize] |= mask1[i];

		}

	}

}

__global__ void construct(uint64_t* d, int size) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < size) {
		d[id] = 7 - id;

	}
}


template<int32_t LineSize, typename Connection>
BitMatrix<LineSize, Connection>::BitMatrix(int _size) :size(_size) {
	randengine = new common::RandEngine(RAND_KERNEL_SIZE);
	//randengine = new common::RandEngine(RAND_KERNEL_SIZE);
	//HostDeviceVector<uint64_t> d1(size * LineSize, 0);
	//HostDeviceVector<uint64_t> lv1(LBitField64::ComputeStorageSize(size), 0);
	//data = &d1;
	//label_v = &lv1;
	data = new HostDeviceVector<uint64_t>(size * LineSize, 0);
	label_v = new HostDeviceVector<uint64_t>(LBitField64::ComputeStorageSize(size), 0);
	//label_v->Resize(LBitField64::ComputeStorageSize(size));
	LBitField64(label_v->HostSpan());
	//data->Resize(size * LineSize); 
};


template<int32_t LineSize, typename Connection>
BitMatrix<LineSize, Connection>::BitMatrix(HostDeviceVector<uint64_t>* _data) : size(std::floor(_data->Size() / LineSize)) {
	randengine = new common::RandEngine(RAND_KERNEL_SIZE);
	//HostDeviceVector<uint64_t> d1(size * LineSize, 0);
	//HostDeviceVector<uint64_t> lv1(LBitField64::ComputeStorageSize(size), 0);
	//data = &d1;
	//label_v = &lv1;
	LOG(INFO) << size;
	data = new HostDeviceVector<uint64_t>(size * LineSize, 0);
	label_v = new HostDeviceVector<uint64_t>(LBitField64::ComputeStorageSize(size), 0);
	//std::cout << LBitField64::ComputeStorageSize(size) << std::endl;
	//label_v->Resize(LBitField64::ComputeStorageSize(size));
	label = LBitField64(label_v->HostSpan());
	LOG(INFO) << size;
	data->Copy(*_data);
};

template<int32_t LineSize, typename Connection>
BitMatrix<LineSize, Connection>::BitMatrix(BitMatrix* bm) : size(bm->Size()) {
	randengine = new common::RandEngine(RAND_KERNEL_SIZE);
	//LOG(DEBUG) << "initializer" << size;
	data = new HostDeviceVector<uint64_t>(size * LineSize, 0);
	label_v = new HostDeviceVector<uint64_t>(LBitField64::ComputeStorageSize(size), 0);
	auto d = bm->Data()->ConstHostVector();
	auto lv = bm->Label_v()->ConstHostVector();
	/*
	for (int i = 0; i < d.size(); ++i) {
		LOG(DEBUG) << d[i];
	}
	LOG(DEBUG) << "here";
	for (int i = 0; i < lv.size(); ++i) {
		LOG(DEBUG) << lv[i];
	}*/
	data->Resize(d.size());
	data->Copy(d);
	label_v->Resize(lv.size());
	label_v->Copy(lv);
	label = LBitField64(label_v->HostSpan());
};

/*
template<int32_t LineSize, typename Connection>
BitMatrix<LineSize, Connection>::BitMatrix(const BitMatrix& bm) const {
	randengine = new common::RandEngine(RAND_KERNEL_SIZE);
	data = new HostDeviceVector<uint64_t>(size * LineSize, 0);
	label_v = new HostDeviceVector<uint64_t>(LBitField64::ComputeStorageSize(size), 0);
	auto d( *(bm.Data());
	const HostDeviceVector<uint64_t> lv = *(bm.Label_v());
	data->Copy(d);
	label_v->Copy(lv);
	label = LBitField64(label_v->HostSpan());
};
*/


template<int32_t LineSize, typename Connection>
BitMatrix<LineSize, Connection>::BitMatrix(std::shared_ptr<BitMatrix> bm) {
	LOG(DEBUG) << "bm here";
	randengine = new common::RandEngine(RAND_KERNEL_SIZE);
	label_v = new HostDeviceVector<uint64_t>(LBitField64::ComputeStorageSize(size), 0);
	data->Copy(*(bm.get()->Data()));
	label_v->Copy(*(bm.get()->Label_v()));
	label = LBitField64(label_v->HostSpan());
};

template<int32_t LineSize, typename Connection>
BitMatrix<LineSize, Connection>::~BitMatrix() {
	delete data;
	delete label_v;
}

template<int32_t LineSize, typename Connection>
void BitMatrix<LineSize, Connection>::Resize(int32_t s) {
	size = s;
	data->Resize(s);
};

template<int32_t LineSize, typename Connection>
void BitMatrix<LineSize, Connection>::sampling() {
	randengine->rand(size * LineSize, data->DevicePointer());
	//construct << < 3, 2 >> > (data->DevicePointer(), size * LineSize);
	//common::Span<uint64_t> dp = data->HostSpan();
	//data->Fill(4);
	//for (int i = 0; i < size * LineSize; ++i)
	//	dp[i] = 3+1;
}



template<int32_t LineSize, typename Connection>
void BitMatrix<LineSize, Connection>::gpuCalLabel() {
	label = LBitField64(label_v->DeviceSpan());
	int32_t block = std::ceil(size / CAL_KERNEL_SIZE) + 1;
	//LOG(INFO) << "block: " << block;
	//LOG(INFO) << "size: " << size;
	//LOG(INFO) << "size: " << CAL_KERNEL_SIZE;
	cal_kernel<Connection> << < CAL_KERNEL_SIZE, block >> > (label, size, LineSize, data->ConstDeviceSpan());
}


template<int32_t LineSize, typename Connection>
void BitMatrix<LineSize, Connection>::cpuCalLabel() {
	//auto data_h = data->HostSpan();
	//std::vector<uint64_t> tt{ 1, 2, 3 };
	//label_v = &HostDeviceVector<uint64_t>(tt);
	//HostDeviceVector<uint64_t> v1;
	//HostDeviceVector<uint64_t>* v = &v1;

	//v->SetDevice(0);
	//v->Resize(4);
	//std::cout << "size: " << data->Size() << std::endl;
	//std::cout << "here: " << label_v->Size() << std::endl;

	//std::vector<uint64_t>& label_h = label_v->HostVector();
	//std::copy_n(thrust::make_counting_iterator(0), 7, label_h.begin());

	
	//abel = LBitField64(label_v->HostSpan());

	//for (int i = 0; i < label.size(); ++i)
	//	std::cout << label.Bits()[i] << std::endl;
	
	auto data_h = data->ConstHostSpan();

#pragma omp parallel for
	for (int i = 0; i < size; ++i) {
		int32_t res = Connection::Cal(data_h.subspan(i * LineSize, LineSize));
		//LOG(IGNORE) << i << " " << res;
		if (res == 1) {
			label.cpuSet(i);
		}
		else if (res == 0) {
			label.cpuClear(i);
		}
	}
}


template<int32_t LineSize, typename Connection>
void BitMatrix<LineSize, Connection>::gpuMask(LBitField64 mask0, LBitField64 mask1) {
	int32_t block = std::ceil(size / CAL_KERNEL_SIZE) + 1;
	mask_kernel << < CAL_KERNEL_SIZE, block >> > (data->DeviceSpan(), size, LineSize, mask0.Bits(), mask1.Bits());
}

template<int32_t LineSize, typename Connection>
void BitMatrix<LineSize, Connection>::cpuMask(LBitField64 mask0, LBitField64 mask1) {
	auto data_h = data->HostSpan();
	common::Span<uint64_t> m0 = mask0.Bits();
	common::Span<uint64_t> m1 = mask1.Bits();

#pragma omp parallel for
	for (int i = 0; i < LineSize; ++i) {
		for (int j = 0; j < size; ++j) {
			data_h[i * LineSize + j] &= m0[i];
			data_h[i * LineSize + j] |= m1[i];
		}
	}
}
template<int32_t LineSize, typename Connection>
common::Span<uint64_t> BitMatrix<LineSize, Connection>::HostLabelSpan() {
	return label_v->ConstHostSpan();
}


template<int32_t LineSize, typename Connection>
common::Span<uint64_t> BitMatrix<LineSize, Connection>::HostdataSpan() {
	return data->ConstHostSpan();
}

template<int32_t LineSize, typename Connection>
HostDeviceVector<uint64_t>* BitMatrix<LineSize, Connection>::Data() {
	return data;
}

template<int32_t LineSize, typename Connection>
HostDeviceVector<uint64_t>* BitMatrix<LineSize, Connection>::Label_v() {
	return label_v;
}


template<int32_t LineSize, typename Connection>
LBitField64 BitMatrix<LineSize, Connection>::DeviceLabel() {
	label = LBitField64(label_v->ConstDeviceSpan());
	return label;
}

template<int32_t LineSize, typename Connection>
LBitField64 BitMatrix<LineSize, Connection>::Label() {
	label = LBitField64(label_v->ConstHostSpan());
	return label;
}


template<int32_t LineSize, typename Connection>
int32_t BitMatrix<LineSize, Connection>::Size() {
	return size;
}