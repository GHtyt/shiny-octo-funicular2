#ifndef SOS_HOST_DEVICE_VECTOR_H_
#define SOS_HOST_DEVICE_VECTOR_H_

#include <initializer_list>
#include <vector>
#include <type_traits>

#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#include <algorithm>
#include <cstdint>
#include <mutex>

#include "common.h"
#include "span.h"


static void (*cudaSetDeviceHandler)(int) = nullptr;  // NOLINT
/*
void SetCudaSetDeviceHandler(void (*handler)(int)) {
	cudaSetDeviceHandler = handler;
}*/

template <typename T> struct HostDeviceVectorImpl;

enum GPUAccess {
	kNone, kRead,
	// write implies read
	kWrite
};

template <typename T>
class HostDeviceVector {
	static_assert(std::is_standard_layout<T>::value, "HostDeviceVector admits only POD types");

public:
	explicit HostDeviceVector(size_t size = 0, T v = T(), int device = -1);
	HostDeviceVector(std::initializer_list<T> init, int device = -1);
	explicit HostDeviceVector(const std::vector<T>& init, int device = -1);
	~HostDeviceVector();

	HostDeviceVector(const HostDeviceVector<T>&) = delete;
	HostDeviceVector(HostDeviceVector<T>&&);

	HostDeviceVector<T>& operator=(const HostDeviceVector<T>&) = delete;
	HostDeviceVector<T>& operator=(HostDeviceVector<T>&&);

	bool Empty() const { return Size() == 0; }
	size_t Size() const;
	int DeviceIdx() const;
	common::Span<T> DeviceSpan();
	common::Span<const T> ConstDeviceSpan() const;
	common::Span<const T> DeviceSpan() const { return ConstDeviceSpan(); }
	T* DevicePointer();
	const T* ConstDevicePointer() const;
	const T* DevicePointer() const { return ConstDevicePointer(); }

	T* HostPointer() { return HostVector().data(); }
	common::Span<T> HostSpan() { return common::Span<T>{HostVector()}; }
	common::Span<T const> HostSpan() const { return common::Span<T const>{HostVector()}; }
	common::Span<T const> ConstHostSpan() const { return HostSpan(); }
	const T* ConstHostPointer() const { return ConstHostVector().data(); }
	const T* HostPointer() const { return ConstHostPointer(); }

	void Fill(T v);
	void Copy(const HostDeviceVector<T>& other);
	void Copy(const std::vector<T>& other);
	void Copy(std::initializer_list<T> other);

	void Extend(const HostDeviceVector<T>& other);

	std::vector<T>& HostVector();
	const std::vector<T>& ConstHostVector() const;
	const std::vector<T>& HostVector() const { return ConstHostVector(); }

	bool HostCanRead() const;
	bool HostCanWrite() const;
	bool DeviceCanRead() const;
	bool DeviceCanWrite() const;
	GPUAccess DeviceAccess() const;

	void SetDevice(int device) const;

	void Resize(size_t new_size, T v = T());


	thrust::device_ptr<T> tbegin() {  // NOLINT
		return thrust::device_ptr<T>(DevicePointer());
	}

	thrust::device_ptr<T> tend() {  // // NOLINT
		return tbegin() + Size();
	}

	thrust::device_ptr<T const> tcbegin() {  // NOLINT
		return thrust::device_ptr<T const>(ConstDevicePointer());
	}

	thrust::device_ptr<T const> tcend() {  // NOLINT
		return tcbegin() + Size();
	}

	template <typename T>
	thrust::device_ptr<T> tbegin() {  // NOLINT
		return thrust::device_ptr<T>(DeviceSpan().data());
	}

	template <typename T>
	thrust::device_ptr<T> tend() {  // NOLINT
		return tbegin() + Size();
	}

	template <typename T>
	thrust::device_ptr<T const> tcbegin() {  // NOLINT
		return thrust::device_ptr<T const>(ConstDeviceSpan().data());
	}

	template <typename T>
	thrust::device_ptr<T const> tcend() {  // NOLINT
		return tcbegin() + Size();
	}

private:
	HostDeviceVectorImpl<T>* impl_;
};








#endif