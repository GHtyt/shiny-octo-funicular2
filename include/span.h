#ifndef SOS_SPAN_H_
#define SOS_SPAN_H_

#include <cinttypes>          // size_t
#include <limits>             // numeric_limits
#include <iterator>
#include <type_traits>
#include <cstdio>
#include <iostream>

#include "base.h"

#if defined(__CUDA__)
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif


#define KERNEL_CHECK(cond)                                                     \
	(XGBOOST_EXPECT((cond), true) ? static_cast<void>(0) : std::terminate())

#define SPAN_CHECK(cond) KERNEL_CHECK(cond)


namespace common
{

constexpr const std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

template <typename SpanType, bool IsConst>
class SpanIterator {
    using ElementType = typename SpanType::element_type;
	
    public:
		using iterator_category = std::random_access_iterator_tag;      // NOLINT
		using value_type = typename SpanType::value_type;  // NOLINT
		using difference_type = int64_t;             // NOLINT

		using reference = typename std::conditional<                    // NOLINT
			IsConst, const ElementType, ElementType>::type&;
		using pointer = typename std::add_pointer<reference>::type;  
		constexpr SpanIterator() = default;

		
		XGBOOST_DEVICE constexpr SpanIterator(
			const SpanType* _span,
			typename SpanType::index_type _idx) noexcept :
											span_(_span), index_(_idx) {}

		
		friend SpanIterator<SpanType, true>;
		template <bool B, typename std::enable_if<!B && IsConst>::type* = nullptr>
		XGBOOST_DEVICE constexpr SpanIterator(                         // NOLINT
			const SpanIterator<SpanType, B>& other_) noexcept
			: SpanIterator(other_.span_, other_.index_) {}

		XGBOOST_DEVICE reference operator*() const {
			//SPAN_CHECK(index_ < span_->size());
			return *(span_->data() + index_);
		}
		XGBOOST_DEVICE reference operator[](difference_type n) const {
			return *(*this + n);
		}


		XGBOOST_DEVICE pointer operator->() const {
			//SPAN_CHECK(index_ != span_->size());
			return span_->data() + index_;
		}

		XGBOOST_DEVICE SpanIterator& operator++() {
			//SPAN_CHECK(index_ != span_->size());
			index_++;
			return *this;
		}

		XGBOOST_DEVICE SpanIterator operator++(int) {
			auto ret = *this;
			++(*this);
			return ret;
		}

		XGBOOST_DEVICE SpanIterator& operator--() {
			//SPAN_CHECK(index_ != 0 && index_ <= span_->size());
			index_--;
			return *this;
		}

		XGBOOST_DEVICE SpanIterator operator--(int) {
			auto ret = *this;
			--(*this);
			return ret;
		}

		XGBOOST_DEVICE SpanIterator operator+(difference_type n) const {
			auto ret = *this;
			return ret += n;
		}

		XGBOOST_DEVICE SpanIterator& operator+=(difference_type n) {
			//SPAN_CHECK((index_ + n) <= span_->size());
			index_ += n;
			return *this;
		}

		XGBOOST_DEVICE difference_type operator-(SpanIterator rhs) const {
			//SPAN_CHECK(span_ == rhs.span_);
			return index_ - rhs.index_;
		}

		XGBOOST_DEVICE SpanIterator operator-(difference_type n) const {
			auto ret = *this;
			return ret -= n;
		}

		XGBOOST_DEVICE SpanIterator& operator-=(difference_type n) {
			return *this += -n;
		}


	protected:
		const SpanType *span_ { nullptr };	
		typename SpanType::index_type index_ { 0 };

};

template <typename T, std::size_t Extent>
struct ExtentAsBytesValue : public std::integral_constant<
	std::size_t,
	Extent == dynamic_extent ?
	Extent : sizeof(T) * Extent> {};

template <std::size_t From, std::size_t To>
struct IsAllowedExtentConversion : public std::integral_constant<
	bool, From == To || From == dynamic_extent || To == dynamic_extent> {};

template <class From, class To>
struct IsAllowedElementTypeConversion : public std::integral_constant<
	bool, std::is_convertible<From(*)[], To(*)[]>::value> {};

template <class T>
struct IsSpanOracle : std::false_type {};
/*
template <class T, std::size_t Extent>
struct IsSpanOracle<Span<T, Extent>> : std::true_type {};
*/
template <class T>
struct IsSpan : public IsSpanOracle<typename std::remove_cv<T>::type> {};



template <typename T, std::size_t Extent = dynamic_extent>
class Span
{
		
	public:
		using element_type = T;                               // NOLINT
		using value_type = typename std::remove_cv<T>::type;  // NOLINT
		using index_type = std::size_t;                       // NOLINT
		using difference_type = int64_t;            // NOLINT
		using pointer = T*;                                   // NOLINT
		using reference = T&;                                 // NOLINT

		using iterator = SpanIterator<Span<T, Extent>, false>;             // NOLINT
		using const_iterator = const SpanIterator<Span<T, Extent>, true>;  // NOLINT
		using reverse_iterator = SpanIterator<Span<T, Extent>, false>;     // NOLINT
		using const_reverse_iterator = const SpanIterator<Span<T, Extent>, true>;  // NOLINT

		constexpr Span() noexcept = default;
		/*
		constexpr Span() {
			std::cout<< element_type << std::endl;
			std::cout<< value_type << std::endl;
			std::cout<< index_type << std::endl;
		}//noexcept = default;
		*/

		XGBOOST_DEVICE Span(pointer _ptr, index_type _count) :
			size_(_count), data_(_ptr) {
			//SPAN_CHECK(!(Extent != dynamic_extent && _count != Extent));
			//SPAN_CHECK(_ptr || _count == 0);
		}

		XGBOOST_DEVICE Span(pointer _first, pointer _last) :
			size_(_last - _first), data_(_first) {
			//SPAN_CHECK(data_ || size_ == 0);
		}

		template <std::size_t N>
		XGBOOST_DEVICE constexpr Span(element_type (&arr)[N])  // NOLINT
			noexcept : size_(N), data_(&arr[0]) {}


	
		
		template <class Container,
			class = typename std::enable_if<
			!std::is_const<element_type>::value &&
			!common::IsSpan<Container>::value&&
			std::is_convertible<typename Container::pointer, pointer>::value&&
			std::is_convertible<typename Container::pointer,
			decltype(std::declval<Container>().data())>::value>::type>
			Span(Container& _cont) :  // NOLINT
			size_(_cont.size()), data_(_cont.data()) {
			static_assert(!common::IsSpan<Container>::value, "Wrong constructor of Span is called.");
		}
		
		template <class Container,
			class = typename std::enable_if<
			std::is_const<element_type>::value &&
			!common::IsSpan<Container>::value&&
			std::is_convertible<typename Container::pointer, pointer>::value&&
			std::is_convertible<typename Container::pointer,
			decltype(std::declval<Container>().data())>::value>::type>
			Span(const Container& _cont) : size_(_cont.size()),  // NOLINT
			data_(_cont.data()) {
			static_assert(!common::IsSpan<Container>::value, "Wrong constructor of Span is called.");
		}
		
		/**/
		template <class U, std::size_t OtherExtent,
			class = typename std::enable_if<
			common::IsAllowedElementTypeConversion<U, T>::value&&
			common::IsAllowedExtentConversion<OtherExtent, Extent>::value>>
			XGBOOST_DEVICE constexpr Span(const Span<U, OtherExtent>& _other)   // NOLINT
			noexcept : size_(_other.size()), data_(const_cast<pointer>(_other.data())) {}
		

		XGBOOST_DEVICE constexpr Span(const Span& _other)
			noexcept : size_(_other.size()), data_(_other.data()) {}

		XGBOOST_DEVICE Span& operator=(const Span& _other) noexcept {
			size_ = _other.size();
			data_ = _other.data();
			return *this;
		}





		XGBOOST_DEVICE ~Span() noexcept {}


		XGBOOST_DEVICE constexpr iterator begin() const noexcept {  // NOLINT
			return {this, 0};
		}

		XGBOOST_DEVICE constexpr iterator end() const noexcept {    // NOLINT
			return {this, size()};
		}

		XGBOOST_DEVICE constexpr const_iterator cbegin() const noexcept {  // NOLINT
			return {this, 0};
		}

		XGBOOST_DEVICE constexpr const_iterator cend() const noexcept {    // NOLINT
			return {this, size()};
		}

		XGBOOST_DEVICE constexpr reverse_iterator rbegin() const noexcept {  // NOLINT
			return reverse_iterator{end()};
		}

		XGBOOST_DEVICE constexpr reverse_iterator rend() const noexcept {    // NOLINT
			return reverse_iterator{begin()};
		}

		XGBOOST_DEVICE constexpr const_reverse_iterator crbegin() const noexcept {  // NOLINT
			return const_reverse_iterator{cend()};
		}

		XGBOOST_DEVICE constexpr const_reverse_iterator crend() const noexcept {    // NOLINT
			return const_reverse_iterator{cbegin()};
		}

		// element access

		XGBOOST_DEVICE reference front() const {  // NOLINT
			return (*this)[0];
		}

		XGBOOST_DEVICE reference back() const {  // NOLINT
			return (*this)[size() - 1];
		}

		XGBOOST_DEVICE reference operator[](index_type _idx) const {
			//SPAN_LT(_idx, size());
			return data()[_idx];
		}

		XGBOOST_DEVICE reference operator()(index_type _idx) const {
			return this->operator[](_idx);
		}

		XGBOOST_DEVICE constexpr pointer data() const noexcept {   // NOLINT
			return data_;
		}

		// Observers
		XGBOOST_DEVICE constexpr index_type size() const noexcept {  // NOLINT
			return size_;
		}

		XGBOOST_DEVICE constexpr index_type size_bytes() const noexcept {  // NOLINT
			return size() * sizeof(T);
		}

		XGBOOST_DEVICE constexpr bool empty() const noexcept {  // NOLINT
			return size() == 0;
		}

		template <std::size_t Extent, std::size_t Offset, std::size_t Count>
		struct ExtentValue : public std::integral_constant<
			std::size_t, Count != dynamic_extent ?
			Count : (Extent != dynamic_extent ? Extent - Offset : Extent)> {};

		template <std::size_t Offset,
			std::size_t Count = dynamic_extent>
			XGBOOST_DEVICE auto subspan() const ->                   // NOLINT
			Span<element_type,
			ExtentValue<Extent, Offset, Count>::value> {
			//SPAN_CHECK((Count == dynamic_extent) ?	(Offset <= size()) : (Offset + Count <= size()));
			return { data() + Offset, Count == dynamic_extent ? size() - Offset : Count };
		}

		XGBOOST_DEVICE Span<element_type, dynamic_extent> subspan(  // NOLINT
			index_type _offset,
			index_type _count = dynamic_extent) const {
			//SPAN_CHECK((_count == dynamic_extent) ?	(_offset <= size()) : (_offset + _count <= size()));
			return { data() + _offset, _count ==
					dynamic_extent ? size() - _offset : _count };
		}



    private:
        index_type size_ { 0 };
        pointer data_ { nullptr };
};


template <typename VectorT, typename T = typename VectorT::value_type,
	typename IndexT = typename Span<T>::index_type>
	Span<T> ToSpan(
		VectorT& vec,
		IndexT offset = 0,
		IndexT size = std::numeric_limits<size_t>::max()) {
	size = size == std::numeric_limits<size_t>::max() ? vec.size() : size;
	//CHECK_LE(offset + size, vec.size());
	return { vec.data().get() + offset, size };
}

template <typename T>
Span<T> ToSpan(thrust::device_vector<T>& vec,
	size_t offset, size_t size) {
	return ToSpan(vec, offset, size);
}

}; // namespace common

#endif