#include <cstdio>
#include "span.h"
//#include "common.h"

//#include <vector>


#include "gtest/gtest.h"


template <typename Iter>
XGBOOST_DEVICE void InitializeRange(Iter _begin, Iter _end) {
    float j = 0;
    for (Iter i = _begin; i != _end; ++i, ++j) {
        *i = j;
    }
}

namespace common {

    TEST(Span, DlfConstructors) {
        // Dynamic extent
        {
            Span<int> s;
            ASSERT_EQ(s.size(), 0);
            ASSERT_EQ(s.data(), nullptr);

            Span<int const> cs;
            ASSERT_EQ(cs.size(), 0);
            ASSERT_EQ(cs.data(), nullptr);
        }

        // Static extent
        {
            Span<int, 0> s;
            ASSERT_EQ(s.size(), 0);
            ASSERT_EQ(s.data(), nullptr);

            Span<int const, 0> cs;
            ASSERT_EQ(cs.size(), 0);
            ASSERT_EQ(cs.data(), nullptr);
        }

        // Init list.
        {
            Span<float> s{};
            ASSERT_EQ(s.size(), 0);
            ASSERT_EQ(s.data(), nullptr);

            Span<int const> cs{};
            ASSERT_EQ(cs.size(), 0);
            ASSERT_EQ(cs.data(), nullptr);
        }
    }
    TEST(Span, FromNullPtr) {
        // dynamic extent
        {
            Span<float> s{ nullptr, static_cast<Span<float>::index_type>(0) };
            ASSERT_EQ(s.size(), 0);
            ASSERT_EQ(s.data(), nullptr);

            Span<float const> cs{ nullptr, static_cast<Span<float>::index_type>(0) };
            ASSERT_EQ(cs.size(), 0);
            ASSERT_EQ(cs.data(), nullptr);
        }
        // static extent
        {
            Span<float, 0> s{ nullptr, static_cast<Span<float>::index_type>(0) };
            ASSERT_EQ(s.size(), 0);
            ASSERT_EQ(s.data(), nullptr);

            Span<float const, 0> cs{ nullptr, static_cast<Span<float>::index_type>(0) };
            ASSERT_EQ(cs.size(), 0);
            ASSERT_EQ(cs.data(), nullptr);
        }
    }

    TEST(Span, FromPtrLen) {
        float arr[16];
        InitializeRange(arr, arr + 16);

        // static extent
        {
            Span<float> s(arr, 16);
            ASSERT_EQ(s.size(), 16);
            ASSERT_EQ(s.data(), arr);

            for (Span<float>::index_type i = 0; i < 16; ++i) {
                ASSERT_EQ(s[i], arr[i]);
            }

            Span<float const> cs(arr, 16);
            ASSERT_EQ(cs.size(), 16);
            ASSERT_EQ(cs.data(), arr);

            for (Span<float const>::index_type i = 0; i < 16; ++i) {
                ASSERT_EQ(cs[i], arr[i]);
            }
        }

        // dynamic extent
        {
            Span<float, 16> s(arr, 16);
            ASSERT_EQ(s.size(), 16);
            ASSERT_EQ(s.data(), arr);

            for (size_t i = 0; i < 16; ++i) {
                ASSERT_EQ(s[i], arr[i]);
            }

            Span<float const, 16> cs(arr, 16);
            ASSERT_EQ(cs.size(), 16);
            ASSERT_EQ(cs.data(), arr);

            for (Span<float const>::index_type i = 0; i < 16; ++i) {
                ASSERT_EQ(cs[i], arr[i]);
            }
        }
    }
}



