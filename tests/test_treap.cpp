#include "treap.h"

#include <gtest/gtest.h>

namespace treap {
	TEST(TREAP2, BASICUsage) {

		int a[] = { 2, 4, 8, 3, 6, 22, 9, 1, 3, 6, 22 };
		Treap<int> treap;
		for (int i = 0; i < 11; ++i) {
			//std::cout << "  k: " << i << std::endl;
			//std::cout << treap.insert(a[i]) << std::endl;
			treap.insert(a[i]);
			//treap.print();
		}
		treap.print();

	}
};