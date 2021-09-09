#include "baseline.h"

#include <gtest/gtest.h>

TEST(BASELINE, BUILD4) {
	//Baseline b;
	//b.print();
	//b.build(4, std::vector<int> {0, 1, 2, 3});
	//b.print();

}

TEST(BASELINE, BUILD8) {
	Baseline b;
	std::vector<int> dh;
	dh.resize(129);
	std::copy_n(thrust::make_counting_iterator(0), 129, dh.begin());
	/*
	dh[0] = 3;
	dh[1] = 7;
	dh[2] = 2;
	dh[3] = 6;
	dh[4] = 1;
	dh[5] = 5;
	dh[6] = 0;
	dh[7] = 4;*/
	/*dh[0] = 3;
	dh[1] = 1;
	dh[2] = 2;
	dh[3] = 0;*/
	b.build(128, dh);
	//b.build(128, std::vector<int> {0, 1, 2, 3, 4, 5, 6, 7}); //64, 65, 66, 67});
	b.print();

}

/*
TEST(BASELINE, BUILD64) {
	int bit = 64;
	Baseline b;
	std::vector<int> dh{ 0, 1, 2, 3, 32, 33, 34, 35 };
	dh.resize(bit);
	//std::copy_n(thrust::make_counting_iterator(0), bit, dh.begin());

	for (int i = 4; i < bit/2; ++i) {
		dh.push_back(i);
		dh.push_back(i+bit/2);
	}
	b.build(bit, dh);
	//b.build(40, std::vector<int> {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 16, 32, 17, 33, 18, 34, 19, 35, 20, 36, 21, 37, 22, 38, 23, 39, 24, 40, 25, 41, 26, 42, 27, 43, 28, 44, 29, 45, 30, 46, 31, 47, 32, 48, 33, 49 });
	b.print();
}*/
