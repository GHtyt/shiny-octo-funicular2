#ifndef SOS_BASELINE_H_
#define SOS_BASELINE_H_

#include "bitfield.h"
#include "bitmatrix.cuh"
#include "host_device_vector.h"

#include <vector>
#include <cmath>
#include <algorithm>

#include "span.h"

#include "treap.h"



__global__ void SetKernel(LBitField64 bits, int pos);

__global__ void ClearKernel(LBitField64 bits, int pos);



using u64_v = HostDeviceVector<uint64_t>;
using u64_sp = common::Span<uint64_t>;

class Node {
	public:
		int son[2];
		u64_v mask0, mask1;  //mask0 or set 1s;  mask1 and set 0s;

		int val{ -1 };

		Node();
		Node(int);
		Node(u64_v m0, u64_v m1, int pos, int direction);
		Node(Node* nod, int pos, int direction);

		Node::Node(const Node& n) { mask0.Copy(n.mask0); mask1.Copy(n.mask1); son[0] = n.son[0]; son[1] = n.son[1]; };


		void print() {
			LOG(INFO) << "leaf:  " << son[0] << "  " << son[1] << " func: " << val;
		}
};

class Baseline {
	public:
		struct Func {
			LBitField64 val;
			int func;

			Func() = delete;
			Func(LBitField64 v, int f) : val(v), func(f) {};

			bool operator== (const Func b) {
				return val == b.val;
			}

			bool operator< (const Func b) {
				return val < b.val;
			}
			bool operator> (const Func b) {
				return val > b.val;
			}

			inline friend std::ostream&	operator<<(std::ostream& os, Func f) {
				os << f.val.Bits()[0];
				return os;
			}
		};

		Baseline() {}

		inline int newnodes(Node* n);

		void build(int, std::vector<int>);

		void print();

	private:
		std::vector<Node*> nodes;


};







#endif