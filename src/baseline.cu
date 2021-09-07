#include "baseline.h"
#include <memory>

__global__ void SetKernel(LBitField64 bits, int pos) {
	//auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos < bits.Size()) {
		bits.gpuSet(pos);
	}
}

__global__ void ClearKernel(LBitField64 bits, int pos) {
	//auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos < bits.Size()) {
		bits.gpuClear(pos);
	}
}

Node::Node() {
	son[0] = -1;
	son[1] = -1;

	std::vector<uint64_t> m0{ 0 };
	std::vector<uint64_t> m1{ 0ULL - 1 };
	mask0 = u64_v(m0);
	mask1 = u64_v(m1);
}


Node::Node(int bit) {
	son[0] = -1;
	son[1] = -1;

	std::vector<uint64_t> m0{ 0 };
	std::vector<uint64_t> m1;
	if (bit < 64)
		m1.push_back((1ULL << bit) - 1 );
	else
		m1.push_back(0ULL - 1);
	//std::vector<uint64_t> m1{ (1ULL<<bit) - 1 };
	mask0 = u64_v(m0);
	mask1 = u64_v(m1);
}

Node::Node(Node* nod, int pos, int direction) {
	son[0] = -1;
	son[1] = -1;
	//LOG(DEBUG) << "constructor";

	mask0.Resize(nod->mask0.Size());
	mask1.Resize(nod->mask1.Size());
	mask0.Copy(nod->mask0);
	mask1.Copy(nod->mask1);


	if (direction == 0) {
		common::Span<uint64_t> storage = mask1.DeviceSpan();
		LBitField64 bits = LBitField64({ storage.data(), storage.size() });
		ClearKernel << <2, 1 >> > (bits, pos);
	}
	else if (direction == 1) {
		common::Span<uint64_t> storage = mask0.DeviceSpan();
		LBitField64 bits = LBitField64({ storage.data(), storage.size() });
		SetKernel << <2, 1 >> > (bits, pos);
	}

}

Node::Node(u64_v m0, u64_v m1, int pos, int direction) {
	son[0] = -1;
	son[1] = -1;

	mask0.Copy(m0);
	mask1.Copy(m1);


	if (direction == 0) {
		common::Span<uint64_t> storage = mask1.DeviceSpan();
		LBitField64 bits = LBitField64({ storage.data(), storage.size() });
		ClearKernel << <2, 1 >> > (bits, pos);
	}
	else if (direction == 1) {
		common::Span<uint64_t> storage = mask0.DeviceSpan();
		LBitField64 bits = LBitField64({ storage.data(), storage.size() });
		SetKernel << <2, 1 >> > (bits, pos);
	}

}



LBitField64 ToHostBit(HostDeviceVector<uint64_t> vec) {
	auto sp = vec.ConstHostVector();
	auto bits = LBitField64({ sp.data(), sp.size() });
	return bits;
}

inline int Baseline::newnodes(Node* n) {
	int res = nodes.size();
	nodes.push_back(n);
	return res;

}


void Baseline::build(int BT, std::vector<int> k) {
	LOG(DEBUG) << "starting build process";
	Node* root = new Node(BT);
	nodes.push_back(root);
	//std::vector<int> k{0, 2, 3, 1};

	//int BT = 4;

	int cur_start = 0;
	for (int i = 0; i < BT; ++i) {
		int end = nodes.size();
		LOG(DEBUG) << "depth: " << i << " " << cur_start << " " << end;
		Treap<Func> t;
		std::vector<uint64_t> v0;
		std::vector<uint64_t> v1;
		if (BT <= 4) {
			v0.push_back(0);
			v1.push_back((1ULL << 16 * BT) - 1);
		}
		else {
			int l = BT / 4;
			for (int i = 0; i < l; ++i) {
				v0.push_back(0);
				v1.push_back((1ULL << 16 * BT) - 1);

			}
		}

		LBitField64 b0({ v0.data(), v0.size() });
		LBitField64 b1({ v1.data(), v1.size() });
		Func f0(b0, 0);
		Func f1(b1, 1);
		t.insert(f0);
		t.insert(f1);

		TestMatrix* tmbase = new TestMatrix(16 * BT);
		tmbase->sampling();

		auto hv = tmbase->Data()->ConstHostVector();

		for (int j = cur_start; j < end; ++j) {
			//LOG(DEBUG) << j;

			TestMatrix *tm = new TestMatrix(tmbase);
			std::vector<uint64_t> m0 = nodes[j]->mask1.ConstHostVector();
			std::vector<uint64_t> m1 = nodes[j]->mask0.ConstHostVector();
			//auto m0 = nodes[j]->mask1.ConstDeviceSpan();
			//auto m1 = nodes[j]->mask0.ConstDeviceSpan();
			auto ma0 = LBitField64(m0);
			auto ma1 = LBitField64(m1);
			tm->cpuMask(ma0, ma1);
			tm->cpuCalLabel();
			//tm->gpuMask(ma0, ma1);
			//tm->gpuCalLabel();


			Func fj(tm->Label(), j);
			//LOG(INFO) << fj.val.Bits()[0];
			int id = t.insert(fj);
			//LOG(DEBUG) << "here" << nodes[j]->son[0] << nodes[j]->son[1];
			//LOG(SILENT) << j << " " << id;

			if (id == -1) {

				Node* left = new Node(nodes[j], k[i], 0);
				Node* right = new Node(nodes[j], k[i], 1);
				nodes[j]->son[0] = newnodes(left);
				nodes[j]->son[1] = newnodes(right);
			}
			else {
				nodes[j]->val = id + cur_start - 1;
			}
		
			//delete tm;
		}
		cur_start = end;

		//t.print();
	}

	//print();
};

void Baseline::print() {
	int m = (nodes.size() < 20) ? nodes.size() : 20;
	for (int i = 0; i < m; ++i) {
		LOG(INFO) << "leaf " <<i<< ":   " << nodes[i]->son[0] << "  " << nodes[i]->son[1] << " func: " << nodes[i]->val;
	}
}
