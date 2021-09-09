#ifndef SOS_TREAP2_H_
#define SOS_TREAP2_H_

#include "log.h"

#include <vector>

namespace treap {
	template<typename T>
	struct TreapNode {
		T val;
		int priority;
		int son[2];

		TreapNode(T _val) : val(_val), priority(rand() % 300) {
			son[0] = -1;
			son[1] = -1;
		}
		void print() {
			LOG(INFO) << val << " " << priority << "  " << son[0] << "  " << son[1];
		}
	};

	template<typename T>
	class Treap {

	public:
		Treap() : root(0) {};
		~Treap() = default;

		//bool valid(Node);
		int left(int node) { return nodes[node].son[0]; };
		int right(int node) { return nodes[node].son[1]; };
		int Size() { return size; }

		void rotate(int n, int direction);
		TreapNode<T>* insert(T val);
		TreapNode<T>* insert(int n, T val);

		void print();

		T get(int node) { return nodes[node]->val; };


	public:
		int root;
		std::vector<TreapNode<T>*> nodes;

	};

	template<typename T>
	void Treap<T>::rotate(int n, int direction) {
		TreapNode<T>* p = nodes[n];
		int k0 = p->son[1 - direction];
		TreapNode<T>* k = nodes[p->son[1 - direction]];
		p->son[1 - direction] = k->son[direction];
		k->son[direction] = k0;
		nodes[n] = k;
		nodes[k0] = p;
	}

	template<typename T>
	TreapNode<T>* Treap<T>::insert(T value) {
		if (nodes.empty()) {
			TreapNode<T>* r = new TreapNode<T>(value);
			nodes.push_back(r);
			return NULL;
		}
		else
			return insert(0, value);
	}

	template<typename T>
	TreapNode<T>* Treap<T>::insert(int n, T value) {
		//std::cout << "inserting" << n << std::endl;
		//nodes[n]->print();
		TreapNode<T>* p = nodes[n];
		TreapNode<T>* res = NULL;
		if (value == p->val) {
			return p;
		}
		else {
			int dir = static_cast<int>(value > p->val);
			if (p->son[dir] == -1) {
				TreapNode<T>* p0 = new TreapNode<T>(value);
				nodes.push_back(p0);
				p->son[dir] = nodes.size() - 1;
			}
			else {
				res = insert(p->son[dir], value);
			}
		}

		if (p->son[0] != -1) {
			TreapNode<T>* p2 = nodes[p->son[0]];
			if (nodes[p->son[0]]->priority > p->priority)
				rotate(n, 1);
		}
		else if (p->son[1] != -1) {
			if (nodes[p->son[1]]->priority < p->priority)
				rotate(n, 0);
		}
		//print();


		return res;

	}

	template<typename T>
	void Treap<T>::print() {
		for (int i = 0; i < nodes.size(); ++i) {
			//std::cout << i << "  root :  " << root << "   ";
			nodes[i]->print();
		}
	}
};
#endif