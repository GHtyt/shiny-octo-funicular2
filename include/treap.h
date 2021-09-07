#ifndef SOS_TREAP2_H_
#define SOS_TREAP2_H_

#include "log.h"

#include <vector>

template<typename T>
class Treap {
	struct Node {
		T val;
		int priority;
		int son[2];

		Node(T _val) : val(_val), priority(rand() % 300) {
			son[0] = -1;
			son[1] = -1;
		}
		void print() {
			LOG(INFO) <<  val << "  " << priority << "  " << son[0] << "  " << son[1];
		}
	};

	public:
		Treap() : root(0) {};
		~Treap() = default;

		//bool valid(Node);
		int left(int node) { return nodes[node].son[0]; };
		int right(int node) { return nodes[node].son[1]; };
		int Size() { return size; }

		void rotate(int n, int direction);
		int insert(T val);
		int insert(int n, T val);

		void print();
	

private:
	int root;
	std::vector<Node*> nodes;

};

template<typename T>
void Treap<T>::rotate(int n, int direction) {
	Node* p = nodes[n];
	int k0 = p->son[1 - direction];
	Node* k = nodes[p->son[1 - direction]];
	p->son[1 - direction] = k->son[direction];
	k->son[direction] = k0;
	nodes[n] = k;
	nodes[k0] = p;
}

template<typename T>
int Treap<T>::insert(T value) {
	if (nodes.empty()) {
		Node* r = new Node(value);
		nodes.push_back(r);
		return -1;
	}
	else
		return insert(0, value);
}

template<typename T>
int Treap<T>::insert(int n, T value) {
	//std::cout << "inserting" << n << std::endl;
	//nodes[n]->print();
	Node* p = nodes[n];
	int res = -1;
	if (value == p->val) {
		return n;
	} 
	else {
		int dir = static_cast<int>(value > p->val);
		if (p->son[dir] == -1) {
			Node* p0 = new Node(value);
			nodes.push_back(p0);
			p->son[dir] = nodes.size()-1;
		}
		else {
			res = insert(p->son[dir], value);
		}
	}

	if (p->son[0] != -1) {
		Node* p2 = nodes[p->son[0]];
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

#endif