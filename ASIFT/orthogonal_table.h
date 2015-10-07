/*
**功能：十字链表总的信息
**作者：孙晓雨
**日期：2015.10.6
*/
#ifndef ORTHOGONAL_TALBE_H
#define ORTHOGONAL_TABLE_H
#include "orthogonal_list.h"
#include "orthogonal_list_item.h"
#include "orthogonal_list_node.h"

template <typename T>
class orthogonal_table{
private:
	vector< orthogonal_list<T> > m_table; //十字链表本身
	int dimension;//一共多少维度
public:
	void initTable();
	bool addList( orthogonal_list<T> m_list);
	bool addNode( orthogonal_list<T> m_node);
};
#endif