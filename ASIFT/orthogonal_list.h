/*
**功能：十字链表一个维度的信息
**作者：孙晓雨
**日期：2015.10.6
*/
#ifndef ORTHOGONAL_LIST_H
#define ORTHOGONAL_LIST_H
#include "global_const.h"
#include <vector>
#include "orthogonal_list_item.h"
#include "orthogonal_list_node.h"
template <typename T>
class orthogonal_list{
private:
	vector< orthogonal_list_item<T> > list;
	int length;
public:
	void initList();//初始化链表
	void clearList();//清楚链表
	void getItemLength();//获取链表的长度
	int setLength(int length);//设置链表的长度
	void addNode(orthogonal_list_node<T> *mNode);//添加节点信息
	void deleteNode(orthogonal_list_node<T> *mNode);// 删除节点信息
	orthogonal_list_node<T>* findNode(orthogonal_list_node *mNode);//查找相应节点
	bool isExistNode(orthogonal_list_node* mNode);//判断查找的节点是否存在
};
#endif