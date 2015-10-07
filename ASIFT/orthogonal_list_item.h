/*
**功能：十字链表一个维度中链表的信息
**作者：孙晓雨
**日期：2015.10.6
*/
#ifndef ORTHOGONAL_LIST_ITEM_H
#define ORTHOGONAL_LIST_ITEM_H
#include "orthogonal_list_node.h"
#include <iostream>
template<typename T>
class orthogonal_list_item{
private:
	int m_length;//链表的长度
	orthogonal_list_node<T>* m_head;
public:
	void initList();//初始化链表
	bool isEmpty();//检测链表是否为空
	bool addNode(const orthogonal_list_node<T>* m_node);//为链表添加节点
	bool deleteNode(orthogonal_list_node<T>* m_mode);//删除链表节点
	int getLength();//获取链表节点
	void setLength(int length);//设置链表的长度
	orthogonal_list_node<T>* find(const orthogonal_list_node<T>* m_node);//查找链表节点
	bool deleteAllNode();//删除链表所有节点
};
#endif