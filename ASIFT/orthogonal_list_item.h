/*
**���ܣ�ʮ������һ��ά�����������Ϣ
**���ߣ�������
**���ڣ�2015.10.6
*/
#ifndef ORTHOGONAL_LIST_ITEM_H
#define ORTHOGONAL_LIST_ITEM_H
#include "orthogonal_list_node.h"
#include <iostream>
template<typename T>
class orthogonal_list_item{
private:
	int m_length;//����ĳ���
	orthogonal_list_node<T>* m_head;
public:
	void initList();//��ʼ������
	bool isEmpty();//��������Ƿ�Ϊ��
	bool addNode(const orthogonal_list_node<T>* m_node);//Ϊ������ӽڵ�
	bool deleteNode(orthogonal_list_node<T>* m_mode);//ɾ������ڵ�
	int getLength();//��ȡ����ڵ�
	void setLength(int length);//��������ĳ���
	orthogonal_list_node<T>* find(const orthogonal_list_node<T>* m_node);//��������ڵ�
	bool deleteAllNode();//ɾ���������нڵ�
};
#endif