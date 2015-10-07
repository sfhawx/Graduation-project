/*
**���ܣ�ʮ������һ��ά�ȵ���Ϣ
**���ߣ�������
**���ڣ�2015.10.6
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
	void initList();//��ʼ������
	void clearList();//�������
	void getItemLength();//��ȡ����ĳ���
	int setLength(int length);//��������ĳ���
	void addNode(orthogonal_list_node<T> *mNode);//��ӽڵ���Ϣ
	void deleteNode(orthogonal_list_node<T> *mNode);// ɾ���ڵ���Ϣ
	orthogonal_list_node<T>* findNode(orthogonal_list_node *mNode);//������Ӧ�ڵ�
	bool isExistNode(orthogonal_list_node* mNode);//�жϲ��ҵĽڵ��Ƿ����
};
#endif