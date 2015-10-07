/*
**功能：对十字链表节点结构的声明
**作者：孙晓雨
**日期：2015.10.6
*/
#ifndef M_NODE_H
#define M_NODE_H
template <typename T>
class orthogonal_list_node{
private:
	T* value;
	orthogonal_list_node *m_down_node;
	orthogonal_list_node *m_tail_node;
public:
	void setDownNode(const orthogonal_list_node *m_node);
	orthogonal_list_node* getDownNode();
	void setTailNode(const orthogonal_list_node *m_node);
	orthogonal_list_node* getTailNode(); 
	T* getValue();
	void setVaule(T* value);
};
#endif