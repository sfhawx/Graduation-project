/*
**功能：对十字链表节点结构的初始化
**作者：孙晓雨
**日期：2015.10.6
*/
#include "orthogonal_list_node.h"
template <typename T>
inline void orthogonal_list_node<T>::setDownNode(const orthogonal_list_node<T> *m_node){
	m_down_node = m_node;
}
template <typename T>
inline orthogonal_list_node<T>* orthogonal_list_node<T>::getDownNode(){
	return m_down_node;
}
template <typename T>
inline void orthogonal_list_node<T>::setTailNode(const orthogonal_list_node<T> *m_node){
	m_tail_node = m_node;
}
template <typename T>
inline orthogonal_list_node<T>* orthogonal_list_node<T>::getTailNode(){
	return m_tail_node;
}
template <typename T>
T* orthogonal_list_node<T>::getValue(){
	return value;
}
template <typename T>
void orthogonal_list_node<T>::setVaule(T* value){
	this->value = value;
}
