#ifndef unode2flynn_h
#define unode2flynn_h

int Unode2Flynn2( int unode_id, // unode with high disloc
		  std::vector<int> tmplist,
                  int *new_child,  // id of swept, new grain
                  int *old_child  // id of swept, new grain
                 );
                 
void UpdateFlynnAges(int iNew,int iOld);

#endif
