#include <iostream>
#include <list>
#include "make_list.h"
#include "bndlist.h"

using namespace std;
list<TNodeData> bplist;

void AddToList(double x, double y, int id, int bnd)
{
    TNodeData node1(x,y,id,bnd);
    list<TNodeData>::iterator it;

    it = bplist.begin();
    while (it != bplist.end() && node1.gt(*it)) it++;
    bplist.insert(it,node1);
}

void NextInList(int lastid, double *x, double *y, int *id)
{
    list<TNodeData>::iterator it;

    it = bplist.begin();
    while (it != bplist.end() && ((*it).idvalue()!=lastid)) it++;
    if (it==bplist.end()) ;// flag error
    else it++;
    if (it==bplist.end()) it = bplist.begin(); // mimic s-linked list
    *x = (*it).xvalue();
    *y = (*it).yvalue();
    *id = (*it).idvalue();
}

int LengthofList()
{
    return(bplist.size());
}

void PrintList()
{
    list<TNodeData>::iterator it;

    for (it=bplist.begin(); it!=bplist.end(); it++)
        cout << (*it);
    cout << endl;
}

