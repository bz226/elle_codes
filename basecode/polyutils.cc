#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <algorithm>
#include "general.h"
#include "attrib.h"
#include "error.h"
#include "polyutils.h"

#include "tripoly.h"


using std::list;
using std::vector;
using std::pair;
using std::find;
using std::cout;
using std::endl;

void sortCoordsOnAngle(vector<Coords> &pts, Coords *ref)
{
    int i=0;
    vector<Coords> tmp(pts.size());
    vector<Coords>::iterator it;
    list< std::pair<int,double> > ordered_list;
    list< std::pair<int,double> >::iterator ito;
    pair<int,double> trial;
    ordered_list.clear();
    for (i=0,it=pts.begin(); it!=pts.end(); i++,it++) {
        trial.first = i;
        trial.second =  polar_angle( ref->x-it->x,ref->y-it->y);
        ito=ordered_list.begin();
        while (ito!=ordered_list.end() &&
                               trial.second > (*ito).second) ito++;
        ordered_list.insert(ito,trial);
    }
    copy(pts.begin(),pts.end(),tmp.begin());
    for (i=0,ito=ordered_list.begin(); ito!=ordered_list.end(); i++,ito++)
        pts[i] = tmp[ito->first];
}

void findPolygon(REAL *pointlist, REAL *normlist, int *edgelist,
                 int nseg, int *seg_count,int *segs_left,
                 list<int> &processed)
{
    unsigned char finished = 0, valid=0;
    int startindx=0, i=0, index[4];
    int end1, end2, startnode, next, curr, tmp;
    list<int>::iterator it;
    std::pair<int,double> trial, nb;
    list<int> seg_index;
    list< std::pair<int,double> > ordered_list;
    list< std::pair<int,double> >::iterator ito;

    startindx=0;
    startnode=NO_NB;
    while (!finished && startindx<nseg) {
        processed.clear();
        // find a valid starting segment
        while (startindx<nseg && (seg_count[startindx]==2 ||
               edgelist[startindx*2]==-1 || 
               edgelist[startindx*2+1]==-1)) startindx++;
        if (startindx<nseg) {
            // process the starting segment
            startnode = curr = end1 = edgelist[startindx*2];
            next = end2 =edgelist[startindx*2+1];
            index[0] = index[1] = startindx;
            processed.push_back(index[1]);
            seg_count[index[1]]++;
            if (seg_count[index[1]]==2) (*segs_left)--;
            /*
             * reverse order to prevent starting a
             * future traversal in the same direction
             */
            if (curr==edgelist[index[1]*2]) {
                edgelist[index[1]*2]=edgelist[index[1]*2+1];
                edgelist[index[1]*2+1]=curr;
            }
            valid = 1;
        }
        while (!finished &&startindx<nseg && valid) {
            /*
             * make a list of nbs in angle order,
             * including curr. The next node is the one
             * before curr in the list
             */
            i=0;
            nb.first = curr;
            nb.second =  polar_angle(
                         pointlist[curr*2]-pointlist[next*2],
                         pointlist[curr*2+1]-pointlist[next*2+1]);
            seg_index.clear();
            seg_index.push_back(index[1]);
            ordered_list.clear();
            ordered_list.push_back(nb);
            while (i<nseg) {
                if (seg_count[i]<2 && i!=index[1] &&
                     ( edgelist[i*2]==next ||
                         edgelist[i*2+1]==next) ) {
                       if (edgelist[i*2]==next)
                           trial.first = edgelist[i*2+1];
                       else trial.first = edgelist[i*2];
                       if (trial.first==-1)
                         trial.second =  polar_angle(
                           normlist[i*2],
                           normlist[i*2+1]);
                       else
                         trial.second =  polar_angle(
                           pointlist[trial.first*2]-
                                            pointlist[next*2],
                           pointlist[trial.first*2+1]-
                                            pointlist[next*2+1]);
                       ito=ordered_list.begin(); it = seg_index.begin();
                       while (ito!=ordered_list.end() &&
                               trial.second > (*ito).second) { ito++;it++; }
                       ordered_list.insert(ito,trial);
                       seg_index.insert(it,i);
                }
                i++;
            }
            /*
             * The list will always include curr
             * if size==1, reached dead-end
             */
            if (ordered_list.size()>1) {
                ito=ordered_list.begin(); it = seg_index.begin();
                while (ito!=ordered_list.end() &&
                      ito->first!=curr) { ito++; it++; }
                if (ito==ordered_list.begin()) {
                    nb=ordered_list.back();
                    index[1] = seg_index.back();
                }
                else {
                    nb = *(--ito);
                    index[1] = *(--it);
                }
       
                processed.push_back(index[1]);
                seg_count[index[1]]++;
                if (seg_count[index[1]]==2) (*segs_left)--;
                index[0] = index[1];
                curr = next;
                next = nb.first;
                if (curr==edgelist[index[1]*2]) {
                    edgelist[index[1]*2]=edgelist[index[1]*2+1];
                    edgelist[index[1]*2+1]=curr;
                }
                finished = (next==startnode);
                if (next==-1) valid = 0;
            }
            else {
            // not valid starting segment, try again
                valid = 0;
            }
        }
    }
    /*nodelist.erase(nodelist.begin(),nodelist.end());*/
    if (finished) {
        /*
        curr=startnode;
        for (it=processed.begin();it!=processed.end();it++) {
            end1 = edgelist[*it * 2];
            end2 = edgelist[*it * 2 + 1];
            if (end1==curr) curr=end2;
            else curr = end1;
            nodelist.push_back(curr);
            cout << *it << ' ' << end1 << ' '<< end2 << ' ';
            cout << endl;      
        }
            cout << endl;      
         */
    }
    else {
        if (*segs_left>2) printf("findPolygon-no start segment found\n");
    }
}

