//*****************************************************************
// Based on unode2flynn_new5_user.cc from Albert's fft implementation
//*****************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>
#include "attrib.h"
#include "nodes.h"
#include "unodes.h"
#include "update.h"
#include "error.h"
#include "parseopts.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "tripoly.h"
#include "setup.h"
#include "init.h"
#include "check.h"
#include "general.h"
#include "polygon.h"
#include "polyutils.h"
#include "gpcclip.h"
#include "splitting.h"
#include "bflynns.h"
#include "error.h"
#include "unode2flynn.h"

using namespace std;

bool ElleCoordRelativePosition(Coords *current,
                              Coords *bnd,
                              int num_bnd_nodes,
                              int *xflags,int *yflags,
                              Coords *rel_pos);
int FindBnode(std::vector<int> nodes, Coords *xy);
int splitFlynn(int flynnindex, int *child1,int *child2, int first, int last,
    				vector<PointData> &boundary);
void updateLists(int node1, int node2, int nn, int curr_rgn);
void CheckNodes();

void CreateBoundary2(vector<PointData> &nodes,int nodenbbnd,
                    int nbnodebnd,
                   int *elle_id,int num,int *end);
                   
const double ELLE_EPS=1.5e-6;

int Unode2Flynn2( int unode_id,     
                 std::vector<int> tmplist, 
                 int *new_child  // id of swept grain
                 )

// information that we can pass are unode_id, tmplist 
// int Unode2Flynn2 ( int unode_id, std::vector<int> &unodelist );

{
	
/*	Nucleation of a new flynn using unodes location. The 
*	The conditions to nucleate a new flynn are: 
*	1) Critical threshold of dislocation density. Only if dislocation density of unode >> d_crit. Last parameter is a user data. 
	   The scalar critical dislocation density can be regarded as an average threshold of low-high angle boundary.  
*	2) Location of critical unode with regard to the boundary nodes. Only allowed unodes near to grain boundaries!!  
* 	3) Critical area of prexistent flynn to split. If flynn area << area_crit, grain split is aborted 
*   4) Unstable process. Need a computer with large free RAM memory, if not nucleation fails because bas memory assignation. Also, unknown overflow errors. 	
*	5) Assume a new grain free of dislocations. The new grain's orientation is the same than the critical unode.  
*   6) Assumption of grain size of nucleated grain. By default take 1st neighbours unodes, but 2nd or 3rd neighbours unodes can be selected.
       At this moment unrelated to local stress, work, stress strain rate, etc. A single-unode is assumed as unstable grain and shrink.  	
* 

CAUTION! A tricky part is an adequate ratio between switchdistance of bnodes and unodes. 	
*/
	bool FSDEBUG =true;
    
    fstream fVoronoiFile;
    fVoronoiFile.open ( "Voronois.txt", fstream::out | fstream::app);     
    
	bool bDisplay = true;
	
    if(FSDEBUG) printf("line: %u\n",__LINE__);
    int i, k;
    int err=0,first, last,endindex, startindex, next, nbnode, curr;
    double area=0;
    Coords xy, vxy, refxy;
    vector<PointData> boundary;
    std::list<int> nbflynns;
    std::list<int> ::iterator lit;
    extern VoronoiData *V_data;
    
	double new_den=0.00; // FULL CLEAN VERSION 
	double area_crit=5e-4, area_k; // minimum flynn area to nucleate a new grain
	int error_alb=0, n_neighbours=40;

	if (bDisplay) printf (" xx %d %d \n", unode_id, tmplist.size());

    vector<int> unodelist, tmplist2;
	double dummy;
   // int subgrain_id;
	
    if (unode_id!=NO_VAL && UnodesActive() && unode_id<ElleMaxUnodes()){


        vector<int> :: iterator vit;
        double roi = ElleUnodeROI();
        Coords rect[4];

        k = ElleUnodeFlynn(unode_id);
        		
		ElleGetUnodeAttribute(unode_id, &dummy, U_DENSITY);
		// subgrain_id=int(dummy);
        nbflynns.push_back(k);
        
           if (unode_id!=NO_VAL && UnodesActive() &&
                                   unode_id<ElleMaxUnodes()) {
                 vector<int> tmplist;
                 if(FSDEBUG) printf("tmplist.size() = %u\n",tmplist.size());
                
                 vector<int> :: iterator vit;
                 double roi = ElleUnodeROI();
                 Coords rect[4];

                 k = ElleUnodeFlynn(unode_id);
                  // NB ElleFlynnNbRegions(k,nbflynns);
                 nbflynns.push_back(k);
                 ElleGetUnodePosition(unode_id,&refxy);
		
// This is a limiting rectangle so the Voronoi data only
// contains info about unodes in the local region.
// The size of the rect should be increased if the 3rd or more
// set of unode neighbours is involved - currently nbs of nbs of unode
// Symptom that this needs increasing is a long boundary shooting
// off to another edge of the flynn (following a triangulation norm)
                 rect[0].x = rect[3].x = refxy.x - 2*n_neighbours*roi;
                 rect[1].x = rect[2].x = refxy.x + 2*n_neighbours*roi;
                 rect[0].y = rect[1].y = refxy.y - 2*n_neighbours*roi;
                 rect[2].y = rect[3].y = refxy.y + 2*n_neighbours*roi;
                 
                 for (lit = nbflynns.begin();lit!=nbflynns.end();lit++) {
                     tmplist.clear();
                     ElleGetFlynnUnodeList(*lit,tmplist2);
                     for (vit = tmplist2.begin();
                            vit != tmplist2.end() && !err; vit++) {
                         ElleGetUnodePosition((*vit),&vxy);
                         ElleCoordsPlotXY(&vxy,&refxy);
                                                                                
                         if( EllePtInRect(rect,4,&vxy) )
                    // inside bounding rect
                             unodelist.push_back(*vit);
                     }
                 }
                 err = ElleVoronoiFlynnUnodes(nbflynns, unodelist);
             }
             else
                 OnError("Invalid unode no.",RANGE_ERR);

	
	
             if (ElleVoronoiActive() && !err) {
                 int numunodes=0, last;
                 Coords pos;
                 vector<int> allnodelist,nodelist;
                 vector<int> unodenbs, tmpnbs, tmpnbs2, tmpnbs3;
                 list<int> total_list, total_unodenbs;
                 list<int> :: iterator it, it1;
                 vector< list<int> > vpt_array;

                 ElleFlynnNodes(k,allnodelist);
                 ElleGetUnodePosition(unode_id,&pos);
                 
	for (i=0;i<allnodelist.size();i++) {
		int j;
		Coords xy, uxy;
		ElleNodePosition(allnodelist[i],&xy);
		if (bDisplay) std::cout << allnodelist[i] << " " << xy.x << " " << xy.y << std::endl;
		ElleCoordsPlotXY(&xy, &pos);

		if ( pointSeparation(&xy, &pos) < ElleUnodeROI()*(n_neighbours))
			nodelist.push_back(allnodelist[i]);
		}
	if (nodelist.size()<2) return(1); // unode not close to boundary
	
	
	/* first version 			
		err = ElleVoronoiFlynnUnodes(nbflynns, tmplist);  // I think there is one only for one flynn ¿?
		
             }
             else
                 OnError("Invalid unode no.",RANGE_ERR);

	
 	if (ElleVoronoiActive() && !err) {
       int numunodes=0, last;
       Coords pos;
       vector<int> allnodelist,nodelist;
       vector<int> unodenbs, tmpnbs, tmpnbs2, tmpnbs3;
       list<int> total_list, total_unodenbs;
       list<int> :: iterator it, it1;
       vector< list<int> > vpt_array;
	      
        nodelist.clear(); 
        ElleFlynnNodes(k,allnodelist);
        ElleGetUnodePosition(unode_id,&pos);
		for (i=0;i<allnodelist.size();i++) {
			int j;
			Coords xy, uxy;
			ElleNodePosition(allnodelist[i],&xy);
			ElleCoordsPlotXY(&xy, &pos);
			if ( pointSeparation(&xy, &pos) < ElleUnodeROI()*(n_neighbours))
				nodelist.push_back(allnodelist[i]);
			}
		if (nodelist.size()<2) return(1); // unode not close to boundary
       
	*/        

       PointData bndnode;       
	// do a list with subgrain's unodes 
	   copy(tmplist.begin(), tmplist.end(), inserter(total_unodenbs, total_unodenbs.end()));	

// total_unodenbs will include multiple instances of some unodes so
// sort numerically
                total_unodenbs.sort();
// then remove the multiple values (could use set_int)
                total_unodenbs.unique();
// then remove the unode itself because I want it first in the array
                total_unodenbs.remove(unode_id);

// vpt_array has a list of Voronoi pts for each unode
// vpt_array[0] is the list for the unode of interest
//   which we know is on the flynn boundary!!!
		 
                ElleUnodeVoronoiPts(unode_id,total_list);		 
                vpt_array.push_back(total_list);
                numunodes = total_unodenbs.size()+1;
                
 				 if (bDisplay) printf("voronoi Pts list %d %d\n", total_list.size(),numunodes);     
 				           
                for (it=total_unodenbs.begin(); it!=total_unodenbs.end();it++) {

                    list<int> pt_list;
                    ElleUnodeVoronoiPts(*it,pt_list);
                    if (pt_list.size() > 0) {
                        vpt_array.push_back(pt_list);
                        total_list.splice(total_list.end(),pt_list);
                    }
                    else numunodes--;
                }
		 
	for (i=0; i<numunodes;i++) {
		for (it=vpt_array[i].begin(); it!=vpt_array[i].end();it++) {
			if (bDisplay) std::cout << *it <<' ' << V_data->vpoints[*it].isBnode()<<' ';
		}
		if (bDisplay) std::cout << std::endl;
	}
	if (bDisplay) std::cout << std::endl;

		 
	#if XY
		for (i=0;i<nodelist.size();i++) {
			int j;
			Coords xy;
			ElleNodePosition(nodelist[i],&xy);
			if (ElleFindVoronoiPt(xy,&j)==false) j=-1;
			if (bDisplay) std::cout << nodelist[i] <<' ' << j;
			if (bDisplay) std::cout << std::endl;
		}
		if (bDisplay) std::cout << std::endl;
		if (bDisplay) std::cout << std::endl;
		list<int> :: iterator it2;
		for (it2=total_list.begin();it2!=total_list.end();it2++) {
			if (bDisplay) std::cout << *it2 <<' ' ;
		}
		if (bDisplay) std::cout << std::endl;
		if (bDisplay) std::cout << std::endl;
		for (it2=total_list.begin();it2!=total_list.end();it2++) {
            if( std::count(total_list.begin(),
                                total_list.end(),*it2)==1) 
			if (bDisplay) std::cout << *it2 <<' ' ;
		}
		if (bDisplay) std::cout << std::endl;
	#endif

// Find the voronoi pts corresponding to Bnodes along the flynn
// boundary - record start and finish
// Note that the voronoi data has created a Bnode for any voronoi
// point on the flynn boundary so the density will greater than
// in the original elle file
                first = last = curr = NO_NB;

                i=startindex=0;
                // Cannot start or end boundary with tj
                while ( i<nodelist.size() &&
                               ElleNodeIsTriple(nodelist[i]) ) i++;
		 
                while ( i<nodelist.size() && first==NO_NB ) {
                    ElleNodePosition(nodelist[i],&xy);
                    if (ElleFindVoronoiPt(xy,&curr)==true) {
                        if (find(total_list.begin(),total_list.end(),curr)
                                                      != total_list.end()) {
                            first=curr;
                            startindex = i;
                        }
                    }
                    i++;
                }
                if (i>=nodelist.size()) {
                    return(1); // only one Bnode is a voronoi point
                }
                bool found = true;
                i++; if (i==nodelist.size()) i=0;
                endindex = i;
                while ( i<nodelist.size() && found ) {
                    ElleNodePosition(nodelist[i],&xy);
                    if (ElleFindVoronoiPt(xy,&curr)==true) {
                        if (find(total_list.begin(),total_list.end(),curr)
                                                      != total_list.end()) {
                            last=curr;
                            endindex = i;
                        }
                    }
                    else {
                        found = false;
                    }
                    i++;
                }
                if (startindex==0) {
                    found = false;
                    i = endindex+1;
                // Cannot start or end boundary with tj
                    while ( i<nodelist.size() &&
                               ElleNodeIsTriple(nodelist[i]) ) i++;
                    while ( i<nodelist.size() && !found ) {
                        ElleNodePosition(nodelist[i],&xy);
                        if (ElleFindVoronoiPt(xy,&curr)==true) {
                            if (find(total_list.begin(),total_list.end(),curr)
                                                      != total_list.end()) {
                                first=curr;
                                startindex=i;
                                found = true;
                            }
                        }
                        i++;
                    }
                }

		
// work on each lot of vpts
// If a point only occurs once in the total list of points,
// it must be on the outer edge so add it to the new boundary list.
// If it occurs more than once, we are at the intersection with another
// unode so find the list of points for that and keep walking
// The new boundary starts with a boundary Bnode, contains a list
// of internal voronoi points which occur in only once and ends with
// the other Bnode.
//
//        if (num_outer<1) this should just join the boundary endpts
//        which may be ok or may be a flat grain

                list<int> :: iterator it2;
                int num_outer= 0;
                for (it2=total_list.begin();
                                  it2!=total_list.end();it2++) {
                    if( std::count(total_list.begin(),
                                total_list.end(),*it2)==1) 
                        num_outer++;
                }
                //if (num_outer<1 || abs(startindex-endindex)<2) {
					
                if (num_outer<1 ) {
                    return(1); //do nothing, flynn cannot be split
                }

                next = last;
                int index = 0;
                found = false;
                i=0;
                while (index <  numunodes  && !found) {
                    if ((it=find(vpt_array[index].begin(),
                                      vpt_array[index].end(),next))
                                     ==vpt_array[index].end()) index++;
                    else found = true;
                }

        		if (!found) {
					 error_alb=1;
		            if (bDisplay) printf(" Did not find boundary node %d\n", error_alb);	
				}
		            		
                if (error_alb == 0) {
					 		
                    int wrap = 0;
                    vector<int> pts_in_bnd;
			        V_data->vpoints[*it].getPosition(&xy);
			        bndnode.setvalues(xy.x,xy.y,i++,1);
 if (bDisplay) std::cout << *it << " " << bndnode;
                    boundary.push_back(bndnode);
                    pts_in_bnd.push_back(*it);
                    it++;
                    if (it==vpt_array[index].end())
                        it = vpt_array[index].begin();
                        
                    while (*it!=first) {
                        if( std::count(total_list.begin(),
                                total_list.end(),*it)==1) {
                            if ((find(pts_in_bnd.begin(),
                                      pts_in_bnd.end(),*it))!= pts_in_bnd.end())
                              return(1);  // back-tracking
                            V_data->vpoints[*it].getPosition(&xy);
                            bndnode.setvalues(xy.x,xy.y,i++,1);
                            
                    //fVoronoiFile << xy.x << " " << xy.y << endl;
                            
 if (bDisplay) std::cout << *it << " " << bndnode;
                            boundary.push_back(bndnode);
                            pts_in_bnd.push_back(*it);
                            it++;
                            if (it==vpt_array[index].end()) {
                                if (wrap==0) {
                                  wrap++; 
                                  it = vpt_array[index].begin();
                                }
                                else {
                                  //dead path - start again
                                  return(1);
                                }
                            }
                        }
                        else {
                            next = *it;
                            index++; if (index==numunodes) index = 0;
                            int count = 0;
                            found = false;
                            while (count <  numunodes  && !found) {
                                if ((it=find(vpt_array[index].begin(),
                                         vpt_array[index].end(),next))
                                             ==vpt_array[index].end()) {
                                    count++;
                                    index++; if (index==numunodes) index=0;
                                }
                                else {
                                    found = true;
                                    wrap = 0;
                            if ((find(pts_in_bnd.begin(),
                                      pts_in_bnd.end(),*it))!= pts_in_bnd.end())
                              return(1);  // back-tracking
                            V_data->vpoints[*it].getPosition(&xy);
                            bndnode.setvalues(xy.x,xy.y,i++,1);                            
                            
 if (bDisplay) std::cout << *it << " " << bndnode;
                            boundary.push_back(bndnode);
                            pts_in_bnd.push_back(*it);
                            it++;
                            
                            if (it==vpt_array[index].end()) {
                                if (wrap==0) {
                                  wrap++; 
                                  it = vpt_array[index].begin();
                                }
                                else {
                                  //dead path - start again
                                  return(1);
                                }
                            }
                                }
                            }
                            if (!found) OnError("Did not find next node",0);

                        }
                    }

	 if (bDisplay) printf("first %d\n",boundary.size() ); 
	 if (bDisplay) printf("bnodes %d %d\n",nodelist[endindex], nodelist[startindex]); 
	 
                   // problem with walk if too many pts??
                    if (boundary.size()<2 ) return(1);
		
                    vector<int> tmplist;
                    vector<int> :: iterator vit;
                    tmplist.clear();

                    ElleGetFlynnUnodeList(k,tmplist);
                    ElleGetFlynnUnodeList(k,tmplist2);
                    		
// This is a version of splitFlynn which will go into the basecode
// when working.  This version accepts the boundary list rather
// than creating it with a directed or random walk.
// printf("nodelist %i\t %i\n",nodelist[endindex], nodelist[startindex]); 	
		
                    int child1, child2, childnew;
            		double area_1, area_2;
		
                    if (ElleNodeIsTriple( nodelist[endindex] ))
                    // need to insert double and set this as end node
                        return(1);
                                                            
                    err =  splitFlynn(k, &child1, &child2, nodelist[endindex],
                                      nodelist[startindex], boundary);
                                      
                                     
                    area_1=ElleRegionArea(child1);
                    area_2=ElleRegionArea(child2);
                    if (area_1>area_2) {
                      //maybe the area is -ve???
                      int tmp = child1;
                      child1 = child2;
                      child2 = tmp;
                    }
                    EllePromoteFlynn(child1);
                    
// This reassignment should be done automatically in the function
// EllePromoteFlynn() when a flynn no longer has a parent
// For now, do it here for speed as this process knows
// child1 is the small flynn
// and can reassign these unodes and give the rest of the list to
// the other child
// The PtInRegion check is needed because the boundary does not
// necessarily follow the exact outline of the unode voronoi

// do  a temp list of unodes ...  ... 

                    for (vit=tmplist.begin(); vit!=tmplist.end();vit++) {
                        ElleGetUnodePosition((*vit),&xy);
                        if (EllePtInRegion(child1,&xy)) {
                            ElleAddUnodeToFlynn(child1, *vit);
                            tmplist2.erase(find(tmplist2.begin(),tmplist2.end(),*vit));;
                        }
                    }
                    EllePromoteFlynn(child2);
                    
                    for (vit=tmplist2.begin(); vit!=tmplist2.end();vit++) {
                        ElleAddUnodeToFlynn(child2, *vit);
                    }
            		tmplist2.clear();
            		tmplist.clear();
	

                    /*	
	                    for (vit=unodelist.begin(); vit!=unodelist.end();vit++) {
                        ElleGetUnodePosition((*vit),&xy);
                        if (EllePtInRegion(child1,&xy)) {
                            ElleAddUnodeToFlynn(child1, *vit);
                            tmplist.erase(find(tmplist2.begin(),tmplist2.end(),*vit));;
                        }
                    }
                    EllePromoteFlynn(child2);

                    for (vit=tmplist.begin(); vit!=tmplist.end();vit++) {
                        ElleAddUnodeToFlynn(child2, *vit);
                    }
            		tmplist.clear();
            		*/
            		
// Add a check area to prevent reset of old grains, with changes of flynn number 
// smallest flynn is always the new flynn RX

                    area_1=ElleRegionArea(child1);
                    area_2=ElleRegionArea(child2);

if (bDisplay) printf ("area_1 & area_2 %e %e\n", area_1, area_2);
if (bDisplay) printf("child %i %i\n", child1, child2); 

		
                    if (area_1<=area_2) childnew=child1;
                    else childnew=child2;
					/*
					// no update unodes properties, only generate new bnodes 	
						tmplist.clear();
						ElleGetFlynnUnodeList(childnew,tmplist);
						if ( tmplist.size() <= 30 ) {  // add to prevent error of big unodes assignation    
							for (vit=tmplist.begin(); vit!=tmplist.end();vit++) {
							// std::cout << *vit <<' ' <<  ElleUnodeFlynn(*vit) << ' ';
							ElleSetUnodeAttribute(*vit,U_DISLOCDEN,new_den);	 // dislocation
							ElleSetUnodeAttribute(*vit,U_ATTRIB_C,double(childnew));
							ElleSetUnodeAttribute(*vit,U_DENSITY, double(subgrain_id));	 // subgrain ID	
		
							}

						}
                    */
                    
                    *new_child = childnew;
                    
	            } // add if error_alb
            }
        }
                            fVoronoiFile.close();

         ElleWriteData("u2f.elle");

if (bDisplay) printf("end\n");
    return(err);
}

                                                                                
int splitFlynn(int flynnindex, int *child1,int *child2,
                   int elle_first, int elle_last,
    				vector<PointData> &boundary)
{
    int *ids=0, i, old;
    int end, nb1, next, last;
    int f1, f2;
    int num=0;
    int err=0;
    ERegion rgn1, rgn2, oldrgn;
    double old_min = ElleminNodeSep();

// Reduce the minimum separation for nodes as
// fft-elle files have a large node sep and
// we need some bnodes along the new internal bnd!
    /*ElleSetMinNodeSep(old_min*0.3);*/

    num = boundary.size();
    ids = new int[num];
    for (i=0;i<num;i++) ids[i] = -1;
    ids[0] = elle_first;
    ids[num-1] = elle_last;
    /*
     * create 2 new children
    f1 = spareFlynn(); // first -> end
    f2 = spareFlynn(); // end -> first
    _a[flynnindex].addChild(f1);
    _a[flynnindex].addChild(f2);
    _a[f1].setParent(flynnindex);
    _a[f2].setParent(flynnindex);
     */
    f1 = ElleFindSpareFlynn();
    f2 = ElleFindSpareFlynn();
    ElleAddFlynnChild(flynnindex,f1);
    ElleAddFlynnChild(flynnindex,f2);
    rgn1 = f1;
    rgn2 = f2;
    oldrgn = flynnindex;
                                                                                     
    CreateBoundary2(boundary,f1,f2,ids,num,&end);
  
    if (ElleCheckForTwoSidedGrain(elle_first,&nb1)) 
        ElleRemoveTripleJLink(elle_first,nb1);
                                                                                
    next = old = end;
    ElleSetFlynnFirstNode(f1,elle_first);
    do {
        if ((nb1=ElleFindBndNode(next,old,flynnindex))==NO_NB)
            OnError("splitFlynn",NONB_ERR);
        if ((i = ElleFindNbIndex(nb1,next))==NO_NB)
            OnError("splitFlynn nb index",NONB_ERR);
        ElleSetRegionEntry(next,i,rgn1);
        old = next;
        next = nb1;
    } while (next!=elle_first);
    if ((old=ElleFindBndNode(next,old,rgn1))==NO_NB)
        OnError("splitFlynn",NONB_ERR);
    last = end;
    ElleSetFlynnFirstNode(f2,last);
    do {
        if ((nb1=ElleFindBndNode(next,old,flynnindex))==NO_NB)
            OnError("splitFlynn",NONB_ERR);
        if ((i = ElleFindNbIndex(nb1,next))==NO_NB)
            OnError("splitFlynn nb index",NONB_ERR);
        ElleSetRegionEntry(next,i,rgn2);
        old = next;
        next = nb1;
    } while (next!=last);
                                                                                                
    *child1 = f1;
    *child2 = f2;
                    
// Reset the node separation

    ElleSetMinNodeSep(old_min);
    delete [] ids;
    return(err);
}

int FindBnode(std::vector<int> nodes, Coords *xy)
{
    bool matched = false;
    int id = NO_VAL;
    double eps=ELLE_EPS; // same eps as crossings.cc and unodes.cc
    Coords nodexy;
    std::vector<int> :: iterator it;

    for (it=nodes.begin(); it!=nodes.end() && !matched; it++) {
        if (ElleNodeIsActive(*it)) ElleNodePlotXY(*it,&nodexy,xy);
        else OnError("Bnode no. invalid",RANGE_ERR);
        if (fabs(nodexy.x-xy->x)<eps && fabs(nodexy.y-xy->y)<eps) {
            matched = true;
            id = *it;
        }
    }
    return(id);
}

void CheckNodes()
{
    int j, max;
// The following lines check the bnode density
// This is equivalent to calling: tidy -i file -u 0
    ElleAddDoubles();
    int newnumdbl=0, numdbl=0, numtrp;
    ElleNumberOfNodes(&newnumdbl,&numtrp);
    while (newnumdbl!=numdbl)
    {
        numdbl = newnumdbl;
        max = ElleMaxNodes();
        for (j=0;j<max;j++)
        {
            if (ElleNodeIsActive(j)) {
                if (ElleNodeIsDouble(j)) ElleCheckDoubleJ(j);
			  // if (ElleNodeIsTriple(j)) ElleCheckTripleJ(j);	  
            }                                                                  
        }
        ElleNumberOfNodes(&newnumdbl,&numtrp);
    }
}
void CreateBoundary2(vector<PointData> &nodes,int nodenbbnd,
                    int nbnodebnd,
                    int *elle_id,int num,int *end)
{
    int next,curr,endindex,i;
    int nb1, nb2, newid, success=0;
    double min,sqx,sqy;
    Coords xy, try_xy, prev_xy;

    endindex = nodes[nodes.size()-1].idvalue();
    
    if (elle_id[endindex]==NO_NB) {
        i=(endindex+1)%num;
        while (elle_id[i]==NO_NB) i=(i+1)%num;
        nb1 = elle_id[i];
        i=(endindex-1+num)%num;
        while (elle_id[i]==NO_NB) i=(i-1+num)%num;
        nb2 = elle_id[i];
        ElleInsertDoubleJ(nb1,nb2,&newid,0.5);
        elle_id[endindex] = newid;
        xy.x = nodes[nodes.size()-1].xvalue();
        xy.y = nodes[nodes.size()-1].yvalue();
        //xy.x = xvals[endindex];
        //xy.y = yvals[endindex];
        ElleSetPosition(newid,&xy);
    }

    curr = next = elle_id[nodes[0].idvalue()];
    xy.x = nodes[0].xvalue();
    xy.y = nodes[0].yvalue();
    min = ElleminNodeSep() * 1.5;
    prev_xy=xy;
    for (i=0; i<nodes.size()-2; i++) {
        try_xy.x = nodes[i+1].xvalue();
        try_xy.y = nodes[i+1].yvalue();
        sqx =(prev_xy.x-try_xy.x)*(prev_xy.x-try_xy.x);
        sqy =(prev_xy.y-try_xy.y)*(prev_xy.y-try_xy.y);
        if((min*min)<(sqx+sqy)){
            xy=try_xy;
            next = ElleFindSpareNode();
            ElleSetPosition(next,&xy);
            ElleSetNeighbour(curr,NO_NB,next,&nodenbbnd);
            ElleSetNeighbour(next,NO_NB,curr,&nbnodebnd);
            prev_xy=xy;
            curr = next;
        } 
    }
    *end = elle_id[endindex];
    ElleSetNeighbour(next,NO_NB,*end,&nodenbbnd);
    ElleSetNeighbour(*end,NO_NB,next,&nbnodebnd);
}
