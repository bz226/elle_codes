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
#include "error.h"

using std::string;
using std::ofstream;
using std::list;
using std::vector;
using std::pair;
using std::cout;
using std::endl;

bool ElleCoordRelativePosition(Coords *current,
                              Coords *bnd,
                              int num_bnd_nodes,
                              int *xflags,int *yflags,
                              Coords *rel_pos);
int FindBnode(std::vector<int> nodes, Coords *xy);
int splitFlynn(int flynnindex, int *child1,int *child2, int first, int last,
    				vector<PointData> &boundary);
void updateLists(int node1, int node2, int nn, int curr_rgn);
int InitThisProcess(), Unode2Flynn();

const double ELLE_EPS=1.5e-6;

main(int argc, char **argv)
{
    int err=0;
    UserData udata;
    extern int InitThisProcess(void);

    /*
     * initialise
     */
    ElleInit();

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);
    ElleUserData(udata);
    ElleSetUserData(udata);

    if (err=ParseOptions(argc,argv))
        OnError("",err);

    /*
     * set the base for naming statistics and elle files
    ElleSetSaveFileRoot("u2f");
     */

    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * run your initialisation function and start the application
     */
    StartApp();

    CleanUp();

    return(0);
} 

/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitThisProcess()
{
    char *infile;
    int err=0;
    
    /*
     * clear the data structures
     */
    ElleReinit();

    ElleSetRunFunction(Unode2Flynn);

    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
    }
}

int Unode2Flynn()
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
	
	
    int i, k, j;
    int err=0,max,first, last,endindex, startindex, next, nbnode, curr;
    int unode_id = NO_NB;
    bool matched=false;
    double area=0;
    Coords xy, vxy, refxy;
    vector<PointData> boundary;
    std::list<int> nbflynns;
    std::list<int> ::iterator lit;
    UserData udata;
    extern VoronoiData *V_data;
	double new_den=0.00; // FULL CLEAN VERSION 
    double d_crit=1e11;
	double area_crit=5e-4, area_k; // minimum flynn area to nucleate a new grain
	double random_angle=5;
	double d_den, old_euler1,old_euler2,old_euler3,old_euler11;
	int error_alb=0, n_neighbours=1;

    /*!
      -u # unode to promote to flynn
     */
    ElleUserData(udata);
	
    unode_id = (int)udata[0];
	n_neighbours = (int)udata[1];
	d_crit = (double)udata[2];

	printf (" xx %i\t%i\t%e\n", unode_id, n_neighbours, d_crit);

    if (strlen(ElleOutFile())==0) ElleSetOutFile("u2f.elle");
    ElleCheckFiles();

    vector<int> unodelist;
	
    if (unode_id!=NO_VAL && UnodesActive() && unode_id<ElleMaxUnodes()){

		ElleGetUnodeAttribute(unode_id,U_DISLOCDEN,&d_den);	
		// printf("unode_id  d_den %i\t%e\n", unode_id, d_den);
        k = ElleUnodeFlynn(unode_id);
        area_k=ElleRegionArea(k);
		
// if Area is higher than a critic value, avoid split of tiny flynns  		
		
	if (d_den>d_crit && area_k>area_crit){ 			
	
	ElleGetUnodeAttribute(unode_id, &old_euler1,E3_ALPHA);
	ElleGetUnodeAttribute(unode_id, &old_euler2,E3_BETA);
	ElleGetUnodeAttribute(unode_id, &old_euler3,E3_GAMMA);
	
    if (unode_id!=NO_VAL && UnodesActive() && unode_id<ElleMaxUnodes()) {
        vector<int> tmplist;
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
        rect[0].x = rect[3].x = refxy.x - 5*roi;
        rect[1].x = rect[2].x = refxy.x + 5*roi;
        rect[0].y = rect[1].y = refxy.y - 5*roi;
        rect[2].y = rect[3].y = refxy.y + 5*roi;
        for (lit = nbflynns.begin();lit!=nbflynns.end();lit++) {
            tmplist.clear();
            ElleGetFlynnUnodeList(*lit,tmplist);
            for (vit = tmplist.begin();
                            vit != tmplist.end() && !err; vit++) {
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
        vector<int> nodelist, unodenbs, tmpnbs, tmpnbs2, tmpnbs3;
        list<int> total_list, total_unodenbs;
        list<int> :: iterator it;
        vector< list<int> > vpt_array;

        ElleFlynnNodes(k,nodelist);
        PointData bndnode;
// The unode's closest nbs
        ElleGetUnodeNearNbs(unode_id,unodenbs);
        copy(unodenbs.begin(),unodenbs.end(),
              inserter(total_unodenbs,total_unodenbs.end()));

if (n_neighbours > 1) {		
// The closest nbs of the unode's closest nbs
        for (i=0; i<unodenbs.size();i++) {
            tmpnbs.clear();
            ElleGetUnodeNearNbs(unodenbs[i],tmpnbs);
            copy(tmpnbs.begin(),tmpnbs.end(),
              inserter(total_unodenbs,total_unodenbs.end()));
        }
	}
// 3rd neighbours		
// The closest nbs of the unode's closest nbs
	
     if (n_neighbours > 2) {		
        for (i=0; i<tmpnbs.size();i++) {
            tmpnbs2.clear();
            ElleGetUnodeNearNbs(tmpnbs[i],tmpnbs2);
            copy(tmpnbs2.begin(),tmpnbs2.end(),
              inserter(total_unodenbs,total_unodenbs.end()));
        }
		printf("3rd neighbour");
	}
	
// 4rd neighbours		
// The closest nbs of the unode's closest nbs
	
     if (n_neighbours > 3) {		
        for (i=0; i<tmpnbs2.size();i++) {
            tmpnbs3.clear();
            ElleGetUnodeNearNbs(tmpnbs2[i],tmpnbs3);
            copy(tmpnbs3.begin(),tmpnbs3.end(),
              inserter(total_unodenbs,total_unodenbs.end()));
        }
		printf("4rt neighbour");
	}	
	
	
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
        for (it=total_unodenbs.begin(); it!=total_unodenbs.end();it++) {
            list<int> pt_list;
            ElleUnodeVoronoiPts(*it,pt_list);
            vpt_array.push_back(pt_list);
            total_list.splice(total_list.end(),pt_list);
        }
#if XY
for (i=0; i<numunodes;i++) {
for (it=vpt_array[i].begin(); it!=vpt_array[i].end();it++) {
std::cout << *it <<' ' << V_data->vpoints[*it].isBnode()<<' ';
}
std::cout << std::endl;
}
std::cout << std::endl;
for (i=0;i<nodelist.size();i++) {
int j;
Coords xy;
ElleNodePosition(nodelist[i],&xy);
if (ElleFindVoronoiPt(xy,&j)==false) j=-1;
std::cout << nodelist[i] <<' ' << j;
std::cout << std::endl;
}
std::cout << std::endl;
std::cout << std::endl;
list<int> :: iterator it2;
for (it2=total_list.begin();it2!=total_list.end();it2++) {
std::cout << *it2 <<' ' ;
}
std::cout << std::endl;
std::cout << std::endl;
for (it2=total_list.begin();it2!=total_list.end();it2++) {
            if( std::count(total_list.begin(),
                                total_list.end(),*it2)==1) 
std::cout << *it2 <<' ' ;
}
std::cout << std::endl;
#endif

// Find the voronoi pts corresponding to Bnodes along the flynn
// boundary - record start and finish
// Note that the voronoi data has created a Bnode for any voronoi
// point on the flynn boundary so the density will greater than
// in the original elle file
        first = last = curr = NO_NB;

        i=startindex=0;
        while ( i<nodelist.size() && first==NO_NB ) {
            ElleNodePosition(nodelist[i],&xy);
            if (ElleFindVoronoiPt(xy,&curr)==true) {
                if (find(total_list.begin(),total_list.end(),curr)
                      != total_list.end()) {
                    first=curr;
                    startindex = i;
                }
            }
            /*else {*/
                i++;
            /*}*/
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

		// printf("first ?? %i\n", first); 
		
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
        if (num_outer<1 || abs(startindex-endindex)<2) {
            ElleWriteData(ElleOutFile());
            return(1); //do nothing, flynn cannot be split
        }

        next = last;
        int index = 0;
        found = false;
        i=0;
        while (index <  numunodes  && !found) {
            if ((it=find(vpt_array[index].begin(),vpt_array[index].end(),next))
                   ==vpt_array[index].end()) index++;
            else found = true;
        }
       // if (!found) OnError("Did not find boundary node",0);
		if (!found) error_alb=1;
			
 if (error_alb == 0) { 		
        V_data->vpoints[*it].getPosition(&xy);
        bndnode.setvalues(xy.x,xy.y,i++,1);
        boundary.push_back(bndnode);
        it++;
        if (it==vpt_array[index].end()) it = vpt_array[index].begin();
        while (*it!=first) {
            if( std::count(total_list.begin(),
                                total_list.end(),*it)==1) {
                V_data->vpoints[*it].getPosition(&xy);
                bndnode.setvalues(xy.x,xy.y,i++,1);
                boundary.push_back(bndnode);
                it++;
                if (it==vpt_array[index].end()) it = vpt_array[index].begin();
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
                    }
                }
                if (!found) OnError("Did not find next node",0);
                it++;
                if (it==vpt_array[index].end()) it = vpt_array[index].begin();
            }
        }
	// printf("first %i\n", first); 
		
        vector<int> tmplist;
        vector<int> :: iterator vit;
        tmplist.clear();
        ElleGetFlynnUnodeList(k,tmplist);
		
// This is a version of splitFlynn which will go into the basecode
// when working.  This version accepts the boundary list rather
// than creating it with a directed or random walk.
// printf("nodelist %i\t %i\n",nodelist[endindex], nodelist[startindex]); 	
		
        int child1, child2, childnew;
		double area_1, area_2;
		
        err =  splitFlynn(k, &child1, &child2, nodelist[endindex],
                          nodelist[startindex], boundary);

        EllePromoteFlynn(child1);
// This reassignment should be done automatically in the function
// EllePromoteFlynn() when a flynn no longer has a parent
// For now, do it here for speed as this process knows
// child1 is the small flynn
// and can reassign these unodes and give the rest of the list to
// the other child
// The PtInRegion check is needed because the boundary does not
// necessarily follow the exact outline of the unode voronoi

        for (vit=unodelist.begin(); vit!=unodelist.end();vit++) {
            ElleGetUnodePosition((*vit),&xy);
            if (EllePtInRegion(child1,&xy)) {
                ElleAddUnodeToFlynn(child1, *vit);
                tmplist.erase(find(tmplist.begin(),tmplist.end(),*vit));;
            }
        }
        EllePromoteFlynn(child2);
        for (vit=tmplist.begin(); vit!=tmplist.end();vit++) {
            ElleAddUnodeToFlynn(child2, *vit);
        }
		tmplist.clear();
	
// Add a check area to prevent reset of old grains, with changes of flynn number 
// smallest flynn is always the new flynn RX

  area_1=ElleRegionArea(child1);
  area_2=ElleRegionArea(child2);

		printf ("area_1 & area_2 %e %e\n", area_1, area_2);
		printf("child %i %i\n", child1, child2); 
		
if (area_1<=area_2) childnew=child1;
else childnew=child2;
	
	ElleGetFlynnUnodeList(childnew,tmplist);
	if ( tmplist.size() <= 30 ) {  // add to prevent error of big unodes assignation    

		for (vit=tmplist.begin(); vit!=tmplist.end();vit++) {
			std::cout << *vit <<' ' <<  ElleUnodeFlynn(*vit) << ' ';
	
		// update information	
	ElleSetUnodeAttribute(*vit,U_DISLOCDEN,new_den);	
	ElleSetUnodeAttribute(*vit,E3_ALPHA,old_euler1);
	ElleSetUnodeAttribute(*vit,E3_BETA,old_euler2); 
	ElleSetUnodeAttribute(*vit,E3_GAMMA,old_euler3);  
	ElleSetUnodeAttribute(*vit,U_ATTRIB_C,double(childnew));
			
	}
// update information flynns
	ElleSetFlynnEuler3(childnew,old_euler1,old_euler2,old_euler3); 
	ElleSetFlynnRealAttribute(childnew,old_euler3,F_ATTRIB_A);

	}
std::cout << std::endl;
#if XY
#endif


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

        ElleWriteData(ElleOutFile());
		
		 } // add if error_alb
    }
}
	if (d_den>d_crit && area_k<area_crit) 	
	// For small flynns, RX with same boundaries, 
	// d_den == 0, and change euler of the critical unode 
	{
		vector<int> tmplist;
	    int kk;
		
		ElleGetFlynnUnodeList(k,tmplist); 
		ElleGetUnodeAttribute(unode_id, &old_euler1,E3_ALPHA);
		ElleGetUnodeAttribute(unode_id, &old_euler2,E3_BETA);
		ElleGetUnodeAttribute(unode_id, &old_euler3,E3_GAMMA);		
		k = ElleUnodeFlynn(unode_id);

		for (kk=0; kk<tmplist.size();kk++) {		
		// update information unodes	
		ElleSetUnodeAttribute(tmplist[kk],U_DISLOCDEN,new_den);	
		ElleSetUnodeAttribute(tmplist[kk],E3_ALPHA,old_euler1);
		ElleSetUnodeAttribute(tmplist[kk],E3_BETA,old_euler2); 
		ElleSetUnodeAttribute(tmplist[kk],E3_GAMMA,old_euler3);
	
		}
		// update information flynns
		ElleSetFlynnEuler3(k,old_euler1,old_euler2,old_euler3); 
		ElleSetFlynnRealAttribute(k,old_euler3,F_ATTRIB_A);		
        ElleWriteData(ElleOutFile());		
	}
}

// general check 
double max_nodes=ElleMaxNodes();
        for (j=0;j<max_nodes;j++) {
            if (ElleNodeIsActive(j))
                if (ElleNodeIsDouble(j)) ElleCheckDoubleJ(j);
                // else if (ElleNodeIsTriple(j)) ElleCheckTripleJ(j);
        }
		

    return(err);
}

                                                                                
int splitFlynn(int flynnindex, int *child1,int *child2,
                   int elle_first, int elle_last,
    				vector<PointData> &boundary)
{
    int *ids=0, i, old, num_elle_nodes;
    int end, nb1, next, last;
    int f1, f2;
    int err=0;
    ERegion rgn1, rgn2, oldrgn;
    double old_min = ElleminNodeSep();

// Reduce the minimum separation for nodes as
// fft-elle files have a large node sep and
// we need some bnodes along the new internal bnd!
       ElleSetMinNodeSep(old_min*0.3);

    ids = new int[boundary.size()];
    for (i=0;i<boundary.size();i++) ids[i] = -1;
    ids[0] = elle_first;
    ids[boundary.size()-1] = elle_last;
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
                                                                                
    CreateBoundary(boundary,f1,f2,ids,num_elle_nodes,&end);
                                                                                
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
#if XY
#endif

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
