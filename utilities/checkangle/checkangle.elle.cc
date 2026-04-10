#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <list>
#include "flynnarray.h"
#include "interface.h"
#include "attrib.h"
#include "nodes.h"
#include "unodes.h"
#include "display.h"
#include "check.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "polygon.h"
#include "stats.h"
#include "init.h"
#include "general.h"
#include "convert.h"
#include "update.h"
#include "setup.h"
#include "parseopts.h"
#include "checkangle.h"

using std::vector;

int InitThisProcess(), ProcessFunction();
int IncreaseAngle(Coords *xy,Coords *xy1,Coords *xy2,Coords *diff);
void CheckAngle (double min_ang);
void CheckArea(double min_area);
void CheckRatio(double max_ratio);

int UpdateAttributes( vector<int> &reassigned, set_int *enrichlist,
                      double density_min);

/*
 *  main
 */
main(int argc, char **argv)
{
    int err=0;
    extern int InitThisProcess(void);
    UserData udata;
 
    /*
     * initialise
     */
    ElleInit();

    ElleUserData(udata);
    udata[INDEXMINANGLE] = MINANGLE;
    udata[INDEXMINAREAFACTOR] = MINAREAFACTOR;
    udata[INDEXMAXRATIO] = MAXRATIO;
    ElleSetUserData(udata);
    
    if (err=ParseOptions(argc,argv))
        OnError("",err);

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);


    /*
     * set the interval for writing to the stats file
    ES_SetstatsInterval(100);
     */
    ElleSetDisplay(0);
    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);
    
    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("checkangle");
    ElleSetSaveFrequency(1);
    /*
     * run your initialisation function and start the application
     */
    StartApp();
    
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

    ElleSetRunFunction(ProcessFunction);

    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
        /*
         * initialise any necessary attributes which may
         * not have been in the elle file
        if (!ElleFlynnAttributeActive(DISLOCDEN)) {
            ElleInitFlynnAttribute(DISLOCDEN);
        }
         */
         
    }
    
}

int ProcessFunction()
{
    int i,j,k,max;
    int err;
    UserData udata;
    double min_area;


    ElleUserData(udata);

    if (udata[INDEXMINANGLE]>0)
        CheckAngle(udata[INDEXMINANGLE]);

    min_area = ElleminNodeSep() * ElleminNodeSep() * SIN60 * 0.5;
    if (udata[INDEXMINAREAFACTOR]>0) {
        min_area *= udata[INDEXMINAREAFACTOR];
        CheckArea(min_area);
    }

    if (udata[INDEXMAXRATIO]>0)
        CheckRatio(udata[INDEXMAXRATIO]);

    max = ElleMaxNodes(); 
    for (k=0;k<max;k++)
        if (ElleNodeIsActive(k)&& ElleNodeIsDouble(k))
            ElleCheckDoubleJ(k);
    ElleAddDoubles();
    if (udata[INDEXMINANGLE]>0)
        CheckAngle(udata[INDEXMINANGLE]);
    ElleUpdate();
    
} 

void CheckAngle(double min_ang)
{
    int moved=1, removed=1,i,j,k,max, count=0;
    int nbnodes[3], nb[3], same, min_type;
    double currang,flynn_area;
    
    double ang,dist;
    Coords xy[3], movedist;
    int *ids, num;

    max = ElleMaxNodes();
    while (moved)  {
    for (k=0,moved=0;k<max;k++) {
        if (ElleNodeIsActive(k)) {
            ElleNodePosition(k,&xy[0]);
            ElleNeighbourNodes(k,nbnodes);
            if (ElleNodeIsDouble(k)) {
                j=0; i=1;
                while (j<3) {
                    if (nbnodes[j]!=NO_NB){
                        nb[i]=nbnodes[j];
                        ElleNodePlotXY(nbnodes[j],&xy[i++],&xy[0]);
                    }
                    j++;
                    
                }
                angle0(xy[0].x,xy[0].y,xy[1].x,xy[1].y,xy[2].x,xy[2].y,
                                                             &currang);
                //if(currang==0) OnError("angle error",0);
                        
                ang = fabs(currang);
                  
                if (ang<min_ang && (EllePhaseBoundary(k,nb[1])==1)){
                  if( EllePhaseBoundary(k,nb[2])==1 ){
                    printf("Problem: both boundaries are two phase boundaries");
                  }
                  else 
                  {
                    count=0;
                    do {
                        IncreaseAngle(&xy[0],&xy[1],&xy[2],&movedist);
                        xy[0].x += movedist.x;
                        xy[0].y += movedist.y;
                        angle0( xy[0].x,xy[0].y,xy[1].x,xy[1].y,
                                            xy[2].x,xy[2].y,&currang);
                        //if (currang==0)   OnError("angle error",0);
                        ang = fabs(currang);
                        dist=(xy[0].x-xy[1].x)*(xy[0].x-xy[1].x)+
                                 (xy[0].y-xy[1].y)*(xy[0].y-xy[1].y);
                        dist=sqrt(dist);
                        count++;
                     } while (ang<min_ang  &&
                                 (dist>ElleSwitchdistance()*0.1));
                     if (count>1) {
                         // moved at least one step before dist too small
                         ElleSetPosition(k,&xy[0]);
                         ElleCheckDoubleJ(k);
                         moved = 1;
                         printf("movedd1 %d\t\n",k);
                     }
                  }
                    
                 }
                 else if (ang<min_ang /*&& (EllePhaseBoundary(k,nb[2])==1)*/)
                 {
                    count=0;
                    do {
                        IncreaseAngle(&xy[0],&xy[2],&xy[1],&movedist);
                        xy[0].x += movedist.x;
                        xy[0].y += movedist.y;
                        angle0( xy[0].x,xy[0].y,xy[2].x,xy[2].y,
                                            xy[1].x,xy[1].y,&currang);
                        //if (currang==0)   OnError("angle error",0);
                        ang = fabs(currang);
                        dist=(xy[0].x-xy[2].x)*(xy[0].x-xy[2].x)+
                                 (xy[0].y-xy[2].y)*(xy[0].y-xy[2].y);
                        dist=sqrt(dist);
                        count++;
                     } while (ang<min_ang  &&
                                 (dist>ElleSwitchdistance()*0.1));
                     if (count>1) {
                         // moved at least one step before dist too small
                        ElleSetPosition(k,&xy[0]);
                        ElleCheckDoubleJ(k);
                        moved = 1;
                        printf("movedd2 %d\t\n",k);
                     }
                  }
               }
               else if (ElleNodeIsTriple(k)) {
                for (j=0;j<3 ;j++) {
                    i = (j+1)%3;
                    ElleNodePlotXY(nbnodes[j],&xy[1],&xy[0]);
                    ElleNodePlotXY(nbnodes[i],&xy[2],&xy[0]);
                      angle0(xy[0].x,xy[0].y,xy[1].x,xy[1].y,xy[2].x,xy[2].y,
                                                             &currang);
                      //if (currang==0)  OnError("angle error",0);
                      ang = fabs(currang);

                      if (ang<min_ang && (EllePhaseBoundary(k,nbnodes[j])==1)){
                        if((EllePhaseBoundary(k,nbnodes[j])==1)&&
                           (EllePhaseBoundary(k,nbnodes[i])==1)){
                           printf("Problem: both boundaries are two phase boundaries\n");
                        }
                      else {

                   // if (ang<min_ang /*|| ang>(M_PI-min_ang)*/) 
                        
                        count=0;
                        do {
                            IncreaseAngle(&xy[0],&xy[1],&xy[2],&movedist);
                            xy[0].x += movedist.x;
                            xy[0].y += movedist.y;
                            angle0(xy[0].x,xy[0].y,xy[1].x,xy[1].y,
                                            xy[2].x,xy[2].y,&currang);
                            //if (currang==0)   OnError("angle error",0);
                            ang = fabs((double)currang);
                            //printf("ang %lf\t\n",ang);
                            dist=(xy[0].x-xy[1].x)*(xy[0].x-xy[1].x)+
                                 (xy[0].y-xy[1].y)*(xy[0].y-xy[1].y);
                            dist=sqrt(dist);
                            count++;
                        } while (ang<min_ang  &&
                                    (dist>ElleSwitchdistance()*0.1)); 
                        if (count>1) {
                         // moved at least one step before dist too small
                            ElleSetPosition(k,&xy[0]);
                            ElleCheckTripleJ(k);
                            moved = 1;
                            printf("movedt1 %d\t\n",k);
                        }
                        j=3;
                      }
                    }
                  else if (ang<min_ang /*&& (EllePhaseBoundary(k,nbnodes[i])==1)*/)
                    {
                       count=0;
                       do {
                            IncreaseAngle(&xy[0],&xy[2],&xy[1],&movedist);
                            xy[0].x += movedist.x;
                            xy[0].y += movedist.y;
                            angle0( xy[0].x,xy[0].y,xy[2].x,xy[2].y,
                                            xy[1].x,xy[1].y,&currang);
                            //if (currang==0)   OnError("angle error",0);
                            ang = fabs(currang);
                            //printf("ang %lf\t\n",ang);
                            dist=(xy[0].x-xy[2].x)*(xy[0].x-xy[2].x)+
                                 (xy[0].y-xy[2].y)*(xy[0].y-xy[2].y);
                            dist=sqrt(dist);
                            count++;
                        } while (ang<min_ang &&
                                    (dist>ElleSwitchdistance()*0.1));
                        if (count>1) {
                         // moved at least one step before dist too small
                            ElleSetPosition(k,&xy[0]);
                            ElleCheckTripleJ(k);
                            moved = 1;
                            printf("movedt2 %d\t\n",k);
                        }
                        j=3;
                    }
                  }
                }
            }
        }
    }
}
 
int IncreaseAngle(Coords *xy,Coords *xy1,Coords *xy2,Coords *diff)
{
    Coords xynew;

    /*
     * 
     * move 0.1 of the distance towards neighbour 1 along boundary
     */
    
    diff->x = xy1->x - xy->x;
    diff->y = xy1->y - xy->y;
    diff->x *= 0.1*ElleSwitchdistance();
    diff->y *= 0.1*ElleSwitchdistance();
}

void CheckArea(double min_area)
{
    int removed=1,merged=1,j,k,max;
    int rgn[3], min_type;
    int min[2];
    int *attribs=0, num_attribs =0;
    double flynn_area;
    double density_min=0.0;
    vector<int> reassigned;
    set_int *unodeset[1];

    if (ElleUnodesActive()) UnodeAttributeList(&attribs, &num_attribs);
    if (num_attribs>0 && ElleUnodeAttributeActive(U_DISLOCDEN))
        ElleGetDefaultUnodeAttribute(&density_min,U_DISLOCDEN);
    min[0] = min[1] = 0;
    while (removed)
    {
      max = ElleMaxFlynns(); 
      for (j=0,removed=0;j<max;j++)
      {
         if (ElleFlynnIsActive(j)){ 
            flynn_area = ElleRegionArea(j);
            if (flynn_area<min_area)
            { 
               reassigned.clear();
               ElleGetFlynnUnodeList(j,reassigned);
               if (ElleFlynnAttributeActive(MINERAL))
                   ElleGetFlynnMineral(j,&min[0]);
			   std::list<int> rgn;
               ElleFlynnNbRegions(j,rgn);

			   std::list<int>::iterator its=rgn.begin();
               for (k=0,merged=0;its!=rgn.end() && !merged;k++,its++) {
                   if (ElleFlynnAttributeActive(MINERAL))
                       ElleGetFlynnMineral(*its,&min[1]);
                   if (min[0]==min[1] && ElleMergeFlynnsNoCheck(*its,j)==0) {
                       merged=1;
                       removed=1;
                       printf("merged %d into %d\n",j,*its);
                       int err=0;
                       unodeset[0] = new set_int;
                       vector<int> unodelist;
                       ElleGetFlynnUnodeList(*its,unodelist);

                       for (vector<int> :: iterator vit = unodelist.begin();
                            vit != unodelist.end() && !err; vit++)
                           unodeset[0]->insert(*vit);
                       if (num_attribs>0 && reassigned.size()>0)
                           err=UpdateAttributes(reassigned,unodeset[0],
                                  density_min);
                       delete unodeset[0];
                   }
               }
               if (!merged) printf("Could not merge %d - two phases\n",j);
            }
          }
        }
     }
     if (attribs!=0) free(attribs);
}

void CheckRatio(double max_ratio)
{
    int removed=1,merged=1,j,k,max;
    int rgn[3], min_type;
    double flynn_area;
    FoliationData fdata;

    while (removed)
    {
      max = ElleMaxFlynns(); 
      for (j=0,removed=0;j<max;j++)
      {
         if (ElleFlynnIsActive(j)){ 
            ES_PanozzoAnalysis(j,&fdata);
printf("%d %f\n",j,fdata.ratio);
            if (fdata.ratio>max_ratio)
            { 
			   std::list<int> rgn;
               ElleFlynnNbRegions(j,rgn);

			   std::list<int>::iterator its=rgn.begin();
               for (k=0,merged=0;its!=rgn.end() && !merged;k++,its++) {
                   if (ElleMergeFlynns(*its,j)==0) {
                       merged=1;
                       removed=1;
                       printf("merged %d into %d\n",j,*its);
                   }
               }
               if (!merged) printf("Could not merge %d - different attributes\n",j);
            }
          }
        }
     }
}

/*!
        \brief          Update unode attributes when a flynn is merged into
                        and adjacent flynn

        \param          reassigned - vector of unodes to be updated
                        enrichlist - set of unodes in new combined flynn
                        density_min - updated disloc density value (use default?)
        \return         \a err - this is not reset in the function 

        \par            Description:
                        For each unode in the reassigned list, find the closest
                        unode in the enrichlist that is not in the reassigned
                        list. Update the orientation of the ressigned unode to
                        that of the closest unode. Update the dislocden density
                        to density_min.  Need to check and update any other
                        active unode attributes.

        \exception
                        No checks, no exceptions trapped

        \par            Example:

\verbatim
                       double density_min=0.0;
                       vector<int> reassigned;
                       set_int *unodeset[1];
*/
int UpdateAttributes( vector<int> &reassigned, set_int *enrichlist,
                      double density_min)
{
    bool not_set=true;
    int err=0, closest;
    double val_euler[3], dist, min_dist;
    vector<int> :: iterator it;
    set_int :: iterator it2;
    Coords xy, ref;

    for (it = reassigned.begin(); it != reassigned.end(); it++) {
        not_set=true;
        ElleGetUnodePosition(*it,&ref);
        for (it2 = enrichlist->begin(); it2 != enrichlist->end(); it2++) {
            if (find(reassigned.begin(),reassigned.end(),*it2)==
                                         reassigned.end()) {
                ElleGetUnodePosition(*it2,&xy);
                ElleCoordsPlotXY (&ref, &xy);
                dist = pointSeparation(&ref,&xy);
                if ( not_set==true) {
                    min_dist = dist;
                    closest = *it2;
                    not_set = false;
                }
                else if ( dist < min_dist) {
                    min_dist = dist;
                    closest = *it2;
                }
            }
        }
        ElleSetUnodeAttribute(*it, density_min, U_DISLOCDEN);
        ElleGetUnodeAttribute(closest, &val_euler[0],E3_ALPHA);
        ElleGetUnodeAttribute(closest, &val_euler[1],E3_BETA);
        ElleGetUnodeAttribute(closest, &val_euler[2],E3_GAMMA);
         ElleSetUnodeAttribute(*it, val_euler[0],E3_ALPHA);
         ElleSetUnodeAttribute(*it, val_euler[1],E3_BETA);
         ElleSetUnodeAttribute(*it, val_euler[2],E3_GAMMA);
    }
    return(err);
}
