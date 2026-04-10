#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "nodes.h"
#include "bflynns.h"
#include "check.h"
#include "error.h"
#include "polygon.h"
#include "runopts.h"
#include "file.h"
#include "errnum.h"
#include "interface.h"
#include "polygon.h"
#include "stats.h"
#include "init.h"
#include "fcntl.h"
#include "mat.h"
#include "misorient.h"

#ifdef _UNIX_
#include <sys/types.h>
#include <sys/stat.h>
#include<fcnt1.h>
#endif //_UNIX_

#define pi 3.1415927
#define MinMax_Orient 1 /*change to 0 if min_max ordering not wanted*/
#define Max_Orient 0    /*change to 0 if max ordering not wanted*/
#define Flynn_Area 1    /*change to 0 if no flynn area is wanted*/
#define Write_Elle 0    /* change to 0 if no elle file will be written*/


int InitThisProcess(), ProcessFunction();

int PanozzoAnalysis(FoliationData *data);

#ifdef __cplusplus
extern "C" {
#endif
    extern float ES_BoundaryLength();
#ifdef __cplusplus
}
#endif

void BinOrientMinMax(int node, double orient, double new_length,
				std::vector <float> &b_length);
void BinOrientMax(int node, double orient, double new_length,
				std::vector <float> &bmax_length);


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
        if (!ElleFlynnAttributeActive(MINERAL) || 
            !ElleFlynnAttributeActive(AGE) ||
            !ElleFlynnAttributeActive(CYCLE) ||
            !ElleFlynnAttributeActive(EULER_3)) 
            OnError(infile,INVALIDF_ERR);
      
*/
     } 
}

const int NUM_BINS=10;

int ProcessFunction()
{
    int i, j, k, max, maxno, size,split,min,grain, min_type, same;
    FILE *tmpstats;  /* declare tmpstats file*/ 
    double flynn_area,age,cycle;
    int sum_grains=0, grain_max;
    float g_max, g_min;
    float total_b_length;
    int nb[3],rgn[3],nn;
    FoliationData fdata;

    double misorient;
    double randorient;
    double sum_area, rgn_area, add_area=0, area,g_area;
    int grain1, grain2;
    double avgrain_size, seg_len;
    ERegion reg;
    double total;
    double sec_moment_size;
	std::vector<float> length (NUM_BINS,0.0);
	std::vector<float> max_length (NUM_BINS,0.0);
    

    tmpstats = fopen("tmpstats.dat","w");
    if (tmpstats == NULL){
        fprintf(stderr,"Error:Unable to locate tmpstats.dat file/n");
        exit(8);
    }

    for (i=0;i<EllemaxStages();i++) 
    {
        //
        // This will be the number of flynns which do not have children
        // but have a parent
        //
        maxno = ElleNumberOfSubgrains();
        fprintf(tmpstats,"%d\nmineral\tnumber\tarea\tcycle\tage\n\n",maxno);

        max = ElleMaxFlynns();
        if (Flynn_Area  ==1 &&
              ElleFlynnAttributeActive(MINERAL) && 
              ElleFlynnAttributeActive(AGE) &&
              ElleFlynnAttributeActive(CYCLE))
        {
          for (j=0;j<max;j++)
      {
            if (ElleFlynnIsActive(j) && !isParent(j)) {
                ElleGetFlynnRealAttribute(j,&cycle,CYCLE);
                ElleGetFlynnIntAttribute(j,&min,MINERAL);
                if (min==QUARTZ && cycle>0){
                    flynn_area = ElleFlynnArea(j);
                    ElleGetFlynnRealAttribute(j,&age,AGE);
                    fprintf(tmpstats,"Quartz\t%d\t%f\t%f\t%f\n",
                     j,flynn_area,cycle,age);                      
                    sum_grains++;
                }
             }  
           }
        }
        
        if ( ElleFlynnAttributeActive(MINERAL) && 
              ElleFlynnAttributeActive(AGE) &&
              ElleFlynnAttributeActive(CYCLE))
        {
        for (j=0;j<max;j++)
        {
            if (ElleFlynnIsActive(j) && !isParent(j)){
               ElleGetFlynnRealAttribute(j,&cycle,CYCLE);
               ElleGetFlynnIntAttribute(j,&min,MINERAL);
               if (min==FELDSPAR && cycle>0){
                   flynn_area = ElleFlynnArea(j);
                   ElleGetFlynnRealAttribute(j,&age,AGE);
                   fprintf(tmpstats,"Fsp\t%d\t%.6lf\t%.6lf\t%.6lf\n",
                       j,flynn_area,cycle,age);                   
                    sum_grains++;
               }
            }
             
        }
        }

        if (sum_grains==0) // minerals not present
            sum_grains = ElleNumberOfGrains();
        avgrain_size=(double)(1.0/sum_grains);
        
        fprintf(tmpstats,"\tgrain number\tarea\n");
        max = ElleMaxFlynns();
        for (i=0,total=0.0;i<max;i++) {
            if (ElleFlynnIsActive(i) && !hasParent(i)) {
                g_area = ElleFlynnArea(i);
                total += ((g_area/avgrain_size - 1) *(g_area/avgrain_size -1));
                fprintf(tmpstats,"\t%d\t%.6lf\n", i, g_area);
            }
        }
        sec_moment_size = ((double)(total/sum_grains));
        fprintf(tmpstats,"total grain number\t%d\n",sum_grains);
        fprintf(tmpstats,"average grain size\t%.6lf\n",avgrain_size);
        fprintf(tmpstats,"second moment grain size\t%.6lf\n",sec_moment_size); 
        
        if (ElleFlynnAttributeActive(EULER_3)) {
        CalculateBoundaryAttribute(MISORIENTATION);
        max = ElleMaxNodes();
        for (j=0;j<max;j++) {
            if (ElleNodeIsActive(j)) {
                ElleNeighbourNodes(j,nb);
                for (i=0;i<3;i++) {
                  // look at each boundary segment once
                  // misorient will be unset (-1) if segment is phase boundary
                    if (nb[i]!=NO_NB && nb[i]<j) {
                        seg_len = ElleNodeSeparation(j,nb[i]);
                        misorient=ElleGetBoundaryAttribute(j,nb[i]);
                        if (misorient>=0 && MinMax_Orient==1)
                            BinOrientMinMax(j,misorient,seg_len,length);
                        if (misorient>=0 && Max_Orient==1)
                            BinOrientMax(j,misorient,seg_len,max_length);
                    }
                }
            }
        } 
        if (MinMax_Orient==1){
           fprintf(tmpstats,"min_max orientation statistics \n");
          
        fprintf(tmpstats,"<=5\t%f\n5-10\t%f\n10-20\t%f\n20-40\t%f\n40-60\t%f\n<90\t%f\n<=180\t%f\n",
                            length[1], length[2],length[3],
                            length[4], length[5],length[6],
                            length[7]);
                                          
        }
        if (Max_Orient==1){
            fprintf(tmpstats,"max orientation statistics \n");
            fprintf(tmpstats,"<=5\t%f\n<=10\t%f\n<=20\t%f\n<=40\t%f\n<=60\t%f\n<=90\t%f\n<=180\t%f\n",
                            max_length[1],max_length[2],max_length[3],
                            max_length[4],max_length[5],max_length[6],
                            max_length[7]);
                                          
        }
        }
        total_b_length=ES_BoundaryLength();
        fprintf(tmpstats,"total_b_length\t%f\n",total_b_length);

        PanozzoAnalysis(&fdata);
        fprintf(tmpstats,"ratio\t%f\n",fdata.ratio);
        fprintf(tmpstats,"maxAng\t%f\n",fdata.maxAng);
        fprintf(tmpstats,"minAng\t%f\n",fdata.minAng);
        fprintf(tmpstats,"accuracy\t%f\n",fdata.accuracy);
/*
        if (Write_Elle == 0) {
           max = ElleMaxNodes();
           for (j=0;j<max;j++) {
               if (ElleNodeIsActive(j) && (ElleNodeAttribute(j,N_ATTRIB_B)==1)) {
                   ElleDeleteDoubleJNoCheck(j);
                   ElleSetNodeAttribute(j,0,N_ATTRIB_B);
               }
           }
        }
  
        if (Write_Elle == 1)  ElleAutoWriteFile(ElleCount());
*/
        
    }
    fprintf(tmpstats,"\n");
    fclose(tmpstats);
    
} 
void BinOrientMinMax(int node, double orient, double new_length,
				std::vector <float> &b_length)
{  
   int index, i, j, nbnodes[3];
   
   // check NUM_BINS if this code is changed
            if (orient<=5) b_length[1]+=(float)new_length;
            else if (orient>5 && orient<=10) b_length[2]+=(float)new_length;
            else if (orient>10 && orient<=20) b_length[3]+=(float)new_length;
            else if (orient>20 && orient<=40) b_length[4]+=(float)new_length;
            else if (orient>40 && orient<=60) b_length[5]+=(float)new_length;
            else if (orient>60 && orient<=90) b_length[6]+=(float)new_length;
            else if (orient>90 && orient<=180) b_length[7]+=(float)new_length;
/*
   ElleNeighbourNodes(node,nbnodes);
   for (j=0;j<3;j++) {
        if (nbnodes[j]!=NO_NB && nbnodes[j]<node){
            if (orient<=5) b_length[1]+=(float)new_length;
            else if (orient>5 && orient<=10) b_length[2]+=(float)new_length;
            else if (orient>10 && orient<=20) b_length[3]+=(float)new_length;
            else if (orient>20 && orient<=40) b_length[4]+=(float)new_length;
            else if (orient>40 && orient<=60) b_length[5]+=(float)new_length;
            else if (orient>60 && orient<=90) b_length[6]+=(float)new_length;
            else if (orient>90 && orient<=180) b_length[7]+=(float)new_length;
        }
        if (nbnodes[j]!=NO_NB && nbnodes[j]>node && ElleNodeIsTriple(nbnodes[j])){
            if (orient<=5) b_length[1]+=(float)new_length;
            else if (orient>5 && orient<=10) b_length[2]+=(float)new_length;
            else if (orient>10 && orient<=20) b_length[3]+=(float)new_length;
            else if (orient>20 && orient<=40) b_length[4]+=(float)new_length;
            else if (orient>40 && orient<=60) b_length[5]+=(float)new_length;
            else if (orient>60 && orient<=90) b_length[6]+=(float)new_length;
            else if (orient>90 && orient<=180) b_length[7]+=(float)new_length;
        }

   }
*/

 
}

void BinOrientMax(int node, double orient, double new_length,
				std::vector <float> &bmax_length)
{  
   int index, i, j, nbnodes[3];
   
            if (orient<=5) bmax_length[1]= bmax_length[1]+new_length;
            if (orient<=10) bmax_length[2]+=new_length;
            if (orient<=20) bmax_length[3]+=new_length;
            if (orient<=40) bmax_length[4]+=new_length;
            if (orient<=60) bmax_length[5]+=new_length;
            if (orient<=90) bmax_length[6]+=new_length;
            if (orient<=180) bmax_length[7]+=new_length;
/*
   float new_length;
   ElleNeighbourNodes(node,nbnodes);
   for (j=0;j<3;j++) {
        if (nbnodes[j]!=NO_NB && nbnodes[j]<node){
            new_length=ElleNodeSeparation(node,nbnodes[j]);
            if (orient<=5) bmax_length[1]+=new_length;
            if (orient<=10) bmax_length[2]+=new_length;
            if (orient<=20) bmax_length[3]+=new_length;
            if (orient<=40) bmax_length[4]+=new_length;
            if (orient<=60) bmax_length[5]+=new_length;
            if (orient<=90) bmax_length[6]+=new_length;
            if (orient<=180) bmax_length[7]+=new_length;
         }
         if (nbnodes[j]!=NO_NB && nbnodes[j]>node && ElleNodeIsTriple(nbnodes[j])){
            new_length=ElleNodeSeparation(node,nbnodes[j]);
            if (orient<=5) bmax_length[1]= bmax_length[1]+new_length;
            if (orient<=10) bmax_length[2]+=new_length;
            if (orient<=20) bmax_length[3]+=new_length;
            if (orient<=40) bmax_length[4]+=new_length;
            if (orient<=60) bmax_length[5]+=new_length;
            if (orient<=90) bmax_length[6]+=new_length;
            if (orient<=180) bmax_length[7]+=new_length;
         }

   }
*/

 
}

int PanozzoAnalysis(FoliationData *data)
{
    int max, i, j, k, nbnodes[3];
    double len, totlen, dx, dy;
    double rosedata[181], pabs, p,alfa, cosalfa;
    double minL, maxL, minA, maxA;
    Coords xy, xynb;

    totlen = 0.0;
    for (i=0; i<181; i++) rosedata[i] = 0.0;

    max = ElleMaxNodes();
    for (i=0; i<max; i++) {
        if (ElleNodeIsActive(i)) {
            ElleNeighbourNodes(i,nbnodes);
            for (j=0;j<3;j++) {
                if (nbnodes[j]!=NO_NB && nbnodes[j]>i) {
                    ElleNodePosition(i,&xy);
                    ElleNodePlotXY(nbnodes[j],&xynb,&xy);
                    dx = (xy.x - xynb.x);
                    dy = (xy.y - xynb.y);
                    len = sqrt(dx*dx + dy*dy);
                    if (len > 0.0) {
                        totlen += len;
                        cosalfa = dy/len;
                        alfa = acos(cosalfa);
                        if (dx < 0.0) alfa = M_PI-alfa;
                        for (k=0;k<180;k++) {
                            pabs = cos(alfa - (double)k/180.0*M_PI);
                            p = len * fabs(pabs);
                            rosedata[k] += p;
                        }
                    }
                    else
                        OnError("PanozzoAnalysis-len 0",0);
                }
            }
        }
    }
    minL = maxL = rosedata[0];
    minA = maxA = 0;
    for (i=0; i<180; i++) {
        if (rosedata[i] < minL) {
            minL = rosedata[i];
            minA = (double)i;
        }
        if (rosedata[i] > maxL) {
            maxL = rosedata[i];
            maxA = (double)i;
        }
    }
    if (maxL > 0.0)
        for (i=0; i<181; i++) rosedata[i] /= maxL;
    if (minL > 0.0) data->ratio = (float)(maxL/minL);
    data->maxAng = (float)maxA;
    data->minAng = (float)minA;
    data->gblength = totlen;
    if (maxA > minA) data->accuracy = (float)fabs(maxA-minA-90.0);
    else data->accuracy = (float)fabs(minA-maxA-90.0);
}
