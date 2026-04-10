#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include "convert.h"
#include "attrib.h"
#include "nodes.h"
#include "update.h"
#include "interface.h"
#include "display.h"
#include "file.h"
#include "error.h"
#include "string_utils.h"
#include "runopts.h"
#include "init.h"
#include "proximity.h"

using std::cout;
using std::endl;

#define MINSTEPS 500
int InitVM();
int Calc_Conc();
void SwapPointers();
void CalcConcAtNode(int n, double dt);
void UpdateConc(int max);
void SetupConc(double *minlen);
int CleanupConc();
void writeprofile(int cnt);

int *Neighbours=0;
int Attrib_id=CONC_A;
double *OldConc=0, *NewConc=0, *SegLength=0, *SegWidth=0;
double Minlen;
double KappaValue;
/*float Kappa=0.000005;*/

int InitVM()
{
    char *infile;
    int err=0, reset=0, start=NO_NB;
    double minlen_elle;
    double lastconc;
    UserData udata;

    ElleReinit();
    ElleSetRunFunction(Calc_Conc);
    ElleSetExitFunction(CleanupConc);
    CleanupConc();

    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
    }
    else /* warn that there is no file open */ ;

    KappaValue=KappaDflt;

    ElleUserData(udata);
    lastconc = udata[AttribValue];
    start = (int)udata[NodeId];
    Attrib_id = (int)udata[AttribId];
    reset = (start!=NO_NB);
    if (reset) {
        if ((err=ElleResetNodeConc((int)udata[NodeId],(int)udata[ResetNum],
                           udata[AttribValue],Attrib_id))==0)
            cout << "Resetting " << (int)udata[ResetNum]*2+1 <<
                 " nodes to " << udata[AttribValue] << endl;
        else
           OnError("Problem resetting concentration",0);
    }
    SetupConc(&minlen_elle);
    Minlen = minlen_elle*ElleUnitLength();

}

int Calc_Conc()
{
    int i, j, k;
    int interval=0,err=0,max;
    int *seq;
    int stages, steps=1;
	int cnt=0;
    int start=NO_NB, reset=0;
    FILE *fp;
    double timestep, dt=1.0, length, tmp;
    double lastconc;
    UserData udata;
    Coords incr;
    double total=0.0;
	int maxf,fid,fid2,mintype,*nodeids,numnodes,nid;
	int empty;
	char fname[100];
	double total_pmap;
	
    ElleUserData(udata);
    lastconc = udata[AttribValue];
    KappaValue = udata[Kappa];
    start = (int)udata[NodeId];
    Attrib_id = (int)udata[AttribId];
    reset = (start!=NO_NB);
    length = ElleUnitLength();

    timestep = ElleTimestep();
    if (EllemaxStages()==1) {
        /* 
         * using value from FE: dt <= len*len/KappaValue
         */
        dt = Minlen*Minlen/KappaValue*0.25;
        tmp = timestep/dt;
        if (tmp > (double)INT_MAX) 
            OnError("Timestep too large for process",0);
        steps = (int)(tmp);
        if (steps<MINSTEPS) {
            fprintf(stderr,"Calculated %d steps\n",steps);
            steps = MINSTEPS;
        }
        dt = timestep/steps;
        fprintf(stderr,"Running %d steps at dt of %le s = %le s\n",steps,dt,steps*dt);
        ElleSetStages(1);
    }
    else if (EllemaxStages()>1) {
        /*
         * if user sets stages then dt=timestep/stages
         */
		dt = timestep/EllemaxStages();
        if ((Minlen*Minlen/KappaValue*0.5) < dt)
            fprintf(stderr,
            "Warning: Kappa, node separation or timestep may need adjustment\n");
        fprintf(stderr,"Running %d stages at dt of %le s = %le s\n",EllemaxStages(),dt,EllemaxStages()*dt);
    }

    /*
     * calculation uses Elle "space" for calculations
     * so convert KappaValue to Elle length units
     */
    KappaValue = KappaValue / length / length;

    //ElleCheckFiles();
    //writeprofile(0);

    max = ElleMaxNodes();
	maxf = ElleMaxFlynns();
	for(fid=0;fid<maxf;fid++) // loop through all flynns
	{
		for(fid2=0;fid2<maxf;fid2++) // loop through all flynns and rest prox map to zero
		{
			if(ElleFlynnIsActive(fid2))
			{
				ElleFlynnNodes(fid2, &nodeids, &numnodes);
				for(nid=0;nid<numnodes;nid++) // reset all boundary concentrations to 0
				{
					if(ElleNodeIsActive(nodeids[nid]))
						NewConc[nodeids[nid]]=0.0;
				}
			}
		}
    	if(ElleFlynnIsActive(fid))
		{
			ElleGetFlynnIntAttribute(fid,&mintype,EXPAND);
			if(mintype == 1) // only for garnets
			{
				std::cout << fid << std::endl;
				ElleFlynnNodes(fid, &nodeids, &numnodes);
				for(nid=0;nid<numnodes;nid++) // reset all garnet boundary concentrations to 1
				{
					if(ElleNodeIsActive(nodeids[nid]))
						NewConc[nodeids[nid]]=1.0;
				}
				
				
				do{ // do diffusion until all boundary nodes non zero

    			  for (k=0;k<steps;k++) {
        			SwapPointers();
        			for (j=0;j<max;j++) {
            			if (ElleNodeIsActive(j)) {
                    			CalcConcAtNode(j,dt);
            			}
        			}
        			/* Update Elle nodes with NewConc */
        			UpdateConc(max);
        			if (reset) {
            			if ((err=ElleResetNodeConc(start,(int)udata[ResetNum],
                        			   udata[AttribValue],Attrib_id))!=0)
            			   OnError("Problem resetting concentration",0);
						cnt++;
        			}
        			//    if(i%100 == 0)
        			//        writeprofile(ElleCount());
        			//if (ElleDisplay())
            			//ElleUpdateDisplay();
    			  }
    			  //ElleUpdate();
				  empty=0;
        		  for (j=0;j<max;j++) { // check top see if all nodes non-zero yet
            			if (ElleNodeIsActive(j)) {
                    			if(NewConc[j] > 0.0)
									; // do nothing
								else
									empty=1;
            			}
        			}
				  				  
    			}while (empty);
				sprintf(fname,"gpm_%04d.elle",fid);
				ElleWriteData(fname);
				for(fid2=0;fid2<max;fid2++) // loop through all nodes and add to total proximity map (assumes starting file had zero N_ATTRIB_A)
				{
					if(ElleNodeIsActive(fid2))
					{
						total_pmap=ElleNodeAttribute(fid2, N_ATTRIB_A);
						ElleSetNodeAttribute(fid2, total_pmap+NewConc[fid2], N_ATTRIB_A);
					}
				}

			}
		}
	}
	//sprintf(fname,"%s_total.elle",ElleSaveFileRoot());
	//ElleWriteData(fname);
	total = ElleTotalNodeMass(Attrib_id);
	printf("mass = %.8e reset nodes %d times\n",total,cnt);
	ElleAutoWriteFile(1);
}

int CleanupConc(void)
{
    if (OldConc) free(OldConc);
    if (NewConc) free(NewConc);
    if (Neighbours) free(Neighbours);
    if (SegLength) free(SegLength);
    if (SegWidth) free(SegWidth);
}

void SetupConc(double *minlen)
{
    int max,i,j,k;
    int nbnodes[3];
    double bndwidth;
    Coords xy;

    bndwidth = ElleBndWidth()/ElleUnitLength();
    *minlen = 1.0;
    if (ElleFindNodeAttribIndex(Attrib_id)==NO_INDX) {
        /*
         * dummy concs only set up if node concentrations
         * not read from file
         */
        ElleAddDoubles();
        ElleInitNodeAttribute(Attrib_id);
        max = ElleMaxNodes();
        for (j=0;j<max;j++) {
            if (ElleNodeIsActive(j)) {
                ElleNodePosition(j,&xy);
                if (xy.x>0.25&&xy.x<0.75)
                    ElleSetNodeAttribute(j,0.3,Attrib_id);
                else ElleSetNodeAttribute(j,0.8,Attrib_id);
                /*if (j%2) ElleSetNodeAttribute(j,0.3,Attrib_id);*/
                /*else ElleSetNodeAttribute(j,0.8,Attrib_id);*/
            }
        }
    }
    
    max = ElleMaxNodes();
    if ((OldConc=(double *)malloc(max*sizeof(double)))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((NewConc=(double *)malloc(max*sizeof(double)))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((SegLength=(double *)malloc(max*sizeof(double)*3))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((SegWidth=(double *)malloc(max*sizeof(double)*3))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((Neighbours=(int *)malloc(max*sizeof(int)*3))==0)
        OnError("SetupConc",MALLOC_ERR);
    for (j=0;j<max;j++) {
        if (ElleNodeIsActive(j)) {
            NewConc[j] = OldConc[j] = ElleNodeAttribute(j,Attrib_id);
            ElleNeighbourNodes(j,nbnodes);
            i=0, k=0;
            while (i<3) {
                Neighbours[j*3+i] = NO_NB;
                SegLength[j*3+i] = 0;
                SegWidth[j*3+i] = bndwidth;
                if (nbnodes[i]!=NO_NB) {
                    Neighbours[j*3+k] = nbnodes[i];
                    SegLength[j*3+k] =
                             ElleNodeSeparation(j,nbnodes[i]);
                    if (SegLength[j*3+k]<(*minlen))
                        *minlen = SegLength[j*3+k];
                    k++;
                }
                i++;
            }
        }
    }
}

void CalcConcAtNode(int n, double dt)
{
    int i,j;
    int nbnodes[3], nb[3];
    double tmp1, tmp2, tmp3;
    double double_kappa; // because of Terence!

    double_kappa=KappaValue*2.0;

    tmp1 = tmp2 = tmp3 = 0;
    for (i=0;i<3;i++) {
        j = Neighbours[n*3+i];
        if (j!=NO_NB) {
            tmp1 += OldConc[j]*SegWidth[n*3+i]/SegLength[n*3+i];
            tmp2 += SegWidth[n*3+i]*SegLength[n*3+i];
            tmp3 += SegWidth[n*3+i]/SegLength[n*3+i];
        }
    }
    NewConc[n] =  OldConc[n]*(1 - double_kappa*dt*(tmp3/tmp2)) +
                                        (double_kappa*dt*tmp1/tmp2);
}

void SwapPointers()
{
    double *tmp;

    tmp = OldConc;
    OldConc = NewConc;
    NewConc = tmp;
}

void UpdateConc(int max)
{
    int j;

    for (j=0;j<max;j++) {
        if (ElleNodeIsActive(j))
            ElleSetNodeAttribute(j,NewConc[j],Attrib_id);
    }
}
// this works with the examples file, circle.elle
void writeprofile(int cnt)
{
    FILE *out;
    int i,min,max,nonodes=79;
//    static int ordering[160]={159, 10, 145, 110, 50, 19, 83, 73, 133, 130, 137, 32, 67, 30, 122, 84, 125, 44, 153, 147, 151, 69, 64, 108, 141, 89, 101, 107, 29, 98, 91, 88, 155, 115, 11, 23, 68, 74, 77, 128, 3, 13, 127, 102, 61, 75, 15, 94, 1, 131, 48, 16, 80, 58, 95, 7, 148, 99, 56, 24, 20, 117, 111, 54, 121, 146, 152, 76, 139, 21, 34, 5, 138, 150, 60, 38, 86, 2, 18, 28, 12, 103, 65, 157, 90, 79, 114, 33, 43, 124, 100, 140, 46, 87, 17, 126, 35, 112, 27, 62, 119, 40, 134, 116, 135, 97, 9, 22, 66, 70, 106, 4, 25, 6, 55, 118, 143, 142, 37, 113, 81, 72, 14, 158, 132, 31, 36, 96, 42, 78, 45, 104, 0, 51, 129, 71, 8, 59, 149, 47, 109, 123, 53, 154, 120, 156, 82, 57, 136, 52, 63, 26, 144, 93, 85, 41, 39, 92, 105, 49};
//    static int ordering[160]={18, 2, 17, 5, 76, 51, 75, 27, 52, 41, 68, 13, 67, 40, 54, 29, 77, 53, 78, 6, 16, 1, 87, 0};
    static int ordering[160]={81,72,14,158,132,31,36,96,42,78,45,104,0,51,129,71,8,59,149,47,109,123,53,154,120,156,82,57,136,52,63,26,144,93,85,41,39,92,105,49,159,10,145,110,50,19,83,73,133,130,137,32,67,30,122,84,125,44,153,147,151,69,64,108,141,89,101,107,29,98,91,88,155,115,11,23,68,74,77,128,3,13,127,102,61,75,15,94,1,131,48,16,80,58,95,7,148,99,56,24,20,117,111,54,121,146,152,76,139,21,34,5,138,150,60,38,86,2,18,28,12,103,65,157,90,79,114,33,43,124,100,140,46,87,17,126,35,112,27,62,119,40,134,116,135,97,9,22,66,70,106,4,25,6,55,118,143,142,37,113};
    UserData udata;

    ElleUserData(udata);

    out=fopen("conc_profile","a");
    fprintf(out,"%s\t%le\t%d\t",ElleFile(),udata[Kappa],cnt);

    for(i=40;i<nonodes+40;i++)
    {
        fprintf(out,"%e\t",ElleNodeAttribute(ordering[i],Attrib_id));
    }
    fprintf(out,"\n");
    fclose(out);
}
