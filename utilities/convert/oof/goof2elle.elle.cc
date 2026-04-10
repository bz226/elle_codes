#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "attrib.h"
#include "nodes.h"
#include "display.h"
#include "check.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "polygon.h"
#include "stats.h"
#include "init.h"
#include "mat.h"
#include "log.h"

int DoSomethingToFlynn(int flynn);
int InitThisProcess(), ProcessFunction();
int ReadPolyFile(FILE *xa, int **x);
int ReadGoofFile(FILE *xs, int xdd, double **x);
int FindGoofIndex(int xd, int xdd, int **x);
int GoofFindBBox(double **nodesgoof,int size,
                  Coords *bl,Coords *br,Coords *tr,Coords *tl);

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
         * ValidAtt is one of: CAXIS, ENERGY, VISCOSITY, MINERAL,
         *  STRAIN, SPLIT, GRAIN
         */
        /*if (!ElleFlynnAttributeActive(EULER_3)) {
            ElleInitFlynnAttribute(EULER_3);
            ElleCheckFlynnDefaults(EULER_3);
        }*/
    }
}

int ProcessFunction()
{
    int maxtri,maxnode,errt,errn,err;
    int **nodeseg;
    double **nodesgoof;
    FILE *polyfile,*newgooffile;
    char fdum1[50],fdum2[50],fdum3[50];
    int dum1,dum2,dum3,max,j,goofindex;
    int ntl,nbr,nbl,ntr;
    Coords delxy,newbbox[4];

    polyfile=fopen("tmp.elle.poly","r");
    if(polyfile)
    {
		fscanf(polyfile,"%d %d %d %d",&maxnode,&dum1,&dum2,&dum3);
        	sprintf(logbuf,"maxnodes %d\n",maxnode);
            Log(0,logbuf);
			maxnode++;
		nodeseg=(int **) imatrix(0,maxnode+1,0,1);
        	nodesgoof=(double **) dmatrix(0,maxnode+1,0,1);
		ReadPolyFile(polyfile,nodeseg);
        	sprintf(logbuf,"maxnodes %d\n",maxnode);
            Log(0,logbuf);
    }
    else
		exit(1);

    newgooffile=fopen("oof.goof","r");

    if(newgooffile)
	    if (ReadGoofFile(newgooffile,maxnode,nodesgoof))
        	exit(1);

    	max = ElleMaxNodes();

    	for (j=0;j<max;j++)
    	{
        	if (ElleNodeIsActive(j))
			{
	    		goofindex=FindGoofIndex(j,maxnode,nodeseg);
	    		delxy.x=nodesgoof[goofindex][0];
	    		delxy.y=nodesgoof[goofindex][1];
	    		//printf("j=%d, goofindex=%d\t%f\t%f\n",j,goofindex,delxy.x,delxy.y);
	    		ElleUpdatePosition(j,&delxy);
			}
    	}

	free_dmatrix(nodesgoof,0,maxnode+1,0,1);
	free_imatrix(nodeseg,0,maxnode+1,0,1);

    if (err=ElleAutoWriteFile(1))
        OnError("",err);
}

/* find goof index that matches equivalent elle index*/
int FindGoofIndex(int j, int maxnode, int **nodeseg)
{
    int i;

    for(i=0;i<maxnode;i++)
    {
	if(nodeseg[i][0]==j)
	    return(nodeseg[i][1]-1);
	    //return(nodeseg[i][1]);
    }

    return(0);

}

/* Read in enough of the goof file to get displacements */
int ReadGoofFile(FILE *newgooffile, int maxnode, double **nodesgoof)
{
    int err=0;
    int i,idum1,idum2,idum3,idum4;
    double fdum1,fdum2,fdum3,fdum4;
    char sdum1[50],sdum2[50],sdum3[50],sdum4[50],sdum5[50],sdum6[50],sdum7[50],sdum8[50],text[1000];
    double **nodesgoof2;
    Coords newbbox[4];
    double minx,maxx,miny,maxy;
    int nbl,nbr,ntr,ntl;
	CellData data;
	double tmp,offset;

	nodesgoof2=(double **) dmatrix(0,maxnode+1,0,1);

    for(i=0;i<15;i++)
		fscanf(newgooffile,"%s",sdum1);

    for(i=0;i<maxnode;i++)
    {
		fscanf(newgooffile,"%s %s %s %s %s %s %s %s"
			,sdum1,sdum2,sdum3,sdum4,sdum5,sdum6,sdum7,sdum8);


		sscanf(&sdum3[2],"%lf",&nodesgoof2[i][0]);
		sscanf(&sdum4[2],"%lf",&nodesgoof2[i][1]);
		sscanf(&sdum5[3],"%lf",&nodesgoof[i][0]);
		sscanf(&sdum6[3],"%lf",&nodesgoof[i][1]);

		nodesgoof2[i][0] += nodesgoof[i][0];
		nodesgoof2[i][1] += nodesgoof[i][1];

		//printf("%d %f %f %f %f\n",i,nodesgoof2[i][0],nodesgoof2[i][1],
			//nodesgoof[i][0],nodesgoof[i][1]);

		/*if (i==0) {
	    	minx = maxx = nodesgoof2[i][0];
	    	miny = maxy = nodesgoof2[i][1];
		}
		else {
	    	if (nodesgoof2[i][0]<minx)
			minx = nodesgoof2[i][0];
	    	else if (nodesgoof2[i][0]>maxx)
			maxx = nodesgoof2[i][0];
	    	if (nodesgoof2[i][1]<miny)
			miny = nodesgoof2[i][1];
	    	else if (nodesgoof2[i][1]>maxy)
			maxy = nodesgoof2[i][1];
		}*/

    }

    do
    {
		fscanf(newgooffile,"%s",&text);
    } while(strcmp(text,"label=bl")!= 0);

	fscanf(newgooffile,"%s",sdum1);
    sscanf(&(sdum1[5]),"%d",&nbl);
    sprintf(logbuf,"text = %s nbl = %d\n",text,nbl);
    Log(0,logbuf);

    do
    {
		fscanf(newgooffile,"%s",text);
    } while(strcmp(text,"label=br")!= 0);

	fscanf(newgooffile,"%s",sdum1);
    sscanf(&(sdum1[5]),"%d",&nbr);
    sprintf(logbuf,"text = %s nbr = %d\n",text,nbr);
    Log(0,logbuf);

    do
    {
		fscanf(newgooffile,"%s",text);
    } while(strcmp(text,"label=tl")!= 0);

    fscanf(newgooffile,"%s",sdum1);
    sscanf(&(sdum1[5]),"%d",&ntl);
    sprintf(logbuf,"text = %s ntl = %d\n",text,ntl);
    Log(0,logbuf);

    do
    {
		fscanf(newgooffile,"%s",text);
    } while(strcmp(text,"label=tr")!= 0);

    fscanf(newgooffile,"%s",sdum1);
    sscanf(&(sdum1[5]),"%d",&ntr);
    sprintf(logbuf,"text = %s ntr = %d\n",text,ntr);
    Log(0,logbuf);

    newbbox[0].x=nodesgoof2[nbl][0];
    newbbox[0].y=nodesgoof2[nbl][1];

    newbbox[1].x=nodesgoof2[nbr][0];
    newbbox[1].y=nodesgoof2[nbr][1];

    newbbox[2].x=nodesgoof2[ntr][0];
    newbbox[2].y=nodesgoof2[ntr][1];

    newbbox[3].x=nodesgoof2[ntl][0];
    newbbox[3].y=nodesgoof2[ntl][1];

    sprintf(logbuf,"bl=%f %f\nbr=%f %f\ntr=%f %f\ntl=%f %f\n",
                    newbbox[0].x,newbbox[0].y,
                    newbbox[1].x,newbbox[1].y,
                    newbbox[2].x,newbbox[2].y,
                    newbbox[3].x,newbbox[3].y
                    );
    Log(0,logbuf);


    //printf("min %f %f max %f %f \n",minx,miny,maxx,maxy);

    //GoofFindBBox(nodesgoof2,maxnode,&newbbox[0],&newbbox[1],&newbbox[2],&newbbox[3]);

	offset = ElleCumSSOffset();

    if (err=ElleUpdateCellBBox(&newbbox[0],&newbbox[1],&newbbox[2],&newbbox[3]))
        fprintf(stderr,"Invalid bounding box - \n%f,%f\n%f,%f\n%f,%f\n%f,%f\n",
                newbbox[0].x,newbbox[0].y,
                newbbox[1].x,newbbox[1].y,
                newbbox[2].x,newbbox[2].y,
                newbbox[3].x,newbbox[3].y
                );
#if XY
    if (err=ElleSetCellBBox(&newbbox[0],&newbbox[1],&newbbox[2],&newbbox[3]))
        fprintf(stderr,"Invalid bounding box - \n%f,%f\n%f,%f\n%f,%f\n%f,%f\n",
                newbbox[0].x,newbbox[0].y,
                newbbox[1].x,newbbox[1].y,
                newbbox[2].x,newbbox[2].y,
                newbbox[3].x,newbbox[3].y
                );

	printf("offset 1: %lf\n",offset);
	ElleCellBBox(&data);
	ElleSetCumSSOffset(offset+nodesgoof[ntr][0]);
	printf("offset 2 nodesgoof: %lf %lf\n",offset,nodesgoof[ntr][0]);
	offset = modf(ElleCumSSOffset(),&tmp);
	printf("offset 3: %lf\n",offset);
	ElleSetSSOffset(offset);
#endif

	free_dmatrix(nodesgoof2,0,maxnode+1,0,1);
    return(err);
}

/*Read in tail end of poly file to get node number matching info*/
int ReadPolyFile(FILE *polyfile, int **nodeseg)
{
    int i,num,dum;
    char sdum1[50];

    do{
		fscanf(polyfile,"%s",sdum1);
    } while(strcmp(sdum1,"ELLE") != 0);

    fscanf(polyfile,"%s",sdum1);
    fscanf(polyfile,"%d %d",&num,&dum);

    for(i=0;i<num;i++)
    {
		fscanf(polyfile,"%d %d",&nodeseg[i][0],&nodeseg[i][1]);
		//printf("%d\t%d\n",nodeseg[i][0],nodeseg[i][1]);
    }


    return(0);
}

int GoofFindBBox(double **nodesgoof,int size,
                  Coords *bl,Coords *br,Coords *tr,Coords *tl)
{
    int i, setmin1=0, setmax1=0, setmin2=0, setmax2=0;
    double xmin,ymin,xmax,ymax;
    double eps;
    eps = 1e-4;
    bl->y = br->y = tr->y = tl->y = nodesgoof[0][1];

    xmin = xmax = nodesgoof[0][0];
    ymin = ymax = nodesgoof[0][1];
    for (i=1;i<size;i++) {
        if (nodesgoof[i][1] < br->y) { br->y = bl->y = nodesgoof[i][1]; }
        if (nodesgoof[i][1] > tr->y) { tr->y = tl->y = nodesgoof[i][1]; }
    }
    ymin = (br->y + eps);
    ymax = (tr->y - eps);
    for (i=0;i<size;i++) {
        if (nodesgoof[i][1] < ymin) {
            if (!setmin1) {
                bl->x = nodesgoof[i][0];
                setmin1 = 1;
            }
            else if (nodesgoof[i][0] < bl->x)
		bl->x = nodesgoof[i][0];

	    if (!setmax1) {
                br->x = nodesgoof[i][0];
                setmax1 = 1;
            }
            else if (nodesgoof[i][0] > br->x)
		br->x = nodesgoof[i][0];
        }
        else if (nodesgoof[i][1] > ymax) {
            if (!setmin2) {
                tl->x = nodesgoof[i][0];
                setmin2 = 1;
            }
            else if (nodesgoof[i][0] < tl->x)
		tl->x = nodesgoof[i][0];

            if (!setmax2) {
                tr->x = nodesgoof[i][0];
                setmax2 = 1;
            }
            else if (nodesgoof[i][0] > tr->x)
		tr->x = nodesgoof[i][0];
        }
    }
}
