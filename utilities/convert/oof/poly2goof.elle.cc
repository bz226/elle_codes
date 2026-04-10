#include <stdio.h>
#include <string.h>
#include <math.h>
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
#include <stdlib.h>
int DoSomethingToFlynn(int flynn);
int InitThisProcess(), ProcessFunction();
int ReadTriEleFile(FILE *x, int xx, int **xxx, int *xxxx);
int ReadTriNodeFile(FILE *x, int xxx, double **xxxx);
int WriteGoofFile(FILE *x, int xxx, int **xyy, int *xyx, int xc, double **xd);

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
        if (!ElleFlynnAttributeActive(EULER_3)) {
            ElleInitFlynnAttribute(EULER_3);
            ElleCheckFlynnDefaults(EULER_3);
        }
    }
}

int ProcessFunction()
{
    int maxtri,maxnode,errt,errn;
    int **tri;
    double **nodes;
    int *ellecode;
    FILE *elefile, *nodefile,*gooffile;
    int dum1,dum2,dum3;




    gooffile=fopen("elle.goof","w");
    elefile=fopen("tmp.elle.3.ele","r");
    nodefile=fopen("tmp.elle.3.node","r");

    if(elefile)
	fscanf(elefile,"%d %d %d",&maxtri,&dum1,&dum2);
    else
	exit(0);

    if(nodefile)
	fscanf(nodefile,"%d %d %d %d",&maxnode,&dum1,&dum2,&dum3);
    else
	exit(0);

    tri=(int **) imatrix(0,maxtri+1,0,3);
    nodes=(double **) dmatrix(0,maxnode+1,0,3);
    if ((ellecode=new int[maxtri+2])==0) {
        free_imatrix(tri,0,maxtri+1,0,3);
        free_dmatrix(nodes,0,maxnode+1,0,3);
        OnError(0,MALLOC_ERR);
    }



    errt=ReadTriEleFile(elefile,maxtri,tri,ellecode);
    errn=ReadTriNodeFile(nodefile,maxnode,nodes);

    sprintf(logbuf,"tri[0][1,2,3]=%d %d %d\n",tri[0][0],tri[0][1],tri[0][2]);
    Log(0,logbuf);
    sprintf(logbuf,"node[0][1,2]=%f %f \n",nodes[0][0],nodes[0][1]);
    Log(0,logbuf);
    sprintf(logbuf,"ellecode[0]=%d\n",ellecode[0]);
    Log(0,logbuf);


    if(!errt && !errn)
	  WriteGoofFile(gooffile,maxtri,tri,ellecode,maxnode,nodes);
    free_imatrix(tri,0,maxtri+1,0,3);
    free_dmatrix(nodes,0,maxnode+1,0,3);
    delete [] ellecode;
    return(0);
}


int ReadTriEleFile(FILE *elefile, int maxtri, int **tri,int *ellecode)
{
    int i,dum1;

    for(i=0;i<maxtri;i++)
    {
	fscanf(elefile,"%d %d %d %d %d",
	    &dum1,&tri[i][0],&tri[i][1],&tri[i][2],&ellecode[i]);
    }
    return(0);
}

int ReadTriNodeFile(FILE *nodefile, int maxnode, double **node)
{
    int i,dum1,dum2;


    for(i=0;i<maxnode;i++)
    {
	fscanf(nodefile,"%d %lf %lf %d",
	    &dum1,&node[i][0],&node[i][1],&dum2);
    }
    return(0);
}


int WriteGoofFile(FILE *gooffile, int maxtri, int **tri, int *ellecode,
int maxnode, double **node)
{

    float curra, currb, currc, gray;
    int i,mintype;
    double tol=0.000001;
    double tl=1e10,tr=-1e10,bl=1e10,br=-1e10;
    int ntl,ntr,nbl,nbr;

	sprintf(logbuf,"WriteGoofFile \n");
    Log(0,logbuf);

	fprintf(gooffile,"version number = 5\n");

	fprintf(gooffile,"Nelements =  %d\n",maxtri);
	fprintf(gooffile,"Nnodes = %d\n",maxnode);

	fprintf(gooffile,"nodes (\n");
	for(i=0;i<maxnode;i++)
	    fprintf(gooffile,"xy i=%d x=%f y=%f dx=0 dy=0\n",i,(float)node[i][0],(float)node[i][1]);
	fprintf(gooffile,")\n");

	fprintf(gooffile,"elements (\n");
	for(i=0;i<maxtri;i++)
	{
	    if (ElleFlynnIsActive(ellecode[i]) && ElleFlynnHasAttribute(ellecode[i],EULER_3))
	    {
		ElleGetFlynnMineral(ellecode[i], &mintype);

		//printf("WriteGoofFile i=%d mintype=%d\n",i,mintype);

		if(mintype==QUARTZ)
		    ElleGetFlynnEuler3(ellecode[i], &curra, &currb, &currc);
	    }

	    gray=fmod((double)(curra+90),15.0)/7.5;
	    if(gray > 1.0)
		gray=2.0-gray;

/*  equivalent code for void problem

	    	fprintf(gooffile,"empty i=%d n1=%d n2=%d n3=%d gray=%f"
		    	,i,tri[i][0]-1,tri[i][1]-1,tri[i][2]-1,gray);
	    	fprintf(gooffile,"\n");
*/


/*  equivalent code for NaCl problem

	    	fprintf(gooffile,"cubic i=%d n1=%d n2=%d n3=%d gray=%f orientation=[%f, %f, %f] "
		    	,i,tri[i][0]-1,tri[i][1]-1,tri[i][2]-1,gray,curra,currb,currc);
	    	fprintf(gooffile," planestrain=true  young=39.98 poisson=0.252 anisotropy=0.696685083 alpha=44e-6\n");
			// all data from http://www.crystaltechno.com/Materials/NaCl.htm
*/

/*  equivalent code for MgO problem

	    	fprintf(gooffile,"cubic i=%d n1=%d n2=%d n3=%d gray=%f orientation=[%f, %f, %f] "
		    	,i,tri[i][0]-1,tri[i][1]-1,tri[i][2]-1,gray,curra,currb,currc);
	    	fprintf(gooffile," planestrain=true  young=269.2 poisson=0.24 anisotropy=1.44 alpha=20.91e-6\n");
			// elasticity data from AGU Handbook Ref
*/

/*  equivalent code for Muscovite problem

	    	fprintf(gooffile,"hexagonal i=%d n1=%d n2=%d n3=%d gray=%f orientation=[%f, %f, %f] "
		    ,i,tri[i][0]-1,tri[i][1]-1,tri[i][2]-1,gray,curra,currb,currc);
	    	fprintf(gooffile,"planestrain=true c11=181.0 c12=48.3 c13=22.8 c33=59.1 c44=16.8 alpha11=20.91e-6 alpha33=12.22e-6\n");
		// pseudo hexagonal elasticity data based on monoclinic averages
*/

/*  equivalent code for Quartz problem
*/
		fprintf(gooffile,"trigonalB i=%d n1=%d n2=%d n3=%d gray=%f orientation=[%f, %f, %f] "
		    ,i,tri[i][0]-1,tri[i][1]-1,tri[i][2]-1,gray,curra,currb,currc);
	    fprintf(gooffile,"planestrain=true c11=86.6 c12=6.7 c13=12.6 c14=-17.8 c33=106.1 c44=57.8 alpha11=20.91e-6 alpha33=12.22e-6\n");
	    // elasticity data from AGU Handbook Ref, themal expansion data from Ward ref (for 500 deg C)
	}
	fprintf(gooffile,")\n");

	fprintf(gooffile,"nodegroup (\n");
	fprintf(gooffile,"label=top\n");
	for(i=0;i<maxnode;i++)
	{
	    if(fabs(node[i][1] - 1.0) < tol)
	    {
		fprintf(gooffile,"node=%d\n",i);
		if(node[i][0]<tl)
		{
		    tl=node[i][0];
		    ntl=i;
		}
		if(node[i][0]>tr)
		{
		    tr=node[i][0];
		    ntr=i;
		}
	    }
	}
	fprintf(gooffile,")\n");

	fprintf(gooffile,"nodegroup (\n");
	fprintf(gooffile,"label=bottom\n");
	for(i=0;i<maxnode;i++)
	{
	    if(fabs(node[i][1] - 0.0) < tol)
	    {
		fprintf(gooffile,"node=%d\n",i);
		if(node[i][0]<bl)
		{
		    bl=node[i][0];
		    nbl=i;
		}
		if(node[i][0]>br)
		{
		    br=node[i][0];
		    nbr=i;
		}
	    }
	}
	fprintf(gooffile,")\n");

	fprintf(gooffile,"nodegroup (\n");
	fprintf(gooffile,"label=left\n");
	for(i=0;i<maxnode;i++)
	{
	    if(fabs(node[i][0] - 0.0) < tol)
	    {
		fprintf(gooffile,"node=%d\n",i);
	    }
	}
	fprintf(gooffile,")\n");

	fprintf(gooffile,"nodegroup (\n");
	fprintf(gooffile,"label=right\n");
	for(i=0;i<maxnode;i++)
	{
	    if(fabs(node[i][0] - 1.0) < tol)
	    {
		fprintf(gooffile,"node=%d\n",i);
	    }
	}
	fprintf(gooffile,")\n");

	fprintf(gooffile,"nodegroup (\n");
	fprintf(gooffile,"label=bl\n");
	fprintf(gooffile,"node=%d\n",nbl);
	fprintf(gooffile,")\n");

	fprintf(gooffile,"nodegroup (\n");
	fprintf(gooffile,"label=br\n");
	fprintf(gooffile,"node=%d\n",nbr);
	fprintf(gooffile,")\n");

	fprintf(gooffile,"nodegroup (\n");
	fprintf(gooffile,"label=tl\n");
	fprintf(gooffile,"node=%d\n",ntl);
	fprintf(gooffile,")\n");

	fprintf(gooffile,"nodegroup (\n");
	fprintf(gooffile,"label=tr\n");
	fprintf(gooffile,"node=%d\n",ntr);
	fprintf(gooffile,")\n");

	sprintf(logbuf,"(tl=%f,1.0)\n(tr=%f,1.0)\n(bl=%f,0.0)\n(br=%f,0.0)\n",tl,tr,bl,br);
    Log(0,logbuf);
	sprintf(logbuf,"(tl=%d)\n(tr=%d)\n(bl=%d)\n(br=%d)\n",ntl,ntr,nbl,nbr);
    Log(0,logbuf);

	fprintf(gooffile,"oof bc fix both top\n");
	fprintf(gooffile,"oof bc fix both bottom\n");
	fprintf(gooffile,"oof bc fix both left\n");
	fprintf(gooffile,"oof bc fix both right\n");

//	fprintf(gooffile,"oof distort set xshift = 0.2\n");
//	fprintf(gooffile,"oof distort set top\n");

	fprintf(gooffile,"oof distort set xshear = 0.2\n");
	fprintf(gooffile,"oof distort set left\n");
	fprintf(gooffile,"oof distort set right\n");
	fprintf(gooffile,"oof distort set top\n");
	fprintf(gooffile,"oof distort set bottom\n");

	fprintf(gooffile,"oof distort increment\n");
	fprintf(gooffile,"oof equilibrate\n");

	fprintf(gooffile,"oof output grid ascii filename=oof.goof\n");
	fprintf(gooffile,"oof output individual stress tensor filename=out_stress\n");
	fprintf(gooffile,"oof output individual strain tensor filename=out_strain\n");

}

