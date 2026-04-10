#include <stdio.h>
#include <math.h>
#include "error.h"
#include "file.h"
#include "init.h"
#include "runopts.h"
#include "interface.h"
#include "nodes.h"
#include "unodes.h"
#include "convert.h"
#include "general.h"
#include "mat.h"
#include "plotaxes.h"

/*!
  Agrees with HKL lower hemisphere C-axis values
  90deg anticlockwise rotation from OIM C-axis lower hemisphere
  Both are equal area
*/
int InitThisProcess(), ProcessFunction();
void change(double *axis, double axis2[3], double rmap[3][3]);
void firo(double *a, double *phi, double *rho);
void startps(FILE *);
void endps(FILE *);
void plotonept(double *axis, double rmap[3][3], double *center, double radius,
				FILE *psout,FILE *polarout);
void splotps(double *center, double radius, double phi, double rho,
				FILE *psout, FILE *polarout);
void startsteronet(double *center, double radius, FILE *psout, char *title, int ngns);
int FindRowsCols(int *rows, int *numperrow);
void old_main(); // no longer used

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
         *  STRAIN, SPLIT, GRAIN*/
        if (!ElleFlynnAttributeActive(EULER_3)) {
            ElleInitFlynnAttribute(EULER_3);
        }
         
    }
    
}

int ProcessFunction()
{
    int i,j,k,l, max, mintype=QUARTZ;
	int err=0;
    int NumPlots=1;
	int Reverse_row_order=0;
    char *infile,psfile[100],polarfile[3][100];
    double curra, currb, currc;
    FILE *psout, *polarout[3]={0,0,0};
    /*double axis_hexagonal[3][3]={0,0,1,1,0,0,1,0,1};*/
    double axis_hexagonal[3][3]={0,0,1,1,1,0,1,0,1};
    double axis_cubic[3][3]={1,1,1,1,0,0,0,1,0};
    double center[3][2]={2,9,2,5,6,9};
    double radius=1.75;
    double rmap[3][3];
	char label[3][7]={"","",""};
    /*char label_hexagonal[3][7]={"c lwr ","a+ upr","r     "};*/
    char label_hexagonal[3][7]={"c lwr ","a     ","r     "};
    char label_cubic[3][7]={"111","100","010"};
    //int ngn_code[3]={-1,0,-1};
    int ngn_code[3]={0,-1,-1};
    char syscall[100];
    UserData udata;

    ElleUserData(udata);

    Reverse_row_order=(int)udata[REVERSE_ORDER];
printf ("%d \n",Reverse_row_order);
    int user_step = (int)udata[SAMPLE_STEP];
    int caxis_output = (int)udata[CAXIS_OUT];

    infile = ElleFile();
    sprintf(psfile,"%s_ax.ps",infile);
    for (i=0;i<3;i++) polarfile[i][0] = '\0';
    if ( caxis_output ) {
        sprintf(polarfile[0],"%s_c_polar.txt",infile);
    }
    //sprintf(polarfile[1],"%s_a_polar.txt",infile);
    //sprintf(polarfile[2],"%s_r_polar.txt",infile);
    sprintf(syscall,"gv %s&",psfile);
    
    psout=fopen(psfile,"w");
    startps(psout);

    if (!ElleUnodeAttributeActive(EULER_3))
    {
		max = ElleMaxFlynns();
		for (j=0;j<max;j++)
			if (ElleFlynnIsActive(j))
			{
				if (strlen(label[0])==0)
				{
                    if (ElleFlynnHasAttribute(j,MINERAL)) {
						ElleGetFlynnMineral(j, &mintype);
						if(mintype==QUARTZ || mintype==CALCITE || mintype==ICE) // "|| mintype==ICE added 12th august 2014, FSteinbach
							for(i=0;i<3;i++)
								strcpy(label[i], label_hexagonal[i]);
						else 
							for(i=0;i<3;i++)
								strcpy(label[i], label_cubic[i]);
                    }
                    else { // assume haxagonal
                        for(i=0;i<3;i++)
                            strcpy(label[i], label_hexagonal[i]);
                    }
				}
				ngn_code[0]++;
			}

		for(i=0;i<NumPlots;i++)
		{
            if (caxis_output && i<1) {
      	    polarout[i]=fopen(polarfile[i],"w");
            if (fprintf(polarout[i],"%s %s %s\n","ID","Azimuth","Dip")<0)
			  OnError(polarfile[i],WRITE_ERR);
          }
			startsteronet(center[i],radius,psout,label[i],ngn_code[i]);

			for (j=0;j<max;j++)
			{

				if (ElleFlynnIsActive(j))
				{
                    if (caxis_output && i<1) {
                      if (fprintf(polarout[i],"%d ",j)<0)
	        		    OnError(polarfile[i],WRITE_ERR);
                    }
					 if (ElleFlynnHasAttribute(j,MINERAL))
                       ElleGetFlynnMineral(j, &mintype);

					if(mintype==QUARTZ || mintype==CALCITE || mintype==ICE) // "|| mintype==ICE added 12th august 2014, FSteinbach
					{
		 			 	ElleGetFlynnEuler3(j, &curra, &currb, &currc); //retrieve euler angles
						eulerZXZ(rmap, curra*M_PI/180, currb*M_PI/180, currc*M_PI/180);

						plotonept(axis_hexagonal[i],rmap,center[i],radius,
									psout,polarout[i]);	    
					}
					else // assume cubic
					{
						ElleGetFlynnEuler3(j, &curra, &currb, &currc); //retrieve euler angles
						eulerZXZ(rmap, curra*M_PI/180, currb*M_PI/180, currc*M_PI/180);

						plotonept(axis_cubic[i],rmap,center[i],radius,
									psout,polarout[i]);	    
					}
				}
			}
		}
	}
	else
	{
		max = ElleMaxUnodes();
		int step = user_step;
		int rows=0, cols=0;
		ngn_code[0]=max/step;
		if (ngn_code[0]<1) ngn_code[0] = 1;
		for(i=0;i<3;i++) {
		  strcpy(label[i], label_hexagonal[i]);
          if (caxis_output && i<1) {
    	  polarout[i]=fopen(polarfile[i],"w");
          if (fprintf(polarout[i],"%s %s %s %s\n","X","Y","Azimuth","Dip")<0)
			OnError(polarfile[i],WRITE_ERR);
          }
        }
        if (err=FindRowsCols(&rows,&cols)!=0) 
			OnError("Problem calculating rows and columns",0);
        int x=0,y=-1;
        if (Reverse_row_order) {
        	x=0;y=rows;
        }
/*  if file contains only unodes, temporarily assume hexagonal  */
        if (ElleMaxFlynns()>0 && ElleFlynnAttributeActive(MINERAL)) {
		    ElleGetFlynnMineral(ElleUnodeFlynn(0), &mintype);
		    if(mintype==QUARTZ || mintype==CALCITE || mintype==ICE) // "|| mintype==ICE added 12th august 2014, FSteinbach
			    for(i=0;i<3;i++)
				    strcpy(label[i], label_hexagonal[i]);
		    else 
			    for(i=0;i<3;i++)
				    strcpy(label[i], label_cubic[i]);
        }

		for(i=0;i<NumPlots;i++)
		{
			startsteronet(center[i],radius,psout,label[i],ngn_code[i]);

			for (j=0;j<max;j+=step)
			{
				ElleGetUnodeAttribute(j, &curra, &currb, &currc,EULER_3); //retrieve euler angles
				orientmatZXZ(rmap, curra*M_PI/180, currb*M_PI/180, currc*M_PI/180);
/*  if file contains only unodes, temporarily assume hexagonal  */
        		if (ElleMaxFlynns()>0 && ElleFlynnAttributeActive(MINERAL)) {
					ElleGetFlynnMineral(ElleUnodeFlynn(j), &mintype);

					if (caxis_output && i<1) {
                        x = j%cols;
                        y = j/rows;
                        if (Reverse_row_order)  y=(rows-1)-y;
                        if (polarout[i]!=0) {
        		  		  if (fprintf(polarout[i],"%d %d ",x,y)<0)
							OnError(polarfile[i],WRITE_ERR);
                        }
					    if(mintype==QUARTZ || mintype==CALCITE || mintype==ICE) // "|| mintype==ICE added 12th august 2014, FSteinbach
					    	plotonept(axis_hexagonal[i],rmap,center[i],radius,
									psout,polarout[i]);
					    else // assume cubic
					    	plotonept(axis_cubic[i],rmap,center[i],radius,
									psout,polarout[i]);	    
                    }
                    else {
					    if(mintype==QUARTZ || mintype==CALCITE || mintype==ICE) // "|| mintype==ICE added 12th august 2014, FSteinbach
					    	plotonept(axis_hexagonal[i],rmap,center[i],radius,
									psout,0);
					    else // assume cubic
						    plotonept(axis_cubic[i],rmap,center[i],radius,
									psout,0);	    
				    }
				}
				else {
					if (caxis_output && i<1) {
                        x = j%cols;
                        if ( (x==0)||((step>1)&&(x<step)) ) {
					        if (Reverse_row_order)  y--;
							else y++;
						}
        				if (fprintf(polarout[i],"%d %d ",x,y)<0)
							OnError(polarfile[i],WRITE_ERR);
						plotonept(axis_hexagonal[i],rmap,center[i],radius,
								psout,polarout[i]);
						/*plotonept(axis_cubic[i],rmap,center[i],radius,
								psout,polarout[i]);*/
                    }
					else {
						plotonept(axis_hexagonal[i],rmap,center[i],radius,
                                psout,0);
						/*plotonept(axis_cubic[i],rmap,center[i],radius,
                                psout,0);*/
					}
				}
			}
		}
    }

    endps(psout);
    for (i=0;i<3;i++)
        if (polarout[i]) fclose(polarout[i]);
#if XY
#endif
    //system(syscall);
} 

void plotonept(double *axis, double rmap [3][3], double *center, double
radius, FILE *psout,FILE *polarout)
{
    int i;
    double axis2[3];
    double phi,rho;

/*
    change(axis,axis2,rmap);
*/
    mataxismult(rmap,axis,axis2);
    firo(axis2,&phi,&rho);

    splotps(center,radius,phi,rho, psout, polarout);

}

void change(double *axis, double axis2[3], double rmap[3][3])
{
    double a, ax[3];
    int i,j;

    for(i=0;i<3;i++)
    {
    	a=0.0;
	for(j=0;j<3;j++)
	{
	    a=a+rmap[j][i]*axis[j];
	}
	ax[i]=a;
    }
    for(i=0;i<3;i++)
	axis2[i]=ax[i];
}

void firo(double *a, double *phi, double *rho)
{
    double z,zz;
    
    z=sqrt((a[0]*a[0])+(a[1]*a[1]));
    zz=sqrt((a[0]*a[0])+(a[1]*a[1])+(a[2]*a[2]));
    
    if(zz >= 0.0001)
    {
	if(z >= 0.00001)
	{
	    *phi=fabs(acos(a[0]/z));
	}
	else
	    *phi=0.0;
	    
	if(a[1] < 0.0) 
	    *phi=-(*phi);
	    
	*rho=acos(a[2]/zz);
    }
    else
    {
	*phi=0.0;
	*rho=0.0;
    }
}

void startps(FILE *psout)
{
    fprintf(psout,"%%!PS-Adobe-2.0\n\n");
    fprintf(psout,"0 setlinewidth\n");
    fprintf(psout,"72 72 scale\n\n");
    fprintf(psout,"/Helvetica findfont\n");
    fprintf(psout,"0.25 scalefont\n");
    fprintf(psout,"setfont\n");

    
}

void endps(FILE *psout)
{
    fprintf(psout,"showpage\n");
    fflush(psout);
    fclose(psout);
 
}
void startsteronet(double *center, double radius, FILE *psout,char *title, int ngns)
{
    char grains[50];

    fprintf(psout,"newpath\n");
    fprintf(psout,"%lf %lf moveto\n",center[0],center[1]);
    fprintf(psout,"%lf %lf %lf 0 360 arc\n",center[0],center[1],radius);
    fprintf(psout,"%lf %lf moveto\n",center[0]-radius,center[1]);
    fprintf(psout,"%lf %lf lineto\n",center[0]+radius,center[1]);
    fprintf(psout,"stroke\n");

    fprintf(psout,"newpath\n");
    fprintf(psout,"%lf %lf moveto\n", center[0]+0.85*radius,center[1]+0.85*radius);
    fprintf(psout,"(%s) show\n",title);
    fprintf(psout,"%lf %lf moveto\n", center[0]+0.85*radius,center[1]-0.85*radius);
    if(ngns > 0)
	fprintf(psout,"(n = %d) show\n",ngns);
    
}

void splotps(double *center, double radius, double phi, double rho,
			FILE *psout, FILE *polarout)
{
    double srad,x,y;
    double ptsize=0.02;
    double angle, azimuth;

    if(rho*180.0/M_PI > 89.0)
    {
	    rho=M_PI-rho;
        
	    if(rho*180.0/M_PI > 89.0)
	    {
	        rho=M_PI-rho;
    	    phi=phi+M_PI;
    	}
    }
    else
    {
        // if phi is zero leave it so that azimuth is 90
    	if (phi!=0) phi=phi+M_PI;
    }
    
    srad=sqrt(2.0)*sin(rho/2.0)*radius;
    x=cos(phi)*srad;
    y=sin(phi)*srad;

    fprintf(psout,"newpath\n");
    fprintf(psout,"%lf %lf moveto\n",x+center[0]+ptsize,y+center[1]);
    fprintf(psout,"%lf %lf %lf 0 360 arc\n",x+center[0],y+center[1],ptsize);
    //fprintf(psout,"closepath\n");
    fprintf(psout,"stroke\n");
   
    if (polarout!=0) {
        //polar angle convention is anticlockwise from y=0
        // geology convention is clockwise from x=0
    	angle=450.0-phi*RTOD; //360.0 - angle + 90.0
        if (angle>=360.0) angle -= 360.0;
    	azimuth=90.0-rho*RTOD;
    	if (fprintf(polarout,"%6.1lf %6.1lf\n",angle,azimuth)<0)
			OnError("polarfile",WRITE_ERR);
    }

}

int FindRowsCols(int *rows, int *numperrow)
{
    int err=0;
    int i, j;
    int max_unodes = ElleMaxUnodes();
    Coords refxy, xy;
    double eps, dx;

    *rows=0;
    ElleGetUnodePosition(0,&refxy);
    ElleGetUnodePosition(1,&xy);
    ElleCoordsPlotXY(&xy,&refxy);
    dx = fabs((xy.x-refxy.x));
    eps = dx*0.1;
    *numperrow=1;
    while (*numperrow<max_unodes && (fabs(xy.y-refxy.y)<eps)) {
        (*numperrow)++;
        ElleGetUnodePosition(*numperrow,&xy);
        ElleCoordsPlotXY(&xy,&refxy);
    }
    if (*numperrow<max_unodes) {
		*rows = max_unodes/(*numperrow);
    }
    else err=1;
    return(err);
}
