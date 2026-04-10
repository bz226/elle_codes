#include "plotaxes.elle.h"

/*!
  Agrees with HKL lower hemisphere C-axis values
  90deg anticlockwise rotation from OIM C-axis lower hemisphere
  Both are equal area
*/
int PlotAxes(wxString filename)
{
    int i,j,k,l, max, mintype=QUARTZ;
    int err=0;
    int Reverse_row_order=0;
    int NumPlots=1;
    double curra, currb, currc;
    FILE *psout, *polarout[3]={0,0,0};
    double axis_hexagonal[3][3]={0,0,1,1,0,0,1,0,1};
    double axis_cubic[3][3]={0,0,1,1,0,0,1,0,1};
    double center[3][2]={2,9,2,5,6,9};
    double radius=1.75;
    double rmap[3][3];
    char label[3][7]={"","",""};
    char label_hexagonal[3][7]={"c lwr","a","r"};
    char label_cubic[3][7]={"111","100","010"};
    //int ngn_code[3]={-1,0,-1}; //plot number of pts after 2nd plot
    int ngn_code[3]={0,-1,-1}; //plot number of pts after 1st plot
    wxString polarfile[3];
    UserData udata;

    ElleUserData(udata);

    Reverse_row_order=(int)udata[REVERSE_ORDER];
    int user_step = (int)udata[SAMPLE_STEP];
    int caxis_output = (int)udata[CAXIS_OUT];
/*This comes from the Init-function above. Needs to be in here so that this function can be called from within 
 * the GUI
         * initialise any necessary attributes which may
         * not have been in the elle file
         * ValidAtt is one of: CAXIS, ENERGY, VISCOSITY, MINERAL,
         *  STRAIN, SPLIT, GRAIN*/
        if (!ElleFlynnAttributeActive(EULER_3)) {
            ElleInitFlynnAttribute(EULER_3);
        }
	if(filename.empty()) {
    	wxString default_ext("_ax.ps");
		filename = ElleFile()+default_ext;
	}
    if ( caxis_output ) {
      polarfile[0] = filename.BeforeLast('.') + wxString("_c_polar.txt");
    }
    //polarfile[1] = filename.BeforeLast('.') + wxString("_a_polar.txt");
    //polarfile[2] = filename.BeforeLast('.') + wxString("_r_polar.txt");

    psout=fopen(filename.c_str(),"w");
    pstartps(psout);
    
    if (!ElleUnodeAttributeActive(EULER_3))
    {
   	 max = ElleMaxFlynns();
   	 for (j=0;j<max;j++)
		if (ElleFlynnIsActive(j))
        {
            if (strlen(label[0])==0)
            {
                   // assumes QUARTZ if no MINERAL attrib
                    if (ElleFlynnHasAttribute(j,MINERAL))
                        ElleGetFlynnMineral(j, &mintype);
                    if(mintype==QUARTZ || mintype==CALCITE)
                        for(i=0;i<3;i++)
                            strcpy(label[i], label_hexagonal[i]);
                    else
                        for(i=0;i<3;i++)
                            strcpy(label[i], label_cubic[i]);
            }
		    ngn_code[0]++;
        }

   	 for(i=0;i<NumPlots;i++)
   	 {
        if (caxis_output && i<1) {
          polarout[i]=fopen(polarfile[i].c_str(),"w");
          if (fprintf(polarout[i],"%s %s %s\n","ID","Azimuth","Dip")<0)
            OnError((char *)(polarfile[i].c_str()),WRITE_ERR);
        }
		startsteronet(center[i],radius,psout,label[i],ngn_code[i]);

		for (j=0;j<max;j++)
		{

		    if (ElleFlynnIsActive(j))
		    {
            
                if (caxis_output && i<1) {
                    if (fprintf(polarout[i],"%d ",j)<0)
                        OnError((char *)(polarfile[i].c_str()),WRITE_ERR);
                }
                if (ElleFlynnHasAttribute(j,MINERAL))
                        ElleGetFlynnMineral(j, &mintype);

				if(mintype==QUARTZ || mintype==CALCITE)
				{
					 //retrieve euler angles
				    ElleGetFlynnEuler3(j, &curra, &currb, &currc);
				    eulerZXZ(rmap, curra*M_PI/180,
                                   currb*M_PI/180,
                                   currc*M_PI/180);
	
				    plotonept(axis_hexagonal[i],rmap,center[i],radius,
                                psout,polarout[i]);	    
				}
                else // assume cubic
                {
                    ElleGetFlynnEuler3(j, &curra, &currb, &currc);
                    eulerZXZ(rmap, curra*M_PI/180, currb*M_PI/180, currc*M_PI/180);

                    plotonept(axis_cubic[i],rmap,center[i],radius,
                                psout,polarout[i]);
                 }
		    }
		}
   	 }
    }
    else {
   	 max = ElleMaxUnodes();
     int rows=0, cols=0;
// Need a gui to allow user to set step, c-axis output flag and
//   reverse_row_order ie plotaxes user options
//temporarily using user data via run opts dialog
   	 int step = user_step;
   	 if (step<1) step = 1;
   	 ngn_code[0]=max/step;
                                                                                
   	 for(i=0;i<3;i++)
   	 {
          strcpy(label[i], label_hexagonal[i]);
          if (i<NumPlots) {
              polarout[i]=fopen(polarfile[i].c_str(),"w");
              if (fprintf(polarout[i],"%s %s %s %s\n",
                                    "X","Y","Azimuth","Dip")<0)
                  OnError((char *)polarfile[i].c_str(),WRITE_ERR);
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
            if(mintype==QUARTZ || mintype==CALCITE)
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
                                                                                
		   	//retrieve euler angles
               ElleGetUnodeAttribute(j, &curra, &currb, &currc,EULER_3);
               orientmatZXZ(rmap, curra*M_PI/180, currb*M_PI/180,
                                     currc*M_PI/180);
/*  if file contains only unodes, temporarily assume hexagonal  */
               if (ElleMaxFlynns()>0 && ElleFlynnAttributeActive(MINERAL)) {
                ElleGetFlynnMineral(ElleUnodeFlynn(j), &mintype);

                if(mintype==QUARTZ || mintype==CALCITE)
                        plotonept(axis_hexagonal[i],rmap,center[i],radius,
                                    psout,0);
                else // assume cubic
                        plotonept(axis_cubic[i],rmap,center[i],radius,
                                    psout,0);
              }
              else {
                    if (caxis_output && i<1) {
                        x = j%cols;
                        y = j/rows;
                        if (Reverse_row_order)  y=(rows-1)-y;
                        if (fprintf(polarout[i],"%d %d ",x,y)<0)
                            OnError((char *)polarfile[i].c_str(),WRITE_ERR);
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

    pendps(psout);
    for (i=0;i<3;i++)
        if (polarout[i]) fclose(polarout[i]);
} 

void plotonept(double *axis, double rmap [3][3], double *center, double
radius, FILE *psout,FILE *polarout)
{
    int i;
    double axis2[3];
    double phi,rho;

    /*change(axis,axis2,rmap);*/
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

void pstartps(FILE *psout)
{
    fprintf(psout,"%%!PS-Adobe-2.0\n\n");
    fprintf(psout,"0 setlinewidth\n");
    fprintf(psout,"72 72 scale\n\n");
    fprintf(psout,"/Helvetica findfont\n");
    fprintf(psout,"0.25 scalefont\n");
    fprintf(psout,"setfont\n");

    
}

void pendps(FILE *psout)
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
    dx = (xy.x-refxy.x);
    eps = dx*0.1;
    *numperrow=1;
    while (*numperrow<max_unodes && (fabs(xy.y-refxy.y)<eps)) {
        (*numperrow)++;
        ElleGetUnodePosition(*numperrow,&xy);
    }
    if (*numperrow<max_unodes) {
        *rows = max_unodes/(*numperrow);
    }
    else err=1;
    return(err);
}
