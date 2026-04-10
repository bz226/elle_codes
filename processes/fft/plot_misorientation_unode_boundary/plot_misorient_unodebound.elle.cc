#include <stdio.h>
#include <math.h>
#include "error.h"
#include "file.h"
#include "init.h"
#include "runopts.h"
#include "interface.h"
#include "unodes.h"
#include "convert.h"
#include "mat.h"
#include "plot_ugrid.h"


int InitThisProcess(), ProcessFunction();
void startps(FILE *);
void endps(FILE *);
double misorientation(int unode_id1,int unode_id2);
void draw_eps (double misorient_angle, int unode_id, int type);
 
int ugrid_size;
// double lagb=4.0, hagb=10.0;
double lagb,hagb;

    FILE *psout;
    double scale=5.0,offset=1.5;	
/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitThisProcess()
{
    char *infile;
    int err=0;
    UserData userdata;

    ElleUserData(userdata);
    ugrid_size=(int)userdata[UGridSize]; // Set grid size
	lagb= userdata[1]; // minimum low angle grain boundary to plot_ugrid  
	hagb= userdata[2]; // transition between low to high angle grain boundary
	
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
    }
}

int ProcessFunction()
{
    int i,j;
    char *infile,psfile[100];
    Coords upt[ugrid_size][ugrid_size];
	int unode_id[ugrid_size][ugrid_size];
	int unode_1, unode_2,unode_3,i_1,j_1,j_2;
	double misangle, mean_misorient,totalangle=0;
	double total_gbe, gbe, mean_gbe, total_segment;
	double inc_shear, dummy1, dummy2; 
	
    CellData unitcell;
 
	inc_shear=ElleSSOffset();
	printf("shear_offsset %e  ",inc_shear);
	
    //char syscall[100];
    UserData userdata;

    infile = ElleFile();
    sprintf(psfile,"%s_mis_gb.ps",infile); 
    //sprintf(syscall,"gv %s&",psfile);
    
    psout=fopen(psfile,"w");
    startps(psout);
    fprintf(psout,"newpath\n");
    fprintf(psout,"%lf %lf moveto\n", offset,offset+scale+1);
    fprintf(psout,"(%s) show\n",psfile);

// store unode_id in an int(matrix)
for (i=0;i<=ugrid_size;i++) {	
	
	for (j=0;j<=ugrid_size;j++) {	
		unode_id[i][j]=i*ugrid_size+j;
		// printf("jjj %d %d %d\n", i,j,unode_id[i][j]);
	}
}

/* 
		for(i=0;i<ugrid_size;i++)
		{
			for(j=0;j<ugrid_size;j++)
			{ 		
    			ElleGetUnodePosition(j+(i*ugrid_size), &upt[j][i]);
    	}
    }
  */

// check unodes neighbours 
for (i=0;i<ugrid_size;i++) {	
 
	for (j=0;j<ugrid_size;j++) {		
			
	unode_1=unode_id[i][j];
		
	// check unodes neigbhbours		
		i_1=i-1;
		j_1=j-1;
		j_2=j+1;
		
	// conditionals to wrapping unode_id information	
		if 	(i_1 < 0) i_1=ugrid_size-1;
		if 	(i_1 > ugrid_size-1) i_1=0;
		if (j_1 <0) j_1=ugrid_size-1;
		if (j_1 >ugrid_size-1) j_1=0;

		// Condtional to wrapping unode_id information 2nd Lower unode 
		
		if 	(j_2 < 0) j_2=ugrid_size-1;
		if 	(j_2 > ugrid_size-1) j_2=0;
			
	// left unode	
		unode_2=unode_id[i][j_1];
	    misangle=misorientation(unode_1,unode_2); 
		
	    if (misangle > lagb) draw_eps(misangle,unode_1,0);
		
		totalangle = totalangle + misangle;
		
 		// printf("total_angle %lf  ", totalangle);
		
		// Grain boundary energy
	   if (misangle <= hagb) {
		gbe=1-log(misangle/hagb);
		gbe=1.0*(misangle/hagb)*gbe;
		if (misangle == 0) gbe=0.0;   
	   }
	   else gbe=1.0; 
        total_gbe += gbe;

	   
	// lower unode
	// CALCULATE misorientation non-ortho mesh 
	   // misangle=0.0;
	   unode_2=unode_id[i_1][j];
	   unode_3=unode_id[i_1][j_2];
	  
	   dummy1=misorientation(unode_1,unode_2)*(1-inc_shear);
	   dummy2= misorientation(unode_1,unode_3); 
	   dummy2=dummy2*inc_shear;
	   misangle=dummy1+dummy2;
	   if (misangle > lagb) draw_eps(misangle,unode_1,1); 
	   totalangle = totalangle+misangle;	

 // printf("total_angle %e  \n", totalangle);
	   
		// Grain boundary energy
	   if (misangle <= hagb) {
		gbe=1-log(misangle/hagb);
		gbe=1.0*(misangle/hagb)*gbe;
		if (misangle == 0) gbe=0.0;  
	   }
	   else gbe=1.0;
		   
        total_gbe += gbe;	  
        total_segment += 2.0;	   
  	}
  }
  
  fprintf(psout,"0 setlinewidth\n");
  fprintf(psout,"1 0 0 setrgbcolor\n"); 
    fprintf(psout,"newpath\n");
		fprintf(psout,"%lf %lf moveto\n",offset,offset);
		fprintf(psout,"%lf %lf lineto\n",offset+scale,offset);
		fprintf(psout,"%lf %lf lineto\n",offset+scale,offset+scale);
		fprintf(psout,"%lf %lf lineto\n",offset,offset+scale);
		fprintf(psout,"%lf %lf lineto\n",offset,offset);
    fprintf(psout,"stroke\n");


    endps(psout);

mean_misorient= totalangle/total_segment;
mean_gbe = total_gbe/total_segment;
  
  printf("mean misorientation %e %e\n", mean_misorient, mean_gbe); 
} 

		
 void draw_eps (double misorient_angle, int unode_id, int type)
// type 0,1 to indicate left and lower boundary

{
	double x_1, y_1, x_2,y_2;
	Coords xy; 
	double dummy1;
    
	ElleGetUnodePosition(unode_id, &xy);
	
	x_1=xy.x-(1/(2*ugrid_size));
	y_1=xy.y-(1/(2*ugrid_size));

	
   if (misorient_angle > hagb) {
	   fprintf(psout,"0 0 0 setrgbcolor\n");
	   fprintf(psout,"0.01 setlinewidth\n");
   }	   
   else	   
    { 
	   dummy1=1-(misorient_angle/hagb);
	   fprintf(psout,"%lf %lf %lf setrgbcolor\n",dummy1,dummy1,dummy1);	
	   fprintf(psout,"0.005 setlinewidth\n");		
   }
   
	if (type == 0) {
    	x_2=x_1;
	    y_2=y_1+(1.0/ugrid_size);
	}
	
	if (type == 1) {
	   x_2=x_1+(1.0/ugrid_size);
	   y_2=y_1;
	}
	
	   // drawline
				fprintf(psout,"newpath\n");
				fprintf(psout,"%lf %lf moveto\n",offset+scale*x_1,offset+scale*y_1);
				fprintf(psout,"%lf %lf lineto\n",offset+scale*x_2,offset+scale*y_2);
    		    fprintf(psout,"stroke\n");
		
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

double misorientation(int unode_id1,int unode_id2)
{
	double euler_1,euler_2;
	double dummy1, dummy2;
	double misorientangle;
	
	ElleGetUnodeAttribute(unode_id1,&euler_1,&dummy1,&dummy2,EULER_3);
	ElleGetUnodeAttribute(unode_id2,&euler_2,&dummy1,&dummy2,EULER_3);
		// printf("unode_id %d %d\n", unode_id1,unode_id2);	
	    // printf("misorient %lf %lf\n", euler_1,euler_2);	
	// only [0,180] 
	if (euler_1<0) euler_1=180+euler_1;
	if (euler_2<0) euler_2=180+euler_2;
	
	misorientangle=fabs(euler_1-euler_2);
	
	if (misorientangle>90) misorientangle=180-misorientangle;

		return(misorientangle);

}
