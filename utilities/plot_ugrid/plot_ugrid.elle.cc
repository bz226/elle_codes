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

int ugrid_size;

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
         * This process assumes unodes are in the elle file
         */
        if (!ElleUnodesActive())
            OnError("No unodes found in file",1);
    }
    
}

int ProcessFunction()
{
    int i,j;
    char *infile,psfile[100];
    Coords upt[ugrid_size][ugrid_size];
    FILE *psout;
    //char syscall[100];
    double scale=7.5,offset=0.5 ;
    UserData userdata;
    int ua,di,dj;
    double thresh=0.15;
    
    ElleUserData(userdata);
    ua=(int)userdata[UGridAngle]; // Set grid angle
		
		if(ua==0)
		{
			di=0;
			dj=1;
		}
		else
		{
			di=1;
			dj=1;
		}
    infile = ElleFile();
    sprintf(psfile,"%s_ugrid.ps",infile);
    

    //sprintf(syscall,"gv %s&",psfile);
    
    psout=fopen(psfile,"w");
    startps(psout);

    fprintf(psout,"newpath\n");
    fprintf(psout,"%lf %lf moveto\n", offset,offset+scale+.25);
    fprintf(psout,"(%s %d) show\n",psfile,ua);
    
    fprintf(psout,".002500 setlinewidth\n");
    fprintf(psout,"0.5 0.5 0.5 setrgbcolor\n");
		
		for(i=0;i<ugrid_size;i++)
		{
			for(j=0;j<ugrid_size;j++)
			{
    		
    			ElleGetUnodePosition(j+(i*ugrid_size), &upt[j][i]);

    	}
    }
    
 		for(i=0;i<ugrid_size;i++)
		{
			for(j=0;j<ugrid_size;j++)
			{
					if(j==0)
					{
						      fprintf(psout,"newpath\n");
    							fprintf(psout,"%lf %lf moveto\n",offset+scale*upt[(int)(fmod(i+(di*j),ugrid_size))][j].x,offset+scale*upt[(int)(fmod(i+(di*j),ugrid_size))][j].y);
					}

    			if(sqrt(((upt[(int)(fmod(i+(di*j),ugrid_size))][j].x-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x)*
    				       (upt[(int)(fmod(i+(di*j),ugrid_size))][j].x-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x))+
    				      ((upt[(int)(fmod(i+(di*j),ugrid_size))][j].y-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y)*
    				       (upt[(int)(fmod(i+(di*j),ugrid_size))][j].y-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y)))>thresh  )
    			{
    				 	fprintf(psout,"stroke\n");
				      fprintf(psout,"newpath\n");
							fprintf(psout,"%lf %lf moveto\n",offset+scale*upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x,
							                                 offset+scale*upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y);
					}		
   			  else
    			    fprintf(psout,"%lf %lf lineto\n",offset+scale*upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x,
							                                 offset+scale*upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y);
    				
    	
    /*		printf("(%d,%d) (%d,%d) (%lf,%lf) (%lf,%lf) %lf\n",(int)(fmod(i+(di*j),ugrid_size)),j,(int)fmod(i+(di*j)+di,ugrid_size),(int)fmod(j+dj,ugrid_size),
    		      upt[(int)(fmod(i+(di*j),ugrid_size))][j].x,
     		      upt[(int)(fmod(i+(di*j),ugrid_size))][j].y,
    					upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x,
    					upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y,
    					sqrt(((upt[(int)(fmod(i+(di*j),ugrid_size))][j].x-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x)*
    				        (upt[(int)(fmod(i+(di*j),ugrid_size))][j].x-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x))+
    				       ((upt[(int)(fmod(i+(di*j),ugrid_size))][j].y-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y)*
    				        (upt[(int)(fmod(i+(di*j),ugrid_size))][j].y-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y))));*/
    			
    	}
    	fprintf(psout,"stroke\n");
    }
 
    fprintf(psout,"1 0 0 setrgbcolor\n");
		fprintf(psout,".0012500 setlinewidth\n");
		if(ua==0)
		{
			di=1;
			dj=0;
  
 		for(j=0;j<ugrid_size;j++)
		{
			for(i=0;i<ugrid_size;i++)
			{
					if(i==0)
					{
						      fprintf(psout,"newpath\n");
    							fprintf(psout,"%lf %lf moveto\n",offset+scale*upt[(int)(fmod(i+(di*j),ugrid_size))][j].x,offset+scale*upt[(int)(fmod(i+(di*j),ugrid_size))][j].y);
					}

    			if(sqrt(((upt[(int)(fmod(i+(di*j),ugrid_size))][j].x-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x)*
    				       (upt[(int)(fmod(i+(di*j),ugrid_size))][j].x-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x))+
    				      ((upt[(int)(fmod(i+(di*j),ugrid_size))][j].y-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y)*
    				       (upt[(int)(fmod(i+(di*j),ugrid_size))][j].y-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y)))>thresh  )
    			{
    				 	fprintf(psout,"stroke\n");
				      fprintf(psout,"newpath\n");
							fprintf(psout,"%lf %lf moveto\n",offset+scale*upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x,
							                                 offset+scale*upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y);
					}		
   			  else
    			    fprintf(psout,"%lf %lf lineto\n",offset+scale*upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x,
							                                 offset+scale*upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y);
    				
    	
    /*		printf("(%d,%d) (%d,%d) (%lf,%lf) (%lf,%lf) %lf\n",
    		      j,
    		      (int)(fmod(i+(di*j),ugrid_size)),
    		      (int)fmod(j+dj,ugrid_size),
    		      (int)fmod(i+(di*j)+di,ugrid_size),
    		      upt[(int)(fmod(i+(di*j),ugrid_size))][j].x,
     		      upt[(int)(fmod(i+(di*j),ugrid_size))][j].y,
    					upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x,
    					upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y,
    					sqrt(((upt[(int)(fmod(i+(di*j),ugrid_size))][j].x-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x)*
    				        (upt[(int)(fmod(i+(di*j),ugrid_size))][j].x-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x))+
    				       ((upt[(int)(fmod(i+(di*j),ugrid_size))][j].y-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y)*
    				        (upt[(int)(fmod(i+(di*j),ugrid_size))][j].y-upt[(int)fmod(i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y))));
    			*/
    	}
    	fprintf(psout,"stroke\n");
 	}
		}
		else
		{
			di=ugrid_size-1;
			dj=1;
			
  
 		for(i=0;i<ugrid_size;i++)
		{
			for(j=0;j<ugrid_size;j++)
			{
					if(j==0)
					{
						      fprintf(psout,"newpath\n");
    							fprintf(psout,"%lf %lf moveto\n",offset+scale*upt[(int)(fmod(ugrid_size-i+(di*j),ugrid_size))][j].x,offset+scale*upt[(int)(fmod(ugrid_size-i+(di*j),ugrid_size))][j].y);
					}

    			if(sqrt(((upt[(int)(fmod(ugrid_size-i+(di*j),ugrid_size))][j].x-upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x)*
    				       (upt[(int)(fmod(ugrid_size-i+(di*j),ugrid_size))][j].x-upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x))+
    				      ((upt[(int)(fmod(ugrid_size-i+(di*j),ugrid_size))][j].y-upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y)*
    				       (upt[(int)(fmod(ugrid_size-i+(di*j),ugrid_size))][j].y-upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y)))>thresh  )
    			{
    				 	fprintf(psout,"stroke\n");
				      fprintf(psout,"newpath\n");
							fprintf(psout,"%lf %lf moveto\n",offset+scale*upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x,
							                                 offset+scale*upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y);
					}		
   			  else
    			    fprintf(psout,"%lf %lf lineto\n",offset+scale*upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x,
							                                 offset+scale*upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y);
    				
    	
  /*  		printf("(%d,%d) (%d,%d) (%lf,%lf) (%lf,%lf) %lf\n",
    		      (int)(fmod(ugrid_size-i+(di*j),ugrid_size)),
    		      j,
    		      (int)fmod(ugrid_size-i+(di*j)+di,ugrid_size),
    		      (int)fmod(j+dj,ugrid_size),
    		      upt[(int)(fmod(ugrid_size-i+(di*j),ugrid_size))][j].x,
     		      upt[(int)(fmod(ugrid_size-i+(di*j),ugrid_size))][j].y,
    					upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x,
    					upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y,
    					sqrt(((upt[(int)(fmod(ugrid_size-i+(di*j),ugrid_size))][j].x-upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x)*
    				        (upt[(int)(fmod(ugrid_size-i+(di*j),ugrid_size))][j].x-upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].x))+
    				       ((upt[(int)(fmod(ugrid_size-i+(di*j),ugrid_size))][j].y-upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y)*
    				        (upt[(int)(fmod(ugrid_size-i+(di*j),ugrid_size))][j].y-upt[(int)fmod(ugrid_size-i+(di*j)+di,ugrid_size)][(int)fmod(j+dj,ugrid_size)].y))));
   */ 			
    	}
    	fprintf(psout,"stroke\n");
 	}
 }
 
    fprintf(psout,"0 0 0 setrgbcolor\n");
    fprintf(psout,"newpath\n");
		fprintf(psout,"%lf %lf moveto\n",offset,offset);
		fprintf(psout,"%lf %lf lineto\n",offset+scale,offset);
		fprintf(psout,"%lf %lf lineto\n",offset+scale,offset+scale);
		fprintf(psout,"%lf %lf lineto\n",offset,offset+scale);
		fprintf(psout,"%lf %lf lineto\n",offset,offset);
    fprintf(psout,"stroke\n");


    endps(psout);
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
