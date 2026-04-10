/*!
This process reads an image and converts it to an elle microstructure
  This process uses the following command line options
  -i option to set the input image file
  batch, (wx, X with the -n option):
  -o option to set the output elle file
  If the output file name is not specified, ".elle" is appended to
  the input filename.

BATCH build:
  Uses the functions in the elle basecode to read the image. Currently
  only ppm and pnm are implemented.
  Converts the file specified and exits

WX build:
  Uses the wxWidget image functions and will convert files in formats read
  by wxWidgets.
  If image specified on command line, convert and display
  Writing the elle file is not automatic. User must save.
  Need to prompt user to save if 'Quit' or 'Open->File' options are chosen.
  Use the Run Options dialog to change the SwitchDistance and alter the
  number of bnodes before saving the elle file
  If Run is chosen, the process will adjust bnode density and display.

X build:
  Uses the functions in the elle basecode to read the image. Currently
  only ppm and pnm are implemented.
  Menu options and behaviour - see WX

Process Init() performs the following steps:
  Set the process run().
  if an input file has been specified:
     If an ouput file has not been specified, set it to the input file
       with ".elle" appended
     Set decimate from the user data
     Clear the Elle data structures
     Create  Elle data from the input image (assumes input
       file is an image file). Unodes are given a U_ATTRIB_A value 
       corresponding to the image value.
     Set the flynn F_ATTRIB_A values from the unode U_ATTRIB_A values
     Clean out the unodes if flag set by user data

Process Run() performs the following steps:
  Adjust bnode density
  If GUI, update display
  else save data to output file (batch or running with -n)


Problems to be fixed:
  GUI File->Open needs filter to be image extensions not .elle
  Need to prompt user to save if 'Quit' or 'Open->File' options are
chosen.
  ElleOutFile setting
  Unode output - do we want it? HEX_GRID, mike hexagonal pattern
inconsistency (doesn't necessarily wrap in Y direction?) Grid pattern
and whether to save unodes should be user data
  
 */

#include <stdio.h>
#include <math.h>
#include <iostream>

#include "attrib.h"
#include "nodes.h"
#include "unodes.h"
#include "update.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "init.h"
#include "stdlib.h"
#include "general.h"
#include "check.h"
#include "version.h"
#include "log.h"
#include "mat.h"

#include "display.h"
#include "image.h"

/*#include "ppm2elle.h"*/

using std::cout;
using std::endl;

int DoSomethingToFlynn(int flynn);
int InitThisProcess(), ProcessFunction(), CheckAndShowFile();
int PromptFileSave();
int image2elle( char *infile, int decimate);
int	FindSegs(int **image, int irows, int icols, double ***segs, int *nsegs);
int Segment(double pts[3][2], double ***segs, int *nsegs, int *colours);
int FindNodes(double ***segs,int nsegs, double **nodes, int *nnodes);
int Same(double a[2], double b[2]);
int MakePolys(double ***segs, int nsegs, double **nodes, int nnodes, int **polys, int *npolys);
int FindRight(double **nodes, int lastnode, int endnode, int trial);
void startps(FILE *);
void endps(FILE *);
int WriteElleFile(int **polys,int npoly,double **nodes, int nnodes,char *fname, int irows, int icols, int **image);		 
int ConvertDataToElle(int **polys,int npoly,double **nodes,int nnodes,
                      int irows,int icols, int **image,
                      std::vector<int> &rgbcols,
                      std::vector<Coords_3D> &orient,
                      double dim);

static int pattern = SQ_GRID;
static int clean=1;
/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitThisProcess()
{
    char *infile; //input image file
    char outfile[FILENAME_MAX]; //output elle file
    int err=0;
    int decimate;

    UserData udata;

    ElleSetRunFunction(CheckAndShowFile);
    ElleSetExitFunction(PromptFileSave);
  
    infile = ElleFile();
    if (strlen(infile)>0) {
       	sprintf(logbuf,"opening %s\n",ElleFile());
        Log(0,logbuf);
        /*
         * if user has not set output filename (with -o)
         *  set it to the input filename with .elle appended
         *  Warning - This keeps the same path
         */
        outfile[0] = '\0';
        if (strlen(ElleOutFile())==0) {
            strcpy(outfile,infile);
	        strcat(outfile,".elle");
            ElleSetOutFile(outfile);
        }
    
        ElleUserData(udata);
        decimate = (int)udata[0];
        clean = ((int)udata[1]!=0)? 0 : 1;

       /*
        * clear the data structures
        */
        ElleReinit();

        if ((err=image2elle(infile, decimate))==0) {

            /*
             * read the data and cleanup the microstructure
            if (strlen(ElleFile())>0) {
                if (err=ElleReadData(ElleFile())) OnError(ElleFile(),err);
            }
             */
            /*
             * cleanup the microstructure and display
             */
            err = ProcessFunction();

        }
    }
    return(err);
}

int CheckAndShowFile()
{
    int i,j,l,max;
    for(i=0;i<5;i++)
    {
        max = ElleMaxNodes();
            for (j=0,l=0;j<max;j++) {
                if (ElleNodeIsActive(j)){
                l++;
            if(l%2 == 0)
                        if (ElleNodeIsDouble(j))
                    ElleCheckDoubleJ(j);
                }
            }
    }
    if (ElleDisplay()) EllePlotRegions(ElleCount());
    else ElleWriteData(ElleOutFile());
}

int ProcessFunction()
{
    int i, k, l,j;
    int err=0,max;
    double val, orient[3];
	int grainlist[10000],ngrains=0,same=0;
    /*ElleCheckFiles();*/
    for (k=0;k<10000;k++) grainlist[k]=-1;
	
	if(ElleUnodesActive())
	{	
	
    	max = ElleMaxFlynns();		// index of maximum flynn used in model

    	for (k=0;k<max;k++) 		// loop though all flynns
		{
        	if (ElleFlynnIsActive(k)) // process flynn if it is active
		    //cout << "Flynn active" << k << endl;
			{
				std::vector<int> unodelist; // define vector list of unodes
    			ElleGetFlynnUnodeList(k,unodelist); // get the list of unodes for a flynn
				val=0.0;
                orient[0] = orient[1] = orient[2] = 0.0;
                if (ElleFlynnAttributeActive(EULER_3)) {
				  if (unodelist.size()>0)
                     //  get the unode's attribute
				    ElleGetUnodeAttribute(unodelist[0],
                                          &orient[0],&orient[1],&orient[2],
                                          EULER_3);
                   // set attribute for this flynn
				  ElleSetFlynnEuler3(k,orient[0],orient[1],orient[2]);
                }
                else {
				    if (unodelist.size()>0)
                     //  get the unode's attribute
				      ElleGetUnodeAttribute(unodelist[0],U_ATTRIB_A, &val);
				    for(i=0,same=0;i<ngrains;i++)
				    {
				  	  if((int)val == grainlist[i])
					  {
						//printf("k, i,grainlist[i],val:%d %d %d %d\n",k, i,grainlist[i],(int)val);
						val=i;
						same=1;
						break;
					  }
				    }
				    if(same==0)
				    {
				  	  //printf("k, ngrains,val:%d %d %d\n",k, ngrains,(int)val);
					  grainlist[ngrains]=(int) val;
					  val=(double) ngrains++;
				    }
                   // set attribute for this flynn
				  ElleSetFlynnRealAttribute(k, (double)val,F_ATTRIB_A);
				}

			}
    	}
    	/*
    	 * update the count and redisplay
    	 * check whether to write an elle file
    	 */
	//cout << "UnodesClean" << endl;	 
	if (clean==1) UnodesClean(); 
	/*clean=1;*/
	}
    ElleWriteData(ElleOutFile());
	for(i=0;i<5;i++)
	{
		max = ElleMaxNodes();
        	for (j=0,l=0;j<max;j++) {
        	    if (ElleNodeIsActive(j)){
	    		l++;
			if(l%2 == 0)
                		if (ElleNodeIsDouble(j)) 
					ElleCheckDoubleJ(j);
        	    }
        	}
	}
    ElleSetFile(ElleOutFile());
    /*
     * stuff that is usually done when Elle file is read
     * only necessary for X, no effect in wx
     */
    if (ElleDisplay()) {
        ElleSetRescale(1);
        ElleUpdateSettings();
    }

	return(err);
} 
             
int PromptFileSave()
{
    
}

int image2elle(char *fname, int decimate)

{
	int **image;  		// decimated image array
	double ***segs;     // segments array
	int nsegs=0;    	// number of segments
	double **nodes;		// node array
	int nnodes=0;		// number of nodes
	int **polys;		// polygon array
	int npolys=0;		// number of polygons
	int irows,icols;	// number of rows and columns in decimated image
    int i,err=0;
    std::vector<int> rgbcols;
    std::vector<Coords_3D> orient;
    double dim;
		
    image = 0;
    dim = ElleUnitLength();
		
    // read and decimate image
	if (err=ElleReadImage(fname, &image, &irows,&icols,decimate,
                          rgbcols, orient, &dim)) return(err);

#if XY
	segs=(double ***)d3tensor(0L,100000L,0L,2,0L,2);
	nodes=(double **)dmatrix(0L,100000L,0L,2);
	polys=(int **)imatrix(0L,10000L,0L,1000L);
#endif
        long maxi = irows*icols*10;
        long maxp = maxi/10;
	segs=(double ***)d3tensor(0L,maxi,0L,2,0L,2);
	nodes=(double **)dmatrix(0L,maxi,0L,2);
	polys=(int **)imatrix(0L,maxp,0L,maxp/10);
	
	FindSegs(image, irows, icols, segs, &nsegs);			// find all segments
	FindNodes(segs, nsegs, nodes, &nnodes);					// find and index all unique nodes
	MakePolys(segs, nsegs, nodes, nnodes, polys, &npolys);	// connect all segments
	// convert to elle
	ConvertDataToElle(polys,npolys,nodes,nnodes,irows,icols,image,
                      rgbcols,orient,dim);

	// write out file
	//WriteElleFile(polys,npolys,nodes,nnodes,ElleOutFile(),irows,icols,image);
	free_imatrix(image,0L,(long)irows,0L,(long)icols);
	free_imatrix(polys,0L,maxp,0L,maxp/10);
	free_dmatrix(nodes,0L,maxi,0L,2L);
	free_d3tensor(segs,0L,maxi,0L,2,0L,2);
    return(err);
}

/*********************************************************

  ReadImage reads in a ppm (portable pixmap) image format file
  and decimates image before stroing in array


int ReadImage(char *fname, int **image, int *rows, int *cols, int decimate)
{
	FILE *in;
	int i,j;
	int icols,irows;
	unsigned long colr;
	int colsP,rowsP;
	pixval maxvalP;
	pixel **pixmap;
	
	printf("filename:",fname);
	in=fopen(fname,"rb"); // assumes binary ppm format image
	if(in == 0L)
	{
		printf("couldn't find file %s\n",fname);
		exit(0);
	}
	
	pixmap=ppm_readppm( in, &colsP, &rowsP, &maxvalP );
	printf("maxvalP:%d\n",(int)maxvalP);
				
	for(i=0;i<rowsP;i++)
	{
		for(j=0;j<colsP;j++)
		{
			//printf("rgb:%d %d %d\n",pixmap[i][j].r,pixmap[i][j].g,pixmap[i][j].b);
			colr=pixmap[i][j].r+(pixmap[i][j].g*maxvalP)+(pixmap[i][j].b*maxvalP*maxvalP);
			
			if(i%decimate==0 && j%decimate==0)
				image[i/decimate][j/decimate]=(int)colr;
		}
	}
	
	for(i=0;i<rowsP/decimate;i++)
		image[i][colsP/decimate]=image[i][0];
	
	for(j=0;j<colsP/decimate;j++)
		image[rowsP/decimate][j]=image[0][j];
		
	image[rowsP/decimate][colsP/decimate]=image[0][0];
	
	fclose(in);
	
	
	*rows=(rowsP/decimate);
	*cols=(colsP/decimate);
	
	for(i=0;i<=*rows/decimate;i++)
	{
		for(j=0;j<=*cols/decimate;j++)
		{
			//printf("%d ",image[i][j]);
		}
		//printf("\n");
	}
	
}
**********************************************************/

/*********************************************************

  FindSegs finds all segments that separate corners of triangles
  defined by image pixels. Two triangles are formed from each small square
  of pixels and simple or triple junction segments are created
  when adjacent corners have a different colour. A Postscript version
  of segments is also written out for debugging purposes only. 

**********************************************************/

int	FindSegs(int **image, int irows, int icols, double ***segs, int *nsegs)
{
	int i,j,k;
	double pts[3][2];
	int colours[3];
	FILE *ps;
	
	for(i=0;i<irows;i++)
	{
		for(j=0;j<icols;j++)
		{
			for(k=0;k<2;k++)
			{
				if(image[i][j] != image[i+1][j+1])
				{
					colours[0]=image[i][j];
					colours[1]=image[i][j+1];
					colours[2]=image[i+1][j];

					pts[0][0]=(j+0.5)/icols;
					pts[0][1]=1.0-((i+0.5)/irows);

					pts[1][0]=pts[0][0]+((double) 1/icols);
					pts[1][1]=pts[0][1];

					pts[2][0]=pts[0][0];
					pts[2][1]=pts[0][1]-((double) 1/irows);

					if(k==1)
					{
						pts[0][0]=pts[1][0];
						pts[0][1]=pts[2][1];					
						colours[0]=image[i+1][j+1];
					}
				}
				else
				{
					colours[0]=image[i][j];
					colours[1]=image[i+1][j+1];
					colours[2]=image[i+1][j];

					pts[0][0]=(j+0.5)/icols;
					pts[0][1]=1.0-((i+0.5)/irows);

					pts[1][0]=pts[0][0]+((double) 1/icols);
					pts[1][1]=pts[0][1]-((double) 1/irows);

					pts[2][0]=pts[0][0];
					pts[2][1]=pts[0][1]-((double) 1/irows);

					if(k==1)
					{
						pts[2][0]=pts[1][0];
						pts[2][1]=pts[0][1];					
						colours[2]=image[i][j+1];
					}
				
				}
				Segment(pts,segs,nsegs,colours);
			}
		}
		//printf("\n");
	}
	//ps=fopen("junk.ps","w");
	//startps(ps);
	for(i=0;i<*nsegs;i++)
	{
		//printf("%d %lf %lf  %lf %lf\n",i,segs[i][0][0],segs[i][0][1],segs[i][1][0],segs[i][1][1]);
    	//fprintf(ps,"newpath\n");
    	//fprintf(ps,"%lf %lf moveto\n",segs[i][0][0]+.5,segs[i][0][1]+.5);
    	//fprintf(ps,"%lf %lf lineto\n",segs[i][1][0]+.5,segs[i][1][1]+.5);
    	//fprintf(ps,"stroke\n");
		if(segs[i][0][0] > 1.0)
			segs[i][0][0] -= 1.0;

		if(segs[i][0][1] < 0.0)
			segs[i][0][1] += 1.0;

		if(segs[i][1][0] > 1.0)
			segs[i][1][0] -= 1.0;

		if(segs[i][1][1] < 0.0)
			segs[i][1][1] += 1.0;
	}
	for(j=0;j<11;j++)
	{
    	//fprintf(ps,"newpath\n");
     	//fprintf(ps,"%lf %lf moveto\n",((double)j/10)+.5,.5);
		//fprintf(ps,"(%d) show\n",j);
/*		for(k=0;k<11;k++)
		{
    		fprintf(ps,"newpath\n");
    		fprintf(ps,"%lf %lf moveto\n",((double)j/10)+.49,((double)k/10)+.5);
    		fprintf(ps,"%lf %lf lineto\n",((double)j/10)+.51,((double)k/10)+.5);
    		fprintf(ps,"stroke\n");
    		fprintf(ps,"newpath\n");
    		fprintf(ps,"%lf %lf moveto\n",((double)j/10)+.5,((double)k/10)+.49);
    		fprintf(ps,"%lf %lf lineto\n",((double)j/10)+.5,((double)k/10)+.51);
    		fprintf(ps,"stroke\n");
    		if(k==0)
			{
    		//	fprintf(ps,"newpath\n");
 			//	fprintf(ps,"%lf %lf moveto\n",.5,((double)k/10)+.5);
			//	fprintf(ps,"(%d) show\n",k);
			}
		}*/
	}
	//endps(ps);
	//exit(0);
}


/*********************************************************

  Segment take a single triangle and actually performs the
  segmentation. Choice of triangle long edge
  is based on sameness of top-left and bottom-right pixels 

		  A1   B			A	C2
		  C   A2 	 0r 	C1	B

**********************************************************/

int Segment(double pts[3][2], double ***segs, int *nsegs, int *colours)
{
	double mid[2];

	mid[0]=(pts[0][0]+pts[1][0]+pts[2][0])/3;
	mid[1]=(pts[0][1]+pts[1][1]+pts[2][1])/3;
	
	if(colours[0]!=colours[1] && colours[0]!=colours[2] && colours[1]!=colours[2]) // ABC
	{
		//printf("%d:ABC\t",*nsegs);
		segs[*nsegs][0][0]=pts[1][0]+((pts[2][0]-pts[1][0])/2);
		segs[*nsegs][0][1]=pts[1][1]+((pts[2][1]-pts[1][1])/2);
		
		segs[*nsegs][1][0]=mid[0];
		segs[*nsegs][1][1]=mid[1];
		
		(*nsegs)++;		
	
		segs[*nsegs][0][0]=pts[0][0]+((pts[1][0]-pts[0][0])/2);
		segs[*nsegs][0][1]=pts[0][1]+((pts[1][1]-pts[0][1])/2);
		
		segs[*nsegs][1][0]=mid[0];
		segs[*nsegs][1][1]=mid[1];
		
		(*nsegs)++;		
	
		segs[*nsegs][0][0]=pts[0][0]+((pts[2][0]-pts[0][0])/2);
		segs[*nsegs][0][1]=pts[0][1]+((pts[2][1]-pts[0][1])/2);
		
		segs[*nsegs][1][0]=mid[0];
		segs[*nsegs][1][1]=mid[1];
		
		(*nsegs)++;		
	
	
	}
	else if(colours[0]==colours[1] && colours[0]!=colours[2] ) // AB
	{
		//printf("%d:AB\t",*nsegs);
		segs[*nsegs][0][0]=pts[0][0]+((pts[2][0]-pts[0][0])/2);
		segs[*nsegs][0][1]=pts[0][1]+((pts[2][1]-pts[0][1])/2);
		
		segs[*nsegs][1][0]=pts[2][0]+((pts[1][0]-pts[2][0])/2);
		segs[*nsegs][1][1]=pts[2][1]+((pts[1][1]-pts[2][1])/2);
		
		(*nsegs)++;		
	}
	else if(colours[1]==colours[2] && colours[0]!=colours[2]) // BC
	{
		//printf("%d:BC\t",*nsegs);
		segs[*nsegs][0][0]=pts[0][0]+((pts[1][0]-pts[0][0])/2);
		segs[*nsegs][0][1]=pts[0][1]+((pts[1][1]-pts[0][1])/2);
		
		segs[*nsegs][1][0]=pts[0][0]+((pts[2][0]-pts[0][0])/2);
		segs[*nsegs][1][1]=pts[0][1]+((pts[2][1]-pts[0][1])/2);
		
		(*nsegs)++;		
	}
	else if(colours[0]==colours[2] && colours[1]!=colours[2]) // AC
	{
		//printf("%d:AC\t",*nsegs);
		segs[*nsegs][0][0]=pts[1][0]+((pts[2][0]-pts[1][0])/2);
		segs[*nsegs][0][1]=pts[1][1]+((pts[2][1]-pts[1][1])/2);
		
		segs[*nsegs][1][0]=pts[0][0]+((pts[1][0]-pts[0][0])/2);
		segs[*nsegs][1][1]=pts[0][1]+((pts[1][1]-pts[0][1])/2);
		
		(*nsegs)++;		
	}
	else
		/*printf("0\t")*/;
}

/*********************************************************

  FindNodes Finds and indexes all unique nodes from segment end points

**********************************************************/

int FindNodes(double ***segs,int nsegs, double **nodes, int *nnodes)
{
	int i,j,k;
	int find;
	
	for(i=0;i<nsegs;i++)
	{
		for(k=0;k<2;k++)
		{
			for(j=0,find=0;j<*nnodes;j++)
			{
				if(Same(segs[i][k],nodes[j])==1)
				{
					segs[i][k][0]=(double) j;
					segs[i][k][1]=0;
					find=1;
				}
			}
			if(find==0)
			{
				//printf("%d %lf %lf\n",*nnodes,segs[i][k][0],segs[i][k][1]);
				nodes[*nnodes][0]=segs[i][k][0];
				nodes[*nnodes][1]=segs[i][k][1];
									
				segs[i][k][0]=(double) (*nnodes);
				segs[i][k][1]=0;
				(*nnodes)++;
			}
		}
	}
	sprintf(logbuf,"nodes:%d\n",*nnodes);
    Log(0,logbuf);
}

/*********************************************************

  Same Checks if two points are coincident

**********************************************************/


int Same(double a[2], double b[2])
{
	double eps=1e-5;
	
		
	if(fabs(a[0]-b[0]) < eps && fabs(a[1]-b[1]) < eps)
		return(1);
	else
		return(0);
}

/*********************************************************

  MakePolys Coalesces all segments in polygons by first doubling
  number of segments to make A/CW and C/W pairs.

**********************************************************/

int MakePolys(double ***segs, int nsegs, double **nodes, int nnodes, int **polys, int *npolys)
{
	int i,j,k,m;
	int pnodes=1;
	int trial[2],ntrials;
	int startnode,endnode,lastnode;
	int whichnode;
	int newpoly=1;
	int segsdone=0;
	
	for(i=0;i<nsegs;i++)
	{		
		segs[i+nsegs][0][0] = segs[i][1][0];
		segs[i+nsegs][0][1] = segs[i][1][1];
		
		segs[i+nsegs][1][0] = segs[i][0][0];
		segs[i+nsegs][1][1] = segs[i][0][1];
	}
	
	nsegs = nsegs*2;
	
	for(i=0;i<nsegs;i++)
	{
		//printf("%d %d %d\n",i, (int) segs[i][0][0],  (int) segs[i][1][0]);
	}
	
	do
	{
		for(i=0;i<nsegs;i++)
		{		
			if((int) segs[i][0][0] >= 0)
			{
				if(newpoly)
				{
					pnodes=1;
					//printf("starting poly %d\n",*npolys);
					startnode=(int)segs[i][0][0];
					lastnode=startnode;
					endnode=(int)segs[i][1][0];
					segs[i][0][0]=-1;
					polys[*npolys][pnodes++]=startnode;
					polys[*npolys][pnodes]=endnode;
					newpoly=0;
					//printf("******\n%d %d\n",startnode,endnode);
					segsdone++;
				}
				do
				{
					for(j=0,ntrials=0;j<nsegs;j++)
					{
						if(segs[j][0][0] >= 0)
						{
							if(segs[j][0][0]==endnode && segs[j][1][0]==lastnode )
							{
							}
							else
							{		  
								if((int) segs[j][0][0] == endnode)
								{
									//printf("seg %d|",j);
									trial[ntrials++]=j;
								}
							}
						}
					}

	 				if(ntrials==1)
					{
						//printf("\n1 trial %d\n",trial[0]);
						lastnode=endnode;
						polys[*npolys][pnodes]=(int)segs[trial[0]][0][0];
						endnode=(int)segs[trial[0]][1][0];	
						segs[trial[0]][0][0]=-1;
						//printf("l,p,e:%d %d %d\n",lastnode,polys[*npolys][pnodes],endnode);
						//printf("%d %d\t*\n",lastnode,endnode);
						pnodes++;
						segsdone++;
					}
					else if(ntrials==2)
					{
						whichnode=FindRight(nodes,lastnode,endnode,(int) segs[trial[0]][1][0]);
						//printf("\n2 trial which %d %d\n",whichnode,(int)segs[trial[whichnode]][0][0]);
						lastnode=endnode;
						polys[*npolys][pnodes]=(int)segs[trial[whichnode]][0][0];
						endnode=(int)segs[trial[whichnode]][1][0];	
						segs[trial[whichnode]][0][0]=-1;
						//printf("l,p,e:%d %d %d\n",lastnode,polys[*npolys][pnodes],endnode);
						//printf("seg %d|%d %d\t**\n",trial[whichnode],lastnode,endnode);
						pnodes++;
						segsdone++;
					}
					else
					{
						//printf("oops %d\n",ntrials);
						exit(0);
					}

					if(startnode==endnode)
					{
						polys[*npolys][0]=pnodes-1;

						//for(k=0;k<pnodes;k++)
							//printf("%d %d %d\n",*npolys,k,polys[*npolys][k]);

						(*npolys)++;
						newpoly=1;
						//segsdone++;
					}
				}while(newpoly==0);				
			}
		}
	    sprintf(logbuf,"segments:%d\n",nsegs);
        Log(0,logbuf);
	}while(segsdone<nsegs);
	sprintf(logbuf,"polygons:%d\n",*npolys);
    Log(0,logbuf);
}

/*********************************************************

  FindRight Determines at a triple junction which is the next 
  correct (actually leftmost!) segment by creating a triangle of
  last node, triple junction node and next node and seeing
  if area is positive (ie Anti-clockwise) or not.

**********************************************************/

int FindRight(double **nodes, int lastnode, int endnode, int trial)
{
	double x[3],y[3],cenx,ceny,area;
	int result;
	
	
	x[0]=nodes[lastnode][0];
	y[0]=nodes[lastnode][1];
	
	
	x[1]=nodes[endnode][0];
	y[1]=nodes[endnode][1];
	
	if(x[0]-x[1] > 0.75)
		x[1]+=1;
	else if(x[1]-x[0] > 0.75)
		x[1]-=1;
	
	if(y[0]-y[1] > 0.75)
		y[1]+=1;
	else if(y[1]-y[0] > 0.75)
		y[1]-=1;
 	
	x[2]=nodes[trial][0];
	y[2]=nodes[trial][1];

	if(x[0]-x[2] > 0.75)
		x[2]+=1;
	else if(x[2]-x[0] > 0.75)
		x[2]-=1;
	
	if(y[0]-y[2] > 0.75)
		y[2]+=1;
	else if(y[2]-y[0] > 0.75)
		y[2]-=1;
	
	result=polyCentroid(x, y, 3, &cenx, &ceny, &area);
	//printf("l e t: %d %d %d\n",lastnode,endnode,trial);
	//printf("result:%d %le %lf %lf\n",result,area,cenx,ceny);	
	if(area<0)
		return(0);
	else
		return(1);
	
}

/*********************************************************

  startps Writes out postcript header

**********************************************************/

void startps(FILE *psout)
{
    fprintf(psout,"%%!PS-Adobe-2.0\n\n");
    fprintf(psout,"0 setlinewidth\n");
    fprintf(psout,"72 72 scale\n\n");
    fprintf(psout,"/Helvetica findfont\n");
    fprintf(psout,"0.25 scalefont\n");
    fprintf(psout,"setfont\n");

    
}
/*********************************************************

  endps Writes out postcript footer

**********************************************************/

void endps(FILE *psout)
{
    fprintf(psout,"showpage\n");
    fflush(psout);
    fclose(psout);
 
}

/*********************************************************

  WriteElleFile Writes out Elle file format file

**********************************************************/
#ifndef HEADER_MAX
#define HEADER_MAX 100
#endif

int WriteElleFile(int **polys,int npoly,double **nodes,int nnodes,char *ellename,int irows,int icols, int **image)
{
	int np,nv,ev;
	int i,j,cntr;
	float xx,yy;     // x and y coordinates for Unodes
	FILE *out;
    char header[HEADER_MAX],*end;

	out=fopen(ellename,"w");
	
	if(out==0L)
		exit(0);
	
                                                                                
    for (i=0;i<HEADER_MAX;i++) header[i]='\0';
                                                                                
    /* write version */
    strncpy(header, ElleGetCreationCString(),HEADER_MAX-1);
    if ((end=strrchr(header,'\n'))!=NULL) {
        end++;
        *end = '\0';
    }
    if (fprintf(out,"%s\n",header)<0) OnError(ellename,WRITE_ERR);
	fprintf(out,"OPTIONS\n");
	fprintf(out,"SwitchDistance %lf\n",0.5/irows);
	fprintf(out,"SpeedUp 1.0\n");	
	fprintf(out,"FLYNNS\n");
	
	for(np=0;np<npoly;np++)
	{
		fprintf(out,"%d %d",np,polys[np][0]);
		
		for(nv=1;nv<=polys[np][0];nv++)
		{
			fprintf(out," %d",polys[np][nv]);
		}
		fprintf(out,"\n");
	}
	
	if (fprintf(out,"LOCATION\n")<0) OnError(ellename,WRITE_ERR);
	
	for(ev=0;ev<nnodes;ev++)
	{		
		fprintf(out,"%d %.8lf %.8lf\n",ev,nodes[ev][0],nodes[ev][1]);
	}

	fprintf(out,"F_ATTRIB_A\nDefault 0.0\n");
	
#if XY
// Unodes written to the file twice ??
	if (fprintf(out,"UNODES\n")<0) OnError(ellename,WRITE_ERR);
	
	for(i=0,cntr=0;i<irows;i++)
	{
		for(j=0;j<icols;j++)
		{
			fprintf(out,"%d %lf %lf\n",cntr++,(j+0.5)/icols,1.0-((i+0.5)/irows));
		}
	}
	if (fprintf(out,"U_CONC_A\n")<0) OnError(ellename,WRITE_ERR);
	
	for(i=0,cntr=0;i<irows;i++)
	{
		for(j=0;j<icols;j++)
		{
			fprintf(out,"%d %ld\n",cntr++,image[i][j]);
		}
	}
#endif


//---------------------------------------------------------------------
// new by Daniel    25.06.2002
//
// now makes triangular lattice for Unodes 
// should be compatible with mike. 
// this routine defines the size of the lattice 
// size is then read by mike. 
//---------------------------------------------------------------------
	
	irows = irows * 1;    // define initial size 
	icols = icols * 1;

	irows = int(irows*(2.0/sqrt(3.0)));   // triangular lattice 

	cntr = 0;                             // counter for number of node 

	cout << irows << endl;       // some output
	cout << icols << endl;

	if (fprintf(out,"UNODES\n")<0) OnError(ellename,WRITE_ERR);
	for(i=0, cntr=0;i<irows;i++)        // y rows
	{
	    //cout << i << endl;
	    for(j=0;j<icols;j++)     // x colums 
	    {
		//cout << j << endl;
		xx = j/(1.0*icols);     // define x 
		yy = (((i+0.5)/(1.0*icols))*(sqrt(3.0)/2.0));   // define y 

		if (i != (2*(i/2)))    // each uneven rows is set radius to right 
		 {
		     xx = xx + (1.0/(2.0*icols));   // checks uneven numbers, this is radius
		 }
		
		// and print to file 

		 fprintf(out,"%d %lf %lf\n",cntr++,xx,yy);
		
	    }
	}

	if (fprintf(out,"U_CONC_A\n")<0) OnError(ellename,WRITE_ERR);
	for(i=0, cntr=0;i<irows;i++)        // y rows
	{
	    for(j=0;j<icols;j++)     // x colums 
	    {
		
		// and print to file 

		 /*fprintf(out,"%d %ld\n",cntr++,image[(int(((irows)-i)*0.866))/4][j/4]);*/
		 fprintf(out,"%d %d\n",cntr++,image[(int)((irows-i)*0.866)][j]);
		
	    }
	}

	fclose(out);
    
}

/*********************************************************

  ConvertDataToElle puts data into Elle 

**********************************************************/

int ConvertDataToElle(int **polys,int npoly,double **nodes,int nnodes,
                      int irows,int icols, int **image,
                      std::vector<int> &rgbcols,
                      std::vector<Coords_3D> &orient,
                      double dim)
{
    int err=0;
	int np,nv,ev;
	int i,j,n,cntr;
    int curr=NO_NB, last=0;
    Coords current;
	Coords xy;     // x and y coordinates for Unodes
    bool orient_data = (orient.size()>0);

    ElleSetSwitchdistance(0.5/irows);
    ElleSetSpeedup(1.0);
    ElleSetUnitLength(dim);

    /*
     * this will be irows*icols in a square grid
     */
    ElleInitUnodes(icols,pattern);
    if (pattern==HEX_GRID) {
        irows = ElleMaxUnodes()/icols;
    }
	
    /*
     * convert polys to flynns
     */
	for(np=0;np<npoly && !err;np++)
	{
        if (curr != np) {
            ElleSetFlynnActive(np);/* fixes np>MaxGrains */
            curr = np;
        }
		
        ElleSetNodeActive(polys[np][1]);
        ElleSetFlynnFirstNode(np,polys[np][1]);

		for(nv=2;nv<=polys[np][0] && !err;nv++)
		{
            ElleSetNodeActive(polys[np][nv]);
            err=ElleSetNeighbour(polys[np][nv],NO_NB,
                                     polys[np][nv-1],&np);
		}
        if (!err) 
            err=ElleSetNeighbour(polys[np][1],NO_NB,
                                 polys[np][polys[np][0]],&np);
	}
	
	for(ev=0;ev<nnodes && !err;ev++)
	{		
        ElleSetNodeActive(ev);
        current.x = nodes[ev][0];
        current.y = nodes[ev][1];
        ElleCopyToPosition(ev,&current);
        ElleCopyToPrevPosition(ev,&current);
        
	}

    if (orient_data) {
	    ElleInitFlynnAttribute(EULER_3);
        ElleInitUnodeAttribute(EULER_3);
    }
    else {
	    ElleInitFlynnAttribute(F_ATTRIB_A);
        ElleInitUnodeAttribute(U_ATTRIB_A);
    }
    long int min=16000000, max=0, range=0;
	for(i=0, cntr=0;i<irows;i++)        // y rows
	{
	    for(j=0;j<icols;j++)     // x colums 
        {
            if ((long)(image[i][j])<min)
               min = (long)(image[i][j]);
            if ((long)(image[i][j])>max)
               max = (long)(image[i][j]);
        }
    }
    if ((range = max-min)<=0) err=RANGE_ERR;

	for(i=0, cntr=0;i<irows && !err;i++)        // y rows
	{
	    for(j=0;j<icols;j++)     // x colums 
        {
#if XY
//LE
// Better to use a square grid of unodes to set the flynn attrib
// and use tidy to create the hexagonal array or 
// after cleaning the unodes, reinit with HEX_GRID, create flynn unode
// lists then set attrib from Flynn attrib  ?


		    xy.x = j/(1.0*icols);     // define x 
		    xy.y = (((i+0.5)/(1.0*icols))*(sqrt(3.0)/2.0));   // define y 

    		if (i != (2*(i/2)))    // each uneven rows is set radius to right 
              xy.x += 1.0/(2.0*icols);   // checks uneven numbers, this is radius
            ElleSetUnodePosition(cntr,&xy);
            err=ElleSetUnodeAttribute(
                                 cntr,
                                 (double)(image[(int)((irows-i)*0.866)][j]),
                                 U_ATTRIB_A);
#endif
            xy.x = (j+0.5)/icols;
            xy.y = (i+0.5)/irows;
            ElleSetUnodePosition(cntr,&xy);
            if (orient_data) {
                Coords_3D xyz;
                xyz.x = xyz.y = xyz.z = 0.0;
                for (n=0;n<orient.size();n++) {
                    if (rgbcols[n*3]==image[irows-1-i][j]) {
                        xyz.x = orient[n].x;
                        xyz.y = orient[n].y;
                        xyz.z = orient[n].z;
                    }
                }
                err=ElleSetUnodeAttribute(
                                 cntr,
                                 xyz.x,xyz.y,xyz.z,
                                 EULER_3);
            }
            else {
                err=ElleSetUnodeAttribute(
                                 cntr,
                                 (double)(image[irows-1-i][j]),
                                 U_ATTRIB_A);
            }
            last = ElleAddUnodeToFlynn(cntr,&xy,last);
            cntr++;
        }
	}
    return(err);
}

