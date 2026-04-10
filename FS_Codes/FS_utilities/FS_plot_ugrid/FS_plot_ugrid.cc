#include "FS_plot_ugrid.h"
using namespace std;

/*
 * Some global variables
 * --> Will mostly be defined in the beginning of "Ugrid" function
 */
int iDim; // dimension of unodes (128,256 etc)
double offset = 0;
double scale = 0;
double thresh = 0;
char *infile;
char fileroot[100]; // to store the elle filename without ".elle" at the end
char psfile[100];
double **UnodesX = 0;
double **UnodesY = 0;
double dNewLineColour[3];

//double dScaleUnodes = 1; // To fir box on ps file, needs adjustment for different simulation boxes

int main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    /*
     * Initialise and set option names
     */
    ElleInit();
    ElleSetOptNames("Plottype","GridSpacing","Scale","ThresholdDist","ExcludePhase","unused","unused","unused","unused");
    /* 
     * Give default values for userdata:
     */
    ElleUserData(userdata);
    userdata[uPlottype]=0; // Plot horizontal (0), vertical (1) or diagonal (2) lines. Default: 0
    userdata[uGridSpacing]=1; // Spacing between the lines, default 1
    userdata[uScale]=7; // Scale to fit box on ps file, 7 is suitable for simple shear... needs proper adjustment when chaning pure shear / simple shear conditions
    userdata[uThresholdDist]=0.15; // threshold distance: two unodes will only be connected with grid line, if distance is smaller than this value
    userdata[uExcludePhase]=0; // ID of the phase that should be excluded Default: 0 (no exclusion)
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    /*
     * Set the function to the one in your process file
     */
    ElleSetInitFunction(InitUgrid);	
    
    /*
     * Set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);
    /*
     * Run your initialisation function and start the application
     */
    StartApp();

    CleanUp();

    return(0);
} 


/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitUgrid()
{
    char *infile;
    int err=0;
    /*
     * Clear the data structures
     */
    ElleSetRunFunction(Ugrid);
    /*
     * Read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) 
    {
        if (err=ElleReadData(infile)) OnError(infile,err);
    }
    /*
     * Check if unodes are in the file
     */
    if(!ElleUnodesActive()) 
    {
        printf("ERROR: No unodes in file\n\n");  
        return 0;
    } 

     
}

int Ugrid()
{
    UserData userdata;
    ElleUserData(userdata);
    
    // Update model box, width/height ratio should stay the same
    //UpdateCell();
        
    /*Prepare the box information*/
    CellData unitcell;
    ElleCellBBox(&unitcell);
    
    double dBoxHeight = unitcell.cellBBox[TOPLEFT].y-unitcell.cellBBox[BASELEFT].y;
    double dBoxWidth  = unitcell.cellBBox[BASERIGHT].x-unitcell.cellBBox[BASELEFT].x;
 
    // Preparations for non-square boxes (pure shear)    
    double dBoxRatio = dBoxWidth/dBoxHeight;
    if (dBoxRatio==1) dBoxRatio=0;
    
    /*For input and output files */
    infile = ElleFile(); // infile is declared at the beginning of the code
    
    /* Remove the last 5 characters and hence the ".elle" from the filename:
     * If elle file was called "myfile.elle", the ps file will be called
     * "myfile_ugrid.ps"*/
    for (int i=0;i<strlen(infile)-5;i++) 
        fileroot[i] = infile[i];
    sprintf(psfile,"%s_ugrid.ps",fileroot); 
    //printf("%s\n",psfile);
    FILE *psout;
    
    /* Spacing and orientation of the lines*/
    int iGridSpacing = (int)userdata[uGridSpacing];
    int iPlottype    = (int)userdata[uPlottype]; 
    scale = userdata[uScale];
    
    /* Plotting parameters (declared at the beginning of the code)*/
    //scale  = 3.8; // needs adjustment to pure shear / simple shear simulations
    offset = 0.1; // was 0.5
    thresh = userdata[uThresholdDist];//0.15;//0.06;
    
    
    /* To exclude a phase (lines will not be drawn)*/
    int iPhaseAttrib = VISCOSITY;
    double dPhase = 0.0; // temporarily storing phase
    int iExcludePhase = (int)userdata[uExcludePhase]; 
    
    /*Prepare unodes and bring them in unit cell*/
    Coords unodexy;
    Coords cTempunodexy;
    
    for (int i=0;i<ElleMaxUnodes();i++)
    {
        /* Also prepare something else: 
         * Check if unode finite position is stored in 
         * unode strain attribute and (if yes) transfer it to unode position
         */
        /* Optional: Scale the unode coordinates up or down to enable better
         * visibility of the image
         */
        if (ElleUnodeAttributeActive(U_FINITE_STRAIN))
        {
            ElleGetUnodeAttribute(i,CURR_S_X,&unodexy.x);  
            ElleGetUnodeAttribute(i,CURR_S_Y,&unodexy.y);  
            //unodexy.x *= dScaleUnodes;
            //unodexy.y *= dScaleUnodes;
            ElleNodeUnitXY(&unodexy);   
            ElleSetUnodePosition(i,&unodexy); 
        }
        else
        {
            ElleGetUnodePosition(i,&unodexy);  
            //unodexy.x *= dScaleUnodes;
            //unodexy.y *= dScaleUnodes;
            ElleNodeUnitXY(&unodexy);
            ElleSetUnodePosition(i,&unodexy);
        }     
        ElleSetUnodePosition(i,&unodexy); 
    }
    
    //if (dScaleUnodes!=1)  
        //printf("Scaling unodes up or down by a factor of %f\n",dScaleUnodes);    
    
    /* Load unode position information and allocate array memory to store all XY 
     * Positions*/
    iDim = sqrt(ElleMaxUnodes());
    UnodesX = dmatrix(0,iDim-1,0,iDim-1);
    UnodesY = dmatrix(0,iDim-1,0,iDim-1);    
    for(int row=0;row<iDim;row++)
    {
        for(int col=0;col<iDim;col++)
        {    		
            ElleGetUnodePosition(UnodeRowCol2ID(col,row), &cTempunodexy);
            UnodesX[col][row] = cTempunodexy.x;
            UnodesY[col][row] = cTempunodexy.y;
        }
    }
    
    /* Initialise and prepare the ps file and set line color*/
    double dLineColor[3];
    dLineColor[0] = 1; // red
    dLineColor[1] = 0; // green 
    dLineColor[2] = 0; // blue
    double dLineWidth = 0.0075;// was: 0.0075;
    psout=fopen(psfile,"w");
    Startps(psout,dLineColor,dLineWidth);    
    
    /*
     * PERFORM THE PLOTTINGS
     */
     
    if (iExcludePhase!=0) printf("Excluding phase %u from drawing\n",iExcludePhase);
     
    switch (iPlottype)
    {
        case 0:
            PlotHorizontalLines(psout,iGridSpacing,iExcludePhase);
            break;
        case 1:
            PlotVerticalLines(psout,iGridSpacing,iExcludePhase);
            break;
        case 2:
            PlotDiagonalLines(psout,iGridSpacing,iExcludePhase);
            break;
        case 3:
            fprintf(psout,"%f %f %f setrgbcolor\n",0.0,0.0,1.0); 
            PlotHorizontalLines(psout,iGridSpacing,iExcludePhase);
            fprintf(psout,"%f %f %f setrgbcolor\n",1.0,0.0,0.0); 
            PlotVerticalLines(psout,iGridSpacing,iExcludePhase);
            break;
    }
    
    
    /* Finish the ps file*/
    Endps(psout,dBoxWidth,dBoxHeight);
    
    return 0;
}

/*
 * Plot grid with initially horizontal parallel lines
 */
void PlotHorizontalLines(FILE *psout,int iGridSpacing,int iExcludePhase)
{
    int col, row; // used in the loops
    
    int idCols = 1; // step in cols (1 for horizontal lines), called "di" before
    int idRows = 0; // step in rows (0 for horizontal lines --> vertical component=0 ), called "dj" before
    
    int iColUnode = 0;
    int iRowUnode = 0;
    Coords cUnodeXY;
    int iColNextUnode = 0;
    int iRowNextUnode = 0;
    Coords cNextUnodeXY;
    
    for (int j=0; j<iDim; j += iGridSpacing) 
    { 
        row = j;
        for(int i=0; i<iDim; i++) 
        {
            col = i;
            /* For the first unode in the line: Start new line and move to the 
             * starting position*/
            if(i==0)
            {
                fprintf(psout,"newpath\n");
                iColUnode = (int)(fmod(col+(idCols*row),iDim));
                iRowUnode = row;
                cUnodeXY.x = UnodesX[iColUnode][iRowUnode];
                cUnodeXY.y = UnodesY[iColUnode][iRowUnode];
                fprintf(psout,"%lf %lf moveto\n",offset+scale*
                    cUnodeXY.x,offset+scale*cUnodeXY.y);
            }
                        
            /* For the remaining unodes in the line: Check if the distance to
             * the next unode is not too large (which would mean it is on the 
             * other side of the box) and draw a line
             */             
            iColUnode = (int)(fmod(col+(idCols*row),iDim));
            iRowUnode = row;
                cUnodeXY.x = UnodesX[iColUnode][iRowUnode];
                cUnodeXY.y = UnodesY[iColUnode][iRowUnode];
            iColNextUnode = (int)fmod(col+(idCols*row)+idCols,iDim);
            iRowNextUnode = (int)fmod(row+idRows,iDim);
                cNextUnodeXY.x = UnodesX[iColNextUnode][iRowNextUnode];
                cNextUnodeXY.y = UnodesY[iColNextUnode][iRowNextUnode];
            
            DrawLine(psout,cUnodeXY,cNextUnodeXY,iColNextUnode,iRowNextUnode,iExcludePhase);
        }
        fprintf(psout,"stroke\n");
    }

}

/*
 * Plot grid with initially vertical parallel lines
 */
void PlotVerticalLines(FILE *psout,int iGridSpacing,int iExcludePhase)
{
    int col, row; // used in the loops
    
    int idCols = 0; // step in cols (1 for horizontal lines), called "di" before
    int idRows = 1; // step in rows (0 for horizontal lines --> vertical component=0 ), called "dj" before
    
    int iColUnode = 0;
    int iRowUnode = 0;
    Coords cUnodeXY;
    int iColNextUnode = 0;
    int iRowNextUnode = 0;
    Coords cNextUnodeXY;
    
    for (int j=0; j<iDim; j += iGridSpacing) 
    { 
        col = j;
        for(int i=0; i<iDim; i++) 
        {
            row = i;
            /* For the first unode in the line: Start new line and move to the 
             * starting position*/
            if(i==0)
            {
                fprintf(psout,"newpath\n");
                iColUnode = (int)(fmod(col+(idCols*row),iDim));
                iRowUnode = row;
                cUnodeXY.x = UnodesX[iColUnode][iRowUnode];
                cUnodeXY.y = UnodesY[iColUnode][iRowUnode];
                fprintf(psout,"%lf %lf moveto\n",offset+scale*
                    cUnodeXY.x,offset+scale*cUnodeXY.y);
            }
                        
            /* For the remaining unodes in the line: Check if the distance to
             * the next unode is not too large (which would mean it is on the 
             * other side of the box) and draw a line
             */             
            iColUnode = (int)(fmod(col+(idCols*row),iDim));
            iRowUnode = row;
                cUnodeXY.x = UnodesX[iColUnode][iRowUnode];
                cUnodeXY.y = UnodesY[iColUnode][iRowUnode];
            iColNextUnode = (int)fmod(col+(idCols*row)+idCols,iDim);
            iRowNextUnode = (int)fmod(row+idRows,iDim);
                cNextUnodeXY.x = UnodesX[iColNextUnode][iRowNextUnode];
                cNextUnodeXY.y = UnodesY[iColNextUnode][iRowNextUnode];
            
            /* To have the option to switch line colors we could define a new
             * colour here depending on unode colum*/
            //double dNewLineColour[3]; --> WAS GLOBALLY DEFINED
            //double val=128.0;
            //if (col < (int)val)
            //{
                //dNewLineColour[0] = (val-(double)col)/val;   
                //dNewLineColour[1] = 0;   
                //dNewLineColour[2] = (double)col/val;                
            //}
            //else
            //{
                //dNewLineColour[0] = 2.0-((double)col/val);
                //dNewLineColour[1] = 0;   
                //dNewLineColour[2] = ((double)col/val)-1.0;  
            //}
        
            //fprintf(psout,"%f %f %f setrgbcolor\n",dNewLineColour[0],dNewLineColour[1],dNewLineColour[2]); 
            
            /* Return to the regular code: */
            DrawLine(psout,cUnodeXY,cNextUnodeXY,iColNextUnode,iRowNextUnode,iExcludePhase);
        }
        fprintf(psout,"stroke\n");
    }

}

/*
 * Plot grid with initially diagonal parallel lines
 */
void PlotDiagonalLines(FILE *psout,int iGridSpacing,int iExcludePhase)
{
    int col, row; // used in the loops
    
    int idCols = 1; // step in cols (1 for horizontal lines), called "di" before
    int idRows = 1; // step in rows (0 for horizontal lines --> vertical component=0 ), called "dj" before
    
    int iColUnode = 0;
    int iRowUnode = 0;
    Coords cUnodeXY;
    int iColNextUnode = 0;
    int iRowNextUnode = 0;
    Coords cNextUnodeXY;
    
    for (int j=0; j<iDim; j += iGridSpacing) 
    { 
        col = j;
        for(int i=0; i<iDim; i++) 
        {
            row = i;
            /* For the first unode in the line: Start new line and move to the 
             * starting position*/
            if(i==0)
            {
                fprintf(psout,"newpath\n");
                iColUnode = (int)(fmod(col+(idCols*row),iDim));
                iRowUnode = row;
                cUnodeXY.x = UnodesX[iColUnode][iRowUnode];
                cUnodeXY.y = UnodesY[iColUnode][iRowUnode];
                fprintf(psout,"%lf %lf moveto\n",offset+scale*
                    cUnodeXY.x,offset+scale*cUnodeXY.y);
            }
                        
            /* For the remaining unodes in the line: Check if the distance to
             * the next unode is not too large (which would mean it is on the 
             * other side of the box) and draw a line
             */             
            iColUnode = (int)(fmod(col+(idCols*row),iDim));
            iRowUnode = row;
                cUnodeXY.x = UnodesX[iColUnode][iRowUnode];
                cUnodeXY.y = UnodesY[iColUnode][iRowUnode];
            iColNextUnode = (int)fmod(col+(idCols*row)+idCols,iDim);
            iRowNextUnode = (int)fmod(row+idRows,iDim);
                cNextUnodeXY.x = UnodesX[iColNextUnode][iRowNextUnode];
                cNextUnodeXY.y = UnodesY[iColNextUnode][iRowNextUnode];
            
            DrawLine(psout,cUnodeXY,cNextUnodeXY,iColNextUnode,iRowNextUnode,iExcludePhase);
        }
        fprintf(psout,"stroke\n");
    }

}

/*
 * Drawing a line from one unode to the next unode if the spacing is not too 
 * large (i.e. <= thresh) and the phase is not excluded
 */
void DrawLine(FILE *psout,Coords cUnodeXY,Coords cNextUnodeXY,int iColNextUnode,int iRowNextUnode, int iExcludePhase)
{
    int iUnodeID = UnodeRowCol2ID(iColNextUnode,iRowNextUnode);
    double dPhase = 0;
    int iDrawLine = 1; // stays 1 if the line should be drawn, 0 if it should not be drawn because unode is in a flynn of the excluded phase
    
    if (CheckUnodeDist(cUnodeXY,cNextUnodeXY))            
    {
        fprintf(psout,"stroke\n");
        fprintf(psout,"newpath\n");                   
        fprintf(psout,"%lf %lf moveto\n",offset+scale*cNextUnodeXY.x,
            offset+scale*cNextUnodeXY.y);
    }
    else
    {
        /*Draw the line instead of moving on to the next unode*/
        
        /*Make sure not to draw a line where F_VISCOSITY == excluded 
         * phase*/
        
        /* Find the flynn the unode sits in to determine the phase at 
         * this position. It wouldn't be correct to just use the unode
         * phase since it is deformed and not in original flynn any more
         */                
        if (ElleFlynnAttributeActive(VISCOSITY) && iExcludePhase!=0)
        {
            int iFlynnID = 0;
          
            // Find the flynn the unode sits in:
            for (iFlynnID=0;iFlynnID<ElleMaxFlynns();iFlynnID++)
            {
                if (ElleFlynnIsActive(iFlynnID) && 
                    EllePtInRegion(iFlynnID,&cNextUnodeXY))
                break;                            
            }                      
          
            ElleGetFlynnRealAttribute(iFlynnID,&dPhase,VISCOSITY);
            if ((int)dPhase==iExcludePhase) iDrawLine=0;
        }
        
        if (iDrawLine==0)
        {
            fprintf(psout,"%lf %lf moveto\n",offset+scale*
                cNextUnodeXY.x,offset+scale*cNextUnodeXY.y);
        }
        else
        {
            fprintf(psout,"%lf %lf lineto\n",offset+scale*
                cNextUnodeXY.x,offset+scale*cNextUnodeXY.y);
        }
    }
}

/*
 * Check if the distance between two unodes is too high to draw a line between
 * them (i.e. if they are on different sides of the box)
 */
int CheckUnodeDist(Coords cUnodeXY,Coords cNextUnodeXY)
{    
    double dUnodeDist = 
        sqrt( pow(cUnodeXY.x-cNextUnodeXY.x,2) + 
              pow(cUnodeXY.y-cNextUnodeXY.y,2)   );
              
    // Return 1 if the distance is too high and 0 if it is below the threshold
    if (dUnodeDist>thresh)
        return (1);
    else
        return (0);
}

/*
 * Set the starting parameters of the ps file such as line width, line color and
 * font size and style. Also the title is set here
 */
void Startps(FILE *psout,double dLineColor[3],double dLineWidth)
{
    //psout=fopen(psfile,"w");
    
    fprintf(psout,"%%!PS-Adobe-2.0\n\n");
    fprintf(psout,"0 setlinewidth\n"); // line width
    fprintf(psout,"72 72 scale\n\n");
    fprintf(psout,"/Helvetica findfont\n");
    fprintf(psout,"0.25 scalefont\n");
    fprintf(psout,"setfont\n");

    fprintf(psout,"newpath\n");
    fprintf(psout,"%lf %lf moveto\n", offset,offset+scale+.25); 
    //fprintf(psout,"(Passive marker grid for file \"%s\") show\n",infile);
    
    fprintf(psout,"%f setlinewidth\n",dLineWidth);
    fprintf(psout,"%f %f %f setrgbcolor\n",dLineColor[0],dLineColor[1],dLineColor[2]);    
}

/*
 * Close ps file file and draw black bounding box around the drawing taking into
 * account the box size of the initial elle file
 */
void Endps(FILE *psout, double dBoxWidth,double dBoxHeight)
{   
    double dScaleUnodes = 1.0; // dummy value to account for a variable that is not used any more
    fprintf(psout,"0 0 0 setrgbcolor\n");
    fprintf(psout,"newpath\n");
		fprintf(psout,"%lf %lf moveto\n",offset,offset);
		fprintf(psout,"%lf %lf lineto\n",offset+scale*dBoxWidth*dScaleUnodes,offset);
		fprintf(psout,"%lf %lf lineto\n",offset+scale*dBoxWidth*dScaleUnodes,offset);
		fprintf(psout,"%lf %lf lineto\n",offset+scale*dBoxWidth*dScaleUnodes,offset+scale*dBoxHeight*dScaleUnodes);
		fprintf(psout,"%lf %lf lineto\n",offset,offset+scale*dBoxHeight*dScaleUnodes);
		fprintf(psout,"%lf %lf lineto\n",offset,offset);
    fprintf(psout,"stroke\n");

    fprintf(psout,"showpage\n");
    fflush(psout);
    fclose(psout);
}

/*
 * Function outputs unode ID from row and column for a given resolution 
 */
int UnodeRowCol2ID(int iCol,int iRow)
{
    int iUnodeID = 0;
    
    iUnodeID = (iDim * iRow)+iCol;
    
    return iUnodeID;
}

/*
 * Function outputs unode row and column from unode ID for a given resolution 
 */
void UnodeID2RowCol(int iUnodeID, int *iCol, int *iRow)
{    
    *iCol = fmod(iUnodeID,iDim);
    *iRow = (iUnodeID-*iCol)/iDim;
}

void UpdateCell()
{
    double dScaleUnodes = 1.0; // dummy value to account for a variable that is not used any more
    // Upate unit cell:
    CellData unitcell;
    ElleCellBBox(&unitcell);
    Coords  corners[4];
    
    corners[BASELEFT].x = unitcell.cellBBox[BASELEFT].x*dScaleUnodes;
    corners[BASELEFT].y = unitcell.cellBBox[BASELEFT].y*dScaleUnodes;
    corners[BASERIGHT].x = unitcell.cellBBox[BASERIGHT].x*dScaleUnodes;
    corners[BASERIGHT].y = unitcell.cellBBox[BASERIGHT].y*dScaleUnodes;
    corners[TOPRIGHT].x = unitcell.cellBBox[TOPRIGHT].x*dScaleUnodes;
    corners[TOPRIGHT].y = unitcell.cellBBox[TOPRIGHT].y*dScaleUnodes;
    corners[TOPLEFT].x = unitcell.cellBBox[TOPLEFT].x*dScaleUnodes;
    corners[TOPLEFT].y = unitcell.cellBBox[TOPRIGHT].y*dScaleUnodes;
    
    ElleSetCellBBox(&corners[BASELEFT], &corners[BASERIGHT],&corners[TOPRIGHT], &corners[TOPLEFT]);
}
