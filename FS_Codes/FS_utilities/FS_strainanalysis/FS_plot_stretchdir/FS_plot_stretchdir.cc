#include "FS_plot_stretchdir.h"
using namespace std;

/*
 * Basic properties of PS file
 */
double dMaxLineLength = 0.01;
double dLineWidth = 0.0075;
double dPSOffset = 0.1;
double dPSScale = 7.0;

int main(int argc, char **argv)
{
    int err=0;
    UserData userdata; 
    
    ElleInit();
    ElleSetOptNames("Spacing","LineLength","MinRatio","MaxRatio","Scale","unused","unused","unused","unused");
    
    ElleUserData(userdata);
    for (int i=0;i<9;i++) 
        userdata[i]=0; // By default, set all user inputs to 0
    userdata[0] = 1; // by default, plot every unode's data
    userdata[1] = dMaxLineLength; // by default 0.01 
    userdata[2] = 1.0; // by default, set min. and max. plotting ratio to ...  
    userdata[3] = 1.0; // ... the same values (always plot with max. line length)
    userdata[4] = dPSScale; // by default, scale = 7 (useful for simple shear)
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    
    ElleSetInitFunction(InitThisProcess);	
    
    if (ElleDisplay()) SetupApp(argc,argv);
    
    StartApp();

    CleanUp();

    return(0);
} 

int InitThisProcess()
{
    char *infile;
    int err=0;
    
    // Clear the data structures
    ElleSetRunFunction(ProcessFunction);
    
    // Read the data
    infile = ElleFile();
    if (strlen(infile)>0) 
        if (err=ElleReadData(infile)) 
            OnError(infile,err);
}

int ProcessFunction()
{
    UserData userdata;
    ElleUserData(userdata);
    int iSpacing = (int)userdata[0];
    if ((int)userdata[1]!=dMaxLineLength) dMaxLineLength = userdata[1];
    if ((int)userdata[4]!=dPSScale) dPSScale = userdata[4];
    double dMinRatio = userdata[2];
    double dMaxRatio = userdata[3];
    double dLineLengthFact = 0.0;
    double dRatio = 0.0;
    
    if (dMaxRatio>dMinRatio)
    {
        // we will make use of scaling line length to ratios
        if (!ElleUnodeAttributeActive(U_ATTRIB_F))
        {
            printf("ERROR: U_ATTRIB_F indicating strain ellipse ratio should ");
            printf("be active.\nTerminating script\n");
            return 1;
        }
    }
        
    /* Prepare*/
    const char *cPSFilePath = ""; // you can indicate where ps file should be stored, "/" at the end!
    char cPSFname[100];
    FILE *fPSout;
    sprintf(cPSFname,"%sStretchDirs.ps",cPSFilePath);
    // Line colors and width
    double dLineColor[3];
    dLineColor[0] = 1; // red
    dLineColor[1] = 0; // green 
    dLineColor[2] = 0; // blue
    Coords cUnodeXY, cPlotXY;
    
    /* Initialise and prepare the ps file*/    
    fPSout=fopen(cPSFname,"w");
    Startps(fPSout,dLineColor,dLineWidth); 
    
    /* Find size of unode array storing the information (must not be NxN)*/    
    int iMaxRows = 0;
    int iMaxCols = 0;
    FindColsRows(&iMaxRows,&iMaxCols);
    
    /* Go through all unodes, find max. stretching direction and plot line */
    int iDim = (int)sqrt((double)ElleMaxUnodes());
    for (int row = iSpacing-1;row<iMaxRows;row+=iSpacing)
    {
        for (int col = iSpacing-1;col<iMaxCols;col+=iSpacing)
        {
            // get real unode id from row and column
            int i = (iMaxCols*row)+col;
            if (i<ElleMaxUnodes())
            {
                // Calculate line length for this unode depending on strain ellipse ratio
                if (dMaxRatio>dMinRatio)
                {
                    ElleGetUnodeAttribute(i,&dRatio,U_ATTRIB_F);
                    dLineLengthFact = (dRatio-dMinRatio)/(dMaxRatio-dMinRatio);
                    
                    if (dLineLengthFact<0.0) dLineLengthFact=0.0;
                    if (dLineLengthFact>1.0) dLineLengthFact=1.0;
                }
                else
                {
                    dLineLengthFact = 1.0;
                }
                
                if (dLineLengthFact>1.0) 
                {
                    printf("Attention: unode %u ",1);
                    printf("dLineLengthFact=%f (>1) ",dLineLengthFact);
                    printf("ratio %f\n",dRatio);
                }
                
                if (dLineLengthFact>0.0) 
                    PlotUnodeData(i,dLineLengthFact*dMaxLineLength,fPSout);
            }
        }
    }
    
    Endps(fPSout);
    
    return 0;
}

void PlotUnodeData(int iID,double dLineLength,FILE *fPSout)
{
    Coords cStartXY,cEndXY; // start/end of the line to plot
    Coords dDiffXY; // difference of unode/point position
    Coords cUnodeXY;
    double dStretchdir = 0.0;
    double dRadius = dLineLength/2.0;
    
    /* Plot stretch direction line for this unode*/    
    ElleGetUnodePosition(iID,&cUnodeXY);
    
    /* Read stretching direction and compute line coords*/
    ElleGetUnodeAttribute(iID,&dStretchdir,U_ATTRIB_E);
    if (dStretchdir>=0.0)
    {
        cStartXY = cUnodeXY;
        cEndXY   = cUnodeXY;
        dDiffXY.x = cos(dStretchdir*DTOR)*dRadius;
        dDiffXY.y = sin(dStretchdir*DTOR)*dRadius;  
        cStartXY.x += dDiffXY.x;  
        cStartXY.y += dDiffXY.y;   
        cEndXY.x   -= dDiffXY.x;
        cEndXY.y   -= dDiffXY.y;     
    }
    else
    {
        dStretchdir += 360.0;     
        cStartXY = cUnodeXY;
        cEndXY   = cUnodeXY;
        dDiffXY.x = cos(dStretchdir*DTOR)*dRadius;
        dDiffXY.y = sin(dStretchdir*DTOR)*dRadius;  
        cStartXY.x += dDiffXY.x;  
        cStartXY.y += dDiffXY.y;   
        cEndXY.x   -= dDiffXY.x;
        cEndXY.y   -= dDiffXY.y;    
    }
    // move to 1st point and draw line to 2nd
    fprintf(fPSout,"newpath\n"); 
    fprintf(fPSout,"%lf %lf moveto\n",dPSOffset+dPSScale*cStartXY.x,
        dPSOffset+dPSScale*cStartXY.y);
    fprintf(fPSout,"%lf %lf lineto\n",dPSOffset+dPSScale*cEndXY.x,
        dPSOffset+dPSScale*cEndXY.y);
    fprintf(fPSout,"stroke\n");
}

void Startps(FILE *psout,double dLineColor[3],double dLineWidth)
{
    /*
     * Set the starting parameters of the ps file such as line width, line color and
     * font size and style. Also the title is set here
     */
    // Before running this code use: psout=fopen(psfile,"w");
    
    fprintf(psout,"%%!PS-Adobe-2.0\n\n");
    fprintf(psout,"0 setlinewidth\n"); // line width
    fprintf(psout,"72 72 scale\n\n");
    fprintf(psout,"/Helvetica findfont\n");
    fprintf(psout,"0.25 scalefont\n");
    fprintf(psout,"setfont\n");

    fprintf(psout,"newpath\n");
    fprintf(psout,"%f %f moveto\n", dPSOffset,dPSOffset+dPSScale+.25); 
    
    fprintf(psout,"%f setlinewidth\n",dLineWidth);
    fprintf(psout,"%f %f %f setrgbcolor\n",dLineColor[0],dLineColor[1],dLineColor[2]);    
}

void Endps(FILE *psout)
{
    /*
     * Close ps file file and draw black bounding box around the drawing taking into
     * account the box size of the initial elle file
     */  
    double dBoxWidth = 0.0, dBoxHeight = 0.0;
    GetBoxInfo(&dBoxWidth,&dBoxHeight);
    
    fprintf(psout,"0 0 0 setrgbcolor\n");
    fprintf(psout,"newpath\n");
		fprintf(psout,"%lf %lf moveto\n",dPSOffset,dPSOffset);
		fprintf(psout,"%lf %lf lineto\n",dPSOffset+dPSScale*dBoxWidth,dPSOffset);
		fprintf(psout,"%lf %lf lineto\n",dPSOffset+dPSScale*dBoxWidth,dPSOffset);
		fprintf(psout,"%lf %lf lineto\n",dPSOffset+dPSScale*dBoxWidth,dPSOffset+dPSScale*dBoxHeight);
		fprintf(psout,"%lf %lf lineto\n",dPSOffset,dPSOffset+dPSScale*dBoxHeight);
		fprintf(psout,"%lf %lf lineto\n",dPSOffset,dPSOffset);
    fprintf(psout,"stroke\n");

    fprintf(psout,"showpage\n");
    fflush(psout);
    fclose(psout);
}

void GetBoxInfo(double *dWidth,double *dHeight)
{
    CellData unitcell;
    ElleCellBBox(&unitcell);
    
    Coords box_xy[4];
    
    box_xy[0].x = unitcell.cellBBox[BASELEFT].x;
    box_xy[0].y = unitcell.cellBBox[BASELEFT].y;
    box_xy[1].x = unitcell.cellBBox[BASERIGHT].x;
    box_xy[1].y = unitcell.cellBBox[BASERIGHT].y;
    box_xy[2].x = unitcell.cellBBox[TOPRIGHT].x;
    box_xy[2].y = unitcell.cellBBox[TOPRIGHT].y;
    box_xy[3].x = unitcell.cellBBox[TOPLEFT].x;
    box_xy[3].y = unitcell.cellBBox[TOPLEFT].y;
    
    //// Calculate with gaussian euqation for polygon areas with n corners:
    //// 2*Area = Σ(i=1 to n)  (y(i)+y(i+1))*(x(i)-x(i+1))
    //// if i+1>n use i=1 again (or here in C++ i=0)
    //int i2 = 0;
    //for (int i=0;i<4;i++)    
    //{
        //i2 = fmod(i,4)+1;
        //*dArea += ( (box_xy[i].y+box_xy[i2].y)*(box_xy[i].x-box_xy[i2].x) )/2;   
    //}
    
    *dWidth  = box_xy[1].x-box_xy[0].x;
    *dHeight = box_xy[3].y-box_xy[0].y;
    //*dSSOffset = ElleSSOffset();
}

void FindColsRows(int *iMaxRows,int *iMaxCols)
{    
    double dBoxWidth = 0.0, dBoxHeight = 0.0, dSpacing = 0.0;
    GetBoxInfo(&dBoxWidth,&dBoxHeight);
    
    if (dBoxHeight!=dBoxWidth)
    {
 
    // Find the spacing of unodes in y-direction, this will indicate the
    // number of unodes in rows and columns:
    // Do this by finding the unode that has the lowest y-coordinate (apart
    // from the unode with id=0) AND the same x-coordinate than the unode 
    // with id=0
    
    /* FIND SPACING in Y: Will be spacing in X if we used 
     * FS_plot_strainanalysis to create this ellefile*/
    Coords cXY1,cXY2,cXYtemp;
    double dMin = 1e3;
    ElleGetUnodePosition(0,&cXY1);
    for (int i=1;i<ElleMaxUnodes();i++)
    {
        ElleGetUnodePosition(i,&cXYtemp);
        if (cXYtemp.x==cXY1.x)
        {
            if (cXYtemp.y<dMin)
            {
                dMin = cXYtemp.y;
                cXY2=cXYtemp;
            }
        }
    }
    ElleCoordsPlotXY(&cXY2,&cXY1);
    dSpacing = fabs(cXY2.y-cXY1.y);
        
    *iMaxRows = (int)ceil(dBoxHeight/dSpacing);
    *iMaxCols = (int)ceil(dBoxWidth/dSpacing);
    // using ceil not to miss any points...when calling unodes, we need to make
    // sure that the id is not > ElleMaxUnodes()
    }
    else
    {
        // 1x1 box:
        *iMaxRows = (int)sqrt((double)ElleMaxUnodes());
        *iMaxCols = *iMaxRows;
    }
    
}

