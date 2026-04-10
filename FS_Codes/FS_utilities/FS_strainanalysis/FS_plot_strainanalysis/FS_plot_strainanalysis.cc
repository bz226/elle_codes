#include "FS_plot_strainanalysis.h"
using namespace std;

/* To handle plot points */
vector<PlotPoints> vAllPlotPoints;
vector<PlotPoints> vDataArray;

int main(int argc, char **argv)
{
    int err=0;
    UserData userdata; 
    
    ElleInit();
    ElleSetOptNames("dUnodeResolution","WriteTxtFile","unused","unused","unused","unused","unused","unused","unused");
    
    ElleUserData(userdata);
    for (int i=0;i<9;i++) 
        userdata[i]=0; // By default, set all user inputs to 0
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    
    ElleSetSaveFrequency(1);
    ElleSetStages(1);
    
    ElleSetInitFunction(InitThisProcess);	
    char cFileroot[] = "FS_plot_strainanalysis.elle";
    ElleSetSaveFileRoot(cFileroot);
    
    if (ElleDisplay()) SetupApp(argc,argv);
    
    StartApp();

    CleanUp();

    return(0);
} 

/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
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

/* 
 * Anything can now be in the ProcessFunction itself:
 */ 
int ProcessFunction()
{
    UserData userdata;             
    ElleUserData(userdata); 
    int iPointDim = (int)userdata[0];
    int iWriteTxt = (int)userdata[1];
    // Set resolution to default (unode dimension):
    if (iPointDim == 0) iPointDim = (int)sqrt(ElleMaxUnodes()); 
    
    printf("Preparing plotting data, please wait ...\n");
    
    /* Check for errors in input data (files exists etc...)*/
    if (Check())
    {
        printf("Script terminated\n");
        return (1);
    }
    
    /* Go through all unodes and calculate strain properties (vorticity, 
     * stretching direction etc...)*/
    for (int i=0;i<ElleMaxUnodes();i++)
    {
        /* 
         * Shift finite unode position to real unode position and store
         * properties in U_FINITE_STRAIN (apart from dilation, which is 
         * stored in U_VISCOSITY):
         */  
        GetUnodeStrainData(i); 
    }
    
    /* Set "plot point" or pixel positions defined by user input. Every pixel 
     * will store the nearest passive marker properties (is this the best 
     * approach??)*/
    PreparePointGrid(iPointDim);
    
    /* Go through all points in "points array", find nearest unode and assign
     * its properties to point */
    int iPointsMax = vAllPlotPoints.size();
    for (int i=0;i<iPointsMax;i++)
    {
        PlotPoints pThisPoint;
        pThisPoint = vAllPlotPoints[0];
        
        // Erase this element, it is added later to data array with the strain 
        // properties:
        vAllPlotPoints.erase(vAllPlotPoints.begin()); 
        
        AssignNearestUnodeProps(&pThisPoint);     
        
        // Add this element (this point) to data array. It now stores the 
        // properties:
        vDataArray.push_back(pThisPoint);
    }
    
    if(iWriteTxt!=0) 
    {
        /* Explanation: Write point data to space delimited textfile. The 
         * textfile will have as many rows and columns as the plot point array*/
        WritePlotPointArray(iWriteTxt);
    }
    
    // Reset unode positions to plot point positions and, from this, write ellefile
    ResetUnodes();
    
    //ElleWriteData("out.elle");
    vAllPlotPoints.clear();
    vDataArray.clear();
    
    
    printf("... Finished!\n");
    
    return 0;
}

int PreparePointGrid(int iRes)
{
    /* Create array of points with equal spacing (regular grid) to plot strain
     * properties on.
     * 
     * Minimum number of points in either x or y direction will be iRes. In case
     * of a non-square box, the number of points is accordingly increased in 
     * direction of the long box edge
     */
    double dBoxWidth = 0.0, dBoxHeight = 0.0, dSpacing = 0.0;
    GetBoxInfo(&dBoxWidth,&dBoxHeight);
    
    if (dBoxHeight>=dBoxWidth)
        dSpacing = dBoxWidth/(double)iRes;
    else
        dSpacing = dBoxHeight/(double)iRes; 
    
    iMaxRows = (int)floor(dBoxHeight/dSpacing);
    iMaxCols = (int)floor(dBoxWidth/dSpacing);
    
    printf("  ... creating %u x %u array of plot points\n",iMaxRows,iMaxCols);
    
    // Create all plot points:
    PlotPoints pTempPoint;
    for (int row=0;row<iMaxRows;row++)
    {
        for (int col=0;col<iMaxCols;col++)
        {
            pTempPoint.x = (double)col*dSpacing; 
            pTempPoint.y = (double)row*dSpacing;         
            vAllPlotPoints.push_back(pTempPoint);
        }
    }
    
    
    return 0;
}

int Check()
{
    if (!ElleUnodesActive)
    {
        printf("ERROR: No unodes in file\n");
        return 1;
    }
    
    // Check if file with pos. gradient data exists:
    if (!ElleUnodeAttributeActive(U_ATTRIB_A))
    {
        printf("ERROR: U_ATTRIB_A is missing\n");
        return 1;
    }
    if (!ElleUnodeAttributeActive(U_ATTRIB_B))
    {
        printf("ERROR: U_ATTRIB_B is missing\n");
        return 1;
    }
    if (!ElleUnodeAttributeActive(U_ATTRIB_C))
    {
        printf("ERROR: U_ATTRIB_C is missing\n");
        return 1;
    }
    if (!ElleUnodeAttributeActive(U_ATTRIB_D))
    {
        printf("ERROR: U_ATTRIB_D is missing\n");
        return 1;
    }
    if (!ElleUnodeAttributeActive(U_ATTRIB_E))
    {
        printf("ERROR: U_ATTRIB_E is missing\n");
        return 1;
    }
    if (!ElleUnodeAttributeActive(U_ATTRIB_F))
    {
        printf("ERROR: U_ATTRIB_F is missing\n");
        return 1;
    }
    
    if (!ElleUnodeAttributeActive(U_VISCOSITY))
        ElleInitUnodeAttribute(U_VISCOSITY); // will store dilation    
    
    if (!ElleUnodeAttributeActive(U_FINITE_STRAIN))
    {
        ElleInitUnodeAttribute(U_FINITE_STRAIN);
        Coords cXY;
        for (int i=0;i<ElleMaxUnodes();i++)
        {            
            ElleGetUnodePosition(i,&cXY);
            ElleSetUnodeAttribute(i,cXY.x,CURR_S_X);
            ElleSetUnodeAttribute(i,cXY.y,CURR_S_Y);
        }
        printf("WARNING: No unode positions found in U_FINITE_STRAIN:\n");
        printf("         Using regular unode positions\n");
    }
    
    return 0;
}


void GetUnodeStrainData(int iUnode)
{
    /* Calculate:
     * Ratio of strain ellipse 
     * Max. and min. strain ("length" of strain ellipse axes) 
     * Max. stretching direction as angle from positive x-axis (math. system) 
     * Vorticity and vorticity number 
     * Dilation (should be one if area is preserved) 
     * 
     * Variables for those properties is in header file of this script:
     * double dVorticity = 0.0, dVorticityNumber = 0.0;
     * double dRatio = 0.0, dEmax = 0.0, dEmin = 0.0, dStretchDir = 0.0;
     * double dDilation = 0.0;
     */
     
     
     /* 
      * First: Shift finite unode position from CURR_S_X and _Y to actual unode 
      * position (U_FINITE_STRAIN will be overwritten by strain properties
      */
    double dTmp[2];
    Coords cUnodeXY;
    dTmp[0]=dTmp[1]=0.0;
    ElleGetUnodeAttribute(iUnode,&dTmp[0],CURR_S_X);
    ElleGetUnodeAttribute(iUnode,&dTmp[1],CURR_S_Y);
    cUnodeXY.x = dTmp[0];
    cUnodeXY.y = dTmp[1];
    ElleNodeUnitXY(&cUnodeXY);
    ElleSetUnodePosition(iUnode,&cUnodeXY);  
     
    double F11,F12,F21,F22;
    Coords cMohrCentre;
    double dMohrRadius;
    
    /* Load tensor elements*/
    ElleGetUnodeAttribute(iUnode,&F11,U_ATTRIB_A);    
    ElleGetUnodeAttribute(iUnode,&F12,U_ATTRIB_B);
    ElleGetUnodeAttribute(iUnode,&F21,U_ATTRIB_C);
    ElleGetUnodeAttribute(iUnode,&F22,U_ATTRIB_D);
    
    /* Some calculations, considered Mohr circle for strain to derive these 
     * equations */
    cMohrCentre.x = (F22+F11)/2.0;
    cMohrCentre.y = (F12-F21)/2.0;
    dMohrRadius = sqrt( pow(F22-F11,2) + pow(F12+F21,2)) / 2.0;
    
    dEmax = sqrt(pow(cMohrCentre.x,2)+pow(cMohrCentre.y,2)) + dMohrRadius;
    dEmin = sqrt(pow(cMohrCentre.x,2)+pow(cMohrCentre.y,2)) - dMohrRadius;
    dRatio = dEmax/dEmin;
    dVorticity = cMohrCentre.y;
    dVorticityNumber = cMohrCentre.y / dMohrRadius;
    dDilation = dEmax*dEmin;
    
    // for stretching direction:
    double dBeta = 0.0, dAlpha = 0.0;
    if (cMohrCentre.x==0.0)
    {
        dBeta = 90.0*DTOR;
    }
    else
    {
        if (cMohrCentre.y==0.0)
        {
            dBeta = 0.0;
        }
        else
        {
            // no special case:
            dBeta = atan(cMohrCentre.y/cMohrCentre.x);
        }
    }
        
    double x1,y1,x2,y2; // temporary stuff
    x1 = F11-cMohrCentre.x;
    y1 = (-F21)-cMohrCentre.y;
    x2 = cos(dBeta)*dEmax;
    y2 = sin(dBeta)*dEmax;
    
    dAlpha = (x1*x2)+(y1*y2);
    dAlpha /= sqrt( (x1*x1)+(y1*y1) ) * sqrt( (x2*x2)+(y2*y2) );
    dAlpha = acos(dAlpha);
    dAlpha /= 2.0; // remember double angles in Mohr circle!
    
    dStretchDir = dAlpha-dBeta; // now in radians!!  
    dStretchDir *= RTOD; // now in degree from positive x-axis (mathematical convention)
    
    /* Store in unode properties (only temporarily in U_FINITE_STRAIN) */
    ElleSetUnodeAttribute(iUnode,dVorticity,START_S_X);
    ElleSetUnodeAttribute(iUnode,dVorticityNumber,START_S_Y);
    ElleSetUnodeAttribute(iUnode,dEmin,PREV_S_X);
    ElleSetUnodeAttribute(iUnode,dEmax,PREV_S_Y);
    ElleSetUnodeAttribute(iUnode,dStretchDir,CURR_S_X);
    ElleSetUnodeAttribute(iUnode,dRatio,CURR_S_Y);
    ElleSetUnodeAttribute(iUnode,dDilation,U_VISCOSITY);
}

void AssignNearestUnodeProps(PlotPoints *pPoint)
{
    /* This function searches for the nearest unode for the given point from
     * the point array that will later form s.th. like the pixels for a 
     * plotted image.
     * The point properties (vorticity,emin,emax,R,etc.) are set to the nearest
     * unode properties
     */
    Coords cPointXY,cUnodeXY;
    double dMinSep = 1e3, dTmpSep = 0.0; // point separations
    int iNearestUnodeID = 0;
    cPointXY.x = pPoint->x;
    cPointXY.y = pPoint->y;
    for (int unode=0;unode<ElleMaxUnodes();unode++)
    {
        ElleGetUnodePosition(unode,&cUnodeXY);
        ElleNodeUnitXY(&cUnodeXY);
        ElleCoordsPlotXY(&cUnodeXY,&cPointXY);
        dTmpSep = pointSeparation(&cUnodeXY,&cPointXY);
        
        if (dTmpSep<dMinSep)
        {
            iNearestUnodeID = unode;
            dMinSep = dTmpSep;
        }
    }
    
    /* Assign unode properties to point */
    ElleGetUnodeAttribute(iNearestUnodeID,&pPoint->vorticity,START_S_X);
    ElleGetUnodeAttribute(iNearestUnodeID,&pPoint->vorticity_number,START_S_Y);
    ElleGetUnodeAttribute(iNearestUnodeID,&pPoint->emin,PREV_S_X);
    ElleGetUnodeAttribute(iNearestUnodeID,&pPoint->emax,PREV_S_Y);
    ElleGetUnodeAttribute(iNearestUnodeID,&pPoint->stretch_dir,CURR_S_X);
    ElleGetUnodeAttribute(iNearestUnodeID,&pPoint->ratio_maxmin,CURR_S_Y);
    ElleGetUnodeAttribute(iNearestUnodeID,&pPoint->dilation,U_VISCOSITY);
}

void WritePlotPointArray(int iPropertyID)
{
    /* 
     * Writing all data from plot point array in sapce separated textfile. Just
     * the NxN array will be written in the textfile 
     * The input int will indicate which property to use:
     * 1 - Vorticity
     * 2 - Vorticity number
     * 3 - Minimum strain
     * 4 - Maximum strain
     * 5 - Ratio of strain ellipse (max/min)
     * 6 - Direction long axis strain ellipse (stretching)
     * 7 - Dilation
     */    
    double vorticity;
    double vorticity_number;
    double emin;
    double emax;
    double ratio_maxmin;
    double stretch_dir;
    double dilation;
    
    /* First, check if user wants to plot ALL arrays and (if yes) do that */
    if (iPropertyID==8)
    {
        printf("  ... writing all arrays of plot data to seven textfiles\n");
        fstream fOutFile1,fOutFile2,fOutFile3,fOutFile4,fOutFile5,fOutFile6,fOutFile7;
        fOutFile1.open ( "VorticityData.txt", fstream::out | fstream::trunc); 
        fOutFile2.open ( "VorticityNumberData.txt", fstream::out | fstream::trunc); 
        fOutFile3.open ( "MinStrainData.txt", fstream::out | fstream::trunc); 
        fOutFile4.open ( "MaxStrainData.txt", fstream::out | fstream::trunc); 
        fOutFile5.open ( "RatioData.txt", fstream::out | fstream::trunc); 
        fOutFile6.open ( "StretchDirData.txt", fstream::out | fstream::trunc); 
        fOutFile7.open ( "DilationData.txt", fstream::out | fstream::trunc);     
        PlotPoints pTmp;
        int iID = 0;
           
        // Go through rows, starting with the TOP row and write to textfile:
        for (int row=iMaxRows-1;row>=0;row--)
        {
            // Go through cols, starting with the LEFT col and write to textfile: 
            for (int col=0;col<iMaxCols;col++)
            {
                iID = (row*iMaxCols)+col;
                pTmp = vDataArray[iID];
                
                // plot all data in textfiles
                fOutFile1 << pTmp.vorticity << " ";  
                fOutFile2 << pTmp.vorticity_number << " "; 
                fOutFile3 << pTmp.emin << " "; 
                fOutFile4 << pTmp.emax << " "; 
                fOutFile5 << pTmp.ratio_maxmin << " "; 
                fOutFile6 << pTmp.stretch_dir << " "; 
                fOutFile7 << pTmp.dilation << " ";                              
            }    
            fOutFile1 << endl; // create new row      
            fOutFile2 << endl;    
            fOutFile3 << endl;    
            fOutFile4 << endl;    
            fOutFile5 << endl;    
            fOutFile6 << endl;    
            fOutFile7 << endl;
        }    
        fOutFile1.close();   
        fOutFile2.close();   
        fOutFile3.close();   
        fOutFile4.close();   
        fOutFile5.close();   
        fOutFile6.close();   
        fOutFile7.close();  
        
        return;      
    }
    
    fstream fOutFile;
    const char *Fname;
    // find correct filename
    switch (iPropertyID) 
    {
        case 1:
            printf("  ... writing vorticity array to textfile\n");
            fOutFile.open ( Fname, fstream::out | fstream::trunc); 
            Fname = "VorticityData.txt"; break;              
        case 2:
            printf("  ... writing vorticity-number array to textfile\n");
            fOutFile.open ( Fname, fstream::out | fstream::trunc); 
            Fname = "VorticityNumberData.txt"; break;
        case 3:
            printf("  ... writing min. strain array to textfile\n");
            fOutFile.open ( Fname, fstream::out | fstream::trunc); 
            Fname = "MinStrainData.txt"; break; 
        case 4:
            printf("  ... writing max. strain array to textfile\n");
            fOutFile.open ( Fname, fstream::out | fstream::trunc); 
            Fname = "MaxStrainData.txt"; break;
        case 5:
            printf("  ... writing strain ratio array to textfile\n");
            fOutFile.open ( Fname, fstream::out | fstream::trunc); 
            Fname = "RatioData.txt"; break;
        case 6:
            printf("  ... writing stretching direction array to textfile\n");
            fOutFile.open ( Fname, fstream::out | fstream::trunc); 
            Fname = "StretchDirData.txt"; break;
        case 7:
            printf("  ... writing dilation array to textfile\n");
            fOutFile.open ( Fname, fstream::out | fstream::trunc); 
            Fname = "DilationData.txt"; break;
        case 8:
            // do not do anything, we did this before
            break;
        default:
            //CODE
          break;
    }
    //fOutFile.open ( Fname, fstream::out | fstream::trunc); 
    PlotPoints pTmp;
    int iID = 0;
           
    // Go through rows, starting with the TOP row and write to textfile:
    for (int row=iMaxRows-1;row>=0;row--)
    {
        // Go through cols, starting with the LEFT col and write to textfile: 
        for (int col=0;col<iMaxCols;col++)
        {
            iID = (row*iMaxCols)+col;
            pTmp = vDataArray[iID];
            // Find the correct property and write it in text file
            switch (iPropertyID) 
            {
                case 1:
                    fOutFile << pTmp.vorticity << " "; break;              
                case 2:
                    fOutFile << pTmp.vorticity_number << " "; break; 
                case 3:
                    fOutFile << pTmp.emin << " "; break; 
                case 4:
                    fOutFile << pTmp.emax << " "; break; 
                case 5:
                    fOutFile << pTmp.ratio_maxmin << " "; break; 
                case 6:
                    fOutFile << pTmp.stretch_dir << " "; break; 
                case 7:
                    fOutFile << pTmp.dilation << " "; break; 
                default:
                    //CODE
                  break;
            }                            
        }    
        fOutFile << endl; // create new row   
    }
    
    fOutFile.close();
}

void ResetUnodes()
{
    printf("  ... resetting unodes\n");
    /* Delete existing unodes and replace with data array (plot points).*/
    UnodesClean();
    /* Save new Elle file - unodes are added manually later */  
    if(ElleWriteData(ElleSaveFileRoot())) OnError("",1);
        
    /* Add unodes manually. Probably there is a more elegant way ;-) */
    fstream fElleFile;
    fElleFile.open (ElleSaveFileRoot(), fstream::out | fstream::app);
    
    // unode positions
    int iNPoints = vDataArray.size();
    fElleFile << "UNODES" << endl;    
    for (int i=0;i<iNPoints;i++)
    {
        PlotPoints pPoint;
        pPoint = vDataArray[i];
        fElleFile << i << " " << pPoint.x << " " << pPoint.y << endl;
    }
    // U_ATTRIB_A
    fElleFile << "U_ATTRIB_A" << endl;    
    for (int i=0;i<iNPoints;i++)
    {
        PlotPoints pPoint;
        pPoint = vDataArray[i];
        fElleFile << i << " " << pPoint.vorticity << endl;
    }
    // U_ATTRIB_B
    fElleFile << "U_ATTRIB_B" << endl;    
    for (int i=0;i<iNPoints;i++)
    {
        PlotPoints pPoint;
        pPoint = vDataArray[i];
        fElleFile << i << " " << pPoint.vorticity_number << endl;
    }
    // U_ATTRIB_C
    fElleFile << "U_ATTRIB_C" << endl;    
    for (int i=0;i<iNPoints;i++)
    {
        PlotPoints pPoint;
        pPoint = vDataArray[i];
        fElleFile << i << " " << pPoint.emin << endl;
    }
    // U_ATTRIB_D
    fElleFile << "U_ATTRIB_D" << endl;    
    for (int i=0;i<iNPoints;i++)
    {
        PlotPoints pPoint;
        pPoint = vDataArray[i];
        fElleFile << i << " " << pPoint.emax << endl;
    }
    // U_ATTRIB_E
    fElleFile << "U_ATTRIB_E" << endl;    
    for (int i=0;i<iNPoints;i++)
    {
        PlotPoints pPoint;
        pPoint = vDataArray[i];
        fElleFile << i << " " << pPoint.stretch_dir << endl;
    }
    // U_ATTRIB_F
    fElleFile << "U_ATTRIB_F" << endl;    
    for (int i=0;i<iNPoints;i++)
    {
        PlotPoints pPoint;
        pPoint = vDataArray[i];
        fElleFile << i << " " << pPoint.ratio_maxmin << endl;
    }
    // U_VISCOSITY
    fElleFile << "U_VISCOSITY" << endl;    
    for (int i=0;i<iNPoints;i++)
    {
        PlotPoints pPoint;
        pPoint = vDataArray[i];
        fElleFile << i << " " << pPoint.dilation << endl;
    }
    fElleFile.close();
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
