#include "FS_fft2elle_strainanalysis.h"
#include "FS_GaussSolve.h"
using namespace std;

///* Some variables */
//int iMaxUnodes = 0;
//int iDim = 0;
//double dBoxArea = 0.0;
//double dBoxWidth = 0.0;
//double dBoxHeight = 0.0;

int main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    /*
     * Initialise and set option names
     */
    ElleInit();
    ElleSetOptNames("importDD","outputPosGradTensor","unused","unused","unused","unused","unused","unused","unused");
    /* 
     * Give default values for userdata:
     */
    ElleUserData(userdata);
    for (int i=0;i<9;i++) 
        userdata[i]=0; // By default, set all user inputs to 0
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    /*
     * Eventually set the interval for writing file, stages etc. if e.g.
	 * this utility should only run once etc.
     */
    ElleSetSaveFrequency(1);
    ElleSetStages(1);
    /*
     * Set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);	
	/*
     * Set the default output name for elle files
     */
    char cFileroot[] = "FS_fft2elle_strainanalysis.elle";  
    ElleSetSaveFileRoot(cFileroot);
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
    int err=0;

    UserData userdata;             
    ElleUserData(userdata); 
    int import_DD = (int)userdata[0];
    int iWritePosGradData = (int)userdata[1];
    bool bWriteFData = false;
    if (iWritePosGradData==1) bWriteFData=true;
    
    // Get some Data 
    GetBoxInfo(&dBoxArea,&dBoxWidth,&dBoxHeight,&dSSOffset);
    iMaxBnodes = ElleMaxNodes();
    iMaxUnodes = ElleMaxUnodes(); 
    iDim = sqrt(iMaxUnodes);
    
    /* Load incremental strains from FFT output file */
    if (LoadBoxIncrStrain("temp-FFT.out"))
        OnError("ERROR (LoadBoxIncrStrain)",err);
    
    /* Temporarily store position of regular square grid unodes after this 
     * strain increment in unode suitable attributes (e.g. U_ATTRIB_D and _E)*/
    if (!fileExists("unodexyz.out")) 
    {
        cout << "ERROR: File unodexyz.out does not exist" << endl;
        return 1;
    }
    if(ReadUnodePos("unodexyz.out"))
    {
        return 1;
    }
        
    printf("dx, dy: %lf, %lf\n",dXStrain,dYStrain);
    printf("Shear strain increment: %lf\n",dShearStrain);
    
    //printf("dy %f, dx %f, ds %f\n",dXStrain,dYStrain,dShearStrain);
    
    /* Reset unode, passive marker and bnode positions */
    err = AnalyseUnodes(bWriteFData);
    if(err) 
    {
        OnError("ERROR (StrainAnalysis)",err);
        return err;
    }
    
    err = AnalyseBnodes();
    if(err) 
    {
        OnError("ERROR (StrainAnalysisBnodes)",err);
        return err;
    }
    
    CleanFile(); // remove temporary unode attribs
    
    /* Reset cell */
    ResetCell(dXStrain,dYStrain,0.0,dShearStrain);
    
    /* Check for bnode topology */
    Coords cNewPos;
    for (int j=0;j<iMaxBnodes;j++)
    {
        if (ElleNodeIsActive(j))
        {
            ElleNodePosition(j,&cNewPos);
            ElleNodeUnitXY(&cNewPos);
            ElleSetPosition(j,&cNewPos);
            if (ElleNodeIsDouble(j)) ElleCheckDoubleJ(j);
                else if (ElleNodeIsTriple(j)) ElleCheckTripleJ(j);
        }
    }
    ElleAddDoubles();
    
    /* Add attributes*/
    /*
     * set the euler angles to the values from the FFT output
     */
    vector<int> attribs(3,NO_VAL);
    attribs[0] = E3_ALPHA;
    attribs[1] = E3_BETA;
    attribs[2] = E3_GAMMA;
    err = SetUnodeAttributesFromFile("unodeang.out",attribs);
    if (err) OnError("unodeang.out",err);
		
	/*
	* set the values of dislocation density from the FFT output
	*/
	if (import_DD == 1)
    {
        if (!ElleUnodeAttributeActive(U_DISLOCDEN))	
        {
	 		ElleInitUnodeAttribute(U_DISLOCDEN);
            ElleSetDefaultUnodeAttribute(0.0,U_DISLOCDEN);
        }
    }

    FS_SetUnodeDD("tex.out");
    
	/*
	* check if unodes changed flynn and (if yes) update its euler angles
	*/
	FS_CheckUnodes();
    
    /* Clean temporary attributes */
    CleanFile();
    
    err=ElleWriteData(ElleSaveFileRoot());
    if(err) OnError("",err);
    
    return err;
}

int StrainAnalysis(Coords *cPrevPosition,Coords *cNewPos,bool bWriteFData,int iPointID)
{
    /* This function finds the position gradient tensor for the point cPrevPos
     * according to the local strain field obtained from FFT output (unodexyz)
     * temporarily stored in unode attributes "iAttribTempPosX" and 
     * "iAttribTempPosY"
     * 
     * Procedure:
     * *
     * 1) Find apos. gradient tensor using 
     *  1.1) Find three nearest neighbour unodes in reg. square grid to the 
     *       point of interest
     *      1.1.1) By finding the 4 nearest ones and deleting the one that is 
     *             furthest away, creates triangle
     *  2.2) Use their actual and new positions to infer pos. grad. tensor
     *      2.2.1) Use the Gaussian system to solve the set of linear equations
     * 2) Find new position of the point of interest and -if user sets the 
     *    boolean input variable to "true", store this position together with
     *    position gradient tensor and the input "iPointID" in a textfile
     */
     
    /* Variable declaration */
    int i4Nbs[4];
    Coords c4Nbs[4];
    int iCounter = 0;
    Coords cTriUnodes[3];
    Coords cTriUnodesDef[3];
    Coords cPrevPos; // will be pushed back in unit cell to search for neighbours, cPrevPosition is preserved
    cPrevPos = *cPrevPosition;
    double dTmp[2];      
    double dPosGradTensor[4]; 
    /* Explanation:
     * If pos. grad. tensor is dPosGradTensor[4]=F, then:
     *    F11 F12
     * F= F21 F22
     * 
     * and
     * 
     * dPosGradTensor[0] = F11
     * dPosGradTensor[1] = F12
     * dPosGradTensor[2] = F21
     * dPosGradTensor[3] = F22
     */  
    double dX,dY; // For rigid body translation
     
    /* Shift current position of point of interest back in model box*/
    ElleNodeUnitXY(&cPrevPos);
    // Found that ElleNodeUnitXY is sometimes insufficient if y-value is
    // just slightly below zero, therefore:
    if (cPrevPos.y<0)
    {
        cPrevPos.y+=dBoxHeight;
        cPrevPos.x+=dSSOffset;
        ElleNodeUnitXY(&cPrevPos);
    }
    
    /* Determine its four nearest unodes in regular square grid. Also get 
     * their positions and delete the one furthest away from marker point */
    int iDelUnode = 0; // will be between 0-3, marks the unode that is furthest away and will be deleted
    iDelUnode= Find4NearestUnodes(cPrevPos,i4Nbs,c4Nbs);
        
        /* Get position before and after thhe the last strain increment for the 
         * 3 unodes closest to point of interest: */
        iCounter = 0;
        for (int ii=0;ii<4;ii++)
        {
            if (ii!=iDelUnode)
            {
                cTriUnodes[iCounter] = c4Nbs[ii];
                dTmp[0]=dTmp[1]=0.0;
                
                /* Get position of this unode from unodexyz.out (FFT output)*/
                ElleGetUnodeAttribute(i4Nbs[ii],&dTmp[0],iAttribTempPosX);
                ElleGetUnodeAttribute(i4Nbs[ii],&dTmp[1],iAttribTempPosY);
                cTriUnodesDef[iCounter].x = dTmp[0];
                cTriUnodesDef[iCounter].y = dTmp[1];
                
                /* FS: 
                 * The next step is important: The position of the points in
                 * cTriUnodesDef (new position) is inside the new model box. 
                 * The position of cTriUnodes (old position) is in old model 
                 * box. Both need to be related to the actual previous position.
                 * 
                 * As we did not updated the model box yet, we need a special 
                 * function to plot the new positions of the three neighbour
                 * points next to the previous deformed position, see 
                 * explanation in NewCell_CoordsPlotXY
                 */
                NewCell_CoordsPlotXY(&cTriUnodesDef[iCounter],cPrevPosition);
                ElleCoordsPlotXY(&cTriUnodes[iCounter],cPrevPosition);
                
                iCounter++;
            }
        }
        
        /* Calculate position gradient tensor and dx, dy:*/
        Solve4PosGradTensor(cTriUnodes,cTriUnodesDef,&dX,&dY,dPosGradTensor);
        
        //printf("dx=%f;\n",dX);
        //printf("dy=%f;\n",dY);
        //printf("f11=%f;\n",dPosGradTensor[0]);
        //printf("f12=%f;\n",dPosGradTensor[1]);
        //printf("f21=%f;\n",dPosGradTensor[2]);
        //printf("f22=%f;\n",dPosGradTensor[3]);
        
        /* Get new position for point of interest*/
        cNewPos->x = dX+dPosGradTensor[0]*cPrevPosition->x+dPosGradTensor[1]*cPrevPosition->y;
        cNewPos->y = dY+dPosGradTensor[2]*cPrevPosition->x+dPosGradTensor[3]*cPrevPosition->y;
           
        /* If bWriteFData is true: Write data in textfile*/
        /* CAREFUL: Only use this for ONE type of points (e.g. all unodes, all
         * passive marker points). The code below just appends data to the 
         * existing textfile!! */
        if (bWriteFData)
        {
                /* Write an info file: */
                if (!fileExists("PosGradTensorInfo.txt"))
                {
                    fstream fOutInfoFile;
                    fOutInfoFile.open ( "PosGradTensorInfo.txt", fstream::out | fstream::trunc); 
                    fOutInfoFile << "Description of columns in ";
                    fOutInfoFile << "PosGradTensor.txt" << endl;
                    fOutInfoFile << "Position gradient tensor F:\n" << endl; 
                    fOutInfoFile << "    F11  F12\nF = \n    F21  F22\n" << endl;
                    fOutInfoFile << "Column 1.: node ID" << endl;
                    fOutInfoFile << "Column 2.: node new x (not repositioned in new model box)" << endl;
                    fOutInfoFile << "Column 3.: node new y (not repositioned in new model box)" << endl;
                    fOutInfoFile << "Column 4.: dX" << endl;
                    fOutInfoFile << "Column 5.: dY" << endl;
                    fOutInfoFile << "Column 6.: F11" << endl;
                    fOutInfoFile << "Column 7.: F12" << endl;
                    fOutInfoFile << "Column 8.: F21" << endl;
                    fOutInfoFile << "Column 9.: F22" << endl;
                    fOutInfoFile.close();                    
                }
                fstream fOutFile;
                fOutFile.open ( "PosGradTensor.txt", fstream::out | fstream::app); 
                
                fOutFile << iPointID << " " << cNewPos->x << " " << cNewPos->y << " ";
                fOutFile << dX << " " << dY << " ";
                fOutFile << dPosGradTensor[0] << " ";
                fOutFile << dPosGradTensor[1] << " ";
                fOutFile << dPosGradTensor[2] << " ";
                fOutFile << dPosGradTensor[3] << endl;
                
                fOutFile.close();
        }
        
     return 0;
}

int AnalyseUnodes(bool bWriteFData)
{
    /* Calculate local strains using the position gradient tensor and update
     * the passive marker point positions */
    
    /* Update attribute for passive markers (set current position to previous) 
     * or initialise it */
    UpdateStrainAttrib();

    Coords cPrevPos;
    Coords cNewPos;
    double dTmp[2];
    
    for (int i=0;i<iMaxUnodes;i++)
    {
        /* Get current position of passive marker and shift back in model box*/
        dTmp[0]=dTmp[1]=0.0;
        ElleGetUnodeAttribute(i,PREV_S_X,&dTmp[0]);
        ElleGetUnodeAttribute(i,PREV_S_Y,&dTmp[1]);
        cPrevPos.x = dTmp[0];
        cPrevPos.y = dTmp[1];
        
        /* Perform strain analysis by calculating position gradient tensor
         * and update poisition */
        StrainAnalysis(&cPrevPos,&cNewPos,bWriteFData,i);
        
        /* Set new position in passive marker grid */
        ElleSetUnodeAttribute(i,CURR_S_X,cNewPos.x);
        ElleSetUnodeAttribute(i,CURR_S_Y,cNewPos.y);
        
        /* Set new unode position after this strain increment */
        dTmp[0]=dTmp[1]=0.0;
        ElleGetUnodeAttribute(i,&dTmp[0],iAttribTempPosX);
        ElleGetUnodeAttribute(i,&dTmp[1],iAttribTempPosY);
        //cNewPos.x = dTmp[0];
        //cNewPos.y = dTmp[1];
        //NewCell_NodeUnitXY(&cNewPos);
        //ElleSetUnodePosition(i,&cNewPos);       
        
        // Update like in old fft2elle:
        Coords cTmpXY;
        ElleGetUnodePosition(i,&cPrevPos); 
        cTmpXY.x = dTmp[0];
        cTmpXY.y = cPrevPos.y;
        
        if (fabs(dYStrain)>1e-6) cTmpXY.y = dTmp[1]; // pure shear
        NewCell_CoordsPlotXY(&cPrevPos,&cTmpXY);
        cNewPos.x = cPrevPos.x+(dTmp[0]-cPrevPos.x);
        cNewPos.y = cPrevPos.y+(dTmp[1]-cPrevPos.y);
        NewCell_NodeUnitXY(&cNewPos);
        ElleSetUnodePosition(i,&cNewPos);        
    }
    
    return 0;
}

int AnalyseBnodes()
{      
    /* Calculate local strains using the position gradient tensor and update
     * the bnode positions */
     
    Coords cPrevPos;
    Coords cNewPos;

    for (int i=0;i<iMaxBnodes;i++)
    {
        if (ElleNodeIsActive(i))
        {
            /* Get current position of bnode*/
            ElleNodePosition(i,&cPrevPos);
            
            /* Perform strain analysis by calculating position gradient tensor
             * and update poisition */
            StrainAnalysis(&cPrevPos,&cNewPos,false,i);
            
            /* Set new bnode position: Use NEARLY the same code than for
             * the ElleSetPosition-function, but without repositioning in model
             * box, this is done later*/
            // THE FOLLOWING WOULD BE INCORRECT: We do not want to reset to 
            // the (still active) old model box
            //   ElleSetPosition(i,&cNewPos); 
            NodeAttrib *p;
            p = ElleNode(i);
            p->prev_x = p->x;
            p->prev_y = p->y;
            p->x = cNewPos.x;
            p->y = cNewPos.y;
        }
    }
    
    return 0;
}

void UpdateStrainAttrib()
{
    /*
     * Read U_FINITE_STRAIN from unodes and shift current x and y to previous x and
     * y. Set current to 0 for the moment, it will be updated by FS_SetEunodeStrain.
     * SPECIAL CASE: If U_FINITE_STRAIN is not active it will be initiated and 
     * default values will be:
     * starting position: Position of the unode in a regular grid of size dim x dim
     * previous position = starting position
     * current position = 0,0
     */
    
    /* If strain attribute is not active at the moment initialize it and 
     * fill with starting values*/
    if (!ElleUnodeAttributeActive(U_FINITE_STRAIN))
    {
        ElleInitUnodeAttribute(U_FINITE_STRAIN);
        
        int iUnodeID = 0;
        double dStartPos[2];
        
        for (int j=0;j<iDim; j++) 
        {
			for (int i=0;i<iDim; i++) 
            {
                dStartPos[0]=dStartPos[1]=0.0;
                
				dStartPos[0]= i*(dBoxWidth/(double)iDim);
				dStartPos[1]= j*(dBoxHeight/(double)iDim);
                
                ElleSetUnodeAttribute(iUnodeID,START_S_X,dStartPos[0]);
                ElleSetUnodeAttribute(iUnodeID,START_S_Y,dStartPos[1]);
                ElleSetUnodeAttribute(iUnodeID,PREV_S_X,dStartPos[0]);
                ElleSetUnodeAttribute(iUnodeID,PREV_S_Y,dStartPos[1]);
                ElleSetUnodeAttribute(iUnodeID,CURR_S_X,0.0);
                ElleSetUnodeAttribute(iUnodeID,CURR_S_Y,0.0);
                
				iUnodeID ++;
			}
		}
    }
    else // U_FINITE_STRAIN is active: Shift current to previous position
    {
        double dPosition[2];
        
        for (int i=0;i<iMaxUnodes;i++)
        {
            dPosition[0]=dPosition[1]=0.0;
            
            ElleGetUnodeAttribute(i,CURR_S_X,&dPosition[0]);
            ElleGetUnodeAttribute(i,CURR_S_Y,&dPosition[1]);
            
            ElleSetUnodeAttribute(i,PREV_S_X,dPosition[0]);
            ElleSetUnodeAttribute(i,PREV_S_Y,dPosition[1]);
            ElleSetUnodeAttribute(i,CURR_S_X,0.0);
            ElleSetUnodeAttribute(i,CURR_S_Y,0.0);            
        }
    }
}

int Find4NearestUnodes(Coords cPoint, int i4NearestUnodeIDs[4], Coords c4NearestUnodesXY[4])
{
    /* 
     * This function finds the 3 nearest unodes to any coordinate in a regular
     * square grid. This coordinate should be in the model box. Model box width
     * and height should be known 
     * imagine the 4 nearest unodes in regular suqare grid are aligned from 
     * 1 to 4 (or here actually 0 to 3) in counter clockwise order and the 1st 
     * one is in the lower left corner
     * 
     * The returned int will be between 0-3 and denote the unode that is 
     * furthest away
     */
     
     /* First, take away simple shear offset from cPoint to correctly determine
      * nearest nodes in regular square grid (will be added again when actual 
      * position of nearest node is found as elle2fft creates regular square 
      * grid WITH simple shear offset) */
    Coords cPointSSCorrected;
    cPointSSCorrected = cPoint;
    cPointSSCorrected.x -= ((cPointSSCorrected.y/dBoxHeight)*dSSOffset);
    ElleNodeUnitXY(&cPointSSCorrected);    
    double dPx=cPointSSCorrected.x;
    double dPy=cPointSSCorrected.y;
    
    // a check, just in case:
    if(dPx<0.0) 
    {   
        dPx=(dBoxWidth-(-dPx));
    }
    
    // initial check:
    //if (dPx>=dBoxWidth) dPx = 0.0;
    //if (dPy>=dBoxHeight+ElleCumSSOffset()) dPy = 0.0;
    
    int i1stID_X = floor( (dPx*(double)iDim)/dBoxWidth );
    int i1stID_Y = floor( (dPy*(double)iDim)/dBoxHeight );
    
    // check (happens if coordinate (x or y) is == box width or height):
    if (i1stID_X>=iDim) i1stID_X-=iDim;
    if (i1stID_Y>=iDim) i1stID_Y-=iDim;
    
    i4NearestUnodeIDs[0] = (i1stID_Y*iDim)+i1stID_X;
    
    if (i1stID_X>=(iDim-1))
    {
        if (i1stID_Y>=(iDim-1))
        {
            //cout << 1 << endl;
            i4NearestUnodeIDs[1] = i4NearestUnodeIDs[0]-(iDim-1);
            i4NearestUnodeIDs[2] = 0;
            i4NearestUnodeIDs[3] = iDim-1;
        }
        else
        {
            //cout << 2 << endl;
            i4NearestUnodeIDs[1] = i4NearestUnodeIDs[0]-(iDim-1);
            i4NearestUnodeIDs[2] = i4NearestUnodeIDs[0]+1;
            i4NearestUnodeIDs[3] = i4NearestUnodeIDs[0]+iDim;
        }
    }
    else
    {
        if (i1stID_Y>=(iDim-1))
        {
            //cout << 3 << endl;
            i4NearestUnodeIDs[1] = i4NearestUnodeIDs[0]+1;
            i4NearestUnodeIDs[3] = (i4NearestUnodeIDs[0]+iDim)-iMaxUnodes;
            i4NearestUnodeIDs[2] = i4NearestUnodeIDs[3]+1;
        }
        else
        {
            //cout << 4 << endl;
            i4NearestUnodeIDs[1] = i4NearestUnodeIDs[0]+1;
            i4NearestUnodeIDs[2] = i4NearestUnodeIDs[1]+iDim;
            i4NearestUnodeIDs[3] = i4NearestUnodeIDs[0]+iDim;
        }
    }
    
    // get unode coords (in regular square grid) and find the unode that is 
    // furthest away from point of interest
    Coords cTemp;
    double dMaxSep = 0.0;
    int iDelUnode = 0;
    double dID = 0.0;
    double dDim = (double)iDim;
    for (int ii=0;ii<4;ii++)
    {
        dID = (double)i4NearestUnodeIDs[ii];
        cTemp.x = cTemp.y = 0.0;
        
        cTemp.x = (fmod(dID,dDim)/dDim)*dBoxWidth;
        cTemp.y = ( (dID-fmod(dID,dDim))/(dDim*dDim) )*dBoxHeight;
                
        // Correct for simple shear offset:
        cTemp.x += ((cTemp.y/dBoxHeight)*dSSOffset);
     
        ElleCoordsPlotXY(&cTemp,&cPoint);
        
        c4NearestUnodesXY[ii] = cTemp;
        
        double dSep = pointSeparation(&cTemp,&cPoint);
        if (dSep>dMaxSep)
        {
            dMaxSep = dSep;
            iDelUnode = ii;            
        }
    }
    
    return(iDelUnode);
}

int ReadUnodePos(const char *cFilename)
{
    ifstream XYZFile(cFilename);
    if (!XYZFile) return(OPEN_ERR);
    
    double dTmpValue[3];
    int iUnodeID = 0;
    
    if (!ElleUnodeAttributeActive(iAttribTempPosX))
        ElleInitUnodeAttribute(iAttribTempPosX);
    if (!ElleUnodeAttributeActive(iAttribTempPosY))
        ElleInitUnodeAttribute(iAttribTempPosY);
        
    while (XYZFile) 
    {
        XYZFile >> iUnodeID >> dTmpValue[0] >> dTmpValue[1] >> dTmpValue[2];
        ElleSetUnodeAttribute(iUnodeID,dTmpValue[0],iAttribTempPosX);
        ElleSetUnodeAttribute(iUnodeID,dTmpValue[1],iAttribTempPosY);
        
        iUnodeID++;
    }
    XYZFile.close();
    //printf("iUnodeID %u iMaxUnodes %u\n",iUnodeID,iMaxUnodes+1);
    
    if (iUnodeID!=(iMaxUnodes+1)) return 1;
              
    return 0;
}

void CleanFile()
{
    /* Remove the temporarily used unode attributes */
    vector<int> vRemoveAttribs;
    
    vRemoveAttribs.push_back(iAttribTempPosX);
    vRemoveAttribs.push_back(iAttribTempPosY);
    ElleRemoveUnodeAttributes(vRemoveAttribs);
    vRemoveAttribs.clear();
}

void GetBoxInfo(double *dArea,double *dWidth,double *dHeight,double *dSSOffset)
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
    
    // Calculate with gaussian euqation for polygon areas with n corners:
    // 2*Area = Σ(i=1 to n)  (y(i)+y(i+1))*(x(i)-x(i+1))
    // if i+1>n use i=1 again (or here in C++ i=0)
    int i2 = 0;
    for (int i=0;i<4;i++)    
    {
        i2 = fmod(i,4)+1;
        *dArea += ( (box_xy[i].y+box_xy[i2].y)*(box_xy[i].x-box_xy[i2].x) )/2;   
    }
    
    *dWidth  = box_xy[1].x-box_xy[0].x;
    *dHeight = box_xy[3].y-box_xy[0].y;
    *dSSOffset = ElleSSOffset();
}

void Solve4PosGradTensor(Coords cUnodePrevXY[3], Coords cUnodeNewXY[3], double *dX, double *dY, double dPosGrad[4])
{   
    /* This function does the following:
     * 1) Prepares a coefficient matrix to solve the linear system 
     *    of equations from current unode position in reg. square grid and new
     *    position after this increment of strain
     * 2) Output the displacement dX and dY and the finite position gradient 
     *    tensor for this set of unodes (triangle) in corresponding parameters.
     * The output position gradient tensor is structured like:
     * 
     * if F is the pos. grad. tensor, then:
     *     F11 F12
     * F = F21 F22
     * 
     *  The output will be: 
     * 
     * dPosGrad[0] = F11
     * dPosGrad[1] = F12
     * dPosGrad[2] = F21
     * dPosGrad[3] = F22
     */
    int N = 6; // always in this setup and in 2D
    double **dCoef = 0;
    dCoef=dmatrix(0,N-1,0,N);
    
    // SETP BACK, THIS IS GOING TO BE HUGE:
    
    // Counting 1st value as rows and 2nd as columns
    dCoef[0][0] = 1; dCoef[0][1] = cUnodePrevXY[0].x; dCoef[0][2] = cUnodePrevXY[0].y; dCoef[0][3] = 0; dCoef[0][4] = 0; dCoef[0][5] = 0; dCoef[0][6] = cUnodeNewXY[0].x;
    dCoef[1][0] = 1; dCoef[1][1] = cUnodePrevXY[1].x; dCoef[1][2] = cUnodePrevXY[1].y; dCoef[1][3] = 0; dCoef[1][4] = 0; dCoef[1][5] = 0; dCoef[1][6] = cUnodeNewXY[1].x;
    dCoef[2][0] = 1; dCoef[2][1] = cUnodePrevXY[2].x; dCoef[2][2] = cUnodePrevXY[2].y; dCoef[2][3] = 0; dCoef[2][4] = 0; dCoef[2][5] = 0; dCoef[2][6] = cUnodeNewXY[2].x;
    dCoef[3][0] = 0; dCoef[3][1] = 0; dCoef[3][2] = 0; dCoef[3][3] = 1; dCoef[3][4] = cUnodePrevXY[0].x; dCoef[3][5] = cUnodePrevXY[0].y; dCoef[3][6] = cUnodeNewXY[0].y; 
    dCoef[4][0] = 0; dCoef[4][1] = 0; dCoef[4][2] = 0; dCoef[4][3] = 1; dCoef[4][4] = cUnodePrevXY[1].x; dCoef[4][5] = cUnodePrevXY[1].y; dCoef[4][6] = cUnodeNewXY[1].y; 
    dCoef[5][0] = 0; dCoef[5][1] = 0; dCoef[5][2] = 0; dCoef[5][3] = 1; dCoef[5][4] = cUnodePrevXY[2].x; dCoef[5][5] = cUnodePrevXY[2].y; dCoef[5][6] = cUnodeNewXY[2].y;
    
    // Solve
    SolveEquationSystem(dCoef,N);
    
    // Move results to output:
    *dX = dCoef[0][6];
    *dY = dCoef[3][6];
    *dY = dCoef[3][6];
    dPosGrad[0] = dCoef[1][6];
    dPosGrad[1] = dCoef[2][6];
    dPosGrad[2] = dCoef[4][6];
    dPosGrad[3] = dCoef[5][6];
    
    free_dmatrix(dCoef,0,N,0,N);
}

bool fileExists(const char *filename)
{
  ifstream ifile(filename);
  return ifile.good();
}

void NewCell_CoordsPlotXY(Coords *xy, Coords *prevxy)
{
    /*
     * Difference to standard Elle function: It outputs 1 or -1 if the coordinates 
     * are on opposite sides of the unit cell in either x-or y -direction.
     * In y-direction:
     * In simple shear this would require correction by subtracting 
     * or adding "offset" to the incremental strain in x-direction and adjustments 
     * in ystrain for pure shear
     * 
     * In x-direction:
     * Requires adjustments of xstrain in pure shear
     */    
    int cnt;
    double unitsize_x,unitsize_y;
    double deformx, deformy;
    
    int iOnOtherSideX, iOnOtherSideY;
    CellData unitcell;

    ElleCellBBox(&unitcell);
    deformx = unitcell.cellBBox[TOPLEFT].x-unitcell.cellBBox[BASELEFT].x;
    deformy = unitcell.cellBBox[BASELEFT].y-unitcell.cellBBox[BASERIGHT].y;
    /* 
     * assumes that the unit cell remains a parallelogram
     * (simple shear ?)
     * assuming yoffset is zero
     * unitcell.xoffset is the simple shear (x) + any cell deformation
     */
    unitsize_x = unitcell.xlength;
    unitsize_y = unitcell.ylength;
    
    // Y-component
    
    if ((xy->y - prevxy->y) >= unitsize_y*MAX_SIZE) 
    {
        xy->y -= unitsize_y;
        xy->x -= unitcell.xoffset;
        while ((xy->y - prevxy->y) >= unitsize_y*MAX_SIZE) 
        {
            xy->y -= unitsize_y;
            xy->x -= unitcell.xoffset;
        }
        iOnOtherSideY = -1;
    }
    else if ((xy->y - prevxy->y) < -unitsize_y*MAX_SIZE) 
    {
        xy->y += unitsize_y;
        xy->x += unitcell.xoffset;
        while ((xy->y - prevxy->y) < -unitsize_y*MAX_SIZE) 
        {
            xy->y += unitsize_y;
            xy->x += unitcell.xoffset;
        }
        iOnOtherSideY=1;
    }
    
    // X-component
    
    if ((xy->x - prevxy->x) >= unitsize_x*MAX_SIZE) 
    {
        xy->x -= unitsize_x;
        xy->y -= unitcell.yoffset;
        while ((xy->x - prevxy->x) >= unitsize_x*MAX_SIZE) 
        {
            xy->x -= unitsize_x;
            xy->y -= unitcell.yoffset;
        }
        iOnOtherSideX = -1;
    }
    else if ((xy->x - prevxy->x) < -unitsize_x*MAX_SIZE) 
    {
        xy->x += unitsize_x;
        xy->y += unitcell.yoffset;
        while ((xy->x - prevxy->x) < -unitsize_x*MAX_SIZE) 
        {
            xy->x += unitsize_x;
            xy->y += unitcell.yoffset;
        }
        iOnOtherSideX = 1;
    }


    /* Now correct for offset that is not in the unit cell yet: (by Florian)
     * If a neighbour is on opposite side of the box in y-direction, a 
     * correction of x-strain is needed in simple or correction of y-strain in 
     * pure shear is needed in order to avoid the errors on top and bottom row 
     * the ugrid images.
     * "dShearStrain" or "dYStrain,dXStrain" have to be either subtracted or 
     * added depending on where the neighbour unode was (e.g. on other side in 
     * x or y direction): This is controlled by "iOnOtherSideY and X" being 
     * either +1 or -1
     */
    if (iOnOtherSideY!=0)
    {
        xy->x = xy->x+(dShearStrain*iOnOtherSideY);
        xy->y = xy->y+(dYStrain*iOnOtherSideY);
    }
    if (iOnOtherSideX!=0)
    {
        xy->x = xy->x+(dXStrain*iOnOtherSideX);
    }
}

void NewCell_NodeUnitXY(Coords *xy)
{
    /* The same the ElleNodeUnitXY, but used for the new, deformed cell, if 
     * model box has not been updated yet. See equivalent: NewCell_CoordsPlotXY
     */
    double eps;
    double minx,maxx,miny,maxy,xp;
    double new_ylength,new_xlength;
    double xoffset;
    CellData unitcell;

    /*
     * assume simple shear in x-direction
     * adjust y & x vals if above or below unit cell
     * adjust if x val outside unit cell
     */
    eps = 1.5e-6;
	/* basil uses float */
    /*eps = 1.0e-5;*/
    ElleCellBBox(&unitcell);
    maxy = unitcell.cellBBox[TOPLEFT].y+dYStrain;
    miny = unitcell.cellBBox[BASELEFT].y;
    maxx = unitcell.cellBBox[BASERIGHT].x+dXStrain;
    minx = unitcell.cellBBox[BASELEFT].x;
    new_ylength = maxy-miny;
    new_xlength = maxx-minx;
    xoffset = dSSOffset+dShearStrain;
    if (xoffset>=new_xlength) xoffset -= new_xlength;
    
    // y-component
    while (xy->y > maxy+eps) {
        xy->y -= new_ylength;
        xy->x -= xoffset;
    }
    while (xy->y < miny-eps) {
        xy->y += new_ylength;
        xy->x += xoffset;
    }
    xp = (xy->y - miny)/new_ylength * xoffset
                              + unitcell.cellBBox[BASELEFT].x;
    
    // x-component
    if (xy->x < (xp-eps)) {
        while (xy->x < xp) {
            xy->x += new_xlength;
            xy->y += unitcell.yoffset; //unitcell-yoffset is usually 0, if not this codes needs an update 
        }
    }
    else {
        xp += new_xlength;
        if (xy->x > (xp+eps)) {
            while (xy->x > xp) {
                xy->x -= new_xlength;
                xy->y -= unitcell.yoffset; //unitcell-yoffset is usually 0, if not this codes needs an update 
            }
        }
    }
}

void ResetCell(double xstrain, double ystrain, double zstrain, double offset)
{
	//assume shortening parallel to y
    CellData unitcell;
    Coords  corners[4]; 
	double cum_offset;
    double dSSOffset = ElleSSOffset(); // FS added for offset update after ElleUpdateCellBBox

	cum_offset = ElleCumSSOffset();
	    ElleCellBBox(&unitcell);
	
// this assumes linear gradient for simple shear 
// 0->offset as y varies 0->1
// xstrain, ystrain are used if pure shear deformation, 0 for simple shear
    corners[BASELEFT].x = unitcell.cellBBox[BASELEFT].x;
    corners[BASELEFT].y = unitcell.cellBBox[BASELEFT].y;
    corners[BASERIGHT].x = unitcell.cellBBox[BASERIGHT].x+xstrain;
    corners[BASERIGHT].y = unitcell.cellBBox[BASERIGHT].y;
    corners[TOPRIGHT].x = unitcell.cellBBox[TOPRIGHT].x+offset+xstrain;
    corners[TOPRIGHT].y = unitcell.cellBBox[TOPRIGHT].y+ystrain;
    corners[TOPLEFT].x = unitcell.cellBBox[TOPLEFT].x+offset;
    corners[TOPLEFT].y = unitcell.cellBBox[TOPLEFT].y+ystrain; // FS corrected this: was "TOPRIGHT" on the right and side of "=" sign before
	
// FS uncommented this #if XY --> #endif part
#if XY
	temp=dd[2][0]; // incremental shear strain
		 	    ElleSetCellBBox(&corners[BASELEFT], &corners[BASERIGHT],
                    &corners[TOPRIGHT], &corners[TOPLEFT]);

	 cum_offset +=temp; 

     ElleSetCumSSOffset(cum_offset);
	  offset=modf(ElleCumSSOffset(),&tmp);
	  ElleSetSSOffset(offset);
#endif
		ElleUpdateCellBBox(&corners[BASELEFT], &corners[BASERIGHT],
                    &corners[TOPRIGHT], &corners[TOPLEFT]);	 
}

int LoadBoxIncrStrain(const char *fname)
{
    /*
     * Read the new cellbox size and displacements
     */
    double dTmp[3][3];
    int err=0;
    int id,i=0,j=0;
    double val[3];

    ifstream datafile(fname);
    if (!datafile) return(OPEN_ERR);
    for (j=0;j<3;j++) 
    {
        datafile >> val[0] >> val[1] >> val[2];
        for (i=0;i<3;i++)  
            dTmp[j][i]=(double)val[i];    
    }
    dShearStrain = dTmp[2][0];
    dXStrain = dTmp[1][0];
    dYStrain = dTmp[1][1];
    
    datafile.close();

    return(err);
}

int SetUnodeAttributesFromFile(const char *fname,vector<int> &attribs)
{
    int id, count, i;
    Coords xy;
    for (i=0, count=0;i<attribs.size();i++) if (attribs[i]!=NO_VAL) count++;
    if (count<1) return (ATTRIBID_ERR);
    double *val = new double[count];
    ifstream datafile(fname);
    if (!datafile) return(OPEN_ERR);
    for (i=0;i<count;i++) {
        ElleInitUnodeAttribute(attribs[i]);
    }
    while (datafile) {
        datafile >> id ;
        for (i=0;i<count && datafile;i++) {
            datafile >> val[i];
            ElleSetUnodeAttribute(id,val[i],attribs[i]);
        }
    }
    datafile.close();
    delete [] val;
    return(0);
}

void FS_SetUnodeDD(const char *fname)
{
    /*
     * FS:
     * This is a new version of the code. It checks if unode changed flynn due 
     * to sweeping boundary and assignes correct flynns to unodes WITHOUT 
     * updating U_ATTRIB_C, because this is done by the FS_topocheck code
     * that always should run after FFT.
     */
    int iUnodeID=0;
	double val[12],dDens=0.0;
    
    UserData userdata;
    ElleUserData(userdata);
    int iFlynnID = 0;
    double dFlynnVisc = 0.0;
    int iUpdateDDs = (int)userdata[0];
    int iExcludeValue = (int)userdata[1];

	
/*	
printf("\t*** Export data from FFT to ELLE ***\n");	
printf("\t*** using as default tex.out file ***\n");
	
printf("Import data to store in U_ATTRIB_A and U_ATTRIB_B\n");
printf("NON import dislocation density to U_DISLOCDEN !!\n");	
printf("[4] normalized strain rate\n");
printf("[5] normalized stress\n");
printf("[6] activity basal mode\n");
printf("[7] activity prismatic mode\n");
printf("[8] Geometrical density of dislocation ()\n");
printf("[9] Stattistical density of dislocation ()\n");	
printf("[10] identification of Fourier Point \n");	
printf("[11] FFT grain nunmber\n");
printf("?? (exp. 4 5 +<return>)\n");	
	
cin >> opt_1 >> opt_2; 
*/
	

    ifstream datafile(fname);
    if (!datafile) 
    {
        printf("ERROR (FS_SetUnodeAttributesFromFile): tex.out missing\n");
    }
    else
    {
        if (iUpdateDDs==1)
        {	
            iUnodeID=0;
            while (datafile) 
            {
                datafile >>val[0]>>val[1]>>val[2]>>val[3]>>val[4]>>val[5]
                         >>val[6]>>val[7]>>val[8]>>val[9]>>val[10]>>val[11];		

                // Get old dislocden:
                ElleGetUnodeAttribute(iUnodeID,&dDens,U_DISLOCDEN);

                /*
                * FS: Check if flynn VISCOSITY is == userdata[1], if yes 
                * this area (like a bubble) is excluded from dislocden update 
                * and gets dislocden = 0
                * 
                * THIS IS OLD
                */
                /*!
                if (ElleFlynnAttributeActive(VISCOSITY))
                {
                    iFlynnID=ElleUnodeFlynn(iUnodeID);
                    ElleGetFlynnRealAttribute(iFlynnID,&dFlynnVisc,VISCOSITY);
                }     

                if (iExcludeValue != 0 && iExcludeValue == (int)dFlynnVisc)
                {
                    // Exclude Unode from updating the dislocden. 
                    // Insead set added dislocden to 0:
                    //ElleSetUnodeAttribute(iUnodeID,(dDens+0.0),U_DISLOCDEN);
                    // Or maybe better: Set e.g. bubble dislocden to zero:
                    ElleSetUnodeAttribute(iUnodeID,0.0,U_DISLOCDEN);
                }
                else
                {
                    // Update dislocden like usual                    
                    ElleSetUnodeAttribute(iUnodeID,(dDens+val[8]),U_DISLOCDEN);
                }	
                */
                ElleSetUnodeAttribute(iUnodeID,(dDens+val[8]),U_DISLOCDEN);
                iUnodeID++;
            }
        }
    }
    datafile.close();

    return;
}

void FS_CheckUnodes()
{
    /* This checks if a unode changed its host flynn during the process and
     * updates its DD and euler_3 if needed */    
    int iFlynnId=0;
    Coords cUnodeXY;
    int iMaxUnodes=ElleMaxUnodes();
    int iMaxFlynns=ElleMaxFlynns();
    vector<int> vReAssign;
    vector<int> vUnodeList;
    
    /* STEP1:
     * Check if attributes storing flynn ID are active, if not assign them 
     * according to the actual situation, updates and corrections will follow */
    if (!ElleUnodeAttributeActive(U_ATTRIB_C))
    {
        ElleInitUnodeAttribute(U_ATTRIB_C);
        for (int unode=0;unode<iMaxUnodes;unode++)
        {
            iFlynnId = ElleUnodeFlynn(unode);
            ElleSetUnodeAttribute(unode,U_ATTRIB_C,(double)iFlynnId);
            // This might not be correct yet, but will be updated during this
            // function
        }
    }
    if (!ElleFlynnAttributeActive(F_ATTRIB_C))
    {
        ElleInitFlynnAttribute(F_ATTRIB_C);    
        for (int flynn=0;flynn<iMaxFlynns;flynn++)
            if (ElleFlynnIsActive(flynn))
                ElleSetFlynnRealAttribute(flynn,(double)flynn,F_ATTRIB_C);
    }
    
    /* STEP 2:
     * Go through all unodes and check if they are in the correct flynn */
    for (int unode=0;unode<iMaxUnodes;unode++)
    {
        bool bFound=false; // will be true once correct host flynn is found
        iFlynnId = ElleUnodeFlynn(unode);
        ElleGetUnodePosition(unode,&cUnodeXY);
        if (ElleFlynnIsActive(iFlynnId))
        {
            if (EllePtInRegion(iFlynnId,&cUnodeXY)) 
            {
                bFound=true;
                ElleSetUnodeAttribute(unode,U_ATTRIB_C,(double)iFlynnId); 
            }
        }
        
        if (!bFound)
        {
            /* Need to search for the correct host flynn*/
            for (int flynn=0;flynn<iMaxFlynns;flynn++)
            {
                if (ElleFlynnIsActive(flynn))
                {
                    if (EllePtInRegion(flynn,&cUnodeXY)) 
                    {
                        ElleAddUnodeToFlynn(flynn,unode); 
                        vReAssign.push_back(unode);
                        // To later identify the unodes that need reassignment:
                        ElleSetUnodeAttribute(unode,U_ATTRIB_C,-1.0); 
                        bFound=true;
                        break;
                    }
                }
            }
        }
    }
    //ElleWriteData("test.elle");
    
    /* STEP 3:
     * Re assign unode attributes euler_3 and dislocden: Use euler3 of closest 
     * unode in that flynn and set dislocden to zero */
    if (ElleUnodeAttributeActive(EULER_3))
        if (ElleUnodeAttributeActive(U_DISLOCDEN))
            for (int i=0;i<vReAssign.size();i++)
            {
                FS_ReAssignAttribsSweptUnodes(vReAssign[i]); 
            }
                    
    vReAssign.clear();
}

void FS_ReAssignAttribsSweptUnodes(int iUnodeID)
{
    /* Re assign unode attributes euler_3 and dislocden: Use euler3 of closest 
     * unode in that flynn and set dislocden to zero */
    int iFlynnId=0;    
    double dRoi = FS_GetROI(8);
    double dTest=0.0;
    double dNewEuler[3];
    double dEulerOld[3];
    double dTmpEuler[3];
    for (int ii=0;ii<3;ii++) 
    {
        dNewEuler[ii]=0.0;
        dEulerOld[ii]=0.0;
        dTmpEuler[ii]=0.0;
    }
    //double dDensityMin = 0.0; // Set new dislocden to this value, i.e. to zero
    vector<int> vUnodeList;
    Coords cUnodeXY;
    Coords cRefXY;
    
    /* Get info about the unode of interest */
    ElleGetUnodePosition(iUnodeID,&cRefXY);    
    iFlynnId = ElleUnodeFlynn(iUnodeID); // that will be the correct flynn
    vUnodeList.clear();
    ElleGetFlynnUnodeList(iFlynnId,vUnodeList);  
    ElleGetUnodeAttribute(iUnodeID,&dEulerOld[0],E3_ALPHA);
    ElleGetUnodeAttribute(iUnodeID,&dEulerOld[1],E3_BETA);
    ElleGetUnodeAttribute(iUnodeID,&dEulerOld[2],E3_GAMMA);   
    
    /* Go to each unode in this list and check if it is NOT a unode that 
     * still needs to be reassigned:
     * If yes: Search for the closest unode to the unode of interest */
    double dMinDist = 1000000.0;
    double dDist    = 0.0;
    int iCount      = 0;
    int iNbUnode    = 0;
    for (int j=0;j<vUnodeList.size();j++)
    {
        ElleGetUnodeAttribute(vUnodeList[j],&dTest,U_ATTRIB_C);
        if (dTest>=0.0) //U_ATTRIB_C will be -1 if unode was swept
        {
            ElleGetUnodePosition(vUnodeList[j],&cUnodeXY);
            ElleCoordsPlotXY(&cRefXY,&cUnodeXY);			  
            dDist = pointSeparation(&cRefXY,&cUnodeXY);
            
            if (dDist<=dRoi && dDist<dMinDist)
            {
                iCount++;      
                dMinDist=dDist;
                iNbUnode = vUnodeList[j];
            }                                 
        }
    }
    
    if (iCount>0)
    {                            
        /* Found the closest nb unode, store its orientation */
        ElleGetUnodeAttribute(iNbUnode,&dNewEuler[0],E3_ALPHA);
        ElleGetUnodeAttribute(iNbUnode,&dNewEuler[1],E3_BETA);
        ElleGetUnodeAttribute(iNbUnode,&dNewEuler[2],E3_GAMMA);       
    }
    else
    {
        /* No unodes found in roi, use mean value of the whole flynn
         * Only if there are no more unodes in flynn (meaning that 
         * vUnodeList.size()==0) keep old orientation */
        if (vUnodeList.size()==0) // unlikely, but may be possible
        {
            for (int ii=0;ii<3;ii++) dNewEuler[ii] = dEulerOld[ii];
            printf("WARNING (FS_ReAssignAttribsSweptUnodes):\nSetting new ");
            printf("orientation of swept unode %u to old value\n",iUnodeID);
        }
        else
        {
            double dDistTotal=0.0;
            dDist=0.0;
            for (int j=0;j<vUnodeList.size();j++)
            {
                ElleGetUnodeAttribute(vUnodeList[j],&dTest,U_ATTRIB_C);
                if (dTest>=0.0)
                {     
                    ElleGetUnodePosition(vUnodeList[j],&cUnodeXY);	
                    ElleCoordsPlotXY(&cRefXY,&cUnodeXY);			  
                    dDist = pointSeparation(&cRefXY,&cUnodeXY);
                    
                    ElleGetUnodeAttribute(vUnodeList[j],&dNewEuler[0],E3_ALPHA);
                    ElleGetUnodeAttribute(vUnodeList[j],&dNewEuler[1],E3_BETA);
                    ElleGetUnodeAttribute(vUnodeList[j],&dNewEuler[2],E3_GAMMA);
                    dTmpEuler[0] += (dNewEuler[0]*dDist);  
                    dTmpEuler[1] += (dNewEuler[1]*dDist);
                    dTmpEuler[2] += (dNewEuler[2]*dDist); 
                    dDistTotal += dDist;                       
                }             
            }
            // to be on the save side: 
            if (dDistTotal<=0.0) // only one unode in this flynn, which is the swept one, reset to old value
            {
                for (int ii=0;ii<3;ii++) dNewEuler[ii] = dEulerOld[ii];
                printf("WARNING (FS_ReAssignAttribsSweptUnodes):\nSetting ");
                printf("new orientation of swept unode ");
                printf("%u to old value\n",iUnodeID);
                dDistTotal = 1.0;
            }
            else
            {
                printf("WARNING (FS_ReAssignAttribsSweptUnodes):\nSetting ");
                printf("new orientation of swept unode ");
                printf("%u to flynn %u mean value\n",iUnodeID,iFlynnId);
            }
            dNewEuler[0] = dTmpEuler[0]/dDistTotal;
            dNewEuler[1] = dTmpEuler[1]/dDistTotal;
            dNewEuler[2] = dTmpEuler[2]/dDistTotal;
            //ElleSetUnodeAttribute(iUnodeID,U_ATTRIB_B,-1.0);
        }
    }
    /* Set freshly determined euler angles and remaining attributes*/
    ElleSetUnodeAttribute(iUnodeID,E3_ALPHA, dNewEuler[0]);
    ElleSetUnodeAttribute(iUnodeID,E3_BETA, dNewEuler[1]);
    ElleSetUnodeAttribute(iUnodeID,E3_GAMMA, dNewEuler[2]); 
    
    // Here: leave existing dislocation density
    //ElleSetUnodeAttribute(iUnodeID,U_DISLOCDEN,dDensityMin);
    ElleSetUnodeAttribute(iUnodeID,U_ATTRIB_C, double(iFlynnId));
    vUnodeList.clear();  
}

double FS_GetROI(int iFact)
{
    /*
     * FS: The product of boxwidth + boxheight will not remain constant in a 
     * pure shear simulation and unode distances change. Hence, a more accurate 
     * determination of ROI is used here using not sqrt(1.0 / (...), but 
     * sqrt(width*height / (...)
     * --> FOR THIS APPROACH THE BOX SHOULD NOT BE A PARALLELOGRAM ETC
     *  --> Added: Well actually it doesn't matter if the box has simple shear 
     *             component (if it is a parallelogram) --> the height and width
     *             as calculated here will stay the same anyway 
     */    
    CellData unitcell;
    ElleCellBBox(&unitcell);
    int iMaxUnodes = ElleMaxUnodes(); 
    double dRoi = 0.0;
    double dBoxHeight = 0.0;
    double dBoxWidth = 0.0;
    double dBoxArea = 0.0;
    dBoxHeight = unitcell.cellBBox[TOPLEFT].y - unitcell.cellBBox[BASELEFT].y;
    dBoxWidth = unitcell.cellBBox[BASERIGHT].x - unitcell.cellBBox[BASELEFT].x;
    
    if (dBoxWidth>dBoxHeight)
        dBoxArea = dBoxWidth*dBoxWidth;
    else
        dBoxArea = dBoxHeight*dBoxHeight;
        
    //dBoxArea = dBoxHeight*dBoxWidth;    
	dRoi = sqrt( dBoxArea/ (double) iMaxUnodes / 3.142 ) * (double)iFact;	// aprox. limit at 2nd neighbours by using iFact = 3
    
    if (dRoi == 0)
    {
        printf("ERROR (FS_GetROI): Roi is zero!\n");
        return (0);
    }
    
    return (dRoi);
}

void TestSomething()
{
    Coords cUnodePrevXY[3];
    Coords cUnodeNewXY[3];
    cUnodePrevXY[0].x = 3;
    cUnodePrevXY[1].x = 3;
    cUnodePrevXY[2].x = 2;
    cUnodePrevXY[0].y = 2.5;
    cUnodePrevXY[1].y = 1.4;
    cUnodePrevXY[2].y = 1.4;
    
    cUnodeNewXY[0].x = 2;
    cUnodeNewXY[1].x = 1;
    cUnodeNewXY[2].x = 4;
    cUnodeNewXY[0].y = 7;
    cUnodeNewXY[1].y = 6;
    cUnodeNewXY[2].y = 2;
    
    double dX,dY;
    double dPosGrad[4];
    
    Solve4PosGradTensor(cUnodePrevXY,cUnodeNewXY, &dX, &dY, dPosGrad);
    
    // Display output:
    printf("dX  = %f\n",dX);
    printf("dY  = %f\n",dY);
    printf("F11 = %f\n",dPosGrad[0]);
    printf("F12 = %f\n",dPosGrad[1]);
    printf("F21 = %f\n",dPosGrad[2]);
    printf("F22 = %f\n",dPosGrad[3]);
    
}
