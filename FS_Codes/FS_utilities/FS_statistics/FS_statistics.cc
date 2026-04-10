#include "FS_statistics.h"
using namespace std;
int iPhaseAttrib = VISCOSITY;
int iPhaseAttribUnodes = U_VISCOSITY;
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
     * Clear the data structures
     */
    ElleSetRunFunction(ProcessFunction);
    /*
     * Read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) 
    {
        if (err=ElleReadData(infile)) OnError(infile,err);
    }
    /*
     * Now that the Elle file is loaded, the user code could potentially check 
     * if e.g. unodes are in the file in case they are necessary or check for
     * attributes in flynns, bnodes or unodes that are needed in this code
     */
    /*! EXAMPLE:
    if(!ElleUnodesActive()) 
    {
        printf("Error: No unodes in file\n\n");  
        return 0;
    } 
    */
     
}

/* 
 * Anything can now be in the ProcessFunction itself:
 */ 
int ProcessFunction()
{
    int err=0;
    /*
     * Read the input data and store it in the array of type "Userdata" called
     * "userdata":
     */
    UserData userdata;              
    ElleUserData(userdata);    
    
    /* 
     * Check which mode of calculating statistics is wihsed:
     * INFO:
     * iMode == 0: Statistics on Areas
     * iMode == 1: Read von Mises stress and strain rate and stress tensor
     *             from all.out file in the same directory as the elle file
     * iMode == 4: Use the function "CodeForFun", where you can input anything 
     *             you like
     */
    int iMode = (int)userdata[uMode]; // 1st input
    int iOption = (int)userdata[uOption]; // 2nd input
    char cFileName[] = "GrainSizesIce.txt"; // if iMode=2
    
    /*
     * Choose correct function:
     */
    switch (iMode)
    {
        case 0:
            // Read flynn areas from phases (phase ID will be flynn VISCOSITY)
            GetFlynnAreaStatistics();
            break;
        case 1:
            ReadAllOutData(0,0);
            break;
        case 2:
            printf("Writing grain size statistics only for grains with phase ID = %u\n",iOption);
            WritePhaseGrainSizeStatistics(cFileName,iOption); // found in special functions
            break;
        case 3:
            GetSlipSystemActivities(2,U_ATTRIB_D,U_ATTRIB_E);
            break;
        case 4:
            CodeForFun();
            break;
    }
    
    /*
     * Finally the data can potentially saved in a new Elle file:
     */  
    //if (ElleUpdate()) OnError("Error saving the outputfile",err);
    
    return 0;
}

/*
 * Function determines area statistics of flynns and writes them to output file.
 */
void GetFlynnAreaStatistics()
{   
    // For output textfile:
    fstream fDataFile;
   
    double dUnitLength = ElleUnitLength(), dUnitArea = dUnitLength*dUnitLength;
    int iCount = 0;
    int iMaxFlynns = ElleMaxFlynns();
    int iFlynnPhase = 0;
    int iNumPhases = (int)GetFlynnMaxRealAttribute(iPhaseAttrib);
    int iNumFlynns = NumberOfFlynns();
    double dFlynnPhase = 0.0, dBoxAreaTotal = GetTotalBoxArea();    
    
    // Declare and define arrays dynamically:
    int *iNumGrainsPhase=0;
    double *dAreaPhase=0;
    
    iNumGrainsPhase = new int[iNumPhases];
    dAreaPhase = new double[iNumPhases];
        
    // ... the ones for clusters:
    int *iArrayFlynnID=0, *iArrayPhaseID=0;
    
    iArrayFlynnID = new int[iNumFlynns];
    iArrayPhaseID = new int[iNumFlynns];
    
    // Preparations:
    
    if (!ElleFlynnAttributeActive(iPhaseAttrib))
    {
        ElleInitFlynnAttribute(iPhaseAttrib);
        ElleSetDefaultFlynnRealAttribute(1,iPhaseAttrib);
    }
    fDataFile.open ( "Statistics_FlynnArea.txt", fstream::out | fstream::app);
    fDataFile << "\n# # # # # # # # # # # # # # # AREA STATISTICS # # # # # # # # # # # # # # #\n" << endl;  
    fDataFile << "# Elle file: " << ElleFile() << endl;     
    fDataFile << "# Created by FS_statistics: elle version " << ElleGetLibVersionString () << " " <<ElleGetLocalTimeString() << endl; 
    fDataFile << "# All Units in Real Units. Elle Unit Length = " << dUnitLength << "\n" << endl;
    
    // Pre-allocate some arrays to zero:
    
    for (int i=0;i<iNumPhases;i++)
    {
        dAreaPhase[i] = 0.0;
        iNumGrainsPhase[i] = 0;
    }
   
    // Loop through all flynns
    
    iCount=0;
    for (int i=0;i<iMaxFlynns;i++)
    {
        if (ElleFlynnIsActive(i))
        {
            // Get basic area information
            ElleGetFlynnRealAttribute(i,&dFlynnPhase,iPhaseAttrib);
            iFlynnPhase = (int)dFlynnPhase;
            
            dAreaPhase[iFlynnPhase-1] += ElleRegionArea(i);
            iNumGrainsPhase[iFlynnPhase-1] ++;
            
            // Prepare arrays for cluster information
            iArrayFlynnID[iCount]=i;
            iArrayPhaseID[iCount]=iFlynnPhase;
            iCount++;
        }
    }
    
    // Get cluster information (maybe put in separate function?)

    GetClusters(iNumPhases, iNumFlynns, iArrayFlynnID, iArrayPhaseID);
    
    // Print data to file and calculate all remaining area information on the run:
    
    fDataFile << "# # # General information # # #" << endl;
    fDataFile << "# A B C D E F\n"
                 "# A) Number of Phases\n"
                 "# B) Total area of box\n"
                 "# C) Total Number of Flynns\n" 
                 "# D) Total Mean Flynn Area\n" 
                 "# E) Total Mean Flynn Diameter (assuming circular flynns)\n" 
                 "# F) Total Mean Flynn Diameter (assuming square flynns)\n" << endl;
                 
    fDataFile << iNumPhases << " " 
              << dBoxAreaTotal * dUnitArea << " "
              << iNumFlynns << " " 
              << ( dBoxAreaTotal / (double)iNumFlynns ) * dUnitArea << " " 
              << ( sqrt( (dBoxAreaTotal / (double)iNumFlynns) / PI ) * dUnitLength * 2) << " " // *2 for diameter instead of radius
              << ( sqrt(dBoxAreaTotal / (double)iNumFlynns ) * dUnitLength ) << endl;
    
    fDataFile << "\n# # # Phase Specific Information # # #" << endl;
    fDataFile << "# G H I\n"
                 "# G) Phase ID\n"
                 "# H) Total Area of Phase\n"
                 "# I) Area Fraction of Phase (0..1)\n" << endl;
                 
    for (int i=0;i<iNumPhases;i++)
    {
        fDataFile << i+1 << " " 
                  << ( dAreaPhase[i] ) * dUnitArea << " " 
                  << dAreaPhase[i] / dBoxAreaTotal << endl;
    }
    // To temporarily output s.th.:
    //cout << dAreaPhase[1] / dBoxAreaTotal << endl;
    
    fDataFile << "\n# # # Grain Specific Information # # #" << endl;
    fDataFile << "# J K L M N O\n"
                 "# J) Phase ID\n"
                 "# K) Number of Grains in Phase\n"
                 "# L) Grain Fraction of Phase(0..1)\n" 
                 "# M) Phase Mean Grain Area\n" 
                 "# N) Phase Mean Grain Diameter (assuming circular grains)\n" 
                 "# O) Phase Mean Grain Diameter (assuming square grains)\n" << endl;
                 
    for (int i=0;i<iNumPhases;i++)
    {
        fDataFile << i+1 << " " 
                  << iNumGrainsPhase[i] << " " 
                  << (double)iNumGrainsPhase[i] / (double)iNumFlynns << " "
                  << ( dAreaPhase[i] / (double)iNumGrainsPhase[i] ) * dUnitArea << " "
                  << ( sqrt( (dAreaPhase[i] / (double)iNumGrainsPhase[i]) / PI ) ) * dUnitLength *2 << " " // if assuming round bubbles
                  << ( sqrt(dAreaPhase[i] / (double)iNumGrainsPhase[i]) ) * dUnitLength << endl; // if assuming square grains
    }
    
    fDataFile << "\n# ATTETION: CLUSTERS OF GRAINS ARE NOT TAKEN INTO ACCOUNT HERE, THINK ABOUT ADDING A PART LIKE:\n" 
              << "# \"CLUSTER INFORMATION\"" << endl;
              
    fDataFile << "END" << endl; // Maybe necessary to read the file afterwards if it contains statistics of more than 1 elle file
           
    // Free some memory and close datafile:
    
    delete[] iNumGrainsPhase; iNumGrainsPhase=0;
    delete[] dAreaPhase; dAreaPhase=0;
    delete[] iArrayFlynnID; iArrayFlynnID=0;
    delete[] iArrayPhaseID; iArrayPhaseID=0;
    
    fDataFile.close();
    
}

/* 
 * Determine the perimeter ratio of each flynn: Always returning the MEAN
 * perimeter ratio of all flynns.
 * 1st input: Set the flynn attribute in which the perimeter ratio for each
 * flynn shall be stored, set to 0 for NO attribute
 * 2nd input is the phase for which we want the perimeter ratio
 */
double GetPerimeterRatiosAllFlynns(int iPerRatioAttrib,int iFlynnPhase)
{
    double dMeanPerRatio = 0.0; 
    double dPerRatio = 0.0;
    double dPhaseTmp = 0.0;   
    int iMaxFlynns = ElleMaxFlynns();
    int iNumPhaseFlynns = 0;
    
    if (iPerRatioAttrib != 0)
    {
        if (!ElleFlynnAttributeActive(iPerRatioAttrib))
        {
            ElleInitFlynnAttribute(iPerRatioAttrib);
            ElleSetDefaultFlynnRealAttribute(0.0,iPerRatioAttrib);            
        }
    }
    
    /* Go through all flynns and check if they are of the phase of interest and 
     * determine perimeter ratio: */
    for (int flynn=0;flynn<iMaxFlynns;flynn++)
    {
        if (ElleFlynnIsActive(flynn))
        {
            ElleGetFlynnRealAttribute(flynn,&dPhaseTmp,iPhaseAttrib);
            
            if ((int)dPhaseTmp == iFlynnPhase)
            {
                //cout << "flynn " << flynn << endl;
                // Flynn is of the phase of interest, determine perimeter ratio:
                dPerRatio = GetFlynnConvexHullPerimeter(flynn) / GetFlynnPerimeter(flynn);   
                dMeanPerRatio += dPerRatio;            
                iNumPhaseFlynns++;
                
                // Write to flynn attribute if user wants to do this:
                if (iPerRatioAttrib != 0)
                    ElleSetFlynnRealAttribute(flynn,dPerRatio,iPerRatioAttrib);
            }
            else
            {
                // Set all other flynns to have a ratio of zero
                if (iPerRatioAttrib != 0)
                    ElleSetFlynnRealAttribute(flynn,0.0,iPerRatioAttrib);
            }                                   
        }
    }
    
    dMeanPerRatio /= (double)iNumPhaseFlynns;
    
    return (dMeanPerRatio);
}

/*
 * Returns the perimeter of a flynn in ELLE UNITS!*/
double GetFlynnPerimeter(int iFlynnID)
{
    /*
     * Returns the perimeter of a single flynn. 
     * If the flynn is inactive, the perimeter 0 will be returned
     */    
    double dFlynnPerimeter = 0.0;
    int iNumbBnodesFlynn = 0;
    vector<int> vBnodeIDs;
    Flynn thisFlynn; // "thisFlynn" is now an instance of the flynn class 
    Coords cThisNodeXY, cNextNodeXY;
    
    if (!ElleFlynnIsActive(iFlynnID))
        return (0.0);
    
    /*
     * Get all bnodes of the flynn
     */
    thisFlynn.flynnNodes(iFlynnID,vBnodeIDs);
    iNumbBnodesFlynn = vBnodeIDs.size();
    
    for (int i=0; i< ( iNumbBnodesFlynn-1 ); i++) // until number of nodes - 1 is enough, because always the following node (i+1) will be accessed as well
    {
        if (ElleNodeIsActive(i))
        {
            // Get position of this and the following node
            ElleNodePosition(vBnodeIDs.at(i),&cThisNodeXY);
            ElleNodePosition(vBnodeIDs.at(i+1),&cNextNodeXY);
            ElleCoordsPlotXY(&cThisNodeXY,&cNextNodeXY);
            // Update perimeter:
            dFlynnPerimeter += pointSeparation(&cThisNodeXY,&cNextNodeXY);
        }
    }
    
    /* Last step: Add distance from last to first bnode: */
    ElleNodePosition(vBnodeIDs.at(iNumbBnodesFlynn-1),&cThisNodeXY);
    ElleNodePosition(vBnodeIDs.at(0),&cNextNodeXY);
    ElleCoordsPlotXY(&cThisNodeXY,&cNextNodeXY);
    // Update perimeter:
    dFlynnPerimeter += pointSeparation(&cThisNodeXY,&cNextNodeXY);
    vBnodeIDs.clear();
    
    return (dFlynnPerimeter);
}

/* 
 * Get the convex hull perimeter of a given flynn using the algorithm by 
 * Jarvis, 1973, also used in Binder (2014) (PhD thesis)
 * -> Returning the convex perimeter in ELLE UNITS!!
 */
double GetFlynnConvexHullPerimeter(int iFlynn)
{    
    int iNode=0,i1stNode=0,iFoundNode=-1, iFoundIndex = -1;
    int iFoundNodeBefore = -1, iFoundIndexBefore = 0;
    double dDist = 0.0, dConvPer = 0.0;  
    double dAngle,dMinAngle,dAngleBefore = -1;  
    vector<int> vBnodes;
    Coords cPointTmp, cPoint, cLastPoint; // Storing the actual and previous node on hull
    
    //fstream fFlynn;// Just for debugging
    //fstream fHull; // Just for debugging
    //fFlynn.open ( "flynnXY.txt", fstream::out | fstream::app); // Just for debugging    
    //fHull.open ( "hullXY.txt", fstream::out | fstream::app); // Just for debugging
    
    /* Store all bnodes in vector*/
    ElleFlynnNodes(iFlynn,vBnodes);
    if (vBnodes.size()<=0)
    {
        printf("No bnodes found for flynn %u\n",iFlynn); 
        printf("Returning hull perimeter = 0 m\n"); 
        return (0.0);       
    }
    
    /* STEP 1: 
     * Search for the 1st hull point, which will be the point with the lowest
     * possible y-coordinate: 
     * First pre-allocate first point to 1st unode in vector */
     
    ElleNodePosition(vBnodes[0],&cPointTmp);
    cLastPoint.x = cPointTmp.x;
    cLastPoint.y = cPointTmp.y;
    
    for (int i=0;i<vBnodes.size();i++)
    {
        iNode = vBnodes.at(i);
        ElleNodePosition(iNode,&cPointTmp);             
        ElleCoordsPlotXY(&cPointTmp,&cLastPoint);
        
        /* JUST FOR DEBUGGING: WRITE FLYNN NODE XY TO TEXTFILE:*/
        //fFlynn << cPointTmp.x << " " << cPointTmp.y << endl;
    
        if (cPointTmp.y <= cLastPoint.y) 
        {
            cPoint.x = cPointTmp.x;
            cPoint.y = cPointTmp.y;
            cLastPoint.x = cPoint.x;
            cLastPoint.y = cPoint.y;
            i1stNode = iNode;
        }
    }
    
    iFoundNodeBefore = i1stNode;
    /* JUST FOR DEBUGGING: WRITE HULL NODE XY TO TEXTFILE:*/
    //fHull << cPoint.x << " " << cPoint.y << endl;
    
    /* STEP 2:
     * Always search for the lowest angle node to find the remaining
     * hull points, but make sure that the angles are always increasing */
     
    /* Find index of i1stNode in vBnodes */
    for (int i=0;i<vBnodes.size();i++)
    {
        if (vBnodes[i]==i1stNode) iFoundIndexBefore = i;
    }
    int iNotFoundCounter = 0;
    int counter=0;
    while (iFoundNode!=i1stNode)
    {
        dAngle = 0; 
        dMinAngle = 1e3; // just set to very high value       
        iFoundNode = -1; // initial dummy value      
        /* Loop through all bnodes to find the next hull point */
        for (int i=0;i<vBnodes.size();i++)
        {
            iNode = vBnodes.at(i);
            ElleNodePosition(iNode,&cPointTmp); 
            // determine the last position again to have correct position, not yet corrected for periodic boundaries
            ElleNodePosition(iFoundNodeBefore,&cLastPoint);            
            ElleCoordsPlotXY(&cPointTmp,&cLastPoint);   
            dAngle = PointAngleWithXAxis(cLastPoint,cPointTmp);
            
            /* Define the criteria that make a point a potential new hull 
             * point:*/
                                    
            // 1.1: Angle must be smaller than the previously measured angle
            bool bAngleHigher=false;
            if (dAngle>=dAngleBefore) bAngleHigher=true;
            // 1.2: If the angles are almost zero, we need to allow some 
            // variation to rule 1.1 and adjust dMinAngle again to dummy value
            if (fabs(dAngle-dAngleBefore)<1e-10) 
                bAngleHigher=true;
                
            // 2.: Angle must be the smallest one measured so far
            bool bSmallestAngle=false;
            if (dAngle<=dMinAngle) bSmallestAngle=true;
            
            // 3.: Node should not be the node we found to be the last hull point:
            bool bNodeCorrect=false;
            if (iFoundNodeBefore!=iNode) bNodeCorrect=true;
            
            //if (dAngle<=dMinAngle && 
                //dAngle>=dAngleBefore && 
                //iFoundNodeBefore!=iNode)
            if (bAngleHigher && bSmallestAngle && bNodeCorrect)
            {                   
                dMinAngle = dAngle;
                iFoundNode = iNode;
                iFoundIndex = i;
                cPoint.x = cPointTmp.x;
                cPoint.y = cPointTmp.y;
            }           
        }
        
        if (iFoundNode==-1)
        {
            // Count how often no new point was found:
            iNotFoundCounter ++;            
        }
        else 
        {
            /////* Set all bnodes in between the new found hull point and the 
             ////* previous one that are on phase boundaries as nodes of the convex 
             ////* hull to avoid effects of e.g. bubble shapes in ice perimeter 
             ////* ratio. 
             ////* Set the last phase boundary bnode in between as the ACTUAL new
             ////* hull point 
             ////*/
            ////printf("this index %u \nlast index %u\n\n",iFoundIndex,iFoundIndexBefore); 
            ////printf("this node %u \nlast node %u\n\n",iFoundNode,iFoundNodeBefore); 
            ////int start,end;
            ////if (iFoundIndexBefore<iFoundIndex)
            ////{
                ////start = iFoundIndexBefore;
                ////end = iFoundIndex;
            ////}
            ////else
            ////{
                ////end = iFoundIndexBefore;
                ////start = iFoundIndex;                    
            ////}
            
            /////* Check if we selected the correct interval (only is correct, 
             ////* if 1st node is not in that interval) */
            ////bool bIsCorrect = true;
            ////for (int k=start+1;k<end;k++) 
                ////if (vBnodes[k] == i1stNode) bIsCorrect = false;
            ////if (!bIsCorrect)
            ////{
                ////int tmp = start;
                ////start = end;
                ////end = tmp;
            ////}
            
            /////* Add all nodes in between to the hull that are on phase 
             ////* boundaries, the last one will be the real new hull point */
            ////bool bAlternativeNewHullPointFound = false;
            ////int iMinIndexDist = 1e10;
            ////int iAlternativeNewIndex = -1, iAlternativeNewNode = -1;
            ////for (int k=start+1;k<end;k++)
            ////{
                ////if(GetBoundaryType(vBnodes.at(k))==1)
                ////{
                    ////ElleNodePosition(vBnodes.at(k),&cPointTmp); 
                    ////ElleCoordsPlotXY(&cPointTmp,&cLastPoint); 
                    ////dDist = pointSeparation(&cPointTmp,&cLastPoint); 
                    ////dConvPer += dDist; 

                    /////* This might be a actual hull point for the next iteration
                     ////* of the while loop, if the distance in index to the new 
                     ////* hull point found before is lowest */
                    ////if (abs(k-iFoundIndex)<iMinIndexDist)
                    ////{
                        /////* Use this as the actual new hull point as long as
                         ////* we do not find a more suitable one*/
                        ////iMinIndexDist = abs(k-iFoundIndex);
                        ////bAlternativeNewHullPointFound = true;
                        ////iAlternativeNewIndex = k;
                        ////dAngle = PointAngleWithXAxis(cLastPoint,cPointTmp);
                    ////}
                    ////cLastPoint.x = cPointTmp.x;
                    ////cLastPoint.y = cPointTmp.y;  
                               
                    /////* JUST FOR DEBUGGING: WRITE HULL NODE XY TO TEXTFILE:*/
                    ////fHull << cPointTmp.x << " " << cPointTmp.y << endl;
                ////}                                   
            ////}     
            bool bAlternativeNewHullPointFound=false;
            /* If no other new hull point was found on the phase boundary use
             * the one determined by the regular convex hull algorithm */
            if (!bAlternativeNewHullPointFound)
            {
                /* New point found at lowest angle setting possible: */
                iFoundNodeBefore = iFoundNode;
                iFoundIndexBefore = iFoundIndex;
                dAngleBefore = dMinAngle;
                    
                /* Update perimeter and last point values */
                dDist = pointSeparation(&cPoint,&cLastPoint);
                dConvPer += dDist;
                cLastPoint.x = cPoint.x;
                cLastPoint.y = cPoint.y; 
                            
                /* JUST FOR DEBUGGING: WRITE HULL NODE XY TO TEXTFILE:*/
                //fHull << cPoint.x << " " << cPoint.y << endl;
            }
            //else
            //{
                //iFoundNode = iAlternativeNewNode;
                //iFoundIndex = iAlternativeNewIndex; 
                //iFoundNodeBefore = iFoundNode;
                //iFoundIndexBefore = iFoundIndex;               
            //}
        }
        counter++;
        /* Sometimes we might be stuck in an endless loop, but we do not want 
         * the program run until eternity, so there is a breakoff: Set the peri-
         * meter to the actual flynn perimeter, which will result in a perimeter
         * ratio of 1 unfortunately: Better solutions??
         */
        if (counter>ElleMaxNodes()) 
        {
            dConvPer = GetFlynnPerimeter(iFlynn);
            iFoundNode=i1stNode;
        }
    }
    vBnodes.clear();
    
    //fFlynn.close(); // just for debugging
    //fHull.close(); // just for debugging
    
    return (dConvPer);   
}

/*
 * Compute the work done in the box by deformation (in J/s) from all.out file
 * The work is calculated using the stress and strain rate tensor for the whole 
 * box:
 * W/t = stressij*strainrateij summed for all ij in J/s
 */
double GetWorkDoneFromAllOut()
{
    double dStressTensor[6];
    double dStrainRateTensor[6];
    double dWorkDone = 0.0;
    
    ReadAllOutData(5,dStressTensor);
    ReadAllOutData(6,dStrainRateTensor);
    
    //for (int i=0;i<6;i++) if(dStrainRateTensor[i]==0) dStrainRateTensor[i]=1;
    
    // W/t = stressij*strainrateij summed for all ij in J/s
    dWorkDone = (dStressTensor[0]*dStrainRateTensor[0] + 
                dStressTensor[1]*dStrainRateTensor[1] +
                dStressTensor[2]*dStrainRateTensor[2] +
                2*dStressTensor[3]*dStrainRateTensor[3] +
                2*dStressTensor[4]*dStrainRateTensor[4] +
                2*dStressTensor[5]*dStrainRateTensor[5]);

    printf("Work done: %e\n",dWorkDone);
    
    // OPTIONAL: WRITE IN OUTPUT FILE
    UserData userdata;              
    ElleUserData(userdata);
    
    if ((int)userdata[uOption]==1)
    {
        cout << "Writing textfile: \"WorkDone.txt\"" << endl;
        fstream fFileWorkDone;
        fFileWorkDone.open ( "WorkDone.txt", fstream::out | fstream::app);
        
        fFileWorkDone << scientific << dWorkDone << endl;
           
        fFileWorkDone.close();
    }       
    
    return (dWorkDone);    
}

/*
 * Function reads all.out data and stores it in speparate textfile if -u A B B=1
 * The input integer iOutputMode causes the function to output a double that 
 * contains one of the values that are read. (Array or Scalar)
 * --> When outputting a scalar the value will be stored in iOutputValue[0]
 * iOutputMode = 0: No output
 * iOutputMode = 1: stress field error
 * iOutputMode = 2: strain rate field error
 * iOutputMode = 3: von Mises stress
 * iOutputMode = 4: von Mises strainrate
 * iOutputMode = 5: stress tensor as double[6]
 * iOutputMode = 6: strain rate tensor double[6]
 * iOutputMode = 7: slip system activity: 3 doubles: basal, prism., pyram
 */
void ReadAllOutData(int iOutputMode, double OutputValue[6])
{
    // Data I want to read:
    double dStressFieldErr = 0.0, dStrainRateFieldErr = 0.0;
    double dStressvonMises = 0.0, dStrainRatevonMises = 0.0;
    double dStressTensor[6];
    double dStrainRateTensor[6];
    double dDiffStress = 0.0;
    double dSlipSystemAct[3];
    
    // Other input:
    int iNSlipSystems = 3; // CANNOT BE LARGER THAN 6 FOR THIS CODE AT THE MOMENT SINCE THE OUTPUT ARRAY HAS 6 VALUES
            
    // Read the data in the header of all.out files:
    
    ifstream fDataAllout ("all.out",ios::binary);
	string sLine;
    stringstream sLineStream; // needs #include <sstream> in header 
    char cFirstChar;
    int iActiveLine=0;
    int iDataStart = 0;
    
    if (fDataAllout.is_open()) 
    {
        while ( getline (fDataAllout,sLine) )
        {
            cFirstChar = sLine.at(0);
            if (cFirstChar!='*' && sLine.length()>0)
            {
                iActiveLine++;
                // Get line as stringstream:
                sLineStream << sLine;
                /*
                 * Set position to the start of the data in that line: 
                 * Unfortunately the positions vary in the header of the 
                 * all.out file, therefore another if loop checking for the 
                 * line in all.out is necessary
                 */
                if (iActiveLine == 1 || iActiveLine == 3 )
                {
                    iDataStart += 22;
                }
                else if (iActiveLine == 6)
                {
                    iDataStart += 24;
                }
                else
                {
                    iDataStart += 23;
                }
                
                sLineStream.seekg(iDataStart);
                
                // Pre-allocate temporary data-array and read data:
                double dVal[6];
                sLineStream >> dVal[0] >> dVal[1] >> dVal[2] >> dVal[3] >> dVal[4] >> dVal[5];
                
                /*
                 * HERE: DECIDE WHAT DATA TO SAVE IN ANOTHER VARIABLE AND/OR
                 * WRITE TO TEXTFILE:
                 */
                switch (iActiveLine)
                {
                case 1:
                    dStressFieldErr = dVal[0];
                    dStrainRateFieldErr = dVal[1];
                    break;
                case 2:
                    dStressvonMises = dVal[0];
                    dStrainRatevonMises = dVal[1];
                    //cout << dStressvonMises << " " << dStrainRatevonMises << endl;
                    break;
                case 3:
                    //cout << "Stress Tensor: s11,s22,s33,s23,s13,s12" << endl;
                    for (int i=0;i<6;i++)
                    {
                        dStressTensor[i] = dVal[i];
                        //cout << dStressTensor[i] << " ";
                    }
                    //cout << endl;
                    dDiffStress = sqrt( pow( (dVal[0]-dVal[1])/2 , 2 ) + pow( dVal[5] , 2 ) );
                    //cout << "Dev. Stress: " << dDiffStress << endl;
                    break;
                case 4:
                    for (int i=0;i<6;i++) 
                    {
                        dStrainRateTensor[i] = dVal[i];
                        //cout << dStrainRateTensor[i] << " ";
                    }
                    //cout << endl;
                    break;
                case 8:
                    for (int i=0;i<iNSlipSystems;i++) 
                    {
                        dSlipSystemAct[i] = dVal[i];
                        //cout << dSlipSystemAct[i] << " ";
                    }
                    //cout << endl;
                    break;
                default:
                    break;
                }                
                
                sLineStream.clear();
                
                /*
                 * Set position to the END of the data in that line: 
                 * Unfortunately the positions vary in the header of the 
                 * all.out file, therefore another if loop checking for the 
                 * line in all.out is necessary
                 */
                if (iActiveLine == 1)
                {
                    iDataStart = iDataStart+sLine.length()-22;
                }
                else if (iActiveLine == 6)
                {
                    iDataStart = iDataStart+sLine.length()-24;
                }
                else
                {
                    iDataStart = iDataStart+sLine.length()-23;
                }
                                
                // Stop while loop if reading the header is finished:                  
                if (iActiveLine == 8)
                    break;
            }
        }
        fDataAllout.close();
    }
    
    // OPTIONAL:
    /*
     * Write some of the data in one row in a textfile:
     * HERE: Macroscopic von Mises stress
     */
    UserData userdata;              
    ElleUserData(userdata);
    
    if ((int)userdata[uOption]==1)
    {
        cout << "Writing textfile: \"AllOutData.txt\"" << endl;
        //cout << "Contains: " << endl;
        //cout << "von Mises stress - von Mises strain rate - differential stress - stress field error - strain rate field error" << endl;
        
        fstream fFile;
        fFile.open ( "AllOutData.txt", fstream::out | fstream::app);
        fFile << "SVM DVM diffStress stressFieldErr strainrateFieldErr ";
        fFile << "basalact prismact pyramact s11 s22 s33 s23 s13 s12 ";
        fFile << "d11 d22 d33 d23 d13 d12 ";
        fFile << scientific << dStressvonMises << " " << dStrainRatevonMises 
            << " " << dDiffStress << " " << dStressFieldErr << " " 
            << dStrainRateFieldErr << " ";
        for (int i=0;i<iNSlipSystems;i++) fFile << dSlipSystemAct[i] << " ";
        for (int i=0;i<6;i++) fFile << dStressTensor[i] << " ";
        for (int i=0;i<6;i++) fFile << dStrainRateTensor[i] << " ";
        fFile << endl;
           
        fFile.close();
    }   
    
    
    // See if a value should be returned
    switch (iOutputMode)
    {
    case 1:
        OutputValue[0] = dStressFieldErr;
        break;
    case 2:
        OutputValue[0] = dStrainRateFieldErr;
        break;
    case 3:
        OutputValue[0] = dStressvonMises;
        break;
    case 4:
        OutputValue[0] = dStrainRatevonMises;
        break;
    case 5:
        for (int i=0;i<6;i++) OutputValue[i] = dStressTensor[i];
        break;
    case 6:
        for (int i=0;i<6;i++) OutputValue[i] = dStrainRateTensor[i];
        break;
    case 7:
        for (int i=0;i<6;i++) 
         {
             /* Fill remaining values with -1 to make sure they're not 
              * mistakenly taken as activities
              */
             if (i<iNSlipSystems) OutputValue[i] = dSlipSystemAct[i];
             else OutputValue[i] = -1;
         }
        break;
    default:
        break;
    }  
}

double GetFlynnMaxRealAttribute(int iAttribID)
{
    int iMaxFlynns = ElleMaxFlynns();
    double dMaxValue = (-1e100), dTempValue = (-1e100);
        
    for (int i=0;i<iMaxFlynns;i++)
    {
        if (ElleFlynnIsActive(i))
        {
            ElleGetFlynnRealAttribute(i,&dTempValue,iAttribID);
            if (dTempValue > dMaxValue)
            {
                dMaxValue = dTempValue;
            }
        }
    }    
    return (dMaxValue);
}

double GetTotalBoxArea()
{
    CellData unitcell;
    ElleCellBBox(&unitcell);
    double dBoxArea = 0.0;
    
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
        dBoxArea += ( (box_xy[i].y+box_xy[i2].y)*(box_xy[i].x-box_xy[i2].x) )/2;   
    }    
    return(dBoxArea);
}

int NumberOfFlynns(void)
{
    int iMaxFlynns = ElleMaxFlynns();
    int iCount=0;
    
    for (int i=0;i<iMaxFlynns;i++)
    {
        if (ElleFlynnIsActive(i))
        {
            iCount++;
        }
    }
    
    return (iCount);
}

int IsNeighbourOfSamePhase(int iFlynn,int iPotentialNeighbour)
{
    /* IsNeighbourOfSamePhase:
     * Returns 1 if iFlynn is a neighbour WITH THE SAME PHASE iPhase of flynn
     * iPotentialNeighbour, otherwise returns 0
     */  
    list<int> lNeighbours;
    int iCheck = 0, iNumNbs = 0, iIdTemp = 0;
    double dPhaseFlynn = 0.0, dPhaseNb = 0.0;
    
    ElleGetFlynnRealAttribute(iFlynn,&dPhaseFlynn,iPhaseAttrib);
    ElleFlynnNbRegions(iFlynn,lNeighbours);
    iNumNbs = lNeighbours.size();
    
    iCheck = 0;
    for (int i=0; i<iNumNbs; i++)
    {
        iIdTemp = lNeighbours.front();
        lNeighbours.pop_front();
        ElleGetFlynnRealAttribute(iIdTemp,&dPhaseNb,iPhaseAttrib);
        if ((int)dPhaseNb == (int)dPhaseFlynn && iIdTemp==iPotentialNeighbour) 
        {
            iCheck=1;
        }
    }
    
    return(iCheck);
    // use ElleFlynnNbRegions
}

vector<int> GetNeighboursSamePhase(int iFlynn)
{
    /* IsNeighbourOfSamePhase:
     * Returns 1 if iFlynn is a neighbour WITH THE SAME PHASE iPhase of flynn
     * iPotentialNeighbour, otherwise returns 0
     */  
    list<int> lNeighbours;
    vector<int> vPhaseNbs;
    int iNumNbs = 0, iIdTemp = 0;
    double dPhaseFlynn = 0.0, dPhaseNb = 0.0;
    
    ElleGetFlynnRealAttribute(iFlynn,&dPhaseFlynn,iPhaseAttrib);
    ElleFlynnNbRegions(iFlynn,lNeighbours);
    iNumNbs = lNeighbours.size();
    
    for (int i=0; i<iNumNbs; i++)
    {
        iIdTemp = lNeighbours.front();
        lNeighbours.pop_front();
        ElleGetFlynnRealAttribute(iIdTemp,&dPhaseNb,iPhaseAttrib);
        if ((int)dPhaseNb == (int)dPhaseFlynn) 
        {
            vPhaseNbs.push_back(iIdTemp);
        }
    }
    return(vPhaseNbs);
}

/* 
 * Function reads mean slip system activities from Elle file (from unodes, as an 
 * input indicate the unode attributes where basal and prismatic activity is 
 * stored). The output tex.out does not indicate pyramidal activity, but this
 * can be calculated since all activities together should be 100% of slip 
 * systems.
 * The output is stored in a textfile if userdata[1]==1
 * To exclude a phase from being read, indicate the phase ID, for no exclusion 
 * type 0.
 */
void GetSlipSystemActivities(int iExcludePhase, int iBasalAttrib, int iPrismAttrib)
{
    printf("Reading slip system activities...\n");
    if (iExcludePhase!=0)
        printf("...not regarding slip systems phase ID = %u\n",iExcludePhase);
    
    /* Some checks */
    if (!ElleUnodesActive())
        printf("Error (GetSlipSystemActivities): No unodes are active.\n");
    if (!ElleUnodeAttributeActive(iBasalAttrib))
        printf("Error (GetSlipSystemActivities): Basal activity attribute not active.\n");
    if (!ElleUnodeAttributeActive(iPrismAttrib))
        printf("Error (GetSlipSystemActivities): Basal prismatic attribute not active.\n");
        
    /* Some checks for phase attributes*/    
    if (!ElleUnodeAttributeActive(iPhaseAttribUnodes) &&
        !ElleFlynnAttributeActive(iPhaseAttrib) &&
        iExcludePhase!=0)
        printf("Error (GetSlipSystemActivities): No attribute for indicating the phase is active.\n");
        
    /* Start reading data */        
    double dBasalAct = 0.0, dSumBasalAct = 0.0, dTmpBasalAct = 0.0;
    double dPrismAct = 0.0, dSumPrismAct = 0.0, dTmpPrismAct = 0.0;
    double dPyramAct = 0.0, dSumPyramAct = 0.0, dTmpPyramAct = 0.0;
    double dPhase = 0.0;
    int iCounter = 0;
        
    for (int i=0;i<ElleMaxUnodes();i++)
    {
        if (iExcludePhase!=0)
        {
            if (!ElleUnodeAttributeActive(iPhaseAttribUnodes))
            {
                // read phase from flynn
                ElleGetFlynnRealAttribute(ElleUnodeFlynn(i),&dPhase,iPhaseAttrib);
            }            
            else
            {
                // read phase from unode itself
                ElleGetUnodeAttribute(i,iPhaseAttribUnodes,&dPhase);
            }
        }
        
        // Read activities if phase is not the excluded phase
        if ((int)dPhase!=iExcludePhase)
        {
            ElleGetUnodeAttribute(i,iBasalAttrib,&dTmpBasalAct);
            ElleGetUnodeAttribute(i,iPrismAttrib,&dTmpPrismAct);    
            dTmpPyramAct = 1.0 - (dTmpBasalAct+dTmpPrismAct);
            dSumBasalAct += dTmpBasalAct;
            dSumPrismAct += dTmpPrismAct;
            dSumPyramAct += dTmpPyramAct;
            iCounter++;
        }
    }      
    
    dBasalAct = dSumBasalAct/(double)iCounter;
    dPrismAct = dSumPrismAct/(double)iCounter;
    dPyramAct = dSumPyramAct/(double)iCounter;
    
    // Write output if user wants to
    UserData userdata;              
    ElleUserData(userdata); 
    
    if ((int)userdata[uOption]==1)
    {
        cout << "Writing textfile: \"SlipSystemAct.txt\"" << endl;
        //cout << "Contains mean activities: " << endl;
        //cout << "Basal - Prismatic - Pyramidal" << endl;
        
        fstream fFile;
        fFile.open ( "SlipSystemAct.txt", fstream::out | fstream::app);
        fFile << "basalact prismact pyramact sum" << " ";
        fFile << scientific << dBasalAct << " " << dPrismAct 
            << " " << dPyramAct << " " << dBasalAct+dPrismAct+dPyramAct << endl;           
        fFile.close();
    } 
}

int GetClusters(int iNumPhases, int iNumFlynns, int *iArrayFlynnID, int *iArrayPhaseID)
{
    //// DELETE LATER!!!
    //time_t t_start;
    //time_t t_end;
    //t_start = clock();
    
    int iErr = 0;
    int iTempID = 0;
    bool bClusterFinished = false, bPhaseFinished = false;
    
    vector< vector< vector <int> > > vClustersPhases;
    vector< vector<int> > vClustersClusters;
    vector<int> vClustersFlynns;
    
    vector<int> vFlynnsPhase; // to store all flynns of one phase temporarily
    vector<int> vPhaseNeighbours;
    
    //for (int i=0;i<iNumFlynns;i++)
    //{
        //cout << iArrayPhaseID[i] <<endl;
    //}
    
    for (int phase=2;phase<=iNumPhases;phase++) //!!!  CHANGE THIS TO int phase = 1
    {
        // Find and store all flynns of that phase
        vFlynnsPhase.clear();
        for (int i=0;i<iNumFlynns;i++)
        {
            if (phase==iArrayPhaseID[i])
            {
                vFlynnsPhase.push_back(iArrayFlynnID[i]);            
            }
        }
        
        // The 1st flynn in vFlynnsPhase will be assigned to the 1st cluster and work as a starting point:
        vClustersFlynns.push_back(vFlynnsPhase.at(0));
        // Now loop through all remaining flynns of that phase to find the ones belonging to that cluster: 
        for (int j=1;j<vFlynnsPhase.size();j++)
        {
            // Get all neighbours of the same phase using GetNeighboursSamePhase:
            vPhaseNeighbours = GetNeighboursSamePhase(vFlynnsPhase.at(j));
            
            // Now: Go through existing Clusters, check if those flynns are 
            // neighbours of at least one of the flynns in the cluster and if
            // they are not already in the cluser. Otherwise do not do anything,
            // a new cluster will be created for them in another iteration
            
        }
    }
    // MAYBE ADD CODE HERE
    //for (int i=0;i<vClustersFlynns.size();i++)
    //{
        //cout<<vClustersFlynns.at(i)<<endl;
    //}
    //cout << endl << vClustersFlynns.size() << endl;
    
    //t_end = clock();
    //double dSeconds = (t_end - t_start) / (double)CLOCKS_PER_SEC;
    //printf("Elapsed time: %e seconds\n",dSeconds);
    
    return (iErr);
}

void CodeForFun()
{
    /*
     * Call function in specialfunctions that writes unode attributes to file:
     */
    
    UserData userdata;              
    ElleUserData(userdata);

    printf("\n# # USING FUN CODE FOR ANY OTHER STATISTICAL OPERATION # #\n\n");
    
    /*! START: REGULAR CODE: KEEP SWITCHED ON FOR STANDARD POST-PROCESSING*/
    
    // HINT FOR FLORIAN FROM FLORIAN: 
    // PLEASE LEAVE THE FOLLOWING IF-ELSE BLOCK UNCOMMENTED WHEN POST-PROCESSING SIMULATIONS
    
    if ((int)userdata[1]<0)
    { 
        printf("Normalising driving forces\n");
        NormaliseDrivingForces();
        ElleWriteData("norm_Estrain.elle");
        return;
    }
    else
    {
        if ((int)userdata[1]==1)
        {
            printf("Writing perimeter ratios\n\n");
            char cFileName[] = "PerimeterRatios.txt";
            WritePerimeterRatiosAllFlynns(cFileName);   
            return;     
        }
    }
    
    GetLocalizationFactor(2);
    
    //WritePhaseGrainSizeStatistics("air_info.txt",2);
    
    //WriteGrainSize2Attribute(1);
    //WriteGrainSize2Textfile(1,0,1);
    //CheckNewFlynns((int)userdata[1]);
    
    //StatTests();

    /*!
    Coords clast,cp;
    
    clast.x = 1.099;
    clast.y = 1.043;
    cp.x = 1.099;
    cp.y = 1.042;
    double dAngle = PointAngleWithXAxis(clast,cp);
    printf("dAngle: %f\n",dAngle*RTOD);
    */
    //printf("Writing grain size statistics only for ice grains\n");
    //char cFileName[] = "GrainSizesIce.txt";
    //WritePhaseGrainSizeStatistics(cFileName,1);
    
    //printf("Writing grain sizes and mean unode attribute in grain\n");
    //printf("All areas in output file in square millimeters\n");
    //int iIcePhase = 1;
    //GetGrainSizeAndMeanUnodeAttribute(iIcePhase,U_ATTRIB_A);
    
    
    //printf("Extracting data from U_ATTRIB_A\n");
    //double dNormaliseToValue = userdata[1];
    //ReadStrainRateDataFromUnode(U_ATTRIB_A,dNormaliseToValue);
    
    
    /*
     * FOR BUBBLE FRACTIONS:
     *
    fstream fFractions;
    double dUnitLength = ElleUnitLength(), dUnitArea = dUnitLength*dUnitLength;
    double dPhase=0.0,dAreas[2];
    double dBoxArea = GetTotalBoxArea();
    fFractions.open ( "PhaseFractions.txt", fstream::out | fstream::app);
    
    dAreas[0]=0;
    dAreas[1]=0;
    
    for (int i=0;i<ElleMaxFlynns();i++)
    {
        if (ElleFlynnIsActive(i))
        {
            ElleGetFlynnRealAttribute(i,&dPhase,VISCOSITY);
            dAreas[(int)dPhase-1]+=ElleRegionArea(i);
        }
    }
    fFractions << dAreas[0]/dBoxArea << " " << dAreas[1]/dBoxArea << endl;
    fFractions.close();
    */
    /*
     * FOR COMPRESSIBILITY TESTS:
     */
    // Write some basic code for a very specific problem:
    // For output textfile:
       
    //fstream fFileRadii, fFileAreas, fileTrackNodePos;
    //double dUnitLength = ElleUnitLength(), dUnitArea = dUnitLength*dUnitLength;
    //double dRadii[4],dAreas[4];
    //int iBubbleFlynnIDs[4];
    
    //iBubbleFlynnIDs[0]=2; // was 11
    ////iBubbleFlynnIDs[1]=14; 
    ////iBubbleFlynnIDs[2]=39;   
    ////iBubbleFlynnIDs[3]=42;   
    
    //// Track Flynn area (circular flynn is assumed): 
    //for (int i=0;i<1;i++)  // was for (int i=0;i<4;i++)
    //{
        //if (ElleFlynnIsActive(iBubbleFlynnIDs[i]))
        //{
            //dAreas[i] = ElleFlynnArea(iBubbleFlynnIDs[i]);
            //dAreas[i] *= dUnitArea;
        
            //// Calculate radii assuming round bubbles:
            //dRadii[i]=sqrt(dAreas[i]/PI);
        //}
        //else // if flynn is not active
        //{
            //dAreas[i] = 0;
            //dRadii[i] = 0;
        //}
    //}
    
    
    //fFileRadii.open ( "BubbleRadiusData.txt", fstream::out | fstream::app);
    //fFileAreas.open ( "BubbleAreasData.txt", fstream::out | fstream::app);
    
    //fFileRadii << dRadii[0] << endl;// " " << dRadii[1] << " " << dRadii[2] << " " << dRadii[3] << endl;
    //fFileAreas << dAreas[0] << endl; // " " << dAreas[1] << " " << dAreas[2] << " " << dAreas[3] << endl;    
        
    //fFileRadii.close();    
    //fFileAreas.close();    
    
    ///*
     //* Track position of a specific bnode:
     //*/
    //int iTrackNode = -124; // set to positive value to use this option
    //if (iTrackNode>=0 && ElleNodeIsActive(iTrackNode))
    //{
        //Coords cNodeXY;
        
        //ElleNodePosition(iTrackNode,&cNodeXY);
                
        //fileTrackNodePos.open ( "TrackNodePos.txt", fstream::out | fstream::app);
        //fileTrackNodePos << cNodeXY.x << " " << cNodeXY.y << endl; 
        //fileTrackNodePos.close(); 
    //}

    /*
     * Call a function in "FS_stat_specialfunctions.cc":
     */
    /*!
    char cFileName[] = "Stats.txt";
    WritePhaseGrainSizeStatistics(cFileName,2);
    */
    



}
