#include "FS_statistics.h"
using namespace std;

void WriteUnodeAttribute2File(char *fFilename,int iAttribute, int iExcludePhase)
{
    int iPhaseAttrib = VISCOSITY;
    /*
     * Information about what this function calculates:
     * 
     * Takes a value from the Attribute iAttribute and writes it in a textfile, 
     * only phases that are not == iExcludePhase are regarded
     * ATTENTION: Phase Information has to be stored in flynns
     * --> Function loads all unodes in a flynn of the correct phase and stores 
     * their data in textfile
    */
    fstream fOutfile;
    fOutfile.open(fFilename,fstream::out | fstream::trunc);
    
    // Using "trunc" all old entries will be overwritten:
    fOutfile << "UnodeID\tUnodeAttribute" << endl; // top line in textfile
    
    fOutfile.close();
    
    // Now the remaining data needs to be appended ("app"):
    fOutfile.open(fFilename,fstream::out | fstream::app);
    
    int iMaxFlynns = ElleMaxFlynns();
    int iUnodeID = 0;
    int iNumbUnodesInFlynn = 0;
    double dTempPhase = 0;
    double dAttribValue = 0;
    
    vector<int> viUnodesInFlynn;
    
    for (int j=0;j<iMaxFlynns;j++)
    {
        if (ElleFlynnIsActive(j))
        {
            /*
             * Check if flynn's phase is excluded, if not store it's unodes in 
             * vector
             */
            ElleGetFlynnRealAttribute(j,&dTempPhase,iPhaseAttrib);
            
            viUnodesInFlynn.clear();
            
            /*
             * If the phase is not excluded, store unodes in vector, otherwise
             * the size of the vector will be zero and the following for-loop
             * will automatically not be used (it runs from 0 to 0).
             */
            if ((int)dTempPhase!=iExcludePhase)
            {
                ElleGetFlynnUnodeList(j,viUnodesInFlynn);
            }
            
            iNumbUnodesInFlynn = viUnodesInFlynn.size(); // the number of unodes in the flynn
    
            for (int i=0;i<iNumbUnodesInFlynn;i++)
            {
                iUnodeID = viUnodesInFlynn.at(i);

                ElleGetUnodeAttribute(iUnodeID,&dAttribValue,iAttribute);
                fOutfile << iUnodeID << "\t" << dAttribValue << endl;                
            }        
        }
    
    }
    
    fOutfile.close();    
}

/* 
 * This function writes the grain size of flynns to attribute F_ATTRIB_A:
 * If you wish type s.th. else than 0 for the "iPhase" input, this way only 
 * grain sizes for respective phase will be used, other phase flynns' grain area
 * will be set to 0 in F_ATTRIB_A
 * 
 * Results will be stored as ellefile "GrainAreas2Attrib.elle"
 */
void WriteGrainSize2Attribute(int iPhase)
{
    if (!ElleFlynnAttributeActive(F_ATTRIB_A))
    {
        ElleInitFlynnAttribute(F_ATTRIB_A);
        ElleSetDefaultFlynnRealAttribute(0.0,F_ATTRIB_A);
    }
    double dArea = 0.0;
    for (int flynn=0;flynn<ElleMaxFlynns();flynn++)
    {
        if (ElleFlynnIsActive(flynn))
        {
            if (iPhase==0)
            {
                dArea=ElleRegionArea(flynn); 
                ElleSetFlynnRealAttribute(flynn,dArea,F_ATTRIB_A);
            }
            else
            {
                // Check phase
                double dPhase = 0.0;
                
                ElleGetFlynnRealAttribute(flynn,&dPhase,VISCOSITY);
                if ((int)dPhase==iPhase)
                {
                    dArea=ElleRegionArea(flynn); 
                    ElleSetFlynnRealAttribute(flynn,dArea,F_ATTRIB_A);
                }
            }
        }
    }
    ElleWriteData("GrainAreas2Attrib.elle");
}

/* 
 * Similar than "WriteGrainSize2Attribute": Writes the ice grain sizes in a 
 * textfile
 * 
 * Results will be stored as ellefile "IceGrainAreas.txt"
 * 
 * If iMode == 0: Write for every grain, overwrite old files
 * if iMode == 1: Write only mean grain area for this phase, add to existing file
 * 
 * if iOutputMM == 1: The grain size will not be stored in Elle units, but mm²
 */
void WriteGrainSize2Textfile(int iIcePhase, int iMode, int iOutputMM)
{
    double dUnitArea = ElleUnitLength()*ElleUnitLength();
    if (iMode==0)
    {
        fstream fMyFile;
        fMyFile.open ( "IceGrainAreas.txt", fstream::out | fstream::trunc);
        double dArea = 0.0;
        for (int flynn=0;flynn<ElleMaxFlynns();flynn++)
        {
            if (ElleFlynnIsActive(flynn))
            {
                if (iIcePhase==0)
                {
                    dArea=ElleRegionArea(flynn); 
                    if (iOutputMM==1)
                        fMyFile << dArea*dUnitArea*1e6 << endl;
                    else
                        fMyFile << dArea << endl;                    
                }
                else
                {
                    // Check phase
                    double dPhase = 0.0;
                    
                    ElleGetFlynnRealAttribute(flynn,&dPhase,VISCOSITY);
                    if ((int)dPhase==iIcePhase)
                    {
                        dArea=ElleRegionArea(flynn); 
                        if (iOutputMM==1)
                            fMyFile << dArea*dUnitArea*1e6 << endl;
                        else
                            fMyFile << dArea << endl;    
                    }
                }
            }
        }
        fMyFile.close();
    }
    else
    {
        fstream fMyFile;
        fMyFile.open ( "MeanIceGrainAreas.txt", fstream::out | fstream::app);
        double dAreaSum = 0.0;
        int iIceGrainCounter=0;
        for (int flynn=0;flynn<ElleMaxFlynns();flynn++)
        {
            if (ElleFlynnIsActive(flynn))
            {
                if (iIcePhase==0) 
                {
                    dAreaSum+=ElleRegionArea(flynn); 
                    iIceGrainCounter++;
                }
                else
                {
                    // Check phase
                    double dPhase = 0.0;
                    
                    ElleGetFlynnRealAttribute(flynn,&dPhase,VISCOSITY);
                    if ((int)dPhase==iIcePhase) 
                    {
                        dAreaSum+=ElleRegionArea(flynn); 
                        iIceGrainCounter++;
                    }
                }
            }
        }
        if (iOutputMM==1)
            dAreaSum = dAreaSum*dUnitArea*1e6;
            
        fMyFile << dAreaSum/(double)iIceGrainCounter << endl;
        fMyFile.close();
    }
}

/*
 * iPhase: Set to the value of flynn VISCOSITY of the phase of interest
 */
void WritePhaseGrainSizeStatistics(char *fFilename,int iPhase)
{
    int iPhaseAttrib = VISCOSITY;
    /*
     * Information about what this function calculates:
     * 
     * Number of ice grains: By loop through all flynns and checking their phase
     * Area fraction of ice: Area of ice / total box area
     * Mean Ice Grain Area: Area of ice / number of ice flynns (*UnitArea)
     * Mean diameter of grains (circular grains): sqrt(Area/pi) (*UnitLength)
     * Mean diameter of grains (circular grains): sqrt(Area) (*UnitLength)
     * Perimeter ratio --> See: Jarvis (1973), Hamann et al. (2007), Weikusat et al. (2009b), Binder (2014)
    */
    fstream fInfoFile;
    fstream fOutfile;
    
    double dUnitLength = ElleUnitLength();
    double dUnitArea = dUnitLength*dUnitLength;
    double dBoxAreaTotal = GetTotalBoxArea();
    double dPerimeterSum = 0.0;
    
    double dAreaIceTotal = 0.0;
    double dAreaFractionIce = 0.0;
    double dMeanIceGrainArea = 0.0;
    double dMeanDiamCirc = 0.0;
    double dMeanDiamSqr = 0.0;
    double dDiamRatio = 0.0;
    double dMeanPerimeterRatio = GetPerimeterRatiosAllFlynns(0,iPhase);
    
    double dMeanPerimeter = 0.0;
    double dMeanRoundFact = 0.0;
    int iNumGrainsIce = 0;
    
    // Temporarily used variables:
    double dFlynnPhase = 0.0;
    
    /*
     * Write info file about what the output file contains:
     */
    fInfoFile.open ("GrainSizeStats_Info.txt", fstream::out | fstream::trunc);
    fInfoFile << "Information about phase with F_VISCOSITY: " << 
                 iPhase << ". Data stored in columns in file \"" << 
                fFilename << "\":\n" << endl;
    fInfoFile << "Col. 1: Number of Grains\n" 
              << "Col. 2: Area Fraction of Phase\n"
              << "Col. 3: Mean Grain Area\n"
              << "Col. 4: Mean Grain Diameter Assuming Circular Grains\n"
              << "Col. 5: Mean Grain Diameter Assuming Square Grains\n" 
              << "Col. 6: Ratio: Circular Grain Diameter / Square Grain Diameter\n"
              << "Col. 7: Mean Perimeter Ratio\n"
              << "\n-ALL DATA IN METER OR SQUAREMETER-" << endl;
    
    fInfoFile.close();
    
    /*
     * Prepare data:
     */
    for (int i=0;i<ElleMaxFlynns();i++)
    {
        if (ElleFlynnIsActive(i))
        {
            // See if the flynn in an ice flynn:
            ElleGetFlynnRealAttribute(i,&dFlynnPhase,iPhaseAttrib);
            
            if (iPhase==(int)dFlynnPhase)
            {
                dAreaIceTotal += ElleRegionArea(i);
                iNumGrainsIce ++;
                dPerimeterSum += GetFlynnPerimeter(i);
            }
            
        }        
    }
    
    /*
     * Calculate data:
     */
    dAreaFractionIce = (dAreaIceTotal / dBoxAreaTotal) ;
    dMeanIceGrainArea = (dAreaIceTotal / iNumGrainsIce) * dUnitArea;
    dMeanDiamCirc = sqrt(dMeanIceGrainArea/PI);
    dMeanDiamSqr = sqrt(dMeanIceGrainArea);
    dDiamRatio = dMeanDiamCirc/dMeanDiamSqr;
    
    dMeanPerimeter = 0.01; // Has to be in real units since mean diameter is also in real units
    dMeanRoundFact = 4*PI*( (dMeanDiamCirc/2)/(dMeanPerimeter*dMeanPerimeter) );
    
    /*
     * Write data to file
     */
    fOutfile.open(fFilename,fstream::out | fstream::app);
    
    fOutfile << iNumGrainsIce << " "; // Write Number of Ice Grains
    fOutfile << dAreaFractionIce << " "; // Write Area Fraction of Ice
    fOutfile << dMeanIceGrainArea << " "; // Write Mean Ice Grain Area
    fOutfile << dMeanDiamCirc << " "; // Write Mean Ice Grain Diameter Assuming Circular Grains
    fOutfile << dMeanDiamSqr << " "; // Write Mean Ice Grain Diameter Assuming Square Grains
    fOutfile << dDiamRatio << " "; // Write Ratio: Circular Grain Diameter / Square Grain Diameter
    fOutfile << dMeanPerimeterRatio; // Perimeter Ratio
    fOutfile << endl;
    
    fOutfile.close();    
}

/*
 * This function is writing two output files: A textfile with grain areas of
 * grains TOUCHING the "neighbour phase" (i.e. air) and those that do not 
 * touch the neighbour phase (do not touch bubbbles)
 */
void WriteGrainSizeStatisticsNeighbourSensitive(int iNbPhase)
{
    int iPhaseAttrib = VISCOSITY;

    fstream fOutFileNoTouch;
    fstream fOutFileTouch;
    
    double dUnitArea = ElleUnitLength()*ElleUnitLength();
    
    // Temporarily used variables:
    double dFlynnPhase = 0.0;
    double dAreaTmp = 0.0;
    vector<int> vFlynnNbs;
    bool bNbPhaseFlynnFound;
    
    /* Open output files*/
    fOutFileTouch.open ("IceAreasBubblesNbs.txt", fstream::out | fstream::trunc);
    fOutFileNoTouch.open ("IceAreasIceNbs.txt", fstream::out | fstream::trunc);
    /*
     * Loop through flynns with phase != iNbPhase, get their area and store it
     * in the correct output file:
     */
    for (int i=0;i<ElleMaxFlynns();i++)
    {
        if (ElleFlynnIsActive(i))
        {
            /* See if the flynn in an ice flynn:*/
            ElleGetFlynnRealAttribute(i,&dFlynnPhase,iPhaseAttrib);
            
            /* Check if this flynn is a flynn of the phase of interest*/
            if ((int)dFlynnPhase!=iNbPhase) 
            {
                dAreaTmp = ElleRegionArea(i);
                dAreaTmp *= dUnitArea;
                
                /* See if this flynn has a flynn as neighbour that is of the
                 * neighbour phase */
                GetFlynnNeighbours(i,vFlynnNbs);
                bNbPhaseFlynnFound = false;
                for (int j=0;j<vFlynnNbs.size();j++)
                {
                    if (ElleFlynnIsActive(vFlynnNbs[j]))
                    {
                        /* Check phase */
                        ElleGetFlynnRealAttribute(vFlynnNbs[j],&dFlynnPhase,iPhaseAttrib);        
                        if ((int)dFlynnPhase==iNbPhase) 
                            bNbPhaseFlynnFound = true;
                    }                    
                }
                vFlynnNbs.clear();
                
                /* Store area in correct output file*/
                if (bNbPhaseFlynnFound) fOutFileTouch << dAreaTmp << endl;
                else fOutFileNoTouch << dAreaTmp << endl;             
            }
            
        }        
    }
    fOutFileTouch.close();
    fOutFileNoTouch.close();
}

/*
 * FS: Use my own function to get all neighbours of a flynn, it's easier for me
 * like this because this outputs a vector<int> and not a list...lists are 
 * harder to handle for me
 */
void GetFlynnNeighbours(int iFlynnID,vector<int> &vNbFlynns)
{
    list<int> lNbFlynns;
    ElleFlynnNbRegions(iFlynnID,lNbFlynns);
    int max_loop = lNbFlynns.size();
    for (int i=0;i<max_loop;i++)
    {
        vNbFlynns.push_back(lNbFlynns.front());
        lNbFlynns.pop_front();
    }
    lNbFlynns.clear();
}

/*
 * Functions reads strain rate data from unode attribue "iAttribute" and writes
 * it in a textfile as a NxN matrix, where N is unode dimension (eg 128, 256) 
 * and each point represents the unode position in a regular square grid
 */
void ReadStrainRateDataFromUnode(int iAttribute,double dNormaliseToValue)
{
    int iMaxUnodes = ElleMaxUnodes();
    int iDim = (int)sqrt(iMaxUnodes);
    
    fstream fOutfile;
    fstream fInfoFile;
    fOutfile.open ( "StrainRates.txt", fstream::out | fstream::app); 
    fInfoFile.open ( "StrainRates_Info.txt", fstream::out | fstream::app); 
        
    // 1. Get max. U_ATTRIB_A value and choose value to normalise data to
    double dTempValue = 0.0;
    double dOutValue = 0.0;
    double dMaxValue = 1e-9;
    
    for (int i=0;i<ElleMaxUnodes();i++)
    {
        ElleGetUnodeAttribute(i,iAttribute,&dTempValue);
        if (dTempValue>dMaxValue) dMaxValue=dTempValue;
    }
    
    if (dNormaliseToValue<0)
        dNormaliseToValue = dMaxValue;
        
    fInfoFile << setprecision(12) << "Max. equivalent strain rate: " << dMaxValue << endl;
    fInfoFile << "Image dimensions: " << iDim << " x " << iDim << endl;
    fInfoFile << "All strain rates are normalised to: " << dNormaliseToValue << endl;
    
    // 2. Loop through all unodes and write value in textfile
    
    
    int iUnode = 0;
    for (int row=iDim-1;row>=0;row--)
    {
        for (int col=0;col<iDim;col++)
        {
            // Get correct unode ID from row and col:
            iUnode = row*iDim + col;
            ElleGetUnodeAttribute(iUnode,iAttribute,&dTempValue);
            dOutValue = (dTempValue/dNormaliseToValue);
            fOutfile << dOutValue << " ";
        }
        fOutfile << endl;
    }
    
    fOutfile.close();
    fInfoFile.close();
}

/*
 * Writing grain size for a flynn and the mean of all the unodes in it for an
 * user defined attribute. Also writes the number of unodes in the flynn
 * 
 * Output file therefore has N rows and 4 columns (N=number of flynns)
 * 
 * Columns: FlynnID FlynnArea(mm^2) AttributeMean NumberOfUnodesInFlynn 
 */
void GetGrainSizeAndMeanUnodeAttribute(int iPhase,int iUnodeAttrib)
{    
    // Some checks
    if (!ElleFlynnAttributeActive(VISCOSITY))
    {
        printf("WARNING (GetGrainSizeAndMeanUnodeAttribute):\n");
        printf("Flynn attribute VISCOSITY is not active:\nSetting it to ");
        printf("active with default = phase of interest\n");
        ElleInitFlynnAttribute(VISCOSITY);
        ElleSetDefaultFlynnRealAttribute((double)iPhase,VISCOSITY);
    }
    if (!ElleUnodeAttributeActive(iUnodeAttrib))
    {
        printf("ERROR (GetGrainSizeAndMeanUnodeAttribute):\n");
        printf("This unode attribute is not active, terminating the code\n");
        return;
    }
    
    // STARTING
    
    fstream fOutfile;
    fOutfile.open ( "GrainSizeAndUnodeAttrib.txt", fstream::out | fstream::trunc); 
    fOutfile << "FlynnID FlynnArea(mm^2) UnodeAttributeMean NumberOfUnodesInFlynn" << endl;
    
    vector<int> vFlynnUnodes;
    int iNumbUnodes;
    double dFlynnPhase,dAttrib,dMeanAttrib,dArea;
    double dUnitArea = ElleUnitLength()*ElleUnitLength();
    
    for (int i=0;i<ElleMaxFlynns();i++)
    {
    if (ElleFlynnIsActive(i))
    {
        // Resetting some values to default:
        iNumbUnodes = 0;
        dFlynnPhase = 0.0;
        dAttrib = 0.0;
        dMeanAttrib = 0.0; 
        dArea = 0.0;
        // Check for correct phase:
        ElleGetFlynnRealAttribute(i,&dFlynnPhase,VISCOSITY);
        if (iPhase==(int)dFlynnPhase)
        {
            // Find Unodes in Flynn
            ElleGetFlynnUnodeList(i,vFlynnUnodes);
            iNumbUnodes = vFlynnUnodes.size();
            
            // Get mean of the attribute
            for (int j=0;j<iNumbUnodes;j++)
            {
                ElleGetUnodeAttribute(vFlynnUnodes[j],&dAttrib,iUnodeAttrib);
                dMeanAttrib += dAttrib;                
            }
            if (iNumbUnodes!=0) dMeanAttrib /= iNumbUnodes;
                        
            // Get Flynn area in mm^2
            dArea = ElleRegionArea(i)*dUnitArea*1e6; // * 1e6 to have it in mm^2
            
            // Write the stuff to output file:
            fOutfile << i << " " << dArea << " " << dMeanAttrib << " " 
                     << iNumbUnodes << endl;
                     
            vFlynnUnodes.clear();
            
        } // end of check if phase is the desired one
    } // end of if flynn is active
    }
    
    fOutfile.close();
}

/* Calculate the angle between the positive x-axis and any point Sk. Use
 * anti-clockwise (mathematical) rotation
 *--> Used for convex hull determination, thereore we also need the previous
 * point Si
 */
double PointAngleWithXAxis(Coords cPrevPoint,Coords cPoint)
{
    // Change naming convention:
    Coords Si,Sk;
    Sk.x = cPoint.x;
    Sk.y = cPoint.y;
    Si.x = cPrevPoint.x;
    Si.y = cPrevPoint.y;
    
    double dTheta = 0.0; // will be in radians
    
    // 1.: Check for special cases:
    
    if (Sk.x-Si.x > 0 && Sk.y == Si.y) return (0);// point on pos. x-axis
    if (Sk.x-Si.x < 0 && Sk.y == Si.y) return (PI);// point on neg. x-axis
    if (Sk.y-Si.y > 0 && Sk.x == Si.x) return (PI/2);// point on pos. y-axis
    if (Sk.y-Si.y < 0 && Sk.x == Si.x) return (3*(PI/2));// point on neg. y-axis
    
    // 2. No special case detected: Check for quadrant and determine the angle
    
    if (Sk.x-Si.x > 0 && Sk.y-Si.y > 0) // 1st quadrant
        dTheta = atan(fabs(Sk.y-Si.y)/fabs(Sk.x-Si.x));
        
    if (Sk.x-Si.x < 0 && Sk.y-Si.y > 0) // 2nd quadrant
        dTheta = atan(fabs(Sk.x-Si.x)/fabs(Sk.y-Si.y)) + (PI/2);
        
    if (Sk.x-Si.x < 0 && Sk.y-Si.y < 0) // 3rd quadrant
        dTheta = atan(fabs(Sk.y-Si.y)/fabs(Sk.x-Si.x)) + PI;
        
    if (Sk.x-Si.x > 0 && Sk.y-Si.y < 0) // 4th quadrant
        dTheta = atan(fabs(Sk.x-Si.x)/fabs(Sk.y-Si.y)) + (3*(PI/2));

    return (dTheta);
}

/*
 * Check if any point is in between a rectangle that is built up by two other
 * points -> Necessary for convex hull algorithm
 * Returns 1 if point is INSIDE
 * Returns 0 if point is OUTSIDE
 */
int CheckPointInRect(Coords cPoint, Coords c1stLast, Coords c2ndLast)
{
    int iInRect = 0;
    Coords cRect[4];
    
    // 1st check for special cases:
    if (c1stLast.x == c2ndLast.x && c1stLast.y == c2ndLast.y)
    {
        printf("points equal\n");
        return (0); // no rectangle possible, cPoint will be new hull point
    }
    
    // Prepare the correct rectangle:
    if (c1stLast.x < c2ndLast.x)
    {
        if (c1stLast.y < c2ndLast.y)
        {
            cRect[BASELEFT].x = c1stLast.x;
            cRect[BASELEFT].y = c1stLast.y; 
            cRect[TOPRIGHT].x = c2ndLast.x;
            cRect[TOPRIGHT].y = c2ndLast.y;       
        }
        else // we need to recalculate a little
        {
            cRect[BASELEFT].x = c1stLast.x;
            cRect[BASELEFT].y = c2ndLast.y;
            cRect[TOPRIGHT].x = c2ndLast.x;
            cRect[TOPRIGHT].y = c1stLast.y; 
        }                    
    }
    else
    {
        if (c1stLast.y > c2ndLast.y)
        {
            cRect[BASELEFT].x = c2ndLast.x;
            cRect[BASELEFT].y = c2ndLast.y; 
            cRect[TOPRIGHT].x = c1stLast.x;
            cRect[TOPRIGHT].y = c1stLast.y;  
        }
        else
        {
            cRect[BASELEFT].x = c2ndLast.x;
            cRect[BASELEFT].y = c1stLast.y; 
            cRect[TOPRIGHT].x = c1stLast.x;
            cRect[TOPRIGHT].y = c2ndLast.y;             
        }
    }
    
    // Check if cPoint is in the rectangle
    Coords cPointTmp;
    cPointTmp.x = cPoint.x;
    cPointTmp.y = cPoint.y;
    
    iInRect = EllePtInRect(cRect,4,&cPointTmp);    
    printf("iInRect: %u\n",iInRect);
    return (iInRect);
}

void NormaliseDrivingForces()
{
    double dEsurf,dEstrain,dEstrainNorm;
    double dEsurfMean;
    int iCounter=0;
    
    for (int i=0;i<ElleMaxNodes();i++)
    {
        if (ElleNodeIsActive(i))
        {
            // only for ice-ice boundaries
            if(GetBoundaryType(i)==0)
            {
                ElleGetNodeAttribute(i,&dEsurf,N_ATTRIB_B);
                dEsurfMean += dEsurf;
                iCounter++;
            }
        }
    }
    
    dEsurfMean /= (double)iCounter;
    
    for (int i=0;i<ElleMaxNodes();i++)
    {
        if (ElleNodeIsActive(i))
        {
            // only for ice-ice boundaries
            if(GetBoundaryType(i)==0)
            {
                ElleGetNodeAttribute(i,&dEsurf,N_ATTRIB_B);
                ElleGetNodeAttribute(i,&dEstrain,N_ATTRIB_C);
                
                
                // Normalise Estrain to EsurfMax to see the factor by which Estrain 
                // is higher or lower
                dEstrainNorm = dEsurf/dEsurfMean;
                //dEstrainNorm = dEstrain/dEsurf;
                
                ElleSetNodeAttribute(i,dEstrainNorm,N_ATTRIB_C);    
            }
            else
                ElleSetNodeAttribute(i,0.0,N_ATTRIB_C);    
        }
    }
    
    
}

/*  
 * This one tells you on which boundary type the node sits on in a 2 phase
 * model. 
 * if return = 0: phase1-phase1 (ice-ice)
 * if return = 1: phase1-phase2 (ice-air)
 * if return = 2: phase2-phase2 (air-air)
 */
int GetBoundaryType(int iNode)
{
    int iRgns[3]; 
    double dPhase[3];
    bool bNodeisDouble = false;
    
    ElleRegions(iNode,iRgns);
    
    for (int i=0;i<3;i++)
    {
        if (iRgns[i]!=NO_NB)
        {
            ElleGetFlynnRealAttribute(iRgns[i],&dPhase[i],VISCOSITY); 
        }
        else
        {
            bNodeisDouble = true;
            dPhase[i] = 0; // phase can never be 0
        }                
    }
    
    // Decide for boundary type by summing up the values in dPhase:
    double dSumValue = dPhase[0]+dPhase[1]+dPhase[2];
    
    if (bNodeisDouble)
    {
        // double junction
        if (dSumValue/2 == 1) return 0;
        if (dSumValue/2 == 1.5) return 1;
        if (dSumValue/2 == 2) return 2;
    }
    else
    {
        // triple junction
        if (dSumValue/3 == 1) return 0;
        if (dSumValue/3 < 2 && dSumValue/3 > 1) return 1;
        if (dSumValue/3 == 2) return 2;
        
    }
}

/*
 * Something to play around with:
 */
void StatTests()
{
    
    // Find dislocden for each unode and store it in a textfile 
    // together with the distance to the nearest ice-air bnode
    fstream fMyFile;
    fMyFile.open ( "DislocdenvsBubbleDistance.txt", fstream::out | fstream::app);
    fMyFile << "ID DD Distance" << endl;
    
    Coords cUnode,cBnode;
    double dDist = 1e20;
    double dPhase = 0.0;
    double dDistTest = 0.0;
    double dDislocden = 0.0;
    
    if (!ElleUnodeAttributeActive(U_VISCOSITY))
    {
        printf("Unode attribute U_VISCOSITY in inactive:\n");
        printf("Use FS_flynn2unode_attribute to add it from flynns\n");
        return;
    }
    if (!ElleUnodeAttributeActive(U_DISLOCDEN))
    {
        printf("Unode attribute U_DISLOCDEN in inactive\n");
        return;
    }
    
    for (int unode=0;unode<ElleMaxUnodes();unode++)
    {
        // only use ice unodes:
        ElleGetUnodeAttribute(unode,&dPhase,U_VISCOSITY);
        
        if ((int)dPhase==1)
        {
            ElleGetUnodePosition(unode,&cUnode);
            ElleGetUnodeAttribute(unode,&dDislocden,U_DISLOCDEN);
            dDist = 1e20;
            // search for smallest ice-air node distance
            for (int j=0;j<ElleMaxNodes();j++)
            {
                if (ElleNodeIsActive(j) )
                { 
                    if (GetBoundaryType(j)==1) //ice-air
                    {
                        ElleNodePosition(j,&cBnode);
                        ElleCoordsPlotXY(&cBnode,&cUnode);
                        dDistTest = pointSeparation(&cBnode,&cUnode);
                        if(dDistTest<dDist) dDist = dDistTest;
                    }
                }
            }
            // Write output in textfile:
            fMyFile << unode << " " << dDislocden << " " << dDist*ElleUnitLength() << endl;
        }                
    }
    
    fMyFile.close();
    
    /*!
    // Find normalised strain energy for each bnode and store it in a textfile 
    // together with the distance to the nearest ice-air bnode, of course only 
    // this for ice-ice bnodes
     
     // First of all normalise all driving forces. Relevant data will be in 
     // F_ATTRIB_C
     
    NormaliseDrivingForces();
    
    // Go through all ice-ice bnodes and seach for closest ice-air bnode. Store
    // result in textfile 
    fstream fMyFile;
    fMyFile.open ( "BnodeDrivForcesvsBubbleDistance.txt", fstream::out | fstream::app);
    
    fMyFile << "This data is showing the normalised strain energy for "<< 
            "each bnode (column 2) and its distance (in meters) to the "<<
            "closest ice-air bnode (column 3). First column is ID" << endl;
    fMyFile << "ID Estrain_norm DisttoNextIceAirBnode" << endl;
    Coords cNode,cBubbleNode;
    double dEstrain = 0.0;
    double dDist = 1e20; 
    double dDistTest = 0.0;
    for (int i=0;i<ElleMaxNodes();i++)
    {
        if (ElleNodeIsActive(i) )
        {        
            // only for ice-ice boundaries
            if(GetBoundaryType(i)==0)
            {
                dDist = 1e20; 
                ElleGetNodeAttribute(i,&dEstrain,N_ATTRIB_C);
                ElleNodePosition(i,&cNode);
                // search for smallest ice-air node distance
                for (int j=0;j<ElleMaxNodes();j++)
                {
                    if (ElleNodeIsActive(j) )
                    { 
                        if (GetBoundaryType(j)==1) //ice-air
                        {
                            ElleNodePosition(j,&cBubbleNode);
                            ElleCoordsPlotXY(&cBubbleNode,&cNode);
                            dDistTest = pointSeparation(&cBubbleNode,&cNode);
                            if(dDistTest<dDist) dDist = dDistTest;
                        }
                    }
                }
                // Write output in textfile:
                fMyFile << i << " " << dEstrain << " " << dDist*ElleUnitLength() << endl;
            }
        }
    }
    
    fMyFile.close(); 
    */  
}

/* 
 * Determine the localization factor after Sornette et al. 1993 and Davy et al.
 * 1995 and write it to a textfile for this Ellefile.
 * 
 * Here we essentially use the following equation, simplified from Sornette et
 * al. (1993) since we can assume always the same area for every unode and an 
 * equal timestep:
 * 
 * Factor = 1/num_unodes * (sum of all von Mises strain rates)^2 / sum of all (von Mises strain rate^2)
 * 
 * or:
 * 
 * Factor = 1/num_unodes * (A^2 / B)  
 * 
 * 
 * The output file has two columns: 1st is localization factor and 2nd is how 
 * many unodes were used to calculate it
 */
void GetLocalizationFactor(int iExcludePhase)
{
    printf("Calculating the localization factor\n");
    if (!ElleUnodeAttributeActive(U_ATTRIB_A))
    {
        printf("Error (GetLocalizationFactor): Unode attribute A is not ");
        printf("active, but should store von Mises normalized strain rate\n");
        return;
    }
    if (!ElleUnodeAttributeActive(U_VISCOSITY))
    {
        printf("Error (GetLocalizationFactor): Unode attribute U_VISCOSITY ");
        printf("is missing\n");
        return;
    }
    
    fstream fMyFile;
    
    double dvonMisesEdot = 0.0;
    double dAvalue = 0.0;
    double dBvalue = 0.0;
    double dPhase = 0.0;
    
    double dLocFact = 0.0;
    
    fMyFile.open ( "LocalizationFactors.txt", fstream::out | fstream::app);
    fMyFile << "Loc-Factor Number_unodes: ";
    
    int iCounter = 0;
    
    for (int i=0;i<ElleMaxUnodes();i++)
    {
        ElleGetUnodeAttribute(i,&dPhase,U_VISCOSITY);
        
        if ((int)dPhase != iExcludePhase)
        {
            ElleGetUnodeAttribute(i,&dvonMisesEdot,U_ATTRIB_A);
            dAvalue += dvonMisesEdot;
            dBvalue += dvonMisesEdot*dvonMisesEdot;
            iCounter++;
        }
    }
    
    dLocFact = (dAvalue*dAvalue)/dBvalue;
    dLocFact /= (double)iCounter;
    dLocFact = 1.0-dLocFact; // to have increasing localisation from 0-1
    fMyFile << dLocFact << " " << iCounter << endl;
    
    fMyFile.close();   
    
}


/* This function needs textfiles to be in the directory with the elle file:
 * FlynnIDs.txt
 * This file contains two columns:
 * 1st column: Flynn IDs of all flynns from the previous step
 * 2nd column: Flynn age for the specific flynn from the previous step
 * 
 * It checks for new flynn IDs not present in the last step and sets their age 
 * to 0 "steps" (Where "steps" is the unit of time here).
 * 
 * Furthermore: FlynnIDs.txt is updated
 * 
 * Input integer "iSimulationStep" is only needed to:
 * Check if it is the first simulation step, if yes (if iSimulationStep==1): 
 * Create the file FlynnIDs.txt in the beginning
 * 
 */
void CheckNewFlynns(int iSimulationStep)
{
    printf("Simulation step: %u\n",iSimulationStep);
    /* Some previous checkings */
    /* Initiate the age attribute*/
    if (!ElleFlynnAttributeActive(AGE)) ElleInitFlynnAttribute(AGE);
    
    /* Always set default attribute to 0: Initially we assume all flynns are new */
    ElleSetDefaultFlynnRealAttribute(0.0,AGE);
    
    /* Create initial FlynnIDs.txt file if this is the first step */
    if (iSimulationStep==1) WriteFlynnIDs2File("FlynnIDs.txt");
    
    /* Prepare vectors storing flynn IDs and ages */    
    vector<int> vFlynnIDsPrevStep;
    vector<int> vFlynnAgePrevStep; 
    
    /* Read list of existing flynn IDs and ages in vector */    
    ReadDataFromIDFile("FlynnIDs.txt",vFlynnIDsPrevStep,vFlynnAgePrevStep);
        
    /* Go through the actual flynn list and check which flynns are new: Set their age to 
     * 0. If a flynn was also present in the previous step, set its age 
     * according to the value stored in the textfile and increase it by 1*/
    for (int flynn=0;flynn<ElleMaxFlynns();flynn++)
    {
        if (ElleFlynnIsActive(flynn))
        {
            /* Check if it is in the previous flynn list: */
            int i=0;
            while (i<vFlynnIDsPrevStep.size())
            {
                if (vFlynnIDsPrevStep[i] == flynn) 
                {
                    // flynn existed in prev. step, set its new age:
                    ElleSetFlynnRealAttribute(flynn,(double)(vFlynnAgePrevStep[i]+1),AGE);
                    
                    // end this loop to save time:
                    i = vFlynnIDsPrevStep.size()+1;
                }
                i++;
            }
        }
    } 
    vFlynnIDsPrevStep.clear();
    vFlynnAgePrevStep.clear();
     
     
    /* Update FlynnIDs.txt */
    WriteFlynnIDs2File("FlynnIDs.txt");
    
    /* Write output Elle file*/
    char new_name[50];
    sprintf(new_name,"with_ages%03u.elle",iSimulationStep);
    ElleWriteData(new_name);
}

/* Read data from FlynnIDs.txt file */
int ReadDataFromIDFile(const char *fname,vector<int> &vIDs,vector<int> &vAges)
{
    ifstream datafile(fname);
    if (!datafile) return(OPEN_ERR);
    
    int iFlynnID = 0;
    double dAge = 0.0;;
    
    while (datafile) 
    {
        datafile >> iFlynnID >> dAge;
        vIDs.push_back(iFlynnID);
        vAges.push_back((int)dAge);
    }
    datafile.close();
}

void WriteFlynnIDs2File (const char *filename)
{
    fstream fFile;
    fFile.open ( filename, fstream::out | fstream::trunc);
    double dAge = 0.0;
    for (int i=0;i<ElleMaxFlynns();i++)
    {
        if (ElleFlynnIsActive(i))
        ElleGetFlynnRealAttribute(i,&dAge,AGE);
        fFile << i << " " << dAge << endl;
    }
    
    fFile.close();
}

/* This function is just showing the statistics for any Elle file in terms of 
 * grain sizes: The mean area and the standard deviation will be written in the
 * terminal window --> Only for the phase of interest (iPhase)
 * 
 * If you want to write an output file containing a list of all areas, set
 * iOutputAreas to 1
 */
void ShowStatistics(int iPhase,int iOutputAreas)
{
    fstream fAreas,fFraction;
    double dUnitArea = ElleUnitLength()*ElleUnitLength();
    double dArea = 0.0, dSumAreas = 0.0; 
    double dTmp = 0.0, dMeanArea = 0.0, dStandardDev = 0.0;
    double dPhase = 0.0;
    int iCounter = 0;
    if (iOutputAreas)
    {
        fAreas.open ( "areas_from_elle_file.txt", fstream::out | fstream::app);
        fFraction.open ( "phase_fraction.txt", fstream::out | fstream::app);
    }
    // Determine mean area
    for (int i=0;i<ElleMaxFlynns();i++)
    {
        if (ElleFlynnIsActive(i))
        {
            ElleGetFlynnRealAttribute(i,&dPhase,VISCOSITY);
            
            if ((int)dPhase==iPhase)
            {
                dArea = ElleRegionArea(i);
                if (iOutputAreas) fAreas << dArea << endl;
                dSumAreas+=dArea;
                iCounter++;
            }
        }
    }   
    dMeanArea = dSumAreas/(double)iCounter;
    
    if (iOutputAreas) fFraction << (dSumAreas/GetTotalBoxArea())*100.0 << endl;
    
    // Determine standard deviation
    for (int j=0;j<ElleMaxFlynns();j++)
    {
        if (ElleFlynnIsActive(j))
        {
            ElleGetFlynnRealAttribute(j,&dPhase,VISCOSITY);
            
            if ((int)dPhase==iPhase)
            {
                dTmp += pow( (ElleRegionArea(j)-dMeanArea),2 );                
            }
        }
    }  
    
    dStandardDev = sqrt( dTmp/(double)iCounter );
    
    if (iOutputAreas) 
    {
        fAreas.close();
        fFraction.close();
    }
        
    // Output in terminal:
    cout << "\nData for Elle file: " << ElleFile() << endl;
    printf("Mean grain area:\n\t%f elleunits^2\n\t%f mm^2\n",dMeanArea,dMeanArea*dUnitArea*(1e6));
    printf("Std. grain area:\n\t%f elleunits^2\n\t%f mm^2\n",dStandardDev,dStandardDev*dUnitArea*(1e6));
    
    printf("Percentage of grains of this phase: %f\n",(dSumAreas/GetTotalBoxArea())*100.0);
}

void WritePerimeterRatiosAllFlynns(char *fFilename)
{
    /* Write convex hull perimter of each flynn to a file. 
     * 1st column: Flynn id
     * 2nd column: Perimeter ratio
     */
    fstream fOutfile;
    fOutfile.open(fFilename,fstream::out | fstream::app);
    double dPerimeterRatio=0.0;
    
    for (int flynn=0;flynn<ElleMaxFlynns();flynn++)
    {
        if (ElleFlynnIsActive(flynn))
        {
            dPerimeterRatio = GetFlynnConvexHullPerimeter(flynn)/GetFlynnPerimeter(flynn);
            fOutfile << flynn << " " << dPerimeterRatio << endl;
        }
    }
    
    fOutfile.close();        
}

/* 
 * Prepare a file to plot a difference image by subtraction of unode attributes:
 * 
 * iAttrib1 - iAttrib2 = iAttrib3
 * 
 * Please make sure all attributes are active in the Elle file
 */
//void PrepareDifferenceImage(int iAttrib1, int iAttrib2, int iAttrib3)
//{
    //if (!ElleUnodeAttributeActive(iAttrib1) ||
        //!ElleUnodeAttributeActive(iAttrib2) ||
        //!ElleUnodeAttributeActive(iAttrib3)     )
    //{   printf("Error (PrepareDifferenceImage): At least one of the selected ");
        //printf("unode attributes is inactive!\n");
    //}
    
    //double dVal1,dVal2,dVal3;
    
    //for (int i=0;i<ElleMaxUnodes();i++)
    //{
        //ElleGetUnodeAttribute(i,&dVal1,iAttrib1);
        //ElleGetUnodeAttribute(i,&dVal2,iAttrib2);
        
        //dVal3 = dVal1-dVal2;
        
        //ElleSetUnodeAttribute(i,dVal3,iAttrib3);
    //} 
    
    //ElleWriteData("diff_image.elle");      
//}


