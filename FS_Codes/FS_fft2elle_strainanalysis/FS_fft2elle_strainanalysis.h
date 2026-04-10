#ifndef FS_fft2elle_strainanalysis_h
#define FS_fft2elle_strainanalysis_h

/* Check the include list for missing or needless header files:*/
#include <string>
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include "attrib.h"
#include "nodes.h"
#include "nodesP.h"
#include "update.h"
#include "error.h"
#include "parseopts.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "init.h"
#include "log.h"
#include "setup.h"
#include "triattrib.h"
#include "unodes.h"
#include "polygon.h"
#include "mat.h"
#include "file.h"
#include "check.h"

#define PI 3.141592654
#define MAX_SIZE  0.5  

/*
 * indices for User data values for this process
 */
const int UInput1=0; 
const int UInput2=1;      
const int UInput3=2;     
const int UInput4=3;      
const int UInput5=4;      
const int UInput6=5;      
const int UInput7=6;      
const int UInput8=7;      
const int UInput9=8; 

/* 
 * Some variables
 */
int iMaxBnodes = 0;
int iMaxUnodes = 0;
int iDim = 0;
double dBoxArea = 0.0;
double dBoxWidth = 0.0;
double dBoxHeight = 0.0;

double dXStrain=0.0;
double dYStrain=0.0;
double dShearStrain=0.0;
double dSSOffset=0.0;

// temporarily storing x and y positions, removed after the process
int iAttribTempPosX = U_ATTRIB_D; 
int iAttribTempPosY = U_ATTRIB_E;

/*
 * Function declaration
 */
int InitThisProcess();
int ProcessFunction();

int StrainAnalysis(Coords *cPrevPosition,Coords *cNewPos,bool bWriteFData,int iPointID);
int AnalyseUnodes(bool bWriteFData);
int AnalyseBnodes();
    void UpdateStrainAttrib();
    int ReadUnodePos(const char *fname);
    
    int Find4NearestUnodes(Coords cPoint,int i4NearestUnodeIDs[3],Coords c4NearestUnodesXY[4]);
    void GetBoxInfo(double *dArea,double *dWidth,double *dHeight,double *dSSOffset);
    int LoadBoxIncrStrain(const char *fname);   
    
    void NewCell_CoordsPlotXY(Coords *xy, Coords *prevxy);
    void NewCell_NodeUnitXY(Coords *xy);
    void ResetCell(double xstrain, double ystrain, double zstrain, double offset);
    
    void Solve4PosGradTensor(Coords cUnodePrevXY[3], Coords cUnodeNewXY[3], double *dX, double *dY, double dPosGrad[4]);
    bool fileExists(const char *filename);
    
int SetUnodeAttributesFromFile(const char *fname,std::vector<int> &attribs);
void FS_SetUnodeDD(const char *fname);
void FS_CheckUnodes(); 
    void FS_ReAssignAttribsSweptUnodes(int iUnodeID);
double FS_GetROI(int iFact);
    
void CleanFile();


    
void TestSomething();

#endif
