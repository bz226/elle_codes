#ifndef FS_statistics_h
#define FS_statistics_h

/* Check the include list for missing or needless header files:*/
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <vector>
#include <iomanip>
#include <math.h>
#include <time.h>
#include "attrib.h"
#include "convert.h"
#include "nodes.h"
#include "update.h"
#include "error.h"
#include "parseopts.h"
#include "polygon.h"
#include "runopts.h"
#include "file.h"
#include "flynns.h"
#include "interface.h"
#include "init.h"
#include "log.h"
#include "setup.h"
#include "triattrib.h"
#include "unodes.h"
#include "polygon.h"
#include "mat.h"
#include "file.h"
#include "version.h"

//#define PI 3.141592654

/*
 * indices for User data values for this process
 */
const int uMode=0; 
const int uOption=1;     

/*
 * Function declaration - Most important functions
 */
int InitThisProcess();
int ProcessFunction();
void GetFlynnAreaStatistics();
double GetPerimeterRatiosAllFlynns(int iWriteTextFile,int iFlynnPhase);
double GetWorkDoneFromAllOut();
void ReadAllOutData(int iOutputMode, double OutputValue[6]);
void GetSlipSystemActivities(int iExcludePhase, int iBasalAttrib, int iPrismAttrib);
void CodeForFun();

/*
 * Other functions:
 */
double GetTotalBoxArea();
void GetFlynnAreaStatistics();
double GetFlynnPerimeter(int iFlynnID);
double GetFlynnConvexHullPerimeter(int iFlynn);
    double PointAngleWithXAxis(Coords cPrevPoint,Coords cPoint);
    int CheckPointInRect(Coords cPoint, Coords c1stLast, Coords c2ndLast);
double GetFlynnMaxRealAttribute(int iAttribID);
int NumberOfFlynns(void);
int IsNeighbourOfSamePhase(int iFlynn,int iPotentialNeighbour);
std::vector<int> GetNeighboursSamePhase(int iFlynn);
int GetClusters(int iNumPhases, int iNumFlynns, int *iArrayFlynnID, int *iArrayPhaseID);

/*
 * Pre-declare some "global" variables:
 */
//int iPhaseAttrib = VISCOSITY;

/*
 * The special functions in "FS_stat_specialfunctions.cc"
 */
void WriteUnodeAttribute2File(char *fFilename,int iAttribute, int iExcludePhase);
void WriteGrainSize2Attribute(int iPhase);
void WriteGrainSize2Textfile(int iIcePhase,int iMode, int iOutputMM);
void WritePhaseGrainSizeStatistics(char *fFilename,int iIcePhase);
void WriteGrainSizeStatisticsNeighbourSensitive(int iNbPhase);
    void GetFlynnNeighbours(int iFlynnID,std::vector<int> &vNbFlynns);
void WritePerimeterRatiosAllFlynns(char *fFilename);

void ReadStrainRateDataFromUnode(int iAttribute,double dNormaliseToValue);
void GetGrainSizeAndMeanUnodeAttribute(int iPhase,int iUnodeAttrib);
void NormaliseDrivingForces();
int GetBoundaryType(int iNode);
void GetLocalizationFactor(int iExcludePhase);
void ShowStatistics(int iPhase,int iOutputAreas);
   
/* Something to play around with:*/
void StatTests();

/* To determine flynn ages*/
void CheckNewFlynns(int iSimulationStep);
int ReadDataFromIDFile(const char *fname,std::vector<int> &vIDs,std::vector<int> &vAges);
void WriteFlynnIDs2File (const char *filename);



#endif
