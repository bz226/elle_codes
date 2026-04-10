#ifndef FS_create_plotlayer_h
#define FS_create_plotlayer_h

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
#include "flynnarray.h"

#define PI 3.141592654

/*
 * indices for User data values for this process
 */
const int UnodeDeletion = 0;  
const int MergeFlynns   = 1;
const int ScaleBox      = 2;
const int Phase2Value   = 3;  
const int Attribute     = 4;

int iPlotLayerAttribute = U_ATTRIB_F;

/*
 * Function declaration
 */
int InitThisProcess();
int ProcessFunction();
int CreatePlotlayer();
int DeleteUnodes();
void ScaleModelBox(double dScaleFactor);
int MergePhaseFlynns(int iPhase);
    int CheckPossibleWrappingFlynn(int iKeepFlynn,int iRemoveFlynn);
void TransferFlynnViscosity2Unodes();
int TransferAttributes(double d2ndPhaseValue,int iAttribute);
int TransferAttributeEuler(double d2ndPhaseValue,int iAttribute);

#endif
