#ifndef FS_startGBS_h
#define FS_startGBS_h

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
#include "check.h"

#define PI 3.141592654

using namespace std;

/* 
 * Some variables
 */
int iUnodePhaseAttrib = U_VISCOSITY;

/*
 * Function declaration
 */
int InitThisProcess();
int ProcessFunction();
    bool CheckFile();
    void GetBoxInfo(double *dArea,double *dWidth,double *dHeight,double *dSSOffset);
    int ReSpaceBnodesTidy(double spacing);
    int FindNearUnodesToBnode(int iBnode,double dROI,vector<int> &vUnodeIds);
    void TmpStoreProps(int iUnode,const char *cFname);

#endif
