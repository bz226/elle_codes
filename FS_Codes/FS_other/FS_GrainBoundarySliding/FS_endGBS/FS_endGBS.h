#ifndef FS_endGBS_h
#define FS_endGBS_h

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
    void UseNearestUnodes();
    void UsePrevAttribs();
    bool IsGBPhase(int iUnodeID,int iGBphase);
    int NearestUnode(int iUnodeID,vector<int> vNbUnodes);
    void ResetUnodeProps(int iReceive,int iDonate);


bool fileExists(const char *cFilename);
#endif
