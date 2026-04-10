#ifndef FS_postprocess_strainanalysis_h
#define FS_postprocess_strainanalysis_h

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

#define PI 3.14159265358979323846264338
#define DTOR PI/180
#define RTOD 180/PI

/*
 * Function declaration
 */
int InitThisProcess();
int ProcessFunction();
    int PrepareProcess(const char *cInroot);
        int StoreIniData(const char *cFilename);
    void WriteData2IncrElleFile(int iStep,const char *cFilename,const char *cWritePath);
    void WriteData2TxtFile();
    
bool fileExists(const char *cFilename);

#endif
