#ifndef FS_plot_strainanalysis_h
#define FS_plot_strainanalysis_h

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
 * More global variables for strain (updated by looping through all unodes
 */
double dVorticity = 0.0, dVorticityNumber = 0.0;
double dRatio = 0.0, dEmax = 0.0, dEmin = 0.0, dStretchDir = 0.0;
double dDilation = 0.0;

int iMaxRows=0,iMaxCols=0;  
/*
 * Prepare a structure for point grid (pixels)
 */
typedef struct
{
    double x; 
    double y; 
    double vorticity;
    double vorticity_number;
    double emin;
    double emax;
    double ratio_maxmin;
    double stretch_dir;
    double dilation;
} PlotPoints;

/*
 * Function declaration
 */
int InitThisProcess();
int ProcessFunction();
    int Check();
    int PreparePointGrid(int iRes);
        void GetBoxInfo(double *dWidth,double *dHeight);
    void GetUnodeStrainData(int iUnode);   
    void AssignNearestUnodeProps(PlotPoints *pPoint);
    void WritePlotPointArray(int iPropertyID);
    void ResetUnodes();

#endif
