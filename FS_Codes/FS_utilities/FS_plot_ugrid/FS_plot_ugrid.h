#ifndef FS_plot_ugrid_h
#define FS_plot_ugrid_h

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

#define PI 3.141592654

/*
 * indices for User data values for this process
 */
const int uPlottype=0; 
const int uGridSpacing=1;  
const int uScale=2;    
const int uThresholdDist=3;  
const int uExcludePhase=4;     

/*
 * Function declaration
 */
int InitUgrid();
int Ugrid();
    void UpdateCell();
    void Startps(FILE *psout,double dLineColor[3],double dLineWidth);
    void PlotHorizontalLines(FILE *psout,int iGridSpacing,int iExcludePhase);
    void PlotVerticalLines(FILE *psout,int iGridSpacing,int iExcludePhase);
    void PlotDiagonalLines(FILE *psout,int iGridSpacing,int iExcludePhase);
        void DrawLine(FILE *psout,Coords cUnodeXY,Coords cNextUnodeXY,int iColNextUnode,int iRowNextUnode, int iExcludePhase);
    void Endps(FILE *psout, double dBoxWidth,double dBoxHeight);
    
// Some auxiliary functions
int CheckUnodeDist(Coords cUnodeXY,Coords cNextUnodeXY);

// Maybe also helpful at some stage
int UnodeRowCol2ID(int iCol,int iRow);
void UnodeID2RowCol(int iUnodeID, int *iCol, int *iRow);

#endif
