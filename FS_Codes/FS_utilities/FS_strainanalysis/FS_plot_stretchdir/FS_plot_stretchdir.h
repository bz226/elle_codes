#ifndef FS_plot_stretchdir_h
#define FS_plot_stretchdir_h

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

int InitThisProcess();
int ProcessFunction();
    void PlotUnodeData(int iID,double dLineLength,FILE *fPSout);

void Startps(FILE *psout,double dLineColor[3],double dLineWidth);
void Endps(FILE *psout);
void GetBoxInfo(double *dWidth,double *dHeight);

void FindColsRows(int *iMaxRows,int *iMaxCols);

#endif
