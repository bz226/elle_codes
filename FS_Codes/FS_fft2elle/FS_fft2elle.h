#ifndef fft2elle_beta_h
#define fft2elle_beta_h

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>
#include <vector>
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
#include "setup.h"
#include "triattrib.h"
#include "unodes.h"
#include "polygon.h"
#include "crossings.h"
#include "check.h"
#include "log.h"
#include "mat.h"

#define PI 3.1415926
#define DTOR PI/180
#define RTOD 180/PI
#define MAX_SIZE  0.5
//using std::ios;
//using std::cout;
//using std::cerr;
//using std::endl;
//using std::ifstream;
//using std::ofstream;
//using std::string;
//using std::vector;

using namespace std;

int InitMoveBnodes(), MoveBnodes();
void SetBnodeStrain(vector<int> & attriblist);
void PositionFromStrain(vector<int> &attriblist);
int SetUnodeAttributesFromFile(const char *fname,vector<int> &attribs);
int SetUnodeStrainFromFile(const char *fname,vector<int> &attribs);
void ResetCell(double xstrain, double ystrain, double zstrain, double ssoffset);
int RemoveNodeAttributes(vector<int> & attriblist);
int LoadDataTemp(const char *fname);
double dd[3][3];
//int SetUnodeAttributeFromNbFlynn(int unum,int flynnid,int attr);
int SetUnodeAttributeFromNbFlynn(int unum,int flynnid,int attr,vector<int> &rlist);
void check_error_UNUSED();

int SetUnodeAttributesFromFile2_UNUSED(const char *fname);
void FS_SetUnodeDD(const char *fname);
void FS_CheckUnodes(); 
    void FS_ReAssignAttribsSweptUnodes(int iUnodeID);

/*!
 * FS: Additions for the passive marker grid:
 */
void FS_SetEunodeStrain(vector<int> & attriblist);
void FS_ReadEunodeStrain();
void FS_CoordsPlotXY(Coords *xy, Coords *prevxy,int *iOnOtherSideX,int *iOnOtherSideY);

// FS: For new Roi determination:
double FS_GetROI(int iFact);

#endif
