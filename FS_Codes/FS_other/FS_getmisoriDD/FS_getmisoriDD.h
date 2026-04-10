#ifndef FS_getmisoriDD_h
#define FS_getmisoriDD_h

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
#include "convert.h"
#include "mineraldb.h"

using namespace std;
/*
 * indices for User data values for this process
 */
const int UHAGB=0;  
const int UExcludePhase=1;

/*
 * Function declaration
 */
int InitThisProcess();
int ProcessFunction();
    int InitialChecks();
    void GetUnodeMisorientationsInFlynn(int iFlynn,vector<int> &vUnodeList,double *dMeanMis);
        double FS_Misori2DD(double dKAM, double dMeanDist, double dBurgersVec);
    
/*
 * From recovery code:
 */
void symmetry(double symm[24][3][3]);
double CME_hex(double curra1,double currb1,double currc1,double curra2,double currb2,double currc2, double symm[8][3][3]);
void orientmat2(double a[3][3], double phi, double rho, double g);
bool fileExists(const char *filename);
int FS_norm_sep(Coords jxy, vector<Coords> &nbxy, double **nbori, int numnbs);

#endif
