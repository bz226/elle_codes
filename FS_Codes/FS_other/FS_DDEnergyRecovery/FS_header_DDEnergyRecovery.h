#ifndef FS_recovery_h
#define FS_recovery_h

/* Check the include list for missing or needless header files:*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <iostream>
#include "nodes.h"
#include "display.h"
#include "check.h"
#include "errnum.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "polygon.h"
#include "unodes.h"
#include "stats.h"
#include "init.h"
#include "mat.h"
#include "convert.h"
#include "update.h"
#include "math.h"
#include "timefn.h"
#include "string_utils.h"
#include "parseopts.h"
#include "setup.h"
#include "triattrib.h"
#include "attrib.h"
#include "mineraldb.h"

#define PIEXACT 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
#define D2R PIEXACT/180
#define R2D 180/PIEXACT

using namespace std;

int 	InitThisProcess();
int FS_RecoveryTesting();

int FS_doannealing();
    int FS_norm_sep(Coords jxy, vector<Coords> &nbxy, double **nbori, int numnbs);
    double FS_GetDDEnergy(double dSumMisoris,double dSumDists,int iNumNbs);
        double FS_Misori2DD(double dKAM, double dMeanDist);//, double dBurgersVec);
    int FS_rot_matrix(int t, double rmap[3][3], double dRotAngle);
        int FS_rot_matrix_axis(double dRotAngle, double rmap[3][3], double axis[3]);
    double FS_MakeAniso(int iRotationMode,double dInVisc);

void 	orientmat2(double a[3][3], double phi, double rho, double g);
void 	symmetry(double symm[24][3][3]);
double 	CME_hex(double curra1,double currb1,double currc1,double curra2,double currb2,double currc2,double symm[24][3][3]);
void 	matvecmultalb (double rmap[3][3], double currentori[3], double newori[3]);


/*
 * indices for User data values for this process
 */
//const int UInput1=0; 
//const int UInput2=1;      
//const int UInput3=2;     
//const int UInput4=3;      
//const int UInput5=4;      
//const int UInput6=5;      
//const int UInput7=6;      
//const int UInput8=7;      
//const int UInput9=8;   

/*
 * Function declaration
 */
//int InitThisProcess();
//int ProcessFunction();

#endif
