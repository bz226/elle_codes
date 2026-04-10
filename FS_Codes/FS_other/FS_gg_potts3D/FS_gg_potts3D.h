#ifndef FS_gg_potts3D_h
#define FS_gg_potts3D_h

/* Check the include list for missing or needless header files:*/
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <list>
#include <vector>
#include <iomanip>
#include <stdio.h>

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time -> also for random number generator*/

#include <math.h>

#define PI 3.141592654
using namespace std;

/*
 * Attributes
 */
const int uFilename = 1; // 1st input
const int uPzero = 2; // 2nd input
const int uC = 3; // 3rd input
const int uT = 4; // 4th input: Temperature in °C
const int uDim = 5; // 5th input
const int uStages = 6; // 6th input
const int uRandomShuffleOff = 7; // 7th input
const int uStartStep = 8; // 8th input, stages already performed in input file

/*
 * Function declaration
 */
int GG3DPotts();
    void ReadInputFile(void);
    void FindNodeNbs(int iNodeID, vector<int>* vNodeNbs);
        void ID2RCS(int iNodeID,int iRCS[3]);
        int RCS2ID(int iRCS[3]);
        void PushRCSIntoBox(int iRCS[3]);
    double GetEnergyState(int iNodeID,vector<int> vNbNodes,double dEsurf);
    double SwitchProbability(double dEnergyChange);
    void PerformSwitch(int iNode, int iNb);
    void Write3DPottsOutput(int iStage);
    
    
    
/////* 
 ////* Using the dmatrix function from Elle modelling platform 
 ////* found in processes/phasefield/feesfield.c
 ////* 
 ////* --> This one has been changed to use only integer instead of double
 ////*/
////#define NR_END  1
////int	**dmatrix(long nrl, long nrh, long ncl, long nch);
////int	**dmatrix(long nrl, long nrh, long ncl, long nch)
////{
	////long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	////int **m;
	
	/////*allocate pointers to rows*/
	////m=(int **) malloc((size_t)((nrow+NR_END)*sizeof(int*)));
	////if (!m) printf("allocation failure 1 in dmatrix()\n");
	////m += NR_END;
	////m -= nrl;
	
	/////*allocate rows and set pointers to them */
	////m[nrl]=(int *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(int)));
	////m[nrl] += NR_END;
	////m[nrl] -= ncl;
	
	////for (i=nrl+1;i<=nrh;i++)	m[i]=m[i-1]+ncol;
	////return m;
////}

#endif
