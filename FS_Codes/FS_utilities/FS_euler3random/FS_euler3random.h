#ifndef FS_euler3random_h
#define FS_euler3random_h

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
const int InUnodes=0;   // If set to 1: Store angles in unodes instead of flynns, default 0
const int NoiseInFlynns = 1; // Setting it to anything higher than 0 means adding a noise to otherwise constant LPO within one flynn, the angles are then stored in unodes and removed from flynns
                               // ATTETION: Only working if InUnodes == 0
const int Anisotropy=2; // If set to 1: Code will create anisotropic distribution of euler angles with maximum values defined in the FOUR following input parameters, default: 0
const int AnisoAlpha=3; // Only if Anisotropy==1: Value for euler alpha 
const int AnisoBeta=4;  // Only if Anisotropy==1: Value for euler beta 
const int AnisoGamma=5; // Only if Anisotropy==1: Value for euler gamma 
const int AnisoNoise=6; // Only if Anisotropy==1: Values for alpha,beta,gamma will have a noise of +- this value
const int AnisoOnlyFor1Phase=7; // If euler angles are already in flynns: If setting to anything else than 0: Only change values in flynns with VISCOSITY = userdata[AnisoOnlyForPhase]

/*
 * Function declaration
 */
int InitThisProcess();
int ProcessFunction();
    //double GetRandomNumber(double dA, double dB);
    double GetRandomNumber(double dA, double dB,double dRandomisation);
    double GetRandomNumberElleFunc(double dA, double dB);
    void AddNoise2Flynns(double dNoisePlusMinus);
    void CheckAllAngles(int iUseUnodes);
    void CheckEulers(double dEulers[3]);
    
/*
 * Some testing
 */
int RanorientQuartz(); // from tidy
int RanorientQuartzUnodes();

#endif
