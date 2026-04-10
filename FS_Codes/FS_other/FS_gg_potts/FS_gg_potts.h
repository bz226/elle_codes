#ifndef FS_gg_potts_h
#define FS_gg_potts_h

/* Check the include list for missing or needless header files:*/
#include <string>
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <iomanip>
#include <stdio.h>

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time -> also for random number generator*/

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
#include "runopts.h"
#include "display.h"

#define PI 3.141592654
using namespace std;

/*
 * Attributes
 */
const int uPzero = 0;
const int uC = 1;
const int uRandomShuffleOff = 2;

/*
 * Function declaration
 */
int InitGGPolyphase();
int GGPolyphase();
    double GetEnergyState(int iUnodeID,double dEsurf);
        void FindUnodeNbs(int iUnodeID, vector<int>* vUnodeNbs);
            int PushIntoBox(int iRowCol);
    double SwitchProbability(double dEnergyChange);
    void PerformSwitch(int iUnode, int iNb);
#endif
