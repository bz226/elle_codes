#ifndef FS_topocheck_h
#define FS_topocheck_h

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
#include "check.h"
#include "crossings.h"
#include "general.h"
#include "bflynns.h"

#define PI 3.141592654 

/*
 * Function declaration
 */
int InitThisProcess();
int TopologyChecks();

/*
 * Topology Check Functions:
 */
void ElleGGTopoChecksOLD();
void UpdateUnodesAndAttribs(bool bWriteUnodeOri2File);
    double FS_GetROI4topochecks(int iFact);
void ElleGGTopoChecks();
int CheckNodesCoincide();
int CheckNodesCoincide2();
void CheckAngle(double min_ang);
    int IncreaseAngle(Coords *xy,Coords *xy1,Coords *xy2,Coords *diff);  
int DeleteSmallFlynns( void );
    int FS_FlynnIsDangling(int flynn);
    int MergeSmallFlynn( int iFlynn1, int iFlynn2 ); 

void CheckIfFlynnsNeedSplit();
    int NoNbNodesTooClose(int iFlynnID, int *iNode1, int *iNode2);
        // for some reason needs declaration in .cc file with the function
        //int FindNoNeighbourBnodes(int iBnode, int iFlynn, vector<int> vFlynnBnodes, vector<int> &vBnodesNoNb);
    int CheckTopoSplitPossible(int iFirst, int iLast, int iFlynn);
    
/* 
 * New Corrsings check moving the node if there is no topological error
 */
int FS_CrossingsCheck(int node, Coords *pos_incr);
    int FS_TripleSwitchPossible(int iTripleJ1,int iTripleJ2);
    int FS_CheckCrossingDJ(int iNode,Coords cNewPosition);
        int FS_CheckFor2SidedFlynn(int iBnode);
        int FS_PtInRect(Coords cPoint,Coords cEdge1,Coords cEdge2);
        
/*
 * To update flynn ages after split:
 */
void UpdateFlynnAges(int iNew,int iOld);

/*
 * Check for things that this topology check cannot solve*/
bool CheckforIslandFlynns();

#endif
