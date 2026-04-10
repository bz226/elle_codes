#ifndef _jkbgbm_elle_h
#define  _jkbgbm_elle_h
#include <cstdio>
#include <cmath>
#include <cstring>
#include "attrib.h"
#include "nodes.h"
#include "file.h"
#include "display.h"
#include "check.h"
#include "error.h"
#include "runopts.h"
#include "init.h"
#include "general.h"
#include "stats.h"
#include "update.h"
#include "interface.h"
#include "crossings.h"
#include "mineraldb.h"
#include "polygon.h"
#include <vector>
#include "movenode.h"
//#include "growthstats.h"
/*#define PI 3.1415926

#define DTOR PI/180
#define RTOD 180/PI*/

#include "unodes.h"
#include "triattrib.h"


int GBMGrowth();

int InitGrowth();



double GetNodeEnergy( int node, Coords * xy );
int GGMoveNode_all( int node, Coords * xy );

// int GGMoveNode(int node,Coords *xy);
int write_data(int stage);
int GetBodyNodeEnergy(int n, double *total_energy);

double GetNodeStoredEnergy( int node, Coords * xy );
double area_swept_gpclip( int node, Coords * xy );
double density_unodes (int node, Coords * xy );
// 


#endif

