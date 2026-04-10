#ifndef _gbm_elle_h
#define  _gbm_elle_h
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
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
#include "movenode.h"
//#include "growthstats.h"
/*#define PI 3.1415926
#define DTOR PI/180
#define RTOD 180/PI*/

int GBMGrowth();

int InitGrowth();

double GetNodeEnergy( int node, Coords * xy );
int GGMoveNode(int node,Coords *xy);
int write_data(int stage);

#endif

