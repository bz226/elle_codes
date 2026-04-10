#ifndef _gbm_movenode_h
  #define _gbm_movenode_h
  #include <stdio.h>
  #include <math.h>
  #include <string.h>
  #include "attrib.h"
  #include "nodes.h"
  #include "file.h"
  #include "display.h"
  #include "check.h"
  #include "error.h"
  #include "runopts.h"
  #include "init.h"
  #include "interface.h"
  #include "general.h"
  #include "convert.h"
  #include "stats.h"
  #include "update.h"
  #include "log.h"
  #include "mineraldb.h"
  #include <vector>

#define TIME_REDUCE 2

int MoveNode( int node1, Coords pvec, Coords * m );
double GetVectorLength( Coords vec );
double DEGCos( Coords vec1, Coords vec2 );
int GetMoveDir( int node, double e1, double e2, double e3, double e4, Coords * newpos,double t );
int GetNewPos( int node, double e1, double e2, double e3, double e4, Coords * newpos,double t );
double GetBoundaryMobility( int node, int nb );

#endif
