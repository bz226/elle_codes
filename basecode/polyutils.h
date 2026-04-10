 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _Elle_polyutils_h
#define _Elle_polyutils_h

#include <list>
#include <vector>
#include "triangle.h"
#include "attrib.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef __cplusplus
}
#endif
void sortCoordsOnAngle(std::vector<Coords> &pts, Coords *ref);
void findPolygon(REAL *pointlist, REAL *normlist,int *edgelist, int nseg,
                 int *seg_count,int *segs_left,
                 std::list<int> &processed);
#endif
