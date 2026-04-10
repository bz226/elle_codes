 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_splitting_h
#define _E_splitting_h

#include <vector>
#include "attrib.h"
#include "pointdata.h"
#include "tripoly.h"

void findNeighbourPts(int j, int prev, struct triangulateio *out,
                      int *cnt, std::vector<int> &nbnodes,
                      unsigned char incl_bnd);
void findNeighbourPts(int j, int prev, struct triangulateio *out,
                      int *cnt, std::vector<int> &nbnodes,
                      unsigned char incl_bnd, Coords *ddnode,int sign);
void CreateBoundary(std::vector<PointData> &nodes,int nodenbbnd,int nbnodebnd,
                    int *elle_id,int num,int *end);
void RemoveBoundary(int *start, int *finish, int rgn1, int rgn2);
int RandomSplit(struct triangulateio *out, std::vector<PointData> &boundary,
                int *ids, int *elle_id, int num, int start);
int DirectSplit(struct triangulateio *out, std::vector<PointData> &boundary,
                double f_area, int *elle_id, int num, int start,
                float dx, float dy);
bool in_array(int idval, std::vector<PointData> &v);
int index_diff(int val1, int val2, int size);
int WriteTriangulation(struct triangulateio *out,const char *name);

#endif
