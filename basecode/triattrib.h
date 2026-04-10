 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_triattrib_h
#define _E_triattrib_h

#include <vector>

#ifndef _E_attrib_h
#include "attrib.h"
#endif

#ifndef _triangle_h
#include "triangle.h"
#endif

typedef struct {
    double init_min, init_max;
    double *elem;
} attr_array;

typedef struct {
    struct triangulateio *tri;
    attr_array *attribute;
    int num_attributes;
    int ref_id;
} Mesh;

void ElleGetTriPtNeighbours(int index, std::vector<int> &nbnodes,
				std::vector<int> &on_bnd, unsigned char bnd );
void ElleGetNodeTriPtNeighbours(int index, std::vector<int> &nbnodes,
				std::vector<int> &on_bnd, int nb_id, unsigned char bnd );
int ElleTriPtOnBoundary(int index);
#ifdef __cplusplus
extern "C" {
#endif
double ElleTriAttributeMax(int index);
double ElleTriAttributeMin(int index);
void ElleSetTriAttributeMax(int index,double val);
void ElleSetTriAttributeMin(int index,double val);
int ElleSetTriAttributeRange(int index);
int ElleMaxTriangles();
double ElleTriAttribute(int id, int index);
void ElleSetTriReferenceId(int ref);
void ElleTriPointPosition(int id, double *x, double *y);
void ElleTrianglePoints(int index, int *pt_ids);
void ElleInitTriAttributes();
void ElleRemoveTriAttributes();
void ElleClearTriAttributes();
void ElleAllocTriAttributes(int);
void ElleSetTriAttributeFromNodes(int index,int nindex);
#ifdef __cplusplus
}
#endif

extern Mesh MeshData;
#endif
