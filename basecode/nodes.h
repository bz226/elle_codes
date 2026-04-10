 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_nodes_h
#define _E_nodes_h

#include <cstdio>
#include <vector>
#include <zlib.h>
#include "attrib.h"
#include "gz_utils.h"

#define SIN60 0.866025
const int MAX_LOCAL_FLYNNS = 9;
const int COMBINED_LIST = MAX_LOCAL_FLYNNS;

int SaveZIPNodeConcA( gzFile out );
int SaveZIPNodeRealAttribute( gzFile out, int id );
int SaveZIPNodeMarginAttribData( gzFile out, int attr_id );
int SaveZIPNodeVelocity( gzFile out );
int SaveZIPStrainData( gzFile out, int * keys, int count );
int SaveZIPStressData( gzFile out, int * keys, int count );
int SaveZIPNodeLocation( gzFile out );
int SaveZIPNodeData( gzFile out );
int LoadZIPLocationData( gzFile in, char str[] );
int LoadZIPVelocityData( gzFile in, char str[] );
int LoadZIPConcAData( gzFile in, char str[] );
int LoadZIPStressData( gzFile in, char str[], int *, int );
int LoadZIPStrainData( gzFile in, char str[], int *, int );
int LoadZIPNodeRealAttribData( gzFile in, char str[], int id );
int LoadZIPNodeMarginAttribData( gzFile in, char str[], int id );
int LoadZIPNodeFile( gzFile in );

#ifdef __cplusplus
extern "C" {
#endif
int ElleInsertDoubleJAtPoint(int node1,int node2,int *nn,Coords *pos);
int ElleInsertDoubleJ(int node1,int node2,int *nn,float frac);
int ElleInsertDoubleJOnBnd(int node1,int bnd1,int node2,int bnd2,
                           int *nn,float frac);
int ElleDeleteDoubleJ(int id);
int ElleDeleteDoubleJNoCheck(int id);
int ElleIdMatch(ERegion *a, ERegion *b);
int ElleTwoSidedGrainId(int node1, int node2);
int ElleFindBndNb(int node_num, ERegion id);
int ElleNewFindBndNb(int node_num, int prev_node,ERegion id);
int ElleFindBndNode(int node_num, int prev_node,int  id);
int ElleReadNodeFile(char *filename);
int ElleWriteNodeFile(char *filename);
int ElleReadLocationData(FILE *fp, char str[]);
int ElleReadVelocityData(FILE *fp, char str[]);
int ElleReadConcAData(FILE *fp, char str[]);
int ElleReadStressData(FILE *fp, char str[],int *,int);
int ElleReadStrainData(FILE *fp, char str[],int *,int);
int ElleReadNodeRealAttribData(FILE *fp, char str[], int id);
int ElleReadNodeMarginAttribData(FILE *fp, char str[], int id);
int ElleJoinTwoDoubleJ(int first, int last,ERegion old,ERegion pnew);
void ElleSwitchTriples(int modfactor);
int ElleSwitchGap(int node1,int node2);
int ElleSmallSwitchGap(int node1,int node2);
int ElleNodesCoincident(int node1,int node2);
int ElleGapTooSmall(int node1,int node2);
int ElleGapTooLarge(int node1,int node2);
int ElleNodeOnRgnBnd(int node, ERegion id);
int ElleSwitchTripleNodes(int node1,int node2);
int ElleSwitchTripleNodesForced(int node1,int node2);
int ElleFindNbIndex(int nb, int node);
int ElleIndexNb(int node, int index);
int ElleFindBndIndex(int node, ERegion full_id);
int ElleFindBndNbIndex(int nb, int node, ERegion full_id);
void ElleRelPosition(Coords *orig, int nb, Coords *xyrel, double *sep);
double ElleNodeSeparation(int node1, int node2);
double ElleNodePrevSeparation(int node1, int node2);
void ElleNumberOfNodes(int *doubles, int *triples);
void ElleFindLocalBBox(Coords *centre, Coords *bbox, int factor);
int ElleResetNodeConc(int start, int reset_cnt, double val, int attrib_id);
double ElleTotalNodeMass(int attr_id);
double ElleNodeConcToMass(int node, int attr_id);
double ElleNodeMassToConc(int node, double mass, int attr_id);
int ElleAddDoubles();
int ElleWriteNodeData(FILE *fp);
int ElleWriteNodeLocation(FILE *fp);
int ElleWriteNodeVelocity(FILE *fp);
int ElleWriteNodeConcA(FILE *fp);
int ElleWriteStressData(FILE *fp, int *keys, int count);
int ElleWriteStrainData(FILE *fp, int *keys, int count);
int ElleWriteNodeRealAttribute(FILE *fp, int id);
int ElleWriteNodeMarginAttribData(FILE *fp, int id);
double ElleConcA();
double ElleVelocityX();
double ElleVelocityY();
int ElleFindNodeAttributeRange(int id,double *amin,double *amax);
/*float ElleTauXX();*/
/*float ElleTauYY();*/
/*float ElleTauZZ();*/
/*float ElleTauXY();*/
/*float EllePressure();*/
/*nodesP.c*/
void ElleInitNodeArray(void);
int *ElleNodeOrderArray();
void ElleCleanNodeArray(void);
void ElleRemoveNodeArray(void);
int ElleReallocateNodeArray(int size);
int ElleNodeIsActive(int node);
int ElleNodeIsSingle(int node);
int ElleNodeIsDouble(int node);
int ElleNodeIsTriple(int node);
void ElleSetNodeActive(int node);
int ElleMaxNodes();
int ElleNodeCONCactive();
int ElleNodeChange();
int ElleNodeCONCAttribute(int attrib_id);
void ElleSetNodeCONCactive(int val);
void ElleSetNodeChange(int val);
void ElleRedistributeNodeCONC(int node,int nb1,int nb2);
void ElleRecalculateNodeCONCrb(int node1,int node2);
int ElleMaxNodeAttributes();
void ElleClearNodeEntry(int indx);
int ElleFindSpareNode();
void ElleSetRegionEntry(int node, int indx, ERegion id);
int ElleSetNeighbour(int node, int indx, int nb, ERegion *id);
int ElleClearNeighbour(int node, int indx);
int ElleNeighbourNodes(int node, int *nbnodes);
void ElleNodeUnitXY(Coords *xy);
void ElleClearNodeArray(int start, int end);
void ElleNodeSameMineral(int node, int *same, int *mintype);
int Elle2NodesSameMineral(int node1, int node2);
unsigned char EllePhaseBoundary(int node1, int node2);
int ElleOrderNbsOnAngle(int node, int *nb);
void ElleSetNodeAttribute(int node, double val, int id);
void ElleGetNodeAttribute(int node, double *val, int id);
void ElleSetNodeMarginAttribute(int node, double val, int id, int rgn);
double ElleNodeMarginConcEstimate(int node, int attr_id,
				std::vector<int> *unodelist, double roi);
void writeNodeLocalPolyFile(int index);
void OnNodeError(int node,const char *message,int err);
void ElleSetNodeDefaultConcA(double val);
void ElleSetNodeDefaultVelocityX(double val);
void ElleSetNodeDefaultVelocityY(double val);
double ElleNodeDefaultConcA();
double ElleNodeDefaultVelocityX();
double ElleNodeDefaultVelocityY();
double ElleNodeAttribute(int node, int index);
double ElleNodeMarginAttribute(int node, int id, int rgn);
int ElleConcAIndex();
int ElleVelocityXIndex();
int ElleVelocityYIndex();
int ElleTauXXIndex();
int ElleTauYYIndex();
int ElleTauZZIndex();
int ElleTauXYIndex();
int ElleTau1Index();
int EllePressureIndex();
int ElleStrainIncrIndex();
int ElleBulkStrainIndex();
void ElleIntrpAttributes(int n);
int ElleNodeNumAttributes();
void ElleNodeAttributeList(int **active, int *max);
void ElleInitNodeMarginAttribute(int index);
void ElleInitNodeAttribute(int index);
void ElleSetBoundaryAttribute(int node, int index, double val);
double ElleGetBoundaryAttribute(int node, int nb);
int ElleRegions(int node, ERegion *rgn);
void ElleNeighbourRegion(int node, int nb, ERegion *region);
void ElleSetPosition(int node, Coords *position);
void ElleSetPrevPosition(int node, Coords *position);
void ElleCopyToPosition(int node, Coords *position);
void ElleCopyToPrevPosition(int node, Coords *position);
void ElleUpdatePosition(int node, Coords *increment);
void ElleNodePosition(int node, Coords *position);
void ElleNodePrevPosition(int node, Coords *previous);
void ElleNodePlotXY(int current, Coords *xy, Coords *prevxy);
void ElleCoordsPlotXY(Coords *xy, Coords *prevxy);
bool ElleCoordRelativePosition(Coords *current,
                              Coords *bnd,
                              int num_bnd_nodes,
                              int *xflags,int *yflags,
                              Coords *rel_pos);
int ellepointplotxy_(double *x, double *y, double *prevx, double *prevy);
unsigned char ElleNodeAttributeActive(int id);
unsigned char ElleNodeMarginAttributeActive(int id);
/* private */
int ElleRemoveNodeAttribute(int index);
int ElleSpareNodeAttribIndex(int index);
int ElleFindNodeAttribIndex(int index);

void ElleDeleteSingleJ(int node );
void EllePrintNodes(FILE *fp);
void ElleRemoveTripleJLink(int node1,int node2);
void ElleRemoveHangingNode(int dblnode,int trplnode);
void ElleMoveToCentre(int node,int node1,int node2,int node3);
void ElleFindCentre(int node,int node1,int node2,int node3,
                    Coords *xy);
void ResetRegionEntries(int start, int *prev, int end,
                        int old, int newrgn);
#ifdef __cplusplus
}
#endif
#endif
