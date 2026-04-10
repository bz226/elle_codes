 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_interface_h
#define _E_interface_h
#include <cstdio>
#include <list>
#include <vector>
#include <zlib.h>
#include "attrib.h"

int SaveZIPParentFlynn( int index, gzFile out );
void ElleFlynnNbRegions(int flynnid, std::list<int> &nbflynns);
void ElleSplitFlynnOnArea(const int type, int index, float dx, float dy);
void ElleSetFlynnRealAttribute(int index, double val, int id);
void ElleGetFlynnRealAttribute(int index, double *valptr, int id);
void ElleSetFlynnEuler3(int index, double alpha, double beta, double gamma);
void ElleGetFlynnEuler3(int index, double *alpha, double *beta, double *gamma);
int ElleGetFlynnEulerCAxis(int flynn_no, Coords_3D *dircos);
void ElleFlynnSetColor(int index,bool c,int r,int b, int g);
void ElleFlynnGetColor(int index,bool *c,int *r,int *b, int *g);
void ElleFlynnNodes(int index, std::vector<int> &ids);
int ElleFlynnNodePositions(int index, std::vector<Coords> &rel_pos,
                           Coords *ref);
int ElleAddUnodeToFlynn(int id, Coords *xy, int start);
int ElleRemoveUnodeFromFlynn(int flynnid, int id);
void ElleAddUnodeToFlynn(int flynnid, int id);
int ElleVoronoiFlynnUnodes();
int ElleVoronoiFlynnUnodes(std::list<int> &flynnlist);
int ElleVoronoiFlynnUnodes(std::list<int> &flynnlist,
                           std::vector<int> &unodelist);
int ElleWriteFlynnUnodePolyFile(int id, const int attrib, char *fname);
void ElleGetFlynnUnodeList(int flynnid, std::vector<int> &l);
unsigned char ElleUnodeAttributeActive(int id);
#if XY
double ElleTotalUnodeMass();
#endif
int ElleCheckSmallFlynn(const int index);

#ifdef __cplusplus
extern "C" {
#endif
int ElleSetFlynnActive(int index);
unsigned char ElleFlynnIsActive(int index);
int ElleMaxFlynns();
int ElleMaxLevels();
int ElleNumberOfGrains();
int ElleNumberOfSubgrains();
int ElleFlynnLevel(const int index);
void ElleSetFlynnFirstNode(int index, int first);
int ElleGetFlynnFirstNode(int index);
int ElleAddFlynnChild(int index, int child);
int getParent(int flynn1);
unsigned char hasParent(int flynn1);
unsigned char isParent(int flynn1);
unsigned char isChildOf(int flynn1,int flynn2);
int ElleFindSpareFlynn();
int ElleWriteParentFlynn(int index, FILE *fp);
void ElleFlynnNodes(int index, int **ids, int *num);
int ElleFlynnNodeCount(int index);
double ElleFlynnArea(const int index);
int ElleSplitFlynn(const int type, int index, int *child1, int *child2,
                             float dx, float dy);
int ElleSplitWrappingFlynn(int index, int node1, int node2,
                            int *child1, int *child2);
int ElleDirectSplitWrappingFlynn(int index, int node1, int node2,
                            int *child1, int *child2, Coords *dir);
int ElleSplitFlynnandPromote(const int type, int index, int *c1, int *c2,
                             float dx, float dy);
void ElleSplitFlynnOnArea(const int type, int index, double maxarea,
                          float dx, float dy);
void ElleSplitFlynnChildrenOnArea(const int type, int index, double maxarea,
                          float dx, float dy);
int ElleMergeFlynns(int keepflynn, int removeflynn);
int ElleMergeFlynnsNoCheck(int keepflynn, int removeflynn);
bool ElleFlynnAttributesAllowMerge(const int flynn1, const int flynn2);
int ElleNucleateFlynnFromNode(int node,int *newid);
void ElleUpdateFirstNodes(int old, int nn, int flynnindex);
void ElleRemoveFlynn(int flynnindex);
void ElleRemoveShrinkingFlynn(int flynnindex);
void EllePromoteFlynn(int flynnindex);
double ElleFindFlynnMinArea(int id);
void ElleCleanFlynnArray();
void ElleCleanDefaultAttributes();
void ElleFlynnDfltAttributeList(int **active, int *maxa);
void ElleCopyFlynnAttributes(int srcindex, int destindex);
unsigned char ElleFlynnAttributeActive(int id);
unsigned char ElleFlynnHasAttribute(int index,int id);
void ElleSetDefaultFlynnRealAttribute(double val,int id);
void ElleSetDefaultFlynnIntAttribute(int val,int id);
void ElleSetDefaultFlynnCAxis(Coords_3D *val);
void ElleSetDefaultFlynnEuler3(double alpha,double beta,double gamma);
void ElleSetDefaultFlynnAge(double val, int type);
int ElleFindFlynnAttributeRange(int attr,double *amin,double *amax);
int ElleFindBndAttributeRange(int attr,double *amin,double *amax);
int ElleWriteFlynnPolyFile(int flynnid, char *fname);
double ElleDefaultFlynnRealAttribute(int id);
int ElleDefaultFlynnIntAttribute(int id);
void ElleSetFlynnRealAttribute(int index, float val, int id);
void ElleGetFlynnRealAttribute(int index, float *valptr, int id);
void ElleSetFlynnIntAttribute(int index, int val, int id);
void ElleGetFlynnIntAttribute(int index, int *valptr, int id);
double ElleDefaultFlynnAge(int type);
void ElleDefaultFlynnCAxis(Coords_3D *valptr);
void ElleDefaultFlynnEuler3(double *alpha,double *beta,double *gamma);
void ElleInitFlynnAttribute(int id);
unsigned char isFlynnIntegerAttribute(int attribid);
unsigned char isFlynnRealAttribute(int attribid);
unsigned char isFlynnAttribute(int attribid);
/*int ElleFlynnAddVoronoiPt(int flynnid, Coords pt);*/
int ElleNumFlynnAttributes(int flynnindex);
void ElleRemoveFlynnAttribute(int index, int idval);
void ElleRemoveDefaultFlynnAttribute(int idval);
int ElleUnodesActive();
void ElleUnodeAttributeList(int **attr_ids,int *maxa);
int ElleFindUnodeAttributeRange(int attr,double *amin,double *amax);
void ElleCleanUnodeArray();
int ElleFindSegmentAttributeRange(int attr,double *amin,double *amax);
void ElleSegmentAttributeList(int **attr_ids,int *maxa);

void ElleInitRandom(unsigned long int seed);
unsigned long int ElleRandom();
double ElleRandomD();
/**********************************************************
 * functions below this line are being phased out
 **********************************************************/
void ElleSetDefaultFlynnExpand(int val);
void ElleSetDefaultFlynnColour(int val);
void ElleSetDefaultFlynnEnergy(float val);
void ElleSetDefaultFlynnViscosity(float val);
void ElleSetDefaultFlynnDislocDen(float val);
void ElleSetDefaultFlynnMineral(int val);
void ElleSetDefaultFlynnSplit(int val);
void ElleSetDefaultFlynnGrain(int val);
int ElleDefaultFlynnColour();
int ElleDefaultFlynnExpand();
float ElleDefaultFlynnEnergy();
float ElleDefaultFlynnViscosity();
float ElleDefaultFlynnDislocDen();
int ElleDefaultFlynnMineral();
int ElleDefaultFlynnSplit();
int ElleDefaultFlynnGrain();
void ElleSetFlynnColour(int index, int val);
void ElleGetFlynnColour(int index, int *valptr);
void ElleSetFlynnCAxis(int index, Coords_3D *val);
void ElleGetFlynnCAxis(int index, Coords_3D *valptr);
void ElleSetFlynnEuler3(int index, float alpha, float beta, float gamma);
void ElleGetFlynnEuler3(int index, float *alpha, float *beta, float *gamma);
void ElleSetFlynnEnergy(int index, float val);
void ElleGetFlynnEnergy(int index, float *valptr);
void ElleSetFlynnViscosity(int index, float val);
void ElleGetFlynnViscosity(int index, float *valptr);
void ElleSetFlynnAge(int index, float val, int type);
void ElleGetFlynnAge(int index, float *valptr, int type);
void ElleSetFlynnDislocDen(int index, float val);
void ElleGetFlynnDislocDen(int index, float *valptr);
void ElleSetFlynnMineral(int index, int val);
void ElleGetFlynnMineral(int index, int *valptr);
void ElleSetFlynnStrain(int index, float val, int type);
void ElleGetFlynnStrain(int index, float *valptr, int type);
void ElleSetFlynnSplit(int index, int val);
void ElleGetFlynnSplit(int index, int *valptr);
void ElleSetFlynnGrain(int index, int val);
void ElleGetFlynnGrain(int index, int *valptr);
void ElleCheckFlynnDefaults(int id);
#ifdef __cplusplus
}
#endif
#endif
