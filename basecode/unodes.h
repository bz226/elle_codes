 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_unodes_h
#define _E_unodes_h

#include <cstdio>
#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <zlib.h>
#include "attrib.h"
#include "gz_utils.h"
#include "triangle.h"
#include "unodesP.h"
#include "voronoi.h"

#define HEX_GRID  0
#define SQ_GRID   1
#define RAN_GRID   2
#define SEMI_RAN_GRID 3

typedef std::list< std::pair<int,double> > list_pair;

typedef struct {
    std::vector<VoronoiPt> vpoints; //coordinates of unode voronoi pts
    std::vector<int> vsegs; // 2 ints per seg, to index into vpoints
} VoronoiData;

Unode *ElleGetParticleUnode(int id);

int ElleInitUnodes();
int ElleInitUnodes(int num);
int ElleInitUnodes(int num, int pattern);
void ElleInitUnodeAttribute(int id);
void ElleRemoveDefaultUnodeAttribute(int id);
void ElleGetDefaultUnodeAttribute(double *val,const int id);
void ElleSetDefaultUnodeAttribute(double val,const int id);
void ElleSetDefaultUnodeAttribute(double val1, double val2, double val3,
                                  const int id);
void ElleGetUnodePolyInfo(int id,const int attrib,
                          Coords *xy,double *val);
void ElleGetUnodePosition(int id,Coords *xy);
void ElleSetUnodePosition(int id,Coords *xy);
void ElleGetUnodeArea(int id,double *area);
void ElleSetUnodeArea(int id,double area);
int ElleRotateUnodes(double rad,Coords *origin);
int ElleGetUnodeAttribute(int id,const int attr_id,double *val);
int ElleGetUnodeAttribute(int id,double *val,const int attr_id); //use this
int ElleGetUnodeAttribute(int id,double *val1,double *val2, double *val3,
                         const int attr_id);
int ElleGetUnodeAttribute(int id,double *val1,double *val2,
                         const int attr_id);
int ElleReadUnodeData(FILE *fp, char str[], int *attr, int maxa);
int ElleReadUnodeData(FILE *fp, char str[], int *attr, int maxa);
int ElleReadUnodeRealAttribData(FILE *fp, char str[], int *attr, int maxa);
int ElleSetUnodeAttribute(int id,const int attr_id,double val);
int ElleSetUnodeAttribute(int id,double val,const int attr_id); //use this
int ElleSetUnodeAttribute(int id,double val1,double val2, double val3,
                         const int attr_id);
int ElleSetUnodeAttribute(int id,double val1,double val2,
                         const int attr_id);
void ElleRemoveUnodeAttribute(int u_id,const int attr_id);
int ElleRemoveUnodeAttributes(std::vector<int> &attriblist);
int ElleMaxUnodes();
int ElleVoronoiActive();
int UnodesActive();
int ElleWriteUnodeData(const char *fname);
int ElleWriteUnodeLocation(std::ofstream &outf);
int ElleWriteUnodeAttributeData(std::ofstream &outf,int *keys,int count);
void TriangulateUnodes(int flynnid, struct triangulateio *out);
void TriangulateUnodes(int flynnid, struct triangulateio *out,
                       int attrid);
unsigned char UnodeAttributeActive(int id);
void UnodeAttributeList(int **attr_ids, int *maxa);
int UnodeAttributeRange(int attr, double *min, double *max);
void ElleSetUnodeCONCactive(int val);
int ElleUnodeCONCactive();
int ElleFindUnodeRow(const double yval);
int ElleFindUnodeColumn(const double xval);
double ElleUnodeROI();
double ElleTotalUnodeMass(int attr_id);
void UnodesClean();
void ElleLocalLists( int n, Coords *incr, set_int **unodeset,
                     set_int **flynnset, double roi );
void CombineLocalLists( set_int **unodelist, set_int **flist);
void ElleReassignUnodes( set_int *unodelist, set_int *flist);
int VoronoiFlynnUnodes(int flynnid);
int VoronoiFlynnUnodes(int flynnid, std::vector<int> &unodelist);

// move these to voronoi.cc
int ElleAddVoronoiPt(Coords xy);
bool ElleFindVoronoiPt(const Coords pt, int *index);
int Voronoi2Flynns(bool voronoi_by_flynn);

void ElleUnodeAddVoronoiPt( int id, int pt_index);
void ElleUnodeInsertVoronoiPt( int id, int post_index, int pt_index);
void ElleUnodeDeleteVoronoiPt( int id, int pt_index);
int ElleUnodeNumberOfVoronoiPts( int id);
void ElleUnodeVoronoiPts( int id, std::list<int> &l);
void ElleUnodeAddNearNb( int id, int nb);
int ElleUnodeNumberOfNearNbs( int id);
void ElleGetUnodeNearNbs( int id, std::vector<int> &nbs );
void ElleUnodeWriteNearNbs( std::ostream & os);
void ElleSetUnodeFlynn( int id, int flynnid);
int ElleUnodeFlynn( int id );
int FindUnode(Coords *ptxy, Coords *origin,
              double dx, double dy,
              int numperrow);
int FindUnode(Coords *ptxy, int start);
int FindUnode(Coords *ptxy, int start, bool wrap);

void UpdateGBMUnodeValues(int n,Coords new_incr,
                          set_int **unodeset, set_int *flynnset,
                          double roi);
double EstimateConcFromUnodeValues(Coords *pt_pos, int attr_id,
				std::vector<int> *unodelist, double roi);
int ElleInitRanUnodes(int uppercell, int subset, int pattern);
int ElleVoronoiUnodes();

int SaveZIPUnodeAttributeData( gzFile out, int * keys, int count );
int SaveZIPUnodeData( gzFile out );
int SaveZIPUnodeLocation( gzFile out );
int LoadZIPUnodeData( gzFile in, char str[], int * attr, int maxa );
int LoadZIPUnodeRealAttribData( gzFile in, char str[], int * keys, int
count );
int LoadZIPUnodeRealAttribData( gzFile in, char str[], int attr_id );
#endif
