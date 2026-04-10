
#ifndef _E_diffusiondata_h
#define _E_diffusiondata_h

#ifndef _E_triattrib_h
#include "triattrib.h"
#endif

typedef struct {
  double DIFF,CBND,dt;
  double *DeltaC, *Area, *Len0, *Len1, *Len2;
} DiffusionData;

void InitDiffusionArrays(int size);
void SetDiffusionDefaultBCond(double CBND);
double DiffusionDefaultBCond();
void SetDiffusionConstant(double DIFF);
double DiffusionConstant();
void SetDiffusionTimestep(double dt);
double DiffusionTimestep();
int SetupDiffusionArrays(struct triangulateio *tri);
void DiffusionArrays(double **len0, double **len1, double **len2,
                     double **area, double **deltaC);

#endif
