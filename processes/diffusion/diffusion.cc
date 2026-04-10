/*****************************************************************************/
/*                                                                           */
/*  (diffusion.c)                                                            */
/*     routine to diffuse some attribute across a delauney mesh given        */
/*        a dimensionless diffusion constant and time step                   */
/*                                                                           */
/*****************************************************************************/

/* If SINGLE is defined when triangle.o is compiled, it should also be       */
/*   defined here.  If not, it should not be defined here.                   */

/* #define SINGLE */

#ifdef SINGLE
#define REAL float
#else /* not SINGLE */
#define REAL double
#endif /* not SINGLE */

#include <stdio.h>
#include "attrib.h"
#include "triangle.h"
#include "diffusiondata.h"

#ifndef _STDLIB_H_
extern void *malloc();
extern void free();
#endif /* _STDLIB_H_ */

int diffusion(struct triangulateio *mesh,double *conc)
{
  int i,j,ne;
  double a0, c0, *deltaC, *area, *len0, *len1, *len2;
  double DIFF,dt;
  int tringh[3];
  double aneigh[3],cneigh[3];

  DiffusionArrays(&len0,&len1,&len2,&area,&deltaC);

  ne = mesh->numberoftriangles;
  DIFF = DiffusionConstant();
  for (i = 0; i < ne; i++) {
    /* setup local concentration variables*/
    c0=conc[i];
    a0=area[i];
    for (j = 0; j < 3; j++) {
       tringh[j]=mesh->neighborlist[i*3+j];
       if(tringh[j] == -1) {
           /*Cneigh[j]=Cbnd;*/  /* for constant C bondary conditions */
           cneigh[j]=c0;  /* for reflecting boundaries */
           aneigh[j]=a0;
       }
       else {
           cneigh[j]=conc[tringh[j]];
           aneigh[j]=area[tringh[j]];
       }
    }
    /* DeltaC is the rate of change in concentration at every element */
    /*
     * valid for a constant DIFF over all elements
     */
    deltaC[i]= (-1.0)*DIFF/a0/a0/2.0*(
                len0[i]*len0[i]*(c0-cneigh[0])*(a0+aneigh[0])/aneigh[0]
               +len1[i]*len1[i]*(c0-cneigh[1])*(a0+aneigh[1])/aneigh[1]
               +len2[i]*len2[i]*(c0-cneigh[2])*(a0+aneigh[2])/aneigh[2]
               );
  }

 /* Now update the concentrations for every triangle */
  dt = DiffusionTimestep();
  for (i = 0; i < ne; i++) {
      conc[i] += deltaC[i]*dt;
  }

} /* finished here */
