#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/* If SINGLE is defined when triangle.o is compiled, it should also be
    */
/*   defined here.  If not, it should not be defined here.
    */

/* #define SINGLE */

#ifdef SINGLE
#define REAL float
#else /* not SINGLE */
#define REAL double
#endif /* not SINGLE */

#include "triattrib.h"
#include "diffusion.h"
#include "diffusiondata.h"
#include "errnum.h"
#include "error.h"

DiffusionData Diffdat;

double DiffusionTimestep()
{
    return(Diffdat.dt);
}

void SetDiffusionTimestep(double dt)
{
    Diffdat.dt=dt;
}

double DiffusionConstant()
{
    return(Diffdat.DIFF);
}

void SetDiffusionConstant(double DIFF)
{
    Diffdat.DIFF=DIFF;
}

double DiffusionDefaultBCond()
{
    return(Diffdat.CBND);
}

void SetDiffusionDefaultBCond(double CBND)
{
    Diffdat.CBND=CBND;
}

void DiffusionArrays(double **len0, double **len1, double **len2,
                     double **area, double **deltaC)
{
    *len0 = Diffdat.Len0;
    *len1 = Diffdat.Len1;
    *len2 = Diffdat.Len2;
    *area = Diffdat.Area;
    *deltaC = Diffdat.DeltaC;
}

void InitDiffusionArrays(int size)
{
  int i;

  if ((Diffdat.Area = (double *) malloc( size * sizeof(double)))==0) {
      fprintf(stderr,"Malloc failed in diffusion\n");
      exit(1);
  }
  if ((Diffdat.Len0 = (double *) malloc( size * sizeof(double)))==0) {
      fprintf(stderr,"Malloc failed in diffusion\n");
      exit(1);
  }
  if ((Diffdat.Len1 = (double *) malloc( size * sizeof(double)))==0) {
      fprintf(stderr,"Malloc failed in diffusion\n");
      exit(1);
  }
  if ((Diffdat.Len2 = (double *) malloc( size * sizeof(double)))==0) {
      fprintf(stderr,"Malloc failed in diffusion\n");
      exit(1);
  }
  if ((Diffdat.DeltaC = (double *) malloc( size * sizeof(double)))==0) {
      fprintf(stderr,"Malloc failed in diffusion\n");
      exit(1);
  }
  for (i=0;i<size;i++) 
      Diffdat.Area[i] = Diffdat.Len0[i] = Diffdat.Len1[i] =
      Diffdat.Len2[i] = Diffdat.DeltaC[i] = 0.0;
}

int SetupDiffusionArrays(struct triangulateio *tri)
{
    int i,j,ne,nn,err=0;
    int pts[3];
    double tris,ex[3],ey[3],eps;

    eps = 1.0E-1;
    ne = tri->numberoftriangles;
    for (i = 0; i < ne; i++) {
      ElleTrianglePoints(i,pts);
      for (j = 0; j < 3; j++) {
          ElleTriPointPosition(pts[j],&ex[j],&ey[j]);
#if XY
         /*ex[j]=(float) MeshData.tri->pointlist[j*2];*/
         /*ey[j] = (float) MeshData.tri->pointlist[j*2+1];*/
/**********   is there an error here - shouldn't it be? ***********/
         /*nn = tri->trianglelist[i*6+j] + 1;*/
         nn = tri->trianglelist[i*6+j];
         ex[j] = tri->pointlist[nn*2];
         ey[j] = tri->pointlist[nn*2 + 1];
#endif
      }
      /* Area, Len[123] - area and length of each side respectively */
      Diffdat.Len0[i]= 
          sqrt((ex[1]-ex[2])*(ex[1]-ex[2])+(ey[1]-ey[2])*(ey[1]-ey[2]));
      Diffdat.Len1[i]=
          sqrt((ex[2]-ex[0])*(ex[2]-ex[0])+(ey[2]-ey[0])*(ey[2]-ey[0]));
      Diffdat.Len2[i]=
          sqrt((ex[0]-ex[1])*(ex[0]-ex[1])+(ey[0]-ey[1])*(ey[0]-ey[1]));
      tris=(Diffdat.Len0[i]+Diffdat.Len1[i]+Diffdat.Len2[i])/2;
      Diffdat.Area[i]=
          sqrt(tris*(tris-Diffdat.Len0[i])*
          (tris-Diffdat.Len1[i])*(tris-Diffdat.Len2[i]));
      if (Diffdat.Area[i]==0.0) {
          Diffdat.Area[i]=eps;
          printf("zero area %d ex %lf %lf %lf ey %lf %lf %lf\n",
                  i,ex[0],ex[1],ex[2],ey[0],ey[1],ey[2]);
      }
    }
    return(err);
}

int CleanDiffusion()
{
  if (Diffdat.Area) free(Diffdat.Area);
  if (Diffdat.Len0) free(Diffdat.Len0);
  if (Diffdat.Len1) free(Diffdat.Len1);
  if (Diffdat.Len2) free(Diffdat.Len2);
  if (Diffdat.DeltaC) free(Diffdat.DeltaC);
}
