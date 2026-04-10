/*****************************************************************************/
/*                                                                           */
/*  (rundiffusion.c)                                                         */
/*     program to set up delauney mesh and run diffusion on some attribute   */
/*        to test diffusion routines                                         */
/*                                                                           */
/*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "attrib.h"
#include "tripoly.h"
#include "triattrib.h"
#include "display.h"
#include "errnum.h"
#include "runopts.h"
#include "error.h"
#include "diffusiondata.h"
#include "diffusion.h"
#include "poly.h"

#ifndef _STDLIB_H_
extern void *malloc();
extern void free();
#endif /* _STDLIB_H_ */

int InitDiffusion();
int RunDiffusion();
extern int diffusion(struct triangulateio *,double *);
extern void GetUpdateFreq(int *);

int InitDiffusion()
{
  char *infile;
  struct flagvals triflags;
  int i,j,k,nn,ne;
  int err=0;
  int *dummypts=0, dummycnt=0; /* just in case poly file has ELLE key */
  int maxrgns;
  double *rgnattribs=0,dflt;
  double tris,ex[3],ey[3];

  ElleClearTriAttributes();

  /**********************************************/
  /******* Read Poly file and triangulate *******/
  /**********************************************/

  triflags.area = 0.001;
  triflags.quality = 5;
  triflags.midptnodes = 1;
  triflags.bndpts = 1;
  /*if ((MeshData.tri = (struct triangulateio *)*/
          /*malloc(sizeof(struct triangulateio)))==0)*/
             /*OnError("InitDiffusion",MALLOC_ERR);*/
  /*initio( MeshData.tri );*/

  infile = ElleFile();
  tripoly(MeshData.tri,infile,&triflags,0,&dummycnt,&dummypts);

  if (dummypts) free(dummypts);

  /*************************************************************/
  /****** Set up initial and final Concentration Arrays ********/
  /*************************************************************/

  ElleAllocTriAttributes(0);

  /* assign attribute array - for now... */
  if (MeshData.tri->triangleattributelist == NULL){
     printf("No attributes set to initialize Cinit with\n");
     exit(1);
  }
  else{
      maxrgns = (int)MeshData.tri->triangleattributelist[0];
      for (i=0; i< MeshData.tri->numberoftriangles; i++){
          k = (int)MeshData.tri->triangleattributelist[i];
          if (k > maxrgns) maxrgns = k;
      }
      if ((rgnattribs= (double *)malloc((maxrgns+1) *
                                    sizeof(double)))==NULL) return(1);
      dflt = 0.0;
      for (i=0;i<maxrgns;i++) rgnattribs[i] = dflt;
      for (i = 0; i < MeshData.tri->numberoftriangles; i++)
          MeshData.attribute[0].elem[i]=dflt;
      if (attribute_set(infile,FLYNN_REAL_ATTRIB_KEY,
                        maxrgns+1,rgnattribs,&dflt)) {
          for (i = 0; i < MeshData.tri->numberoftriangles; i++) {
              k = (int)MeshData.tri->triangleattributelist[i];
              MeshData.attribute[0].elem[i]=rgnattribs[k];
          }
      }
     ElleSetTriAttributeRange(0);
  }

  SetDiffusionTimestep(0.5e-6);    /* time step */
  SetDiffusionConstant(1.0);  /* diffusion constant */
                              /* diffusion constant */
  SetDiffusionConstant(0.5e3/MeshData.tri->numberoftriangles);
  SetDiffusionDefaultBCond(0.0);  /* default boundary condition */

  /**************************************/
  /******* Allocate Diffusion Arrays *******/
  /**************************************/

  InitDiffusionArrays(MeshData.tri->numberoftriangles);
  /**************************************/
  /******* Setup Diffusion Arrays *******/
  /**************************************/

  if (err=SetupDiffusionArrays(MeshData.tri))
      OnError("InitDiffusion",0);
}

int RunDiffusion()
{
  int i, update;
  double time;
  /**************************************/
  /******* Run Diffusion Here     *******/
  /**************************************/


  if (ElleDisplay()) EllePlotTriangles();
  GetUpdateFreq(&update);
  for (i=0;i<EllemaxStages();i++) {
      diffusion(MeshData.tri,MeshData.attribute[0].elem);
      time=(double)ElleCount()*DiffusionTimestep();
      /* print results for Time=time here */
        if (ElleDisplay()) {
            if (update && ElleCount()%update==0) EllePlotTriangles();
			ElleIncrementCount();
            ElleShowStages( ElleCount() );
        }
  }
  return(0);
}
