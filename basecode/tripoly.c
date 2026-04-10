#include <stdio.h>
#include <string.h>
#include <math.h>

 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include "tripoly.h"
#include "poly.h"
#include "error.h"

#ifndef _STDLIB_H_
extern void *malloc();
extern void free();
#endif /* _STDLIB_H_ */


/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

/*****************************************************************************


  tripoly()   Triangulate polygons



*****************************************************************************/

int tripoly( struct triangulateio *out, char *infile,
             struct flagvals *flags, char *optionaltxt,
             int *maxindx, int **ielle)
{
  struct triangulateio inp, mid, mid2;
  void showvar();
  int err=0;
  int nbm=0, internal_bnd=0;
  char str[32], buf[81];

  initio( &inp );
  initio( &mid );
  initio( &mid2 );

  if ((err=readpolyinput(&inp,infile,flags,optionaltxt,maxindx,ielle,
                         &nbm,&internal_bnd)) !=0) return (err);
  printf("Now Triangulate");

  /********************************/
  /******* Setup Structures *******/
  /********************************/
  /* Make necessary initializations so that Triangle can return a   */
  /*   triangulation in `mid' and a refined triangulation in 'out'. */


  /* Make necessary initializations so that Triangle can return a   */
  /*   triangulation in `mid' and a refined triangulation in 'out'. */

  /* Make necessary initializations so that Triangle can return a */
  /*   triangulation in `out'.                                    */


  /**************************************/
  /******* Triangulate and Refine *******/
  /**************************************/
  /* Triangulate the points.            */
  /*   flags as defined in the triangle routine:
          -p  Triangulates a Planar Straight Line Graph (.poly file).
          -r  Refines a previously generated mesh.
          -q  Quality mesh generation.  A minimum angle may be specified[20].
          -a  Applies a maximum triangle area constraint.
          -A  Applies attributes to identify elements in certain regions.
          -c  Encloses the convex hull with segments.
          -e  Generates an edge list.
          -v  Generates a Voronoi diagram.
          -n  Generates a list of triangle neighbors.
          -g  Generates an .off file for Geomview.
          -B  Suppresses output of boundary information.
          -P  Suppresses output of .poly file.
          -N  Suppresses output of .node file.
          -E  Suppresses output of .ele file.
          -I  Suppresses mesh iteration numbers.
          -O  Ignores holes in .poly file.
          -X  Suppresses use of exact arithmetic.
          -z  Numbers all items starting from zero (rather than one).
          -o2 Generates second-order elements (6 nodes per triangle).
          -Y  Suppresses boundary segment splitting.
          -S  Specifies maximum number of added Steiner points.
          -i  Uses incremental method, rather than divide-and-conquer.
          -F  Uses Fortune's sweepline algorithm, rather than d-and-c.
          -l  Uses vertical cuts only, rather than alternating cuts.
          -s  Force segments into mesh by splitting (instead of using CDT).
          -C  Check consistency of final mesh.
          -Q  Quiet:  No terminal output except errors.
          -V  Verbose:  Detailed information on what I'm doing.
  */
  /*   Switches are chosen for the triangle routine as follows:  */
  /*   Read a PSLG (p), preserve the convex hull (c), number everything */
  /*   from  zero (z), assign a regional attribute to each element (A), */
  /*   and  produce an edge list (e), and a triangle neighbor list (n). */

#if XY
  strcpy(str,"pzAen");
  /*strcpy(str,"pczAen");*/
  if (flags->bndpts==0) {
    strcat(str,"Y");
  }
  triangulate(str, &inp, &mid, (struct triangulateio *) NULL);

  /* Attach area constraints to the triangles in preparation for */
  /*   refining the triangulation.                               */

  /* Needed only if -r and -a switches used: */
/*
  if ((mid.trianglearealist = (REAL *)malloc(
                   mid.numberoftriangles * sizeof(REAL))==0) {
      fprintf(stderr,"Malloc failed in tripoly\n");
      return(1);
  }
  mid.trianglearealist[0] = 3.0;
  mid.trianglearealist[1] = 1.0;
*/

  /* Refine the triangulation according to the attached constraints.    */
  /*   Switches are chosen for the triangle routine as follows:         */
  /*   Read a PSLG (p), refine a previos mesh (r), quality mesh         */
  /*   generation (q), number everything from  zero (z), suppress output */
  /*   of poly file (P), produce an edge list (e), and a triangle neighbor */
  /*   list (n) and 6 nodes per element */

  printf("Triangulating 2nd time\n");
  
  strcpy(str,"pzrAen");
  if (flags->bndpts==0) {
    strcat(str,"Y");
  }
  triangulate(str, &mid, &mid2, (struct triangulateio *) NULL);

/*
  printf("Triangulating 3rd time\n");
  triangulate("prAeq10zn", &mid2, &mid, (struct triangulateio *) NULL);
*/

  printf("Triangulating last time\n");
  strcpy(str,"pzrAPn");
  /*strcpy(str,"przcPn");*/
  /*triangulate("przqPno2", &mid2, out, (struct triangulateio *) NULL);*/
  if (flags->area>0.0) {
    sprintf(buf,"a%f",flags->area);
    strcat(str,buf);
  }
  if (flags->quality>0) {
    sprintf(buf,"q%d",flags->quality);
    strcat(str,buf);
  }
  if (flags->bndpts==0) {
    strcat(str,"Y");
  }
  if (flags->midptnodes) strcat(str,"o2");
  if (flags->edges) strcat(str,"e");
  if (flags->voronoi) strcat(str,"v");
  triangulate(str, &mid2, out, (struct triangulateio *) NULL);
#endif
  printf("Triangulating last time\n");
  strcpy(str,"pzAePnQ");
  /*triangulate("przqPno2", &mid2, out, (struct triangulateio *) NULL);*/
  if (flags->area>0.0) {
    sprintf(buf,"a%f",flags->area);
    strcat(str,buf);
  }
  if (flags->quality>0) {
    sprintf(buf,"q%d",flags->quality);
    strcat(str,buf);
  }
  /*
   * if an area constraint is set but quality is not (ie q0)
   * can have problems with narrow polygons - gives large maxnbs
   * which causes malloc failure even though triangulation is
   * successful.
   * In this case, works better with a small q
   */
  else if (flags->area>0.0) {
    sprintf(buf,"q5");
    strcat(str,buf);
  }
  if (flags->bndpts==0) {
    strcat(str,"Y");
    if (internal_bnd) strcat(str,"Y");
  }
  if (flags->midptnodes) strcat(str,"o2");
  if (flags->edges) strcat(str,"e");
  if (flags->voronoi) strcat(str,"v");
  triangulate(str, &inp, out, (struct triangulateio *) NULL);


  /************************************/
  /******* Output for debugging *******/
  /************************************/

  /*printf("Initial triangulation:\n\n");*/
  /*showvar(&mid, 1, 1, 1, 1, 1, 0);*/
  /*printf("Refined triangulation:\n\n");*/
  /*showvar(out, 0, 1, 0, 0, 0, 0);*/


  /***********************/
  /******* Cleanup *******/
  /***********************/
  /* Free all allocated arrays, including those allocated by Triangle. */

  cleanio( &inp );
  cleanio( &mid );
  cleanio( &mid2 );

  return 0;
}

int readpolyinput( struct triangulateio *inp, char *infile,
             struct flagvals *flags, char *optionaltxt,
             int *maxindx, int **ielle, int*nbm, int *internal_bnd)
{
  char str[32], buf[81];
  FILE *fp;
  long curr_pos;
  int i,j,k,ip,is,p1,p2,bmark;
  int np,ns,natt,ndim,nh,nr,npp;
  float xval, yval, attr, area;
  float tmp=0;
  /**************************/
  /******* Read Input *******/
  /**************************/

  np=ns=natt=ndim=nh=nr=npp=0;
  str[0] = '\0';

  if((fp=fopen(infile,"r"))==NULL) {
      fprintf(stderr,"cannot open file %s\n",infile);
      return(1);
  }

  /* Input File Format:
   First line:    <# of pts> <dim (=2)> <# of attr> <# of bndry markers (0 or 1)
   Next np lines: <point #> <x> <y> [attributes] [boundary marker]
   One line:      <# of segments> <# of boundary markers (0 or 1)
   Next ns lines: <segment #> <endpoint> <endpoint> [boundary marker]
   One line:      <# of holes>
   Next nh lines: <hole #> <x> <y>
   One line: <# of regional attributes and/or area constraints>
   Next nr lines:  <constraint #> <x> <y> <attrib> <max area>
   Optional line: Keyword followed by ascii data which will be interpreted
               according to the keyword
  */

  if (fgets(buf,80,fp)==NULL) {
      fprintf(stderr,"Read failed in tripoly\n");
      return(1);
  }
  if (sscanf(buf,"%d %d %d %d",&np, &ndim, &natt, nbm)!=4) {
      fprintf(stderr,"points read failed in tripoly\n");
      return(1);
  }
  printf("%d %d %d %d",np, ndim, natt, *nbm);
printf("sizeof REAL %ld \n",sizeof(REAL));

  inp->numberofpoints = np;
  inp->numberofpointattributes = natt;
  if ((inp->pointlist = (REAL *) malloc(
             inp->numberofpoints * 2 * sizeof(REAL)))==0) {
      fprintf(stderr,"Malloc failed in tripoly\n");
      return(1);
  }
  if (inp->numberofpointattributes != 0) {
    if ((inp->pointattributelist = (REAL *) malloc(inp->numberofpoints *
                                          inp->numberofpointattributes *
                                          sizeof(REAL)))==0) {
        fprintf(stderr,"Malloc failed in tripoly\n");
        return(1);
    }
    for (i=0;i<np*natt;i++) inp->pointattributelist[i] = 0.0;
  }
  if ((inp->pointmarkerlist = (int *) malloc(
             inp->numberofpoints * sizeof(int)))==0) {
      fprintf(stderr,"Malloc failed in tripoly\n");
      return(1);
  }

  for (i=0;i<np;i++){
     inp->pointmarkerlist[i]=0;
     if (fscanf(fp,"%d %f %f",&ip, &xval, &yval)!=3) {
       fprintf(stderr,"point read failed in tripoly\n");
       return(1);
     }
     inp->pointlist[i*2] = xval;
     inp->pointlist[i*2+1] = yval;
     if(natt>0){
        for (j=0;j<natt;j++){
           if (fscanf(fp,"%f",&attr)!=1) {
             fprintf(stderr,"point attrib read failed in tripoly\n");
             return(1);
           }
           inp->pointattributelist[i*natt+j] = attr;
        }
     }
     if(*nbm>0){
        if (fscanf(fp,"%d",&bmark)!=1) {
          fprintf(stderr,"point marker read failed in tripoly\n");
          return(1);
        }
        inp->pointmarkerlist[i] = bmark;
        if (bmark==99) *internal_bnd = 1;
     }
  }

  fscanf(fp,"^\n");
  fscanf(fp,"\n");
  if (fgets(buf,80,fp)==NULL) {
      fprintf(stderr,"segment header read failed in tripoly\n");
      return(1);
  }
  if (sscanf(buf,"%d %d",&ns, nbm)!=2) {
      fprintf(stderr,"segment header read failed in tripoly\n");
      return(1);
  }
  inp->numberofsegments = ns;
  if ((inp->segmentlist = (int *) malloc(
                 inp->numberofsegments * 2 * sizeof(int)))==0) {
      fprintf(stderr,"Malloc failed in tripoly\n");
      return(1);
  }
  if ((inp->segmentmarkerlist = (int *) malloc(
                 inp->numberofsegments * sizeof(int)))==0) {
      fprintf(stderr,"Malloc failed in tripoly\n");
      return(1);
  }
  for (i=0;i<ns;i++){ 
     inp->segmentmarkerlist[i] = 0;
     if (fscanf(fp,"%d %d %d",&is, &p1, &p2)!=3) { 
          fprintf(stderr,"seg read failed in tripoly\n");
          return(1);
     }
     inp->segmentlist[i*2] = p1-1;      
     inp->segmentlist[i*2+1] = p2-1;
         /* -1 in 2 lines above because point number is array address+1 */
     if(*nbm>0){
        if (fscanf(fp,"%d",&bmark)!=1) {
          fprintf(stderr,"seg marker read failed in tripoly\n");
          return(1);
        }
        inp->segmentmarkerlist[i] = bmark;
     }
  }

  if (fscanf(fp,"%d",&nh)!=1) {
    fprintf(stderr,"num holes read failed in tripoly\n");
    return(1);
  }
  inp->numberofholes = nh;
  if (nh > 0) {
    if ((inp->holelist = (REAL *) malloc(
                 inp->numberofholes * 2 * sizeof(REAL)))==0) {
      fprintf(stderr,"Malloc failed in tripoly\n");
      return(1);
    }
    for (i=0;i<nh;i++){ 
       if (fscanf(fp,"%d %f %f",&is, &xval, &yval)!=3) { 
           inp->holelist[i*2] = xval;      
           inp->holelist[i*2+1] = yval;
           fprintf(stderr,"hole read failed in tripoly\n");
           return(1);
        }
    }
  }

  if (fscanf(fp,"%d",&nr)!=1) {
    fprintf(stderr,"num regions read failed in tripoly\n");
    return(1);
  }
  inp->numberofregions = nr;
  if (nr > 0) {
    if ((inp->regionlist = (REAL *) malloc(
                   inp->numberofregions * 4 * sizeof(REAL)))==0) {
        fprintf(stderr,"Malloc failed in tripoly\n");
        return(1);
    }
    for (i=0;i<nr;i++){ 
       if (fscanf(fp,"%d %f %f %f %f",&is, &xval, &yval, &attr, &area)!=5) { 
          fprintf(stderr,"region read failed in tripoly\n");
          return(1);
        }
        inp->regionlist[i*4] = xval;      
        inp->regionlist[i*4+1] = yval;
        inp->regionlist[i*4+2] = attr;
        inp->regionlist[i*4+3] = area;
    }
  }
  fscanf(fp,"^\n");
  fscanf(fp,"\n");
  /* optional keyword lines */
  while (fgets(buf,80,fp)!=0) {
    if (sscanf(buf,"%s",str)!=1) {
        fprintf(stderr,"read failed in tripoly\n");
        return(1);
    }
    if (!strcmp(str,ELLE_KEY)) {
        if (optionaltxt) {
            strcpy(optionaltxt,buf);
            i=strlen(optionaltxt);
            if (optionaltxt[i-1]=='\n') optionaltxt[i-1] = '\0';
        }
        /* check if elle-poly numbers follow */
        curr_pos = ftell(fp);
        if (fgets(buf,80,fp)!=0) {
            if (sscanf(buf,"%d %d",&npp,&j)==2) {
              fprintf(stderr,"num elle-poly points read failed in tripoly\n");
              return(1);
            }
            if (npp>0) {
                *maxindx=j+1;
                if ((*ielle = (int *)
                     malloc( (*maxindx) * sizeof(int)))==0) {
                  fprintf(stderr,"Malloc failed in tripoly\n");
                  return(1);
                }
                for (i=0;i<*maxindx;i++) (*ielle)[i] = -1;
                for (i=0;i<npp;i++){ 
                  if (fscanf(fp,"%d %d",&j, &k)!=2) { 
                    fprintf(stderr,"elle-poly read failed in tripoly\n");
                    return(1);
                  }
                  if (j<*maxindx) (*ielle)[j] = k;
                  else fprintf(stderr,"elle index out of range %d\n",j);
                }
            }
            
            else fseek( fp,curr_pos,SEEK_SET );
        }
    }
    else if (!strcmp(str,AREA_KEY)) {
        if (sscanf(buf,"%s %f",str,&tmp)!=2) {
          fprintf(stderr,"area read failed in tripoly\n");
          return(1);
        }
        if (tmp>0.0) flags->area=(float)tmp;
    }
    else if (!strcmp(str,QUALITY_KEY)) {
        if (sscanf(buf,"%s %d",str,&i)!=2) {
          fprintf(stderr,"quality read failed in tripoly\n");
          return(1);
        }
        if (i>0) flags->quality=i;
    }
    fscanf(fp,"^\n");
    fscanf(fp,"\n");
  }

  /*printf("Input point set:\n\n");*/
  /*showvar(inp, 1, 0, 0, 1, 0, 0);*/
}

int WritePoly(struct triangulateio *out,char *name)
{
    int i, j;
    FILE *fp;

    if ((fp=fopen(name,"w"))==0)
        OnError("WritePoly",OPEN_ERR);
    fprintf(fp,"%d 2 %d ", out->numberofpoints,
                                  out->numberofpointattributes);
    fprintf(fp,"%d\n", (out->pointmarkerlist!=0?1:0));
    for (i=0;i<out->numberofpoints;i++) {
#ifdef SINGLE
        fprintf(fp,"%d %f %f",i,out->pointlist[i*2],
#else
        fprintf(fp,"%d %lf %lf",i,out->pointlist[i*2],
#endif
                                     out->pointlist[i*2+1]);
        for (j=0;j<out->numberofpointattributes;j++)
#ifdef SINGLE
          fprintf(fp," %f",
#else
          fprintf(fp," %lf",
#endif
            out->pointattributelist[i*out->numberofpointattributes+j]);
        if (out->pointmarkerlist) fprintf(fp," %d", out->pointmarkerlist[i]);
		fprintf(fp,"\n");
    }
    if (out->numberofedges>0) {
        fprintf(fp,"%d ",out->numberofedges);
        fprintf(fp,"%d\n", (out->edgemarkerlist!=0?1:0));
        for (i=0;i<out->numberofedges;i++) {
            fprintf(fp,"%d %d %d ",i,out->edgelist[i*2],
                                      out->edgelist[i*2+1]);
            if (out->edgemarkerlist)
                fprintf(fp," %d", out->edgemarkerlist[i]);
	    	fprintf(fp,"\n");
        }
    }
    if (out->numberofsegments>0) {
        fprintf(fp,"%d ",out->numberofsegments);
        fprintf(fp,"%d\n", (out->segmentmarkerlist!=0?1:0));
        for (i=0;i<out->numberofsegments;i++) {
            fprintf(fp,"%d %d %d ",i,out->segmentlist[i*2],
                                      out->segmentlist[i*2+1]);
            if (out->segmentmarkerlist)
                fprintf(fp," %d", out->segmentmarkerlist[i]);
	    	fprintf(fp,"\n");
        }
    }
    fprintf(fp,"0\n");
    fprintf(fp,"0\n");
    fclose(fp);
    return(0);
}

/*****************************************************

  tripolypts()   Triangulate polygons (from point list)
  This is pre voronoi and calls trivorpolypts
  with NULL as pointer to voronoi output

*****************************************************/
int tripolypts( struct triangulateio *out, 
				struct flagvals *flags,
             int numbndpts, double *xvals, double *yvals, 
             double *attrib_vals, int num, int num_attr )
{
	trivorpolypts(out,(struct triangulateio *)NULL,flags,
					numbndpts,xvals,yvals,attrib_vals,num,num_attr);
}

/*****************************************************

  trivorpolypts()   Triangulate polygons (from point list)
  if num>numbndpts, triangulate a polygon with internal
    pts and use YY flag so only these pts are used.
  Assumes boundary pts are first in the list, internal pts
    follow
  Writes point attribute values if attrib_vals not 0
    (presently, assumes 1 attribute).
  Voronoi normlist will be in vorout if voronoi flag set and 
	vorout !=0


*****************************************************/

int trivorpolypts( struct triangulateio *out, 
				struct triangulateio *vorout,
				struct flagvals *flags,
             int numbndpts, double *xvals, double *yvals, 
             double *attrib_vals, int num, int num_attr )
{
  char str[30], buf[10];
  struct triangulateio inp;
  int np,ns,natt,nbm,ndim,nh,nr;
  int i,j,ip,is,p1,p2,bmark;

  np=ns=natt=nbm=ndim=nh=nr=0;

  /**************************/
  /******* Setup Input *******/
  /**************************/

  initio( &inp );
  initio( out );
  if (vorout) {
	initio( vorout );
	flags->voronoi=1;
  }

  inp.numberofpoints = np = num;
  if ((inp.pointlist = (REAL *) malloc(
             inp.numberofpoints * 2 * sizeof(REAL)))==0) {
      fprintf(stderr,"Malloc failed in tripolypts\n");
      return(1);
  }
  if ((inp.pointmarkerlist = (int *) malloc(
             inp.numberofpoints * sizeof(int)))==0) {
      fprintf(stderr,"Malloc failed in tripolypts\n");
      return(1);
  }
  if (attrib_vals) {
      inp.numberofpointattributes=num_attr;
      if ((inp.pointattributelist = (REAL *) malloc(
             inp.numberofpointattributes * inp.numberofpoints *
                                         sizeof(REAL)))==0) {
          fprintf(stderr,"Malloc failed in tripolypts\n");
          return(1);
      }
      for(i=0;i<np*num_attr;i++)
          inp.pointattributelist[i] = attrib_vals[i];
  }
  for (i=0;i<np;i++){
      inp.pointlist[i*2] = xvals[i];
      inp.pointlist[i*2+1] = yvals[i];
      if (i<numbndpts) inp.pointmarkerlist[i] = 1;
      else inp.pointmarkerlist[i] = 0;
  }

  inp.numberofsegments = ns = numbndpts;
  if ((inp.segmentlist = (int *) malloc(
                 inp.numberofsegments * 2 * sizeof(int)))==0) {
      fprintf(stderr,"Malloc failed in tripolypts\n");
      return(1);
  }
  if ((inp.segmentmarkerlist = (int *) malloc(
                 inp.numberofsegments * sizeof(int)))==0) {
      fprintf(stderr,"Malloc failed in tripolypts\n");
      return(1);
  }
  for (i=0;i<ns;i++){ 
      inp.segmentlist[i*2] = i;      
      inp.segmentlist[i*2+1] = (i+1)%ns;
      inp.segmentmarkerlist[i] = 1;
  }

  /*
   * no holes
   */
  /*   Switches are chosen for the triangle routine as follows:  */
  /*   Read a PSLG (p), number everything from  zero (z),
  /*   and  produce an edge list (e), and a triangle neighbor list (n).
*/

  strcpy(str,"QzAen");
  if (ns>0) strcat(str,"p");
  if (flags->bndpts==0) {
      strcat(str,"Y");
  }
  if (num>numbndpts) {
      strcat(str,"Y");
  }
  if (flags->edges) strcat(str,"e");
  if (flags->voronoi && vorout!=0) {
      strcat(str,"v");
  }
  if (flags->area>0.0) {
      sprintf(buf,"a%g",flags->area);
      strcat(str,buf);
  }
#ifdef DEBUG
  WritePoly(&inp,"b4.poly");
#endif
  triangulate(str, &inp, out, vorout);


  /***********************/
  /******* Cleanup *******/
  /***********************/
  /* Free all allocated arrays, including those allocated by Triangle.
*/

  if (inp.regionlist==out->regionlist)  out->regionlist=NULL;
  cleanio( &inp );

  return 0;
}

void WriteVoronoiAsPoly(struct triangulateio *vorout)
{
    int i, j, extras=0, validedges=0, unum=0;
    int *normindex=0;

REAL eps=1.5e-6; // same as crossings.cc
                                                                                
    for (i=0, extras=0;i<vorout->numberofedges;i++) {
        if (vorout->edgelist[i*2]==-1||vorout->edgelist[i*2+1]==-1) {
            extras++;
        }
    }
    if (extras>0) {
        normindex = (int *)malloc(sizeof(int)*extras);
        for (i=0,j=0;i<vorout->numberofedges;i++) {
            if (vorout->edgelist[i*2]==-1||vorout->edgelist[i*2+1]==-1)
                normindex[j++]=i;
        }
    }
    validedges = vorout->numberofedges-extras;
    /*extras=0;*/
        printf("%d 2 %d\n",vorout->numberofpoints+unum,
vorout->numberofpointattributes);
    for (i=0;i<vorout->numberofpoints;i++)  {
    for (j=0;j<vorout->numberofpoints;j++)  {
if (j!=i  &&
(fabs((double)(vorout->pointlist[j*2]-vorout->pointlist[i*2]))<eps)
  && (fabs((double)(vorout->pointlist[j*2+1]-vorout->pointlist[i*2+1]))<eps))
    printf("match %d %d\n",i,j);
    }
        printf("%d %f %f %d\n",i,(float)vorout->pointlist[i*2],
           (float)vorout->pointlist[i*2+1],
           (int)vorout->pointattributelist[i]);
    }
/*
    for (i=0;i<extras;i++)  {
       printf("%d %f %f -1\n",vorout->numberofpoints+i,
           (float)vorout->normlist[normindex[i]*2],
           (float)vorout->normlist[normindex[i]*2+1]);
    }
*/
    printf("%d 0\n",validedges);
    for (i=0,j=0, extras=0;i<vorout->numberofedges;i++) {
       if (vorout->edgelist[i*2]!=-1 && vorout->edgelist[i*2+1]!=-1) {
           printf("%d %d %d\n", j,vorout->edgelist[i*2],
                    vorout->edgelist[i*2+1]);
       j++;
       }
    }
    printf("0\n0\n");
}


void initio( struct triangulateio *io )
{
  /* Not needed if -N switch used. */
  io->pointlist=(REAL *)NULL;
  /* Not needed if -N switch used or number of point attributes is zero: */
  io->pointattributelist=(REAL *)NULL;
  /* Not needed if -N or -B switch used. */
  io->pointmarkerlist=(int *)NULL;

  io->numberofpoints=0;
  io->numberofpointattributes=0;

  /* Not needed if -E switch used. */
  io->trianglelist=(int *)NULL;
  /* Not needed if -E switch used or number of triangle attributes is zero: */
  io->triangleattributelist=(REAL *)NULL;
  io->trianglearealist=(REAL *)NULL;

  io->neighborlist=(int *)NULL; /* Needed only if -n switch used. */
  io->numberoftriangles=0;
  io->numberofcorners=0;
  io->numberoftriangleattributes=0;

  /* Needed only if segments are output (-p or -c) and -P not used: */
  io->segmentlist=(int *)NULL;
  /* Needed only if segments are output (-p or -c) and -P and -B not used: */
  io->segmentmarkerlist=(int *)NULL;
  io->numberofsegments=0;

  io->holelist=(REAL *)NULL;
  io->numberofholes=0;
  io->regionlist=(REAL *)NULL;
  io->numberofregions=0;

  /* Needed only if -e switch used. */
  io->edgelist=(int *)NULL;
  /* Needed if -e used and -B not used. */
  io->edgemarkerlist=(int *)NULL;
  io->normlist=(REAL *)NULL;
  io->numberofedges=0;

}

void cleanio( struct triangulateio *io )
{
  if (io->pointlist!=NULL) free(io->pointlist);
  if (io->pointattributelist!=NULL) free(io->pointattributelist);
  if (io->pointmarkerlist!=NULL) free(io->pointmarkerlist);
  if (io->trianglelist!=NULL) free(io->trianglelist);
  if (io->triangleattributelist!=NULL) free(io->triangleattributelist);
  if (io->trianglearealist!=NULL) free(io->trianglearealist);
  if (io->neighborlist!=NULL) free(io->neighborlist);

  if (io->segmentlist!=NULL) free(io->segmentlist);
  if (io->segmentmarkerlist!=NULL) free(io->segmentmarkerlist);
  if (io->holelist!=NULL) free(io->holelist);
  if (io->regionlist!=NULL) free(io->regionlist);

  if (io->edgelist!=NULL) free(io->edgelist);
  if (io->edgemarkerlist!=NULL) free(io->edgemarkerlist);
  if (io->normlist!=NULL) free(io->normlist);

  initio(io);

}


/*****************************************************************************/
/*                                                                           */
/*  showvar()   Print the input or output.                                    */
/*                 for debugging                                             */
/*                                                                           */
/*****************************************************************************/

void showvar(io, markers, reporttriangles, reportneighbors, reportsegments,
            reportedges, reportnorms)

struct triangulateio *io;
int markers;
int reporttriangles;
int reportneighbors;
int reportsegments;
int reportedges;
int reportnorms;
{
  int i, j;

  for (i = 0; i < io->numberofpoints; i++) {
    printf("Point %4d:", i);
    for (j = 0; j < 2; j++) {
      printf("  %.6g", io->pointlist[i * 2 + j]);
    }
    if (io->numberofpointattributes > 0) {
      printf("   attributes");
    }
    for (j = 0; j < io->numberofpointattributes; j++) {
      printf("  %.6g",
             io->pointattributelist[i * io->numberofpointattributes + j]);
    }
    if (markers) {
      printf("   marker %d\n", io->pointmarkerlist[i]);
    } else {
      printf("\n");
    }
  }
  printf("\n");

  if (reporttriangles || reportneighbors) {
    for (i = 0; i < io->numberoftriangles; i++) {
      if (reporttriangles) {
        printf("Triangle %4d points:", i);
        for (j = 0; j < io->numberofcorners; j++) {
          printf("  %4d", io->trianglelist[i * io->numberofcorners + j]);
        }
        if (io->numberoftriangleattributes > 0) {
          printf("   attributes");
        }
        for (j = 0; j < io->numberoftriangleattributes; j++) {
          printf("  %.6g", io->triangleattributelist[i *
                                         io->numberoftriangleattributes + j]);
        }
        printf("\n");
      }
      if (reportneighbors) {
        printf("Triangle %4d neighbors:", i);
        for (j = 0; j < 3; j++) {
          printf("  %4d", io->neighborlist[i * 3 + j]);
        }
        printf("\n");
      }
    }
    printf("\n");
  }

  if (reportsegments) {
    for (i = 0; i < io->numberofsegments; i++) {
      printf("Segment %4d points:", i);
      for (j = 0; j < 2; j++) {
        printf("  %4d", io->segmentlist[i * 2 + j]);
      }
      if (markers) {
        printf("   marker %d\n", io->segmentmarkerlist[i]);
      } else {
        printf("\n");
      }
    }
    printf("\n");
  }

  if (reportedges) {
    for (i = 0; i < io->numberofedges; i++) {
      printf("Edge %4d points:", i);
      for (j = 0; j < 2; j++) {
        printf("  %4d", io->edgelist[i * 2 + j]);
      }
      if (reportnorms && (io->edgelist[i * 2 + 1] == -1)) {
        for (j = 0; j < 2; j++) {
          printf("  %.6g", io->normlist[i * 2 + j]);
        }
      }
      if (markers) {
        printf("   marker %d\n", io->edgemarkerlist[i]);
      } else {
        printf("\n");
      }
    }
    printf("\n");
  }
}

