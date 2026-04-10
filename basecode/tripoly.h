 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _tripoly_h
#define _tripoly_h

#ifndef _triangle_h
#include "triangle.h"
#endif

struct flagvals {
    unsigned char midptnodes;
    unsigned char bndpts;
    unsigned char edges;
    unsigned char voronoi;
    int quality;
    float area;
};

#ifndef SUN

#ifdef __cplusplus
extern "C" {
#endif
int tripoly(struct triangulateio *out, char *infile,
            struct flagvals *flags, char *optionaltxt,
            int *npolyp,int **ipoly);

int readpolyinput( struct triangulateio *inp, char *infile,
             struct flagvals *flags, char *optionaltxt,
             int *maxindx, int **ielle, int*nbm, int *internal_bnd);
int tripolypts(struct triangulateio *out,
            struct flagvals *flags,
            int numbndpts, double *xvals, double *yvals, double *attr,
            int num, int num_attr );
int trivorpolypts( struct triangulateio *out, 
				struct triangulateio *vorout,
				struct flagvals *flags,
             int numbndpts, double *xvals, double *yvals, 
             double *attrib_vals, int num, int num_attr );
int WritePoly(struct triangulateio *out, char *fname);
void WriteVoronoiAsPoly(struct triangulateio *vorout);
void initio(struct triangulateio *io);
void cleanio(struct triangulateio *io);
#ifdef __cplusplus
}
#endif
#else /* SUN */
int tripoly();
int tripolypts();
void initio();
void cleanio();
#endif /* SUN */

#endif
