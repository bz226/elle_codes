#include <stdio.h>
#include <math.h>
#include "general.h"

#ifdef __cplusplus
extern "C" {
#endif
int crossingstest_( double  *pgon, int *nverts, double  point[2],
                    int *res );
int pointonsegment_( double  *x1, double *y1,
                    double *x2, double *y2,
                    double *x, double *y,
                    int *res );
#ifdef __cplusplus
}
#endif

/* ======= Crossings algorithm ============================================ */

/*
 * calling Graphics Gem in general.cc from f77
 */
int crossingstest_( double  *pgon, int *nverts, double  point[2],
                    int *res )
{
     *res = CrossingsTest(pgon,*nverts,point);
}

int pointonsegment_(  double  *x1, double *y1,
                    double *x2, double *y2,
                    double *x, double *y,
                    int *res )
{
     *res = PointOnSegment(*x1,*y1,*x2,*y2,*x,*y);
}
