 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <stdio.h>
#include <math.h>
#include "general.h"
#include "convert.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

#define XY 0

/*
 * ANSI C code from the article
 * "Centroid of a Polygon"
 * by Gerard Bashein and Paul R. Detmer,
	(gb@locke.hs.washington.edu, pdetmer@u.washington.edu)
 * in "Graphics Gems IV", Academic Press, 1994
 */

/*********************************************************************
polyCentroid: Calculates the centroid (xCentroid, yCentroid) and area
of a polygon, given its vertices (x[0], y[0]) ... (x[n-1], y[n-1]). It
is assumed that the contour is closed, i.e., that the vertex following
(x[n-1], y[n-1]) is (x[0], y[0]).  The algebraic sign of the area is
positive for counterclockwise ordering of vertices in x-y plane;
otherwise negative.

Returned values:  0 for normal execution;  1 if the polygon is
degenerate (number of vertices < 3);  and 2 if area = 0 (and the
centroid is undefined).
**********************************************************************/
int polyCentroid(double x[], double y[], int n,
		 double *xCentroid, double *yCentroid, double *area)
     {
     register int i, j;
     double ai, atmp = 0, xtmp = 0, ytmp = 0;
     if (n < 3) return 1;
     for (i = n-1, j = 0; j < n; i = j, j++)
	  {
	  ai = x[i] * y[j] - x[j] * y[i];
	  atmp += ai;
	  xtmp += (x[j] + x[i]) * ai;
	  ytmp += (y[j] + y[i]) * ai;
	  }
     *area = atmp / 2;
     if (atmp != 0)
	  {
	  *xCentroid =	xtmp / (3 * atmp);
	  *yCentroid =	ytmp / (3 * atmp);
	  return 0;
	  }
     return 2;
     }
/*
 * only change is that  x and y are passed as float
 * instead of double
 */
int polyCentroid(float x[], float y[], int n,
         double *xCentroid, double *yCentroid, double *area)
     {
     register int i, j;
     double ai, atmp = 0, xtmp = 0, ytmp = 0;
     if (n < 3) return 1;
     for (i = n-1, j = 0; j < n; i = j, j++)
      {
      ai = (double)x[j] * y[i] - (double)x[i] * y[j];
      /*ai = (double)x[i] * y[j] - (double)x[j] * y[i];*/
      atmp += ai;
      xtmp += ((double)x[j] + x[i]) * ai;
      ytmp += ((double)y[j] + y[i]) * ai;
      }
     *area = atmp / 2;
     if (atmp != 0)
      {
      *xCentroid =  xtmp / (3 * atmp);
      *yCentroid =  ytmp / (3 * atmp);
      return 0;
      }
     return 2;
     }

#define WINDING

#define X	0
#define Y	1

#ifndef TRUE
#define TRUE	1
#define FALSE	0
#endif


/* test if a & b are within epsilon.  Favors cases where a < b */
#define Near(a,b,eps)	( ((b)-(eps)<(a)) && ((a)-(eps)<(b)) )


/* ======= Crossings algorithm ============================================ */

/* Shoot a test ray along +X axis.  The strategy, from MacMartin, is to
 * compare vertex Y values to the testing point's Y and quickly discard
 * edges which are entirely to one side of the test ray.
 *
 * Input 2D polygon _pgon_ with _numverts_ number of vertices and test point
 * _point_, returns 1 if inside, 0 if outside.	WINDING and CONVEX can be
 * defined for this test.
 */
int CrossingsTest( double  *pgon, int numverts, double  point[2])
{
#ifdef	WINDING
register int	crossings ;
#endif
register int	j, yflag0, yflag1, inside_flag, xflag0 ;
register double ty, tx, *vtx0, *vtx1 ;
#ifdef	CONVEX
register int	line_flag ;
#endif

    tx = point[X] ;
    ty = point[Y] ;

    vtx0 = pgon+ 2*(numverts-1) ;
    /* get test bit for above/below X axis */
    yflag0 = ( vtx0[Y] >= ty ) ;
    vtx1 = pgon;

#ifdef	WINDING
    crossings = 0 ;
#else
    inside_flag = 0 ;
#endif
#ifdef	CONVEX
    line_flag = 0 ;
#endif
    for ( j = numverts+1 ; --j ; ) {

	yflag1 = ( vtx1[Y] >= ty ) ;
	/* check if endpoints straddle (are on opposite sides) of X axis
	 * (i.e. the Y's differ); if so, +X ray could intersect this edge.
	 */
	if ( yflag0 != yflag1 ) {
	    xflag0 = ( vtx0[X] >= tx ) ;
	    /* check if endpoints are on same side of the Y axis (i.e. X's
	     * are the same); if so, it's easy to test if edge hits or misses.
	     */
	    if ( xflag0 == ( vtx1[X] >= tx ) ) {

		/* if edge's X values both right of the point, must hit */
#ifdef	WINDING
		if ( xflag0 ) crossings += ( yflag0 ? -1 : 1 ) ;
#else
		if ( xflag0 ) inside_flag = !inside_flag ;
#endif
	    } else {
		/* compute intersection of pgon segment with +X ray, note
		 * if >= point's X; if so, the ray hits it.
		 */
		if ( (vtx1[X] - (vtx1[Y]-ty)*
		     ( vtx0[X]-vtx1[X])/(vtx0[Y]-vtx1[Y])) >= tx ) {
#ifdef	WINDING
		    crossings += ( yflag0 ? -1 : 1 ) ;
#else
		    inside_flag = !inside_flag ;
#endif
		}
	    }
#ifdef	CONVEX
	    /* if this is second edge hit, then done testing */
	    if ( line_flag ) goto Exit ;

	    /* note that one edge has been hit by the ray's line */
	    line_flag = TRUE ;
#endif
	}

	/* move to next pair of vertices, retaining info as possible */
	yflag0 = yflag1 ;
	vtx0 = vtx1 ;
	vtx1 += 2 ;
    }
#ifdef	CONVEX
    Exit: ;
#endif
#ifdef	WINDING
    /* test if crossings is not zero */
    inside_flag = (crossings != 0) ;
#endif

    return( inside_flag ) ;
}

/* ======= Crossings Multiply algorithm =================================== */

/*
 * This version is usually somewhat faster than the original published in
 * Graphics Gems IV; by turning the division for testing the X axis crossing
 * into a tricky multiplication test this part of the test became faster,
 * which had the additional effect of making the test for "both to left or
 * both to right" a bit slower for triangles than simply computing the
 * intersection each time.  The main increase is in triangle testing speed,
 * which was about 15% faster; all other polygon complexities were pretty much
 * the same as before.  On machines where division is very expensive (not the
 * case on the HP 9000 series on which I tested) this test should be much
 * faster overall than the old code.  Your mileage may (in fact, will) vary,
 * depending on the machine and the test data, but in general I believe this
 * code is both shorter and faster.  This test was inspired by unpublished
 * Graphics Gems submitted by Joseph Samosky and Mark Haigh-Hutchinson.
 * Related work by Samosky is in:
 *
 * Samosky, Joseph, "SectionView: A system for interactively specifying and
 * visualizing sections through three-dimensional medical image data",
 * M.S. Thesis, Department of Electrical Engineering and Computer Science,
 * Massachusetts Institute of Technology, 1993.
 *
 */

/* Shoot a test ray along +X axis.  The strategy is to compare vertex Y values
 * to the testing point's Y and quickly discard edges which are entirely to one
 * side of the test ray.  Note that CONVEX and WINDING code can be added as
 * for the CrossingsTest() code; it is left out here for clarity.
 *
 * Input 2D polygon _pgon_ with _numverts_ number of vertices and test point
 * _point_, returns 1 if inside, 0 if outside.
 */
int CrossingsMultiplyTest( double  *pgon, int numverts, double  point[2])
{
register int	j, yflag0, yflag1, inside_flag ;
register double	ty, tx, *vtx0, *vtx1 ;

    tx = point[X] ;
    ty = point[Y] ;

    vtx0 = &pgon[numverts-1] ;
    /* get test bit for above/below X axis */
    yflag0 = ( vtx0[Y] >= ty ) ;
    vtx1 = &pgon[0] ;

    inside_flag = 0 ;
    for ( j = numverts+1 ; --j ; ) {

	yflag1 = ( vtx1[Y] >= ty ) ;
	/* Check if endpoints straddle (are on opposite sides) of X axis
	 * (i.e. the Y's differ); if so, +X ray could intersect this edge.
	 * The old test also checked whether the endpoints are both to the
	 * right or to the left of the test point.  However, given the faster
	 * intersection point computation used below, this test was found to
	 * be a break-even proposition for most polygons and a loser for
	 * triangles (where 50% or more of the edges which survive this test
	 * will cross quadrants and so have to have the X intersection computed
	 * anyway).  I credit Joseph Samosky with inspiring me to try dropping
	 * the "both left or both right" part of my code.
	 */
	if ( yflag0 != yflag1 ) {
	    /* Check intersection of pgon segment with +X ray.
	     * Note if >= point's X; if so, the ray hits it.
	     * The division operation is avoided for the ">=" test by checking
	     * the sign of the first vertex wrto the test point; idea inspired
	     * by Joseph Samosky's and Mark Haigh-Hutchinson's different
	     * polygon inclusion tests.
	     */
	    if ( ((vtx1[Y]-ty) * (vtx0[X]-vtx1[X]) >=
		    (vtx1[X]-tx) * (vtx0[Y]-vtx1[Y])) == yflag1 ) {
		inside_flag = !inside_flag ;
	    }
	}

	/* Move to the next pair of vertices, retaining info as possible. */
	yflag0 = yflag1 ;
	vtx0 = vtx1 ;
	vtx1 += 2 ;
    }

    return( inside_flag ) ;
}


/* FROM GRAPHICS GEMS II
 *    altered for floating point calculations
 */
/* lines_intersect:  AUTHOR: Mukesh Prasad
 *
 *   This function computes whether two line segments,
 *   respectively joining the input points (x1,y1) -- (x2,y2)
 *   and the input points (x3,y3) -- (x4,y4) intersect.
 *   If the lines intersect, the output variables x, y are
 *   set to coordinates of the point of intersection.
 *
 *   All values are in integers.  The returned value is rounded
 *   to the nearest integer point.
 *
 *   If non-integral grid points are relevant, the function
 *   can easily be transformed by substituting floating point
 *   calculations instead of integer calculations.
 *
 *   Entry
 *        x1, y1,  x2, y2   Coordinates of endpoints of one segment.
 *        x3, y3,  x4, y4   Coordinates of endpoints of other segment.
 *
 *   Exit
 *        x, y              Coordinates of intersection point.
 *
 *   The value returned by the function is one of:
 *
 *        DONT_INTERSECT    0
 *        DO_INTERSECT      1
 *        COLLINEAR         2
 *
 * Error conditions:
 *
 *     Depending upon the possible ranges, and particularly on 16-bit
 *     computers, care should be taken to protect from overflow.
 *
 *     In the following code, 'long' values have been used for this
 *     purpose, instead of 'int'.
 *
 */

#define	DONT_INTERSECT    0
#define	DO_INTERSECT      1
#define COLLINEAR         2

/**************************************************************
 *                                                            *
 *    NOTE:  The following macro to determine if two numbers  *
 *    have the same sign, is for 2's complement number        *
 *    representation.  It will need to be modified for other  *
 *    number systems.                                         *
 *                                                            *
 **************************************************************/

#define SAME_SIGNS( a, b )	\
		(((long) ((unsigned long) a ^ (unsigned long) b)) >= 0 )

int lines_intersect( double x1, double y1,   /* First line segment */
		     double x2, double y2,

		     double x3, double y3,   /* Second line segment */
		     double x4, double y4,

		     double *x,
		     double *y         /* Output value:
		                * point of intersection */
               )
{
    double a1, a2, b1, b2, c1, c2; /* Coefficients of line eqns. */
    double r1, r2, r3, r4;         /* 'Sign' values */
    double denom, offset, num;     /* Intermediate values */
	double eps = 1e-9;            /* used in COLINEAR determination */

    /* Compute a1, b1, c1, where line joining points 1 and 2
     * is "a1 x  +  b1 y  +  c1  =  0".
     */

    a1 = y2 - y1;
    b1 = x1 - x2;
    c1 = x2 * y1 - x1 * y2;

    /* Compute r3 and r4.
     */


    r3 = a1 * x3 + b1 * y3 + c1;
    r4 = a1 * x4 + b1 * y4 + c1;

    /* Check signs of r3 and r4.  If both point 3 and point 4 lie on
     * same side of line 1, the line segments do not intersect.
     */

    if ( r3 != 0 &&
         r4 != 0 &&
/*
         SAME_SIGNS( r3, r4 ))
*/
         ((r3>0.0&&r4>0.0)||(r3<0.0&&r4<0.0)))
        return ( DONT_INTERSECT );

    /* Compute a2, b2, c2 */

    a2 = y4 - y3;
    b2 = x3 - x4;
    c2 = x4 * y3 - x3 * y4;

    /* Compute r1 and r2 */

    r1 = a2 * x1 + b2 * y1 + c2;
    r2 = a2 * x2 + b2 * y2 + c2;

    /* Check signs of r1 and r2.  If both point 1 and point 2 lie
     * on same side of second line segment, the line segments do
     * not intersect.
     */

    if ( r1 != 0 &&
         r2 != 0 &&
/*
         SAME_SIGNS( r1, r2 ))
*/
         (r1>0.0&&r2>0.0)||(r1<0.0&&r2<0.0))
        return ( DONT_INTERSECT );

    /* Line segments intersect: compute intersection point. 
     */

    denom = a1 * b2 - a2 * b1;
/*
    if ( denom == 0 )
*/
    if ( denom < eps && denom > -eps )
        return ( COLLINEAR );
    offset = denom < 0 ? - denom / 2 : denom / 2;

    /* The denom/2 is to get rounding instead of truncating.  It
     * is added or subtracted to the numerator, depending upon the
     * sign of the numerator.
     */

    num = b1 * c2 - b2 * c1;
/*
    *x = ( num < 0 ? num - offset : num + offset ) / denom;
*/
    *x = num/denom;

    num = a2 * c1 - a1 * c2;
/*
    *y = ( num < 0 ? num - offset : num + offset ) / denom;
*/
    *y = num/denom;
    return ( DO_INTERSECT );
    } /* lines_intersect */

/*
 * Based on lines_intersect but finds the intersection
 * of the lines defined by the segments x1y1-x2y2 and x3y3-x4y4
 */
int intersection_pt( double x1, double y1,   /* First line segment */
		     double x2, double y2,

		     double x3, double y3,   /* Second line segment */
		     double x4, double y4,

		     double *x,
		     double *y         /* Output value:
		                * point of intersection */
               )
{
    double a1, a2, b1, b2, c1, c2; /* Coefficients of line eqns. */
    double denom, num;     /* Intermediate values */
	double eps = 1e-9;

    /* Compute a1, b1, c1, where line joining points 1 and 2
     * is "a1 x  +  b1 y  +  c1  =  0".
     */

    a1 = y2 - y1;
    b1 = x1 - x2;
    c1 = x2 * y1 - x1 * y2;


    /* Compute a2, b2, c2 */

    a2 = y4 - y3;
    b2 = x3 - x4;
    c2 = x4 * y3 - x3 * y4;

    /*  compute intersection point. 
     */

    denom = a1 * b2 - a2 * b1;
    if ( denom < eps && denom > -eps )
        return ( COLLINEAR );

    num = b1 * c2 - b2 * c1;
    *x = num/denom;

    num = a2 * c1 - a1 * c2;
    *y = num/denom;

    return ( DO_INTERSECT );
} /* intersection_pt */


/* A main program to test the function intersection_pt

main()
{
    double x1, x2, x3, x4, y1, y2, y3, y4;
    double x, y;

    for (;;) {
        printf( "X1, Y1: " );
	scanf( "%lf %lf", &x1, &y1 );
        printf( "X2, Y2: " );
	scanf( "%lf %lf", &x2, &y2 );
        printf( "X3, Y3: " );
	scanf( "%lf %lf", &x3, &y3 );
        printf( "X4, Y4: " );
	scanf( "%lf %lf", &x4, &y4 );

        switch ( intersection_pt( x1, y1, x2, y2, x3, y3, x4, y4, &x, &y )) {
            case DONT_INTERSECT:
			 printf( "Lines don't intersect\n" );
			 break;
            case COLLINEAR:
                         printf( "Lines are collinear\n" );
                         break;
            case DO_INTERSECT:
			 printf( "Lines intersect at %lf,%lf\n", x, y );
                         break;
        }
    }
}
 */

#if XY
int angle(float x,float y,
          float x1,float y1,
          float x2,float y2,
          float *angle)
{
    /*
     * returns the angle between the lines x,y - x1,y1 (a1X+b1Y+c1=0)
     * and x,y - x2,y2 (a2X+b2Y+c2=0)
     * tan ang = (a1b2 - a2b1)/(a1a2 + b1b2)
     */
    float a1,a2,b1,b2,num,denom;

    /* Compute a1, b1 */
    a1 = y - y1;
    b1 = x1 - x;
    /* Compute a2, b2 */
    a2 = y - y2;
    b2 = x2 - x;

    num = a1*b2- a2*b1;
    denom = a1*a2 + b1*b2;
    if (num==0) *angle=0;
    else if (denom==0) *angle=PI_2;
    else *angle = atanf(num/denom);
    return(0);
}
#endif
/*
 * version of angle which uses double parameters
 * elle2.2  Feb 01
 */
int angle(double x,double y,double x1,double y1,double x2,double y2,
          double *angle)
{
    /*
     * returns the angle between the lines x,y - x2,y2
     * and x,y - x1,y1 (anticlockwise is +ve)
     */
    double phi;
    double a,b,sinphi,cosphi,yval,xval;

    phi = atan2((y2 - y),(x2 - x));
    sinphi = sin(phi);
    cosphi = cos(phi);
    a = x1 - x;
    b = y1 - y;
    yval = b*cosphi - a*sinphi;
    xval = a*cosphi + b*sinphi;

    if (xval==0.0 && yval==0.0) {
         /*fprintf(stderr,"angle:zero-points coincident\n");*/
         *angle = 0.0;
    }
    else *angle = atan2(yval,xval);
    return(0);
}


int angle(float x,float y,float x1,float y1,float x2,float y2,float *fangle)
{
    /*
     * returns the angle between the lines x,y - x2,y2
     * and x,y - x1,y1 (anticlockwise is +ve)
     */
    double dblang=0;
    int err = 0;

    err = angle((double)x,(double)y,(double)x1,(double)y1,
                    (double)x2,(double)y2,&dblang);

    if (err==0) *fangle = (float)dblang;
    return(err);
}

/*
 * version of angle changed March 00
 * extra code for angle of 0 and 180
 */

int angle0(float x,float y,float x1,float y1,float x2,float y2,float *fangle)
{
    /*
     * returns the angle between the lines x,y - x2,y2
     * and x,y - x1,y1 (anticlockwise is +ve)
     */
    int err=0;
    double dblang=0;

    err = angle0((double)x,(double)y,(double)x1,(double)y1,
                 (double)x2,(double)y2,&dblang);
    if (err==0) *fangle = (float)dblang;
    return(err);
}

int angle0(double x,double y,double x1,double y1,double x2,double y2,
           double *angle)
{
    /*
     * returns the angle between the lines x,y - x2,y2
     * and x,y - x1,y1 (anticlockwise is +ve)
     */
    int err=0;
    double phi;
    double a1,a2,b1,b2,sinphi,cosphi,yval,xval;

    /* Compute a1, b1 */
    a1 = x1 - x;
    b1 = y1 - y;
    /* Compute a2, b2 */
    a2 = x2 - x;
    b2 = y2 - y;
    phi = atan2(b2,a2);
    sinphi = sin(phi);
    cosphi = cos(phi);
    yval = b1*cosphi - a1*sinphi;
    xval = a1*cosphi + b1*sinphi;

    if (xval==0.0 && yval==0.0) {
        if (b1!=b2 && ((b1<=0 && b2>=0) || (b2<=0 && b1>=0)))
           *angle = PI_2;
        else if (a1!=a2 && ((a1<=0 && a2>=0) || (a2<=0 && a1>=0)))
           *angle = PI_2;
        else {
            *angle = 0;
            err = 1;
        }
    }
    else {
        *angle = atan2(yval,xval);
    }
    return(err);
}

/* A main program to test the function angle

main()
{
    float x1, x2, y1, y2;
    float x, y;
    float ang;

    for (;;) {
        printf( "X, Y: " );
	scanf( "%f %f", &x, &y );
        printf( "X1, Y1: " );
	scanf( "%f %f", &x1, &y1 );
        printf( "X2, Y2: " );
	scanf( "%f %f", &x2, &y2 );

    angle(x,y,x1,y1,x2,y2,&ang);
    printf("angle = %f\n",ang);
    }
}
 */

int rotate_coords(double x, double y,
                  double x_orig, double y_orig,
                  double *x_new, double *y_new,
                  double angle)
{
    /* angle in radians
     * clockwise rotation by angle a - multiply by matrix
     *            | cos a   -sin a |
     *        M = | sin a    cos a |
     *
     *  x_new,y_new
     *            \
     *        angle\
     *   x,y  ______\x_orig,y_orig
     */
    *x_new = cos(angle)*(x-x_orig) - sin(angle)*(y-y_orig) + x_orig;
    *y_new = sin(angle)*(x-x_orig) + cos(angle)*(y-y_orig) + y_orig;
    return(0);
}

int rotate_coords_ac(double x, double y,
                  double x_orig, double y_orig,
                  double *x_new, double *y_new,
                  double angle)
{
    /* angle in radians
     * counterclockwise rotation by angle a - multiply by matrix
     *            | cos a   -sin a |
     *        M = | sin a    cos a |
     *
     *  x_new,y_new
     *         /
     *        /angle
     *   x,y /_______ x_orig,y_orig
     */
    *x_new = cos(-angle)*(x-x_orig) - sin(-angle)*(y-y_orig) + x_orig;
    *y_new = sin(-angle)*(x-x_orig) + cos(-angle)*(y-y_orig) + y_orig;
    return(0);
}


/* 
A Fast 2D Point-On-Line Test
by Alan Paeth
from "Graphics Gems", Academic Press, 1990
*/

#define ABS(a)          (((a)<0) ? -(a) : (a))

/* find minimum of a and b */
#define MIN(a,b)        (((a)<(b))?(a):(b))

/* find maximum of a and b */
#define MAX(a,b)        (((a)>(b))?(a):(b))

int PntOnLine(int ipx,int ipy,int iqx,int iqy,int itx,int ity)
   {
/*
 * given a line through P:(px,py) Q:(qx,qy) and T:(tx,ty)
 * return 0 if T is not on the line through      <--P--Q-->
 *        1 if T is on the open ray ending at P: <--P
 *        2 if T is on the closed interior along:   P--Q
 *        3 if T is on the open ray beginning at Q:    Q-->
 *
 * Example: consider the line P = (3,2), Q = (17,7). A plot
 * of the test points T(x,y) (with 0 mapped onto '.') yields:
 *
 *     8| . . . . . . . . . . . . . . . . . 3 3
 *  Y  7| . . . . . . . . . . . . . . 2 2 Q 3 3    Q = 2
 *     6| . . . . . . . . . . . 2 2 2 2 2 . . .
 *  a  5| . . . . . . . . 2 2 2 2 2 2 . . . . .
 *  x  4| . . . . . 2 2 2 2 2 2 . . . . . . . .
 *  i  3| . . . 2 2 2 2 2 . . . . . . . . . . .
 *  s  2| 1 1 P 2 2 . . . . . . . . . . . . . .    P = 2
 *     1| 1 1 . . . . . . . . . . . . . . . . .
 *      +--------------------------------------
 *        1 2 3 4 5 X-axis 10        15      19
 *
 * Point-Line distance is normalized with the Infinity Norm
 * avoiding square-root code and tightening the test vs the
 * Manhattan Norm. All math is done on the field of integers.
 * The latter replaces the initial ">= MAX(...)" test with
 * "> (ABS(qx-px) + ABS(qy-py))" loosening both inequality
 * and norm, yielding a broader target line for selection.
 * The tightest test is employed here for best discrimination
 * in merging collinear (to grid coordinates) vertex chains
 * into a larger, spanning vectors within the Lemming editor.
 */

    long px,py,qx,qy,tx,ty;

    px=(long)ipx;
    py=(long)ipy;
    qx=(long)iqx;
    qy=(long)iqy;
    tx=(long)itx;
    ty=(long)ity;
    if ( ABS((qy-py)*(tx-px)-(ty-py)*(qx-px)) >=
        (MAX(ABS(qx-px), ABS(qy-py)))) return(0);
    if (((qx<px)&&(px<tx)) || ((qy<py)&&(py<ty))) return(1);
    if (((tx<px)&&(px<qx)) || ((ty<py)&&(py<qy))) return(1);
    if (((px<qx)&&(qx<tx)) || ((py<qy)&&(qy<ty))) return(3);
    if (((tx<qx)&&(qx<px)) || ((ty<qy)&&(qy<py))) return(3);
    return(2);
    }

int PointOnSegment(double x1, double y1,
                   double x2, double y2,
                   double x, double y)
{
    int res=0;

    /*
     * this eps is smaller than that in the equiv basil
     * pntonsegment (c_funcs.c). Ensures that nodes very close
     * to a boundary will be moved to the boundary by elle2poly.
     * Else can have case where elle2poly marks a point as being
     * on a boundary but basil just fails and does not set bnd
     * conditions.. (Fix by having basil use bnd info from trimesh -
     *               needs new ibctype flag)
     */
    double eps = 3.0e-6;
    double xmin,xmax,ymin,ymax;
    double val1,val2;

    if (x1<x2) {
        xmin = x1;
        xmax = x2;
    }
    else {
        xmin = x2;
        xmax = x1;
    }
    if (y1<y2) {
        ymin = y1;
        ymax = y2;
    }
    else {
        ymin = y2;
        ymax = y1;
    }
    if ((x > xmin-eps)&&(x < xmax+eps)&&
        (y > ymin-eps)&&(y < ymax+eps)) {
        val1 = ABS((y-y1)*(xmax-xmin));
        val2 = ABS((ymax-ymin)*(x-x1));
        if (val1<=(val2+eps) && val1>=(val2-eps)) res=2;
    }
    
    return((res==2)?1:0);

}

int PointOnSegment(float x1, float y1,
                   float x2, float y2,
                   float x, float y)
{
    int res=0;

#if XY
    int ix1, ix2, iy1, iy2, ix, iy;
    if (x1>=0) ix1 = (int)((x1+5E-7)*1.0E6);
    else ix1 = (int)((x1-5E-7)*1.0E6);
    if (y1>=0) iy1 = (int)((y1+5E-7)*1.0E6);
    else iy1 = (int)((y1-5E-7)*1.0E6);
    if (x2>=0) ix2 = (int)((x2+5E-7)*1.0E6);
    else ix2 = (int)((x2-5E-7)*1.0E6);
    if (y2>=0) iy2 = (int)((y2+5E-7)*1.0E6);
    else iy2 = (int)((y2-5E-7)*1.0E6);
    if (x>=0) ix = (int)((x+5E-7)*1.0E6);
    else  ix = (int)((x-5E-7)*1.0E6);
    if (y>=0) iy = (int)((y+5E-7)*1.0E6);
    else  iy = (int)((y-5E-7)*1.0E6);
    res = PntOnLine(ix1,iy1,ix2,iy2,ix,iy);
#endif
    /*
     * this eps is smaller than that in the equiv basil
     * pntonsegment (c_funcs.c). Ensures that nodes very close
     * to a boundary will be moved to the boundary by elle2poly.
     * Else can have case where elle2poly marks a point as being
     * on a boundary but basil just fails and does not set bnd
     * conditions.. (Fix by having basil use bnd info from trimesh -
     *               needs new ibctype flag)
     */
    float eps = 3.0e-6;
    float xmin,xmax,ymin,ymax;
    float val1,val2;

    if (x1<x2) {
        xmin = x1;
        xmax = x2;
    }
    else {
        xmin = x2;
        xmax = x1;
    }
    if (y1<y2) {
        ymin = y1;
        ymax = y2;
    }
    else {
        ymin = y2;
        ymax = y1;
    }
    if ((x > xmin-eps)&&(x < xmax+eps)&&
        (y > ymin-eps)&&(y < ymax+eps)) {
        val1 = ABS((y-y1)*(xmax-xmin));
        val2 = ABS((ymax-ymin)*(x-x1));
        if (val1<=(val2+eps) && val1>=(val2-eps)) res=2;
    }
    
    return((res==2)?1:0);

}
/* A main program to test the pntonline

main()
{
    int res;
    float x1, x2, y1, y2;
    float x, y;
    int ix1, ix2, iy1, iy2;
    int ix, iy;

    for (;;) {
        printf( "Point X, Y: " );
    	scanf( "%f %f", &x, &y );
        printf( "Line X1, Y1: " );
    	scanf( "%f %f", &x1, &y1 );
        printf( "X2, Y2: " );
    	scanf( "%f %f", &x2, &y2 );

        ix1 = (int)(x1*1.0E6);
        iy1 = (int)(y1*1.0E6);
        ix2 = (int)(x2*1.0E6);
        iy2 = (int)(y2*1.0E6);
        ix = (int)(x*1.0E6);
        iy = (int)(y*1.0E6);
        res = PntOnLine(ix1,iy1,ix2,iy2,ix,iy);
        switch(res) {
        case 0: printf("Point not on line through X1,Y1 X2,Y2\n");
                break;
        case 1: printf("Point on ray ending at X1,Y1\n");
                break;
        case 2: printf("Point on segment\n");
                break;
        case 3: printf("Point on ray beg at X2,Y2\n");
                break;
        }
    }
}
 */

/*
 * These 4 routines have only been used with CAXIS - return values 0-359
 */
int PolarToCartesian(double *x, double *y, double *z,
                     double angxy, double angz)
{
    /*
     * angles in radians
     */
    double r;

    r = cos(angz);
    *x = r * cos(angxy);
    *y = r * sin(angxy);
    *z = sin(angz);
    return(0);
}

int CartesianToPolar(double x, double y, double z,
                     double *angxy, double *angz)
{
    double eps = 1e-9;
    *angz = asin(z);
    if (x==0.0) {
        if (y>=0.0) *angxy = PI*0.5;
        else        *angxy = PI * 1.5;
    }
    else {
        // *angxy = atan(y/x);
        // if (x<0.0) *angxy += PI;
        // else if (y<0) *angxy += 2.0*PI;
        *angxy = atan2(y,x);
        if (*angxy < -eps) *angxy += 2.0*PI;
        if (*angxy > (2.0*PI-eps)) *angxy -= 2.0*PI;
    }
    return(0);
}

int PolarToCartesian(float *x, float *y, float *z,
                     double angxy, double angz)
{
    /*
     * angles in radians
     */
    double r;

    r = cos(angz);
    *x = (float)(r * cos(angxy));
    *y = (float)(r * sin(angxy));
    *z = (float)(sin(angz));
    return(0);
}
int CartesianToPolar(float x, float y, float z,
                     double *angxy, double *angz)
{
    float eps = 1e-6;
    *angz = asin((double)z);
    if (x==0.0) {
        if (y>=0.0) *angxy = PI*0.5;
        else        *angxy = PI * 1.5;
    }
    else {
        *angxy = atan((double)y/x);
        if (x<0.0) *angxy += PI;
        if (*angxy < -eps) *angxy += 2.0*PI;
        else if (*angxy > (2.0*PI-eps)) *angxy -= 2.0*PI;
    }
    return(0);
}

/*!
 * convert geo xy convention 0deg north, cw +ve
 * to math convention 0deg +x axis, anticw +ve
 * assumes range 0-359
 */
int GeoToMath(double *xy, double *dip)
{
    // geology convention is lower hemisphere
 // math is upper hemisphere
 //*xy = 450.0 - *xy +180.0;
    *xy = 450.0 - *xy;
    if (*xy>=360.0) *xy -= 360.0;
}

/*!
 * convert math convention 0deg +x axis, anticw +ve
 * to geo xy convention 0deg north, cw +ve
 * assumes range 0-359
 */
int MathToGeo(double *xy, double *dip)
{
    // geology convention is lower hemisphere
    // math is upper hemisphere
    //*xy = 450 - *xy -180.0;
    *xy = 450 - *xy;
    if (*xy < 0.0) *xy += 360.0;
    if (*xy >= 360.0) *xy -= 360.0;
}

/*!
 * returns angle between line 0,0-x,y and y=0 +ve axis
 * anticlockwise is +ve, range 0,2*PI
 */
double polar_angle(double x,double y)
{
    int err = 0;
    double theta=0.0, angz=0.0, z=0.0;

    err=CartesianToPolar(x,y,z,&theta,&angz);
    if (theta<0) theta += 2.0*PI;
    return(theta);
}

double calcfmod(double numer, double denom, double *fl)
{
    double ret = 0.0;

    *fl = floor(numer/denom);
    ret = numer - floor(numer/denom) * denom;
    return (ret);

#if XY
or (assuming denom is positive):

if(numer >= 0)
  return fmod(numer,denom);
else
{
  ret = fmod(-numer,denom);
  if(ret == 0)
    return ret;
  else
    return denom - ret;
#endif
}


/* function to grow a triangle so it has a buffer equal to the ROI */
void grow_rect(double tri[3][2], double temp[2][2],double roi)
{
    double maxx=-1e10,maxy=-1e10,minx=1e10,miny=1e10;
    int i;

/*
    minx=tri[0][0];
    maxx=tri[0][0];
    miny=tri[0][1];
    maxy=tri[0][1];
*/
    for(i=0;i<3;i++)
    {
        if(tri[i][0]<minx)
            minx=tri[i][0];
        if(tri[i][0]>maxx)
            maxx=tri[i][0];

        if(tri[i][1]<miny)
            miny=tri[i][1];
        if(tri[i][1]>maxy)
            maxy=tri[i][1];

    }

    temp[0][0]=minx-roi;
    temp[0][1]=miny-roi;
    temp[1][0]=maxx+roi;
    temp[1][1]=maxy+roi;
}

int dump_comments(FILE *fp)
{
    unsigned char done=0;
    int c;

    while (!done) {
        c = getc(fp);
        done = (c=='\n'||c==EOF);
    }
    return(0);
}
