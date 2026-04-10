 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_general_h
#define _E_general_h

#include <cstdio>

int polyCentroid(double x[], double y[], int n,
		 double *xCentroid, double *yCentroid, double *area);
int PointOnSegment(double x1, double y1,
                   double x2, double y2,
                   double x, double y);
int PolarToCartesian(double *x, double *y, double *z,
                     double angxy, double angz);
int angle(double x,double y,double x1,double y1,double x2,double y2,
            double *angle);
int angle0(double x,double y,double x1,double y1,double x2,double y2,
           double *angle);
double polar_angle(double x,double y);
#ifdef __cplusplus
extern "C" {
#endif
int polyCentroid(float x[], float y[], int n,
		 double *xCentroid, double *yCentroid, double *area);
int lines_intersect( double x1, double y1, double x2, double y2,
		     double x3, double y3, double x4, double y4,
		     double *x, double *y );
int intersection_pt( double x1, double y1, double x2, double y2,
		     double x3, double y3, double x4, double y4,
		     double *x, double *y );
int angle(float x,float y,float x1,float y1,float x2,float y2,float *angle);
int angle0(float x,float y,float x1,float y1,float x2,float y2,float *angle);
int rotate_coords(double x, double y,
                  double x_orig, double y_orig,
                  double *x_new, double *y_new,
                  double angle);
int rotate_coords_ac(double x, double y,
                  double x_orig, double y_orig,
                  double *x_new, double *y_new,
                  double angle);
int PntOnLine(int px,int py,int qx,int qy,int tx,int ty);
int PointOnSegment(float x1, float y1,
                   float x2, float y2,
                   float x, float y);
int PolarToCartesian(float *x, float *y, float *z,
                     double angxy, double angz);
int CartesianToPolar(float x, float y, float z,
                     double *angxy, double *angz);
int CrossingsMultiplyTest(double *pgon, int numverts, double point[2] );
int CrossingsTest( double *pgon, int numverts, double point[2] );
double calcfmod(double numer, double denom, double *fl);
void grow_rect(double tri[3][2], double temp[2][2],double roi);
int dump_comments(FILE *fp);
#ifdef __cplusplus
}
#endif
#endif
