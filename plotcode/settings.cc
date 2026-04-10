/*----------------------------------------------------------------
 *    Elle:   settings.cc  1.0  22 June 2005
 *
 *    Copyright (c) 2005 by L.A. Evans
 *----------------------------------------------------------------*/
#include "settings.h"

extern void PlotUnodes(int,int,int,double,double);
extern void PlotSegments(int,int,int,double,double);
#ifdef __cplusplus
extern "C" {
#endif
extern int fillpoly_(float *xpts,float *ypts,int *num_nodes,
                int *col,int *outline);
extern int drawcircle_(float *xpt,float *ypt,float *size,
             int *col,int *outline);
#ifdef __cplusplus
}
#endif

Settings * user_options = new Settings();

int DrawNode(int dd,float *xpt,float *ypt,float *size,
             int *col,int *outline)
{
    *xpt -= *size;
    *ypt += *size;
    *size *= 2;
    return(drawcircle_(xpt,ypt,size,col,outline));
}

int DrawUnodes(int col_range,int col_min,int attr,
                double min,double max)
{
    PlotUnodes(col_range,col_min,attr,min,max);
}

int DrawSegments(int col_range,int col_min,int attr,
                double min,double max)
{
    PlotSegments(col_range,col_min,attr,min,max);
}

int FillPolygon(int index,float *xpts,float *ypts,int *num_nodes,
                int *col,int *outline)
{
    return(fillpoly_(xpts,ypts,num_nodes,col,outline));
}

Settings *GetUserOptions()
{
    return (user_options);
}

void DrawGradient( double list[] [3], Coords p1, Coords gvector )
{
}
void Log( int level,char *msg )
{
    fprintf(stderr,"%s\n",msg);
}
void Error()
{
}

