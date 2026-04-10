 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_display_h
#define _E_display_h

/* matches def in menus.c */
#define GRAINS    12
#define SUBGRAINS 13
#define UNITS     14
#define TRIANGLES 15

#ifndef NODSP

int ElleMarkNode(int dd,int *col,float x,float y,float size);
float ElleGetPointRadius(int max);
#ifdef __cplusplus
extern "C" {
#endif
void ElleUpdateDisplay();
void EllePlotRegions(int count);
void EllePlotTriangles();
void GetRegionOption(int *val);
void GetDrawingOption(int *val);
void GetLblRgnOption(int *val);
void GetUpdFreqOption(int *val);
void ElleShowStages(int stage);
void ElleRescaleWin();
void ElleUpdateSettings();

#ifdef __cplusplus
}
#endif

#else

#ifdef __cplusplus
extern "C" {
#endif
void ElleUpdateDisplay();
void ElleUpdateSettings();
void ElleShowStages(int stage);
void EllePlotRegions(int count);
void ElleRescaleWin();

#ifdef __cplusplus
}
#endif

#endif

#endif
