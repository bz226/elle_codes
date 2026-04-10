 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_attrib_h
#define _E_attrib_h

/*
 *  Allowed values for node attributes
 */
/* types values */
#define  GRAIN_B      2
#define  SUBGRAIN_B   4
#define  UNIT_B       8
#define  TRIPLE_J    16
#define  DOUBLE_J    32

/* neighbour values */
#define  NO_NB     -1
#define  DUMMY     -2

/* state values */
#define  INACTIVE    0
#define  ACTIVE      1

/* array index values */
#define  NO_INDX    -1
#define  MAX_INDX   2147483647
#define  INDX_CHECK 1431655764 /* MAX_INDX/1.5 */

/* node default attribute indices ?? */
#define  I_VEL_X      0
#define  I_VEL_Y      1
#define  I_TAU_XX     2
#define  I_TAU_YY     3
#define  I_TAU_ZZ     4
#define  I_TAU_XY     5
#define  I_TAU_1      6
#define  I_PRESSURE   7
#define  I_SR_INCR    8
#define  I_STRN_BULK  9

/* region attribute indices */
#define  I_COLOUR    0
#define  I_EXPAND    1
#define  I_VISC      2
#define  I_CAXIS_X   3
#define  I_CAXIS_Y   4
#define  I_CAXIS_Z   5
#define  I_ENERGY    6
#define  I_CONC_A    7

/* define unitcell corners */
#define BASELEFT  0
#define BASERIGHT 1
#define TOPRIGHT  2
#define TOPLEFT   3

#define  NO_VAL    -1

typedef struct {
    double x;
    double y;
} Coords;

typedef struct {
    double x;
    double y;
    double z;
} Coords_3D;

typedef struct {
    Coords start;
    Coords prev;
    Coords curr;
} FiniteStrainData;

/*
typedef struct {
    int grain;
    int subgrain;
    int unit;
} ERegion;
*/
#define ERegion int

#define MAX_U_DATA  9

typedef double UserData[MAX_U_DATA];

typedef struct {
    Coords cellBBox[4];
    double xoffset;
    double yoffset;
    double xlength;
    double ylength;
    double cum_xoffset;
} CellData;

#endif
