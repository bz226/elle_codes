/*
 *----------------------------------------------------
 * Copyright: (c) L. A. Evans, M. W. Jessell
 * File:      $RCSfile: mineraldb.h,v $
 * Revision:  $Revision: 1.3 $
 * Date:      $Date: 2014/06/16 23:13:17 $
 * Author:    $Author: levans $
 *
 *----------------------------------------------------
 */
#ifndef _E_min_attrib_h
#define _E_min_attrib_h

#ifdef __cplusplus
extern "C" {
#endif
double GetMineralAttribute(int mineral, int attribute);
int ElleLoadMineralSymmetry(double symm[24][3][3], int mineral);
int symmetry_set(double symm[24][3][3], const int symmetry);

#ifdef __cplusplus
}
#endif

/*
 *  Allowed values for mineral attributes
 */
/* types values */
#define GB_MOBILITY           1001
#define RECOVERY_RATE         1002
#define CRITICAL_SG_ENERGY    1003
#define CRITICAL_RX_ENERGY    1004
#define VISCOSITY_BASE        1005
#define SURFACE_ENERGY        1006
#define DD_ENERGY             1007
#define MIN_FLYNN_AREA        1008
#define MAX_FLYNN_AREA        1009
#define CRITICAL_MISORIENT    1010
#define GB_MOBILITY_2         1011
#define GB_ACTIVATION_ENERGY  1012
#define SYMMETRY              1013
#define BURGERS_VECTOR        1014

#define CUBIC                 1100
#define HEXAGONAL             1101
#define ORTHORHOMBIC          1102
#define MONOCLINIC            1103
#define TRIGONAL              1104
#endif
