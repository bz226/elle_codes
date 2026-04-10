 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_lut_h
#define _E_lut_h
#include <zlib.h>
#include "gz_utils.h"

const int LUT_MAX = 1024;

typedef struct {
    int size;
    double *data;
} EnergyLUT;


int LoadZIPGBEnergyLUT(gzFile in, char str[]);
int SaveZIPGBEnergyLUT(gzFile in);
#ifdef __cplusplus
extern "C" {
#endif
int ElleReadGBEnergyLUT(FILE *fp, char str[]);
int ElleWriteGBEnergyLUT(FILE *fp);
void ElleInitEnergyLUT(int size);
int ElleEnergyLUTSize();
void ElleSetEnergyLUT(int index,double val);
double ElleEnergyLUTValue(int index);
void ElleRemoveEnergyLUT();

#ifdef __cplusplus
}
#endif
#endif
