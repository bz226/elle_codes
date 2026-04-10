#ifndef jkbmelt_h
  #define jkbmelt_h

#include <stdio.h>
#include <stdlib.h>
#include "error.h"
#include "parseopts.h"
#include "init.h"
#include "runopts.h"
#include "stats.h"
#include "setup.h"

/* * indices for User data values for this process */
const int Mobility = 0; // mobility
const int Magic = 1; // magic number (area fudge factor)
const int EquilMeltFrac = 2; // equilibrium melt fraction (not currently used)
const int XlXlEnergy = 3; // xl-xl interface energy
const int LiqXlEnergy = 4; // liq-xl interface energy
const int LiqLiqEnergy = 5; // liq-liq interface energy
const int IncMeltfraction = 6; // Incremental change in Melt Fraction

#endif
