#ifndef _MELT_ELLE_H_
#define _MELT_ELLE_H_
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nodes.h"
#include "display.h"
#include "check.h"
#include "errnum.h"
#include "error.h"
#include "general.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "polygon.h"
#include "stats.h"
#include "init.h"
#include "crossings.h"
#include "convert.h"
#include "lut.h"
#include "timefn.h"
#include "mineraldb.h"
#include "update.h"
#include "jkbmelt.h"
#include "meltmovenode.h"
#include "jkbmeltstats.h"


/** physical constants******************************** */
double energyofdislocations; // in Jm-1    (from mineraldb)
double energyofsurface; // in Jm-2    (from mineraldb)
double mobilityofboundary; // in m2s-1J-1    (aka fudge factor)
double dislocationdensityscaling = 10e13; // in m-2    (ie 1.5 in elle file = 1.5e13 in real world)



//this is not necessarily the right time step!
double truetimestep = 3.1536e10; // in s     (1000yrs here, not including leap years)
//it should be whatever is set in the input file!
//truetimestep=ElleTimestep();



double lengthscale = 1e-3; // in m     (sides of box)
double R = 8.314472; // in Jmol-1K-1    (Gas constant)
double Qgbm = 200; // in Jmole-1    (Activation energy for GBM  (made up number))
double oldarealiq;
/**  */

using std::cout;
using std::endl;

//#define DEBUG


int Melt();
int InitMelt();

UserData CurrData;
const int MeltId = F_ATTRIB_A; // melt phase

double GetAngleEnergy( double angle );
int Melt_MoveNode4Pos( int flynn,int n, Coords * movedir, int * same, double arealiq, double mobfac, double fudge, double areaequil,
                  double energyxlxl, double energyliqxl, double energyliqliq );
int Melt_MoveNode3Pos( int flynn,int n, Coords * movedir, int * same, double arealiq, double mobfac, double fudge, double areaequil,
                  double energyxlxl, double energyliqxl, double energyliqliq );
double GetSurfaceNodeEnergy( int n, int meltmineral, Coords * xy, double * energy, double arealiq, double fudge, double areaequil,
                          double energyxlxl, double energyliqxl, double energyliqliq );
int GetBodyNodeEnergy( int n, double * total_energy );
int GetCSLFactor( Coords_3D xyz[3], float * mobfac );
int ElleGetFlynnEulerCAxis( int flynn_no, Coords_3D * dircos );
int GetCSLFactor( Coords_3D xyz[3], float * mobfac );
int ElleGetFlynnEulerCAxis( int flynn_no, Coords_3D * dircos );
double CalcAreaLiq( int meltmineral );
void ElleNodeSameAttrib( int node, int * same, int * samemin, int attrib_id );

extern double ElleSwitchLength();

double incroffset = 0.002;
double oldfudge;



#endif //_MELT_ELLE_H_
