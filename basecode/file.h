 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_file_h
#define _E_file_h

#include "string_utils.h"
/* id values */
#define  E_OPTIONS  20
#define  REGIONS    21
#define  COMMENT    22
#define  FLYNNS     23
#define  PARENTS    24
#define  UNODES     25
#define  EXPAND     26
#define  COLOUR     27
#define  BNODES     28
#define  BOUNDARIES 29
#define  SEGMENTS   30

#define  LOCATION   130
#define  VELOCITY   131
#define  VEL_X      132
#define  VEL_Y      133

#define  STRESS     140
#define  TAU_XX     141
#define  TAU_YY     142
#define  TAU_ZZ     143
#define  TAU_XY     144
#define  TAU_1      145
#define  PRESSURE   146

#define  NUM_STRESS_VALS 6

#define  CONC_A       147
#define  NUM_NB       148
#define  TRPL_ONLY    149

#define  ATTRIB_A   150
#define  ATTRIB_B   151
#define  ATTRIB_C   152
#define  ATTRIB_D   153
#define  ATTRIB_E   154
#define  ATTRIB_F   155

#define  N_ATTRIB_A   150
#define  N_ATTRIB_B   151
#define  N_ATTRIB_C   152
#define  N_ATTRIB_D   153
#define  N_ATTRIB_E   154
#define  N_ATTRIB_F   155

#define  N_MCONC_A    160
#define  N_MCONC_A_1  161
#define  N_MCONC_A_2  162
#define  N_MCONC_A_3  163
#define  N_MATTRIB_A    164
#define  N_MATTRIB_A_1  165
#define  N_MATTRIB_A_2  166
#define  N_MATTRIB_A_3  167
#define  N_MATTRIB_B    168
#define  N_MATTRIB_B_1  169
#define  N_MATTRIB_B_2  170
#define  N_MATTRIB_B_3  171
#define  N_MATTRIB_C    172
#define  N_MATTRIB_C_1  173
#define  N_MATTRIB_C_2  174
#define  N_MATTRIB_C_3  175

#define  CAXIS      50
#define  CAXIS_X    51
#define  CAXIS_Y    52
#define  CAXIS_Z    53
#define  EULER_3    55
#define  E3_ALPHA   56
#define  E3_BETA    57
#define  E3_GAMMA   58

#define  ENERGY     60
#define  GBE_LUT    61

#define VISCOSITY   70
#define S_EXPONENT  71

#define AGE         72
#define CYCLE       73
#define  NUM_FLYNN_AGE_VALS 2

#define DISLOCDEN   74

#define  MINERAL    80
#define  QUARTZ     81
#define  FELDSPAR   82
#define  GARNET     83
#define  MICA       84
#define  CALCITE    85
#define  MINERAL_A  86
#define  MINERAL_B  87
#define  MINERAL_C  88
#define  ICE           381
#define  OLIVINE       382
#define  MAGNESIUM     383
#define  OMPHACITE     384
#define  CLINOPYROXENE 385
#define  HALITE        386

#define  STRAIN     90
#define  INCR_S     91
#define  BULK_S     92

#define  NUM_STRAIN_VALS 2

#define FLYNN_STRAIN 100
#define E_XX         101
#define E_XY         102
#define E_YY         103
#define E_YX         104
#define E_ZZ         105
#define F_INCR_S     106
#define F_BULK_S     107

#define  NUM_FLYNN_STRAIN_VALS 7

#define FLYNN_STRESS 108

/*
 *general dummy flynn attributes
 */
#define  SPLIT     111

#define  GRAIN     112
#define  F_ATTRIB_A 113
#define  F_ATTRIB_B 114
#define  F_ATTRIB_C 115
#define  F_ATTRIB_D 119
#define  F_ATTRIB_E 120
#define  F_ATTRIB_F 121
#define  F_ATTRIB_I 116
#define  F_ATTRIB_J 117
#define  F_ATTRIB_K 118

/*
 * boundary attributes 200-299
 */
#define MISORIENTATION 201
#define STANDARD       202
#define NONE           0
#define RAINBOW        203
/*
 * unode attributes 400-499
 */
#define  U_FINITE_STRAIN    400
#define  START_S_X   401
#define  START_S_Y   402
#define  PREV_S_X    403
#define  PREV_S_Y    404
#define  CURR_S_X    405
#define  CURR_S_Y    406
#define  NUM_FINITE_STRAIN_VALS 6
#define  U_STRAIN    407
#define  NUM_UNODE_STRAIN_VALS 7
#define  U_S_EXPONENT 408

#define  U_ATTRIB_A  413
#define  U_ATTRIB_B  414
#define  U_ATTRIB_C  415
#define  U_ATTRIB_D  424
#define  U_ATTRIB_E  425
#define  U_ATTRIB_F  426

#define U_FRACTURES 417
#define U_PHASE 418
#define U_TEMPERATURE 419
#define U_DIF_STRESS 420
#define U_MEAN_STRESS 421
#define U_DENSITY 422
#define U_YOUNGSMODULUS 423

#define  U_CONC_A    447
#define  U_CAXIS     450
#define  U_EULER_3   455

#define  U_ENERGY    460
#define  U_VISCOSITY 470
#define  U_DISLOCDEN 474

#define  U_LOCATION  499

/*
 * segment attributes 500-
 */
#define S_PERMEABILITY 500
#define S_FLUID_FLUX 501
#define S_LOCATION 599

#define  MAX_VALS 10

/*!
  File keyword and corresponding id for the data in an
  elle file (.elle) including structure and
  attributes of the elle objects (flynns, nodes, etc)
 */
static valid_terms FileKeys[] = {
                         { "OPTIONS", E_OPTIONS },
                         { "REGIONS", REGIONS },
                         { "FLYNNS", FLYNNS },
                         { "PARENTS", PARENTS },
                         { "UNODES", UNODES },
                         { "SEGMENTS", SEGMENTS },
                         { "EXPAND", EXPAND },
                         { "COLOUR", COLOUR },
                         { "LOCATION", LOCATION },
                         { "VELOCITY", VELOCITY },
                         { "STRESS", STRESS },
                         { "CONC_A", CONC_A },
                         { "N_MCONC_A", N_MCONC_A },
                         { "N_ATTRIB_A", ATTRIB_A },
                         { "N_ATTRIB_B", ATTRIB_B },
                         { "N_ATTRIB_C", ATTRIB_C },
                         { "N_ATTRIB_D", ATTRIB_D },
                         { "N_ATTRIB_E", ATTRIB_E },
                         { "N_ATTRIB_F", ATTRIB_F },
                         { "N_MATTRIB_A", N_MATTRIB_A },
                         { "N_MATTRIB_B", N_MATTRIB_B },
                         { "N_MATTRIB_C", N_MATTRIB_C },
                         { "STRAIN", STRAIN },
                         { "CAXIS", CAXIS },
                         { "EULER_3", EULER_3 },
                         { "ENERGY", ENERGY },
                         { "VISCOSITY", VISCOSITY },
                         { "S_EXPONENT", S_EXPONENT },
                         { "AGE", AGE },
                         { "DISLOCDEN", DISLOCDEN },
                         { "GBENERGY_LUT", GBE_LUT },
                         { "MINERAL", MINERAL },
                         { "FLYNN_STRAIN", FLYNN_STRAIN },
                         { "FLYNN_STRESS", FLYNN_STRESS },
                         { "GRAIN", GRAIN },
                         { "SPLIT", SPLIT },
                         { "F_ATTRIB_A", F_ATTRIB_A },
                         { "F_ATTRIB_B", F_ATTRIB_B },
                         { "F_ATTRIB_C", F_ATTRIB_C },
                         { "F_ATTRIB_D", F_ATTRIB_D },
                         { "F_ATTRIB_E", F_ATTRIB_E },
                         { "F_ATTRIB_F", F_ATTRIB_F },
                         { "F_ATTRIB_I", F_ATTRIB_I },
                         { "F_ATTRIB_J", F_ATTRIB_J },
                         { "F_ATTRIB_K", F_ATTRIB_K },
                         { "U_CONC_A", U_CONC_A },
                         { "U_ATTRIB_A", U_ATTRIB_A },
                         { "U_ATTRIB_B", U_ATTRIB_B },
                         { "U_ATTRIB_C", U_ATTRIB_C },
                         { "U_ATTRIB_D", U_ATTRIB_D },
                         { "U_ATTRIB_E", U_ATTRIB_E },
                         { "U_ATTRIB_F", U_ATTRIB_F },
                         { "U_STRAIN", U_STRAIN },
                         { "U_FINITE_STRAIN", U_FINITE_STRAIN },
                         { "U_CAXIS", U_CAXIS },
                         { "U_EULER_3", U_EULER_3 },
                         { "U_ENERGY", U_ENERGY },
                         { "U_S_EXPONENT", U_S_EXPONENT },
                         { "U_DISLOCDEN", U_DISLOCDEN },
                         { "U_VISCOSITY", U_VISCOSITY },
                         { "U_FRACTURES",U_FRACTURES},
                         { "U_PHASE",U_PHASE},
                         { "U_TEMPERATURE",U_TEMPERATURE},
                         { "U_DIF_STRESS",U_DIF_STRESS},
                         { "U_MEAN_STRESS",U_MEAN_STRESS},
                         { "U_DENSITY",U_DENSITY},
                         { "U_YOUNGSMODULUS",U_YOUNGSMODULUS},
                         { "S_PERMEABILITY",S_PERMEABILITY},
                         { "S_FLUID_FLUX",S_FLUID_FLUX},
                           NULL
                          };

#define MAX_MINERAL_LEN 20
static valid_terms MineralKeys[] = {
                         { "QUARTZ", QUARTZ },
                         { "FELDSPAR", FELDSPAR },
                         { "MICA", MICA },
                         { "GARNET", GARNET },
                         { "CALCITE", CALCITE },
                         { "ICE", ICE },
                         { "OLIVINE", OLIVINE },
                         { "MAGNESIUM", MAGNESIUM },
                         { "OMPHACITE", OMPHACITE },
                         { "CLINOPYROXENE", CLINOPYROXENE },
                         { "MINERAL_A", MINERAL_A },
                         { "MINERAL_B", MINERAL_B },
                         { "MINERAL_C", MINERAL_C },
                           NULL
                          };

static valid_terms VelocityKeys[] = {
                         { "VEL_X", VEL_X },
                         { "VEL_Y", VEL_Y },
                           NULL
                          };

static valid_terms StressKeys[] = {
                         { "TAU_XX", TAU_XX },
                         { "TAU_YY", TAU_YY },
                         { "TAU_ZZ", TAU_ZZ },
                         { "TAU_XY", TAU_XY },
                         { "TAU_1", TAU_1 },
                         { "PRESSURE", PRESSURE },
                           NULL
                          };

static valid_terms StrainKeys[] = {
                         { "INCR_S", INCR_S },
                         { "BULK_S", BULK_S },
                           NULL
                          };

static valid_terms FiniteStrainKeys[] = {
                         { "START_S_X", START_S_X },
                         { "START_S_Y", START_S_Y },
                         { "PREV_S_X", PREV_S_X },
                         { "PREV_S_Y", PREV_S_Y },
                         { "CURR_S_X", CURR_S_X },
                         { "CURR_S_Y", CURR_S_Y },
                           NULL
                          };

/*
 * F_BULK_S and F_INCR_S for backward compatibility
 */
static valid_terms FlynnStrainKeys[] = {
                         { "INCR_S", INCR_S },
                         { "BULK_S", BULK_S },
                         { "E_XX", E_XX },
                         { "E_XY", E_XY },
                         { "E_YX", E_YX },
                         { "E_YY", E_YY },
                         { "E_ZZ", E_ZZ },
                         { "INCR_S", F_INCR_S },
                         { "BULK_S", F_BULK_S },
                           NULL
                          };

static valid_terms FlynnAgeKeys[] = {
                         { "AGE", AGE },
                         { "CYCLE", CYCLE },
                           NULL
                          };

static valid_terms Leftovers[] =
{
  { "NONE", NONE },
  { "TRIPLE", TRPL_ONLY },
  { "NEIGHBOURS",NUM_NB },
  { "RAINBOW",RAINBOW },
  { "E3_ALPHA", E3_ALPHA },
  { "E3_BETA", E3_BETA },
  { "E3_GAMMA", E3_GAMMA },
    NULL
};
                                                                                
static valid_terms BoundaryKeys[] =
{
  { "MISORIENTATION", MISORIENTATION },
  { "STANDARD", STANDARD },
  { "NONE", NONE },
  { "RAINBOW",RAINBOW },
    NULL
};
#define SET_ALL "Default"

#ifdef __cplusplus
extern "C" {
#endif
int ElleReadData(char *filename);
int ElleWriteData(const char *filename);
int ElleAutoWriteFile(int cnt);
int ElleCheckFiles();
void ElleAttributeNotInFile(char *fname,int id);
#ifdef __cplusplus
}
#endif
#endif
