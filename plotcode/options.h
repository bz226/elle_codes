/*----------------------------------------------------------------
 *    Elle:   options.h  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/
#ifndef _opts_h
#define _opts_h

#include "file.h"

#define MAX_OPTION_NAME 20
#define CMNTS_FILE "logcomments.txt"

#define GRAINS    12
#define SUBGRAINS 13
/*
 * allowed option types
 */
#define SYB_BOOL   201
#define SYB_INT    202
#define SYB_FLOAT  203
#define SYB_STRING 204
#define SYB_DOUBLE 205

/*
 * options - used when reading and writing options
 */
#define O_UPDFREQ       301
#define O_REGION        302
#define O_DRWUNODETYP   303
#define O_OVERLAY       304
#define O_NODEATTRIB    305
#define O_NODEATTRMIN   306
#define O_NODEATTRMAX   307
#define O_LBLREGION     308
#define O_FLYNNATTRIB   309
#define O_FLYNNATTRMIN  310
#define O_FLYNNATTRMAX  311
#define O_BNDATTRIB     312
#define O_BNDATTRMIN    313
#define O_BNDATTRMAX    314
#define O_BNDLEVEL      315
#define O_UNODEATTRIB   316
#define O_UNODEATTRMIN  317
#define O_UNODEATTRMAX  318

static valid_terms option_terms[] = {
                              { "Update_Interval",O_UPDFREQ },
                              { "Overlay_Plot",O_OVERLAY },
                              { "Unode_triangulation",O_DRWUNODETYP },
                              { "Flynn_Labels",O_LBLREGION },
                              { "Flynn_Attribute",O_FLYNNATTRIB },
                              { "Node_Attribute",O_NODEATTRIB },
                              { "Unode_Attribute",O_UNODEATTRIB },
                              { "Boundary_Attribute",O_BNDATTRIB },
                              { "Boundary_Level",O_BNDLEVEL },
                                 NULL };
static valid_terms region_terms[] = { 
                         { "NONE", NONE },
                         { "CAXIS", CAXIS },
                         { "EULER_3", E3_ALPHA },
                         { "ENERGY", ENERGY },
                         { "VISCOSITY", VISCOSITY },
                         { "S_EXPONENT", S_EXPONENT },
                         { "MINERAL", MINERAL },
                         { "STRAIN_E_XX", E_XX },
                         { "STRAIN_E_XY", E_XY },
                         { "STRAIN_E_YX", E_YX },
                         { "STRAIN_E_YY", E_YY },
                         { "STRAIN_INCR", F_INCR_S },
                         { "STRAIN_BULK", F_BULK_S },
                         { "GRAIN", GRAIN },
                         { "EXPAND", EXPAND },
                         { "AGE", AGE },
                         { "CYCLE", CYCLE },
                         { "DISLOCATION_DENSITY", DISLOCDEN },
                         { "ATTRIB_A", F_ATTRIB_A },
                         { "ATTRIB_B", F_ATTRIB_B },
                         { "ATTRIB_C", F_ATTRIB_C },
                                 NULL };
#define MAX_RGN_OPTIONS 25 /* must be greater than no. of region_terms */

static valid_terms node_terms[] = {
                         { "NONE", NONE },
                         { "NEIGHBOURS",NUM_NB },
                         { "TRIPLES",TRPL_ONLY },
                         { "VELOCITY_X",VEL_X },
                         { "VELOCITY_Y",VEL_Y },
                         { "TAU_XX",TAU_XX },
                         { "TAU_YY",TAU_YY },
                         { "TAU_ZZ",TAU_XY },
                         { "TAU_XY",TAU_ZZ },
                         { "TAU_1",TAU_1 },
                         { "PRESSURE",PRESSURE },
                         { "CONC_A",CONC_A },
                         { "N_ATTRIB_A",N_ATTRIB_A },
                         { "N_ATTRIB_B",N_ATTRIB_B },
                         { "N_ATTRIB_C",N_ATTRIB_C },
                                 NULL };
#define MAX_NODE_OPTIONS 20 /* must be greater than no. of node_terms */

static valid_terms unode_terms[] = {
                         { "NONE", NONE },
                         { "LOCATION", U_LOCATION },
                         { "CONC_A",CONC_A },
                         { "CAXIS", CAXIS },
                         { "EULER_3", E3_ALPHA },
                         { "EULER_3_B", E3_BETA },
                         { "ENERGY", U_ENERGY },
                         { "VISCOSITY", U_VISCOSITY },
                         { "DISLOCATION_DENSITY", U_DISLOCDEN },
                         { "STRAIN_INCR", F_INCR_S },
                         { "STRAIN_BULK", F_BULK_S },
                         { "U_ATTRIB_A",U_ATTRIB_A },
                         { "U_ATTRIB_B",U_ATTRIB_B },
                         { "U_ATTRIB_C",U_ATTRIB_C },
                                 NULL };

static valid_terms boundary_terms[] = { 
                         { "NONE", NONE },
                         { "STANDARD", STANDARD },
                         { "RAINBOW", RAINBOW },
                         { "MISORIENTATION", MISORIENTATION },
                                 NULL };

#define MAX_BND_OPTIONS 5 /* must be greater than no. of boundary_terms */

#define MAX_LEVEL_OPTIONS 9 /* max levels that can be displayed */

#endif
