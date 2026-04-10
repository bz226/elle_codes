 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_runopts_h
#define _E_runopts_h

#ifndef FILENAME_MAX
#include <stdio.h>
#endif

#ifndef _E_string_utils_h
#include "string_utils.h"
#endif

#ifndef _E_attrib_h
#include "attrib.h"
#endif

#include <string.h>

typedef int (*ElleRunFunc)(void);

struct save_params {
    int do_save;
    int frequency;
    char fileroot[FILENAME_MAX];
};
struct physical_params {
    double time;
    double timestep;
    double unitlength;
    double bndwidth;
    double temperature;
    double pressure;
    double massincr;
    UserData user_data;
};

struct control_params {
    char file[FILENAME_MAX];
    char extrafile[FILENAME_MAX];
    char dspoptsfile[FILENAME_MAX];
    char outfile[FILENAME_MAX];
    double switchDist;
    double speedup;
    double maxNodeSep;
    double minNodeSep;
    double bndvel;
    double minFlynnArea;
    double maxFlynnArea;
    CellData unitcell;
};

typedef struct {
    int display;
    int rescale;
    int stages;
    int verbose;
    ElleRunFunc InitFunc;
    ElleRunFunc RunFunc;
    ElleRunFunc ExitFunc;
    int Count;
    char AppName[FILENAME_MAX];
    char* OptNames[9];
    struct control_params control;
    struct save_params save;
    struct physical_params physical;
} runtime_opts;

/*
 * some default option values
 */
#define DFLT_BNDWIDTH        1e-9 /* boundary width in metres */
#define DFLT_MIN_FLYNN_AREA  0.0
#define DFLT_MAX_FLYNN_AREA  0.00625 /* split on area, if no mineral */
/*
 * definitions used when accessing option fields
 */
#define RO_DISPLAY     201
#define RO_RESCALE     203
#define RO_STAGES      204
#define RO_VERBOSE     205
#define RO_INITFUNC    206
#define RO_RUNFUNC     207
#define RO_FILE        208
#define RO_EXTRAFILE   209
#define RO_SWITCHDIST  210
#define RO_MAXNODESEP  211
#define RO_MINNODESEP  212
#define RO_SPEEDUP     213
#define RO_SVFREQ      214
#define RO_SVFILEROOT  215
#define RO_CELLBBOX    216
#define RO_SSOFFSET    217
#define RO_CUMSSOFFSET 218
#define RO_PSOFFSET    219
#define RO_CUMPSOFFSET 220
#define RO_PRESSURE    221
#define RO_TEMPERATURE 222
#define RO_TIME        223
#define RO_TIMESTEP    224
#define RO_UNITLENGTH  225
#define RO_MINFLYNNAREA 226
#define RO_MAXFLYNNAREA 227
#define RO_MASSINCR    228
#define RO_BNDWIDTH    229

static valid_terms run_option_terms[] = {
                             { "Stages",RO_STAGES },
                             { "Display", RO_DISPLAY },
                             { "Verbose", RO_VERBOSE },
                             { "Init Function", RO_INITFUNC },
                             { "Run Function", RO_RUNFUNC },
                             { "SaveInterval", RO_SVFREQ },
                             { "Save_File Root",RO_SVFILEROOT },
                             { "SwitchDistance",RO_SWITCHDIST },
                             { "MaxNodeSeparation",RO_MAXNODESEP },
                             { "MinNodeSeparation",RO_MINNODESEP },
                             { "SpeedUp",RO_SPEEDUP },
                             { "CellBoundingBox",RO_CELLBBOX },
                             { "SimpleShearOffset",RO_SSOFFSET },
                             { "CumulativeSimpleShear",RO_CUMSSOFFSET },
                             { "PureShearOffset",RO_PSOFFSET },
                             { "Time",RO_TIME },
                             { "Timestep",RO_TIMESTEP },
                             { "UnitLength",RO_UNITLENGTH },
                             { "Temperature",RO_TEMPERATURE },
                             { "Pressure",RO_PRESSURE },
                             { "BoundaryWidth",RO_BNDWIDTH },
                             { "MassIncrement",RO_MASSINCR },
                             { "MinFlynnArea",RO_MINFLYNNAREA },
                             { "MaxFlynnArea",RO_MAXFLYNNAREA },
                               NULL
                              };

#ifdef __cplusplus
extern "C" {
#endif
void ElleSetSaveFrequency(int freq);
void ElleSetSaveFileRoot(char *name);
void ElleSetVerbose(int state);
void ElleSetRescale(int state);
void ElleSetDisplay(int state);
void ElleSetStages(int state);
void ElleSetInitFunction(ElleRunFunc);
void ElleSetRunFunction(ElleRunFunc);
void ElleSetExitFunction(ElleRunFunc);
void ElleSetCount(int count);
void ElleIncrementCount();
void ElleSetAppName(char *progname);
void ElleSetFile(char *filename);
void ElleSetExtraFile(char *filename);
void ElleSetDspOptsFile(char *filename);
void ElleSetOutFile(char *filename);
void ElleSetBndVelocity(double vel);
void ElleSetSwitchdistance(double val);
void ElleSetSpeedup(double val);
void ElleSetMinNodeSep(double val);
void ElleSetMaxNodeSep(double val);
void ElleSetTime(double val);
void ElleSetTimestep(double val);
void ElleSetUnitLength(double val);
void ElleSetTemperature(double val);
void ElleSetPressure(double val);
void ElleSetBndWidth(double val);
void ElleSetMassIncr(double val);
void ElleSetUserData(UserData value);
void ElleSetSSOffset(double val);
void ElleSetCumSSOffset(double val);
void ElleSetMinFlynnArea(double val);
void ElleSetMaxFlynnArea(double val);
void ElleSetRunOptionValue(int id,int *val);
void ElleGetRunOptionValue(int id,int *val);
void ElleRestoreRunOptions(void);
void ElleCopyRunOptions(void);
int ElleReadOptionData(FILE *fp,char *str);
int ElleWriteOptionData(FILE *fp);
ElleRunFunc ElleInitFunction();
ElleRunFunc ElleRunFunction();
ElleRunFunc ElleExitFunction();
int ElleCount();
char *ElleAppName(void);
double ElleSwitchdistance(void);
double ElleSpeedup(void);
char *ElleFile(void);
char *ElleExtraFile(void);
char *ElleDspOptsFile(void);
char *ElleOutFile(void);
char *ElleSaveFileRoot(void);
double EllemaxNodeSep(void);
double ElleminNodeSep(void);
double ElleBndVelocity(void);
int EllemaxStages(void);
int EllesaveInterval(void);
int ElleSetCellBBox(Coords *baseleft, Coords *baseright,
                     Coords *topright, Coords *topleft);
int ElleUpdateCellBBox(Coords *baseleft, Coords *baseright,
                     Coords *topright, Coords *topleft);
void ElleCellBBox(CellData *data);
double ElleTime();
double ElleTimestep();
double ElleUnitLength();
double ElleTemperature();
double EllePressure();
double ElleBndWidth();
double ElleMassIncr();
void ElleUserData(UserData value);
double ElleSSOffset();
double ElleCumSSOffset();
double EllePSOffset();
double ElleMinFlynnArea();
double ElleMaxFlynnArea();
extern int ElleDisplay();
int ElleRescale();
void ElleInitRunOptions();
int StartApp();
int ElleDisplayHelpMsg();
int ElleSetOptName(char* names[9]);
int ElleSetOptNames(const char* n1,const char* n2,const char* n3,
                    const char* n4,const char* n5,const char* n6,
                    const char* n7,const char* n8,const char* n9);
int ElleGetOptNames(char* names[9]);
#ifdef __cplusplus
}
#endif
#endif
