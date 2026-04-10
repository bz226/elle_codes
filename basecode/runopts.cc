 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "attrib.h"
#include "error.h"
#include "file.h"
#include "file_utils.h"
#include "general.h"
#include "setup.h"
#include "stats.h"
#include "init.h"
#include "runopts.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/
Coords DefaultTopLeft;
Coords DefaultTopRight;
Coords DefaultBaseLeft;
Coords DefaultBaseRight;
runtime_opts Settings_run, Tmp_run;

void ElleInitRunOptions()
{
    int i;
    double sw_dist;

    Settings_run.display = 1;
    Settings_run.verbose = 0;
    Settings_run.stages = 20;
    Settings_run.AppName[0] = '\0';
    Settings_run.Count = 0;
    Settings_run.InitFunc = 0;
    Settings_run.RunFunc = 0;
    Settings_run.ExitFunc = ElleExit;
    Settings_run.control.file[0] = '\0';
    Settings_run.control.extrafile[0] = '\0';
    Settings_run.control.dspoptsfile[0] = '\0';
    Settings_run.control.outfile[0] = '\0';
    Settings_run.control.bndvel = 0;
    Settings_run.control.minFlynnArea = DFLT_MIN_FLYNN_AREA;
    Settings_run.control.maxFlynnArea = DFLT_MAX_FLYNN_AREA;
    Settings_run.save.fileroot[0] = '\0';
    Settings_run.save.do_save = 0;
    Settings_run.save.frequency = 0;
    Settings_run.physical.time = 0;
    Settings_run.physical.timestep = 3.15e7;
    Settings_run.physical.unitlength = 1e-2;
    Settings_run.physical.temperature = 25;
    Settings_run.physical.pressure = 1;
    Settings_run.physical.bndwidth = DFLT_BNDWIDTH;
    Settings_run.physical.massincr = 0;
    for (i=0;i<MAX_U_DATA;i++) Settings_run.physical.user_data[i]=1.0;
    sw_dist = 1.0/40;
    ElleSetSwitchdistance(sw_dist);
    Settings_run.control.maxNodeSep = sw_dist*2.2;
    Settings_run.control.minNodeSep = sw_dist;
    /* speedup <= 0.5 * 0.2 * switchdist * switchdist */
    /*ElleSetSpeedup(sw_dist*sw_dist*0.1);*/
    ElleSetSpeedup(1);

    DefaultTopLeft.x = 0.0;
    DefaultTopLeft.y = 1.0;
    DefaultTopRight.x = 1.0;
    DefaultTopRight.y = 1.0;
    DefaultBaseLeft.x = 0.0;
    DefaultBaseLeft.y = 0.0;
    DefaultBaseRight.x = 1.0;
    DefaultBaseRight.y = 0.0;
    ElleSetCellBBox(&DefaultBaseLeft,&DefaultBaseRight,
                    &DefaultTopRight,&DefaultTopLeft);
    ElleSetSSOffset(0);
    ElleSetCumSSOffset(0);
    ElleSetOptNames("1","2","3","4","5","6","7","8","9");
}
int ElleSetOptNames(const char* n1,const char* n2,const char* n3,
                    const char* n4,const char* n5,const char* n6,
                    const char* n7,const char* n8,const char* n9)
{
    char* names[9];
    names[0]=(char *)n1;
    names[1]=(char *)n2;
    names[2]=(char *)n3;
    names[3]=(char *)n4;
    names[4]=(char *)n5;
    names[5]=(char *)n6;
    names[6]=(char *)n7;
    names[7]=(char *)n8;
    names[8]=(char *)n9;   
    ElleSetOptName(names);
}

int ElleDisplayHelpMsg()
{
    printf("help msg\n");
    return(0);
}

void ElleCopyRunOptions()
{
    Tmp_run = Settings_run;
}

void ElleRestoreRunOptions()
{
    Settings_run = Tmp_run;
}

/*
 * set a field in the runtime_options structure
 */
void ElleSetRunOptionValue(int id,int *val)
{
    double fval;

    switch(id) {
    case RO_DISPLAY   : ElleSetDisplay(*val);
                        break;
    case RO_STAGES:     ElleSetStages(*val);
                        break;
    case RO_SWITCHDIST: fval = *((double *)val);
                        ElleSetSwitchdistance(fval);
                        break;
    case RO_SPEEDUP:    fval = *((double *)val);
                        ElleSetSpeedup(fval);
                        break;
    case RO_VERBOSE:    ElleSetVerbose(*val);
                        break;
    case RO_RESCALE:    ElleSetRescale(*val);
                        break;
    case RO_SVFREQ :    ElleSetSaveFrequency(*val);
                        break;
    case RO_SVFILEROOT :ElleSetSaveFileRoot((char *)val);
                        break;
    default:            break;
    }
}

/*
 * get a field from the runtime_options structure
 */
void ElleGetRunOptionValue(int id,int *val)
{
    switch(id) {
    case RO_DISPLAY:     *val = Settings_run.display;
                         break;
    case RO_STAGES:      *val = Settings_run.stages;
                         break;
    case RO_SWITCHDIST:  *((double *)val) = Settings_run.control.switchDist;
                         break;
    case RO_MAXNODESEP:  *((double *)val) = Settings_run.control.maxNodeSep;
                         break;
    case RO_MINNODESEP:  *((double *)val) = Settings_run.control.minNodeSep;
                         break;
    case RO_SPEEDUP:     *((double *)val) = Settings_run.control.speedup;
                         break;
    case RO_VERBOSE:     *val = Settings_run.verbose;
                         break;
    case RO_RESCALE:     *val = Settings_run.rescale;
                         break;
    case RO_SVFREQ :     *val = Settings_run.save.frequency;
                         break;
    case RO_SVFILEROOT : strcpy((char *)val,Settings_run.save.fileroot);
                         break;
    default:             break;
    }
}

void ElleSetVerbose(int state)
{
    Settings_run.verbose = state;
}

void ElleSetRescale(int state)
{
    Settings_run.rescale = state;
}

void ElleSetDisplay(int state)
{
    Settings_run.display = state;
}

void ElleSetRunFunction(ElleRunFunc func)
{
    Settings_run.RunFunc = func;
}

void ElleSetInitFunction(ElleRunFunc func)
{
    Settings_run.InitFunc = func;
}

void ElleSetExitFunction(ElleRunFunc func)
{
    Settings_run.ExitFunc = func;
}

void ElleIncrementCount()
{
    Settings_run.Count++;
    ElleCheckFiles();
}

void ElleSetCount(int count)
{
    Settings_run.Count = count;
}

void ElleSetAppName(char *progname)
{
    char *pname = 0;

    /* program name without path */
    if ((pname=strrchr(progname,E_DIR_SEPARATOR))==NULL)
        pname=progname;
    else pname++;
    /* in case the program name ended with '/' */
    if (pname && (strlen(pname) < 1)) pname=progname;
    if (pname!=0)
        strcpy(Settings_run.AppName,pname);
}

void ElleSetStages(int state)
{
    Settings_run.stages = state;
}

void ElleSetFile(char *filename)
{
    if (filename!=0)
        strncpy(Settings_run.control.file,filename,
                                         FILENAME_MAX-1);
}

void ElleSetOutFile(char *filename)
{
    if (filename!=0)
        strncpy(Settings_run.control.outfile,filename,
                                         FILENAME_MAX-1);
}

void ElleSetExtraFile(char *filename)
{
    if (filename!=0)
        strncpy(Settings_run.control.extrafile,filename,
                                         FILENAME_MAX-1);
}

void ElleSetDspOptsFile(char *filename)
{
    if (filename!=0)
        strncpy(Settings_run.control.dspoptsfile,filename,
                                         FILENAME_MAX-1);
}

void ElleSetSwitchdistance(double val)
{
    Settings_run.control.switchDist = val;
    Settings_run.control.maxNodeSep = val*2.2;
    Settings_run.control.minNodeSep = val;
}

void ElleSetSpeedup(double val)
{
    Settings_run.control.speedup = val;
}

void ElleSetMinNodeSep(double val)
{
    Settings_run.control.minNodeSep = val;
}

void ElleSetMaxNodeSep(double val)
{
    Settings_run.control.maxNodeSep = val;
}

void ElleSetMinFlynnArea(double val)
{
    Settings_run.control.minFlynnArea = val;
}

void ElleSetMaxFlynnArea(double val)
{
    Settings_run.control.maxFlynnArea = val;
}

void ElleSetBndVelocity(double val)
{
    Settings_run.control.bndvel = val;
}

void ElleSetSaveFrequency(int freq)
{
    Settings_run.save.frequency = freq;
    if (freq>0) Settings_run.save.do_save = 1;
}

void ElleSetSaveFileRoot(char *name)
{
    if (name) {
        strcpy(Settings_run.save.fileroot,name);
        ES_SetStatsFileRoot(name);
    }
}


int EllemaxStages()
{
    int val;

    ElleGetRunOptionValue(RO_STAGES,&val);
    return( val );
}

int EllesaveInterval()
{
    int val;

    ElleGetRunOptionValue(RO_SVFREQ,&val);
    return( val );
}

void ElleSetPressure(double value)
{
    Settings_run.physical.pressure = value;
}

void ElleSetTemperature(double value)
{
    Settings_run.physical.temperature = value;
}

void ElleSetUnitLength(double value)
{
    Settings_run.physical.unitlength = value;
}

void ElleSetTimestep(double value)
{
    Settings_run.physical.timestep = value;
}

void ElleSetTime(double value)
{
    Settings_run.physical.time = value;
}

void ElleSetMassIncr(double value)
{
    Settings_run.physical.massincr = value;
}

void ElleSetBndWidth(double value)
{
    Settings_run.physical.bndwidth = value;
}

void ElleSetUserData(UserData value)
{
    int i;

    for (i=0;i<MAX_U_DATA;i++)
        Settings_run.physical.user_data[i] = value[i];
}

void ElleSetSSOffset(double value)
{
    Settings_run.control.unitcell.xoffset = value;
}

void ElleSetCumSSOffset(double value)
{
    Settings_run.control.unitcell.cum_xoffset = value;
}

double ElleSSOffset()
{
    return(Settings_run.control.unitcell.xoffset);
}

double EllePSOffset()
{
    /*return(Settings_run.control.unitcell.ylength-1.0);*/
    return(Settings_run.control.unitcell.cellBBox[BASELEFT].y);
}

double ElleCumSSOffset()
{
    return(Settings_run.control.unitcell.cum_xoffset);
}

int ElleRescale()
{
    return( Settings_run.rescale );
}

int ElleDisplay()
{
    int val;

    ElleGetRunOptionValue(RO_DISPLAY,&val);
    return( val );
}

double EllePressure()
{
    return(Settings_run.physical.pressure);
}

double ElleTemperature()
{
    return(Settings_run.physical.temperature);
}

double ElleUnitLength()
{
    return(Settings_run.physical.unitlength);
}

double ElleTimestep()
{
    return(Settings_run.physical.timestep);
}

double ElleTime()
{
    return(Settings_run.physical.time);
}

double ElleBndWidth()
{
    return(Settings_run.physical.bndwidth);
}

double ElleMassIncr()
{
    return(Settings_run.physical.massincr);
}

void ElleUserData(UserData value)
{
    int i;

    for (i=0;i<MAX_U_DATA;i++)
        value[i] = Settings_run.physical.user_data[i];
}

ElleRunFunc ElleInitFunction()
{
    return( Settings_run.InitFunc);
}

ElleRunFunc ElleRunFunction()
{
    return( Settings_run.RunFunc);
}

ElleRunFunc ElleExitFunction()
{
    return( Settings_run.ExitFunc);
}

int ElleCount()
{
    return( Settings_run.Count);
}

char *ElleAppName()
{
    return( &(Settings_run.AppName[0]) );
}

int ElleSetOptName(char* names[9])
{
       for(int n=0;n<9;n++)
        Settings_run.OptNames[n]=names[n];
}
int ElleGetOptNames(char* names[9])
{
    for(int n=0;n<9;n++)
        names[n]=Settings_run.OptNames[n];
}

char *ElleSaveFileRoot()
{
    return( &(Settings_run.save.fileroot[0]) );
}

char *ElleFile()
{
    return( &(Settings_run.control.file[0]) );
}

char *ElleExtraFile()
{
    return( &(Settings_run.control.extrafile[0]) );
}

char *ElleDspOptsFile()
{
    return( &(Settings_run.control.dspoptsfile[0]) );
}

char *ElleOutFile()
{
    return( &(Settings_run.control.outfile[0]) );
}

double ElleMinFlynnArea()
{
    return( Settings_run.control.minFlynnArea );
}

double ElleMaxFlynnArea()
{
    return( Settings_run.control.maxFlynnArea );
}

double ElleBndVelocity()
{
    return( Settings_run.control.bndvel );
}

double ElleSwitchdistance()
{
    double val;

    ElleGetRunOptionValue(RO_SWITCHDIST,(int *)(&val));
    return( val );
}

double ElleSpeedup()
{
    double val;

    ElleGetRunOptionValue(RO_SPEEDUP,(int *)(&val));
    return( val );
}

double EllemaxNodeSep()
{
    double val;

    ElleGetRunOptionValue(RO_MAXNODESEP,(int *)(&val));
    return( val );
}

double ElleminNodeSep()
{
    double val;

    ElleGetRunOptionValue(RO_MINNODESEP,(int *)(&val));
    return( val );
}

/*!
 *  ElleSetCellBBox just sets the unitcell corners -
 *  used by reposition which wraps nodes back into the unitcell
 *  and by ElleReadOptions when a file is read
 */
int ElleSetCellBBox(Coords *baseleft, Coords *baseright,
                     Coords *topright, Coords *topleft)
{
    double tmp;
    double eps = 1e-6;
    if (topleft->y<=baseleft->y || topright->y<=baseright->y ||
        topright->x<=topleft->x || baseright->x<=baseleft->x)
        return(1);
    Settings_run.control.unitcell.cellBBox[BASELEFT] = *baseleft;
    Settings_run.control.unitcell.cellBBox[BASERIGHT] = *baseright;
    Settings_run.control.unitcell.cellBBox[TOPRIGHT] = *topright;
    Settings_run.control.unitcell.cellBBox[TOPLEFT] = *topleft;
    Settings_run.control.unitcell.xlength = baseright->x - baseleft->x;
    Settings_run.control.unitcell.ylength = topleft->y - baseleft->y;
    Settings_run.control.unitcell.yoffset = (baseleft->y-baseright->y);
#if XY
    Settings_run.control.unitcell.xoffset = topleft->x - baseleft->x;
    Settings_run.control.unitcell.yoffset =
					(baseleft->y-baseright->y);
    if (fabs(Settings_run.control.unitcell.ylength-1.0)<eps &&
        Settings_run.control.unitcell.xoffset >= 
                  Settings_run.control.unitcell.xlength) {
		tmp = Settings_run.control.unitcell.xoffset-
					Settings_run.control.unitcell.xlength;
        Settings_run.control.unitcell.xoffset = tmp;
        //Settings_run.control.unitcell.xoffset = 
         //modf(Settings_run.control.unitcell.xoffset,&tmp);
        Settings_run.control.unitcell.cum_xoffset += tmp;
    }
    else {
		Settings_run.control.unitcell.xoffset = 
                 Settings_run.control.unitcell.cum_xoffset;
        if (Settings_run.control.unitcell.xoffset>
				Settings_run.control.unitcell.xlength) {
			Settings_run.control.unitcell.xoffset = 
				Settings_run.control.unitcell.xoffset-
					Settings_run.control.unitcell.xlength;
		}
	}
#endif
    return(0);
}

/*!
 *  ElleUpdateCellBBox sets the unitcell corners and
 *  compares with the previous unitcell to calculate
 *  the new offset and cumulative offset
 *  used by processes which deform the unitcell
 */
int ElleUpdateCellBBox(Coords *baseleft, Coords *baseright,
                     Coords *topright, Coords *topleft)
{
    double tmp, cumoffset, offset, tmpfl;
    double last_xoffset, last_deformx, last_yoffset, last_deformy;
    double deformx, deformy, incrx, incry;
    double eps = 1e-6;
    CellData data;

    ElleCellBBox(&data);
    last_xoffset = data.xoffset;
    last_deformx = data.cellBBox[TOPLEFT].x-data.cellBBox[BASELEFT].x;
    last_yoffset = data.yoffset;
    last_deformy = data.cellBBox[BASELEFT].y-data.cellBBox[BASERIGHT].y;
    cumoffset = data.cum_xoffset;
    double oldlength= data.xlength;

    if (topleft->y<=baseleft->y || topright->y<=baseright->y ||
        topright->x<=topleft->x || baseright->x<=baseleft->x)
        return(1);
    Settings_run.control.unitcell.cellBBox[BASELEFT] = *baseleft;
    Settings_run.control.unitcell.cellBBox[BASERIGHT] = *baseright;
    Settings_run.control.unitcell.cellBBox[TOPRIGHT] = *topright;
    Settings_run.control.unitcell.cellBBox[TOPLEFT] = *topleft;
    Settings_run.control.unitcell.xlength = baseright->x - baseleft->x;
    Settings_run.control.unitcell.ylength = topleft->y - baseleft->y;
    deformx = topleft->x - baseleft->x;
    incrx = (deformx - last_deformx);
    Settings_run.control.unitcell.xoffset += incrx;
    deformy = baseleft->y - baseright->y;
    incry = (deformy - last_deformy);
    Settings_run.control.unitcell.yoffset += incry;
/*
    Settings_run.control.unitcell.xoffset +=
(Settings_run.control.unitcell.xlength-oldlength);
    offset += Settings_run.control.unitcell.xoffset;
*/
    cumoffset += incrx;
	Settings_run.control.unitcell.cum_xoffset = cumoffset;
    if ((fabs(data.ylength- data.xlength)<eps) &&
        (Settings_run.control.unitcell.xoffset >= 
                  Settings_run.control.unitcell.xlength)) {
        Settings_run.control.unitcell.xoffset = 
            calcfmod(cumoffset,Settings_run.control.unitcell.xlength,&tmpfl);
/*
		tmp = Settings_run.control.unitcell.xoffset-
					Settings_run.control.unitcell.xlength;
        Settings_run.control.unitcell.xoffset = tmp;
        Settings_run.control.unitcell.cum_xoffset += tmp;
*/
    }
    else {
        Settings_run.control.unitcell.xoffset = 
            calcfmod(cumoffset,Settings_run.control.unitcell.xlength,&tmpfl);
/*
		Settings_run.control.unitcell.cum_xoffset = cumoffset;
        if (Settings_run.control.unitcell.xoffset>
				Settings_run.control.unitcell.xlength) {
			Settings_run.control.unitcell.xoffset = 
				Settings_run.control.unitcell.xoffset-
					Settings_run.control.unitcell.xlength;
		}
*/
	}
    return(0);
}

void ElleCellBBox(CellData *data)
{
    *data = Settings_run.control.unitcell;
/*
    *baseleft = Settings_run.control.cellBBox[BASELEFT];
    *baseright = Settings_run.control.cellBBox[BASERIGHT];
    *topright = Settings_run.control.cellBBox[TOPRIGHT];
    *topleft = Settings_run.control.cellBBox[TOPLEFT];
*/
}

int ElleReadOptionData(FILE *fp,char *str)
{
    int err=0, i, num, finished=0;
    int cellset=0;
    double val;
    Coords bl,br,tr,tl;

    while (!feof(fp) && !err && !finished) {
        if ((num = fscanf(fp,"%s", str))!=1) return(READ_ERR);
        else if (str[0] == '#') dump_comments( fp );
        else {
            validate(str,&i,run_option_terms);
            switch(i) {
            case RO_SWITCHDIST: if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                else ElleSetSwitchdistance(val); 
                                break;
                     /*
                      * currently, setting switchdist sets node
                      * separations 
                      */
            case RO_MAXNODESEP: if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                break;
            case RO_MINNODESEP: if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                break;
            case RO_SPEEDUP   : if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                else ElleSetSpeedup(val);
                                break;
            case RO_TEMPERATURE : if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                else ElleSetTemperature(val);
                                break;
            case RO_PRESSURE : if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                else ElleSetPressure(val);
                                break;
            case RO_UNITLENGTH: if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                else ElleSetUnitLength(val);
                                break;
            case RO_TIMESTEP : if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                else ElleSetTimestep(val);
                                break;
            case RO_TIME     : if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                else ElleSetTime(val);
                                break;
            case RO_BNDWIDTH : if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                else ElleSetBndWidth(val);
                                break;
            case RO_MASSINCR : if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                else ElleSetMassIncr(val);
                                break;
            case RO_SSOFFSET : if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                else ElleSetSSOffset(val);
                                break;
            case RO_CUMSSOFFSET : if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                else ElleSetCumSSOffset(val);
                                break;
            case RO_MINFLYNNAREA : if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                else ElleSetMinFlynnArea(val);
                                break;
            case RO_MAXFLYNNAREA : if (fscanf(fp,"%lf\n",&val)!=1)
                                    err = READ_ERR;
                                else ElleSetMaxFlynnArea(val);
                                break;
            case RO_CELLBBOX:   if (fscanf(fp,"%lf %lf\n",&bl.x,&bl.y)!=2)
                                        err = READ_ERR;
                                else if (fscanf(fp,"%lf %lf\n",&br.x,&br.y)!=2)
                                        err = READ_ERR;
                                else if (fscanf(fp,"%lf %lf\n",&tr.x,&tr.y)!=2)
                                        err = READ_ERR;
                                else if (fscanf(fp,"%lf %lf\n",&tl.x,&tl.y)!=2)
                                        err = READ_ERR;
                                if (!err) {
                                     ElleSetCellBBox(&bl,&br,&tr,&tl);
                                     cellset = 1;
                                }
                                break;
            default:            finished = 1;
                                break;
            }
        }
    }
    if (!err && !cellset) {
        bl.x = 0; bl.y = 0;
        br.x = 1; br.y = 0;
        tr.x = 1; tr.y = 1;
        tl.x = 0; tl.y = 1;
        ElleSetCellBBox(&bl,&br,&tr,&tl);
    }
    return(err);
}

int ElleWriteOptionData(FILE *fp)
{
    char label[20];
    int err=0, i=0;
    double val;
    CellData unitcell;

    if (!id_match(FileKeys,E_OPTIONS,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    while (run_option_terms[i].name!=NULL && !err) {
        switch(run_option_terms[i].id) {
        case RO_SWITCHDIST: fprintf(fp,"%s ",run_option_terms[i].name);
                            val = ElleSwitchdistance();
                            if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            break;
        case RO_MAXNODESEP: fprintf(fp,"%s ",run_option_terms[i].name);
                            val = EllemaxNodeSep();
                            if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            break;
        case RO_MINNODESEP: fprintf(fp,"%s ",run_option_terms[i].name);
                            val = ElleminNodeSep();
                            if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            break;
        case RO_SPEEDUP   : fprintf(fp,"%s ",run_option_terms[i].name);
                            val = ElleSpeedup();
                            if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            break;
        case RO_TEMPERATURE: fprintf(fp,"%s ",run_option_terms[i].name);
                            val = ElleTemperature();
                            if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            break;
        case RO_PRESSURE  : fprintf(fp,"%s ",run_option_terms[i].name);
                            val = EllePressure();
                            if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            break;
        case RO_BNDWIDTH  : fprintf(fp,"%s ",run_option_terms[i].name);
                            val = ElleBndWidth();
                            if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            break;
        case RO_MASSINCR  : fprintf(fp,"%s ",run_option_terms[i].name);
                            val = ElleMassIncr();
                            if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            break;
        case RO_UNITLENGTH: val = ElleUnitLength();
                            if (val > 0) {
                                fprintf(fp,"%s ",run_option_terms[i].name);
                                if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            }
                            break;
        case RO_TIMESTEP  : val = ElleTimestep();
                            if (val > 0) {
                                fprintf(fp,"%s ",run_option_terms[i].name);
                                if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            }
                            break;
        case RO_TIME      : val = ElleTime();
                            if (val > 0) {
                                fprintf(fp,"%s ",run_option_terms[i].name);
                                if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            }
                            break;
        case RO_CELLBBOX  : ElleCellBBox(&unitcell);
                            fprintf(fp,"%s ",run_option_terms[i].name);
                            if (fprintf(fp,"%.8e %.8e\n",
                                unitcell.cellBBox[BASELEFT].x,
                                unitcell.cellBBox[BASELEFT].y)<0)
                                err = WRITE_ERR;
                            else {
                                sprintf(label,"%c%ld%c%s",'%',
                                    strlen(run_option_terms[i].name)+1,
                                    's',"%.8e %.8e %c");
                                if (fprintf(fp,label," ",
                                    unitcell.cellBBox[BASERIGHT].x,
                                    unitcell.cellBBox[BASERIGHT].y, '\n')<0)
                                    err = WRITE_ERR;
                                else if (fprintf(fp,label," ",
                                    unitcell.cellBBox[TOPRIGHT].x,
                                    unitcell.cellBBox[TOPRIGHT].y, '\n')<0)
                                    err = WRITE_ERR;
                                else if (fprintf(fp,label," ",
                                    unitcell.cellBBox[TOPLEFT].x,
                                    unitcell.cellBBox[TOPLEFT].y, '\n')<0)
                                    err = WRITE_ERR;
                            }
                            break;
        case RO_SSOFFSET : fprintf(fp,"%s ",run_option_terms[i].name);
                            val = ElleSSOffset();
                            if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            break;
        case RO_CUMSSOFFSET: fprintf(fp,"%s ",run_option_terms[i].name);
                            val = ElleCumSSOffset();
                            if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            break;
        case RO_MINFLYNNAREA: val = ElleMinFlynnArea();
                            if (val != DFLT_MIN_FLYNN_AREA) {
                                fprintf(fp,"%s ",run_option_terms[i].name);
                                if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            }
                            break;
        case RO_MAXFLYNNAREA: val = ElleMaxFlynnArea();
                            if (val != DFLT_MAX_FLYNN_AREA) {
                                fprintf(fp,"%s ",run_option_terms[i].name);
                                if (fprintf(fp,"%.8e\n",val)<0) err = WRITE_ERR;
                            }
                            break;
        default:            break;
        }
        i++;
    }
    return(err);
}

int StartApp()
{
    ElleRunFunc init, run;

    if ((init = ElleInitFunction())==NULL)
        OnError("InitFunc not set\n",0);
#ifndef NODISP
    if (ElleDisplay())
        Run_App( 0 );
    else {
#else
    ElleSetDisplay(0);
#endif
        (*init)();
        if (strlen(ElleFile())!=0) {
            if ((run = ElleRunFunction())) (*run)();
            else OnError("RunFunc not set",0);
        }
        else
            OnError("No data file open",0);
#ifndef NODISP
    }
#endif
    return(0);
}

#ifdef NODISP

int Run_App(FILE *)
{
}
int SetupApp(int, char **)
{
}

#endif
