
/*----------------------------------------------------------------
 *    Elle:   settings.h  1.0  22 June 2005
 *
 *    Copyright (c) 2005 by L.A. Evans
 *----------------------------------------------------------------*/
#ifndef _settings_h
#define _settings_h

#include <stdio.h>
#include "attrib.h"
#include "types.h"
#include "csettings.h"

#define E_DFLT_DNODE_COL 4
#define E_DFLT_TNODE_COL 5
#define E_COL_MAX      127
#define E_COL_MIN        9
#define EULER_RGB      699

#ifdef __cplusplus
//std::cout<<"in settings.h __cplusplus "<<__cplusplus<<std::endl;
extern "C" {
#endif
extern void Init_Options(void);
extern void GetCUnodePlotOption(int *val);
extern void GetCRegionOption(int *val);
extern void GetCUpdFreqOption(int *val);
extern void GetCOverlayOption(int *val);
extern void GetCNodeAttribOption(int *val);
extern void SetCNodeAttribOption(int val,double min,double max);
extern void GetCFlynnAttribOption(int *val);
extern void SetCFlynnAttribOption(int val,double min,double max);
extern void GetCNodeAttribMinOption(double *val);
extern void GetCNodeAttribMaxOption(double *val);
extern void GetCFlynnAttribMinOption(double *val);
extern void GetCFlynnAttribMaxOption(double *val);
extern void GetCBndAttribOption(int *val);
extern void GetCBndAttribMinOption(double *val);
extern void GetCBndAttribMaxOption(double *val);
extern void SetCBndAttribOption(int val,double min,double max);
extern void GetCBndLevelOption(int *val);
extern void SetCBndLevelOption(int val);
extern void GetCUnodeAttribOption(int *val);
extern void GetCUnodeAttribMinOption(double *val);
extern void GetCUnodeAttribMaxOption(double *val);
extern void SetCUnodeAttribOption(int val,double min,double max);
extern void GetCSegmentAttribOption(int *val);
extern void GetCSegmentAttribMinOption(double *val);
extern void GetCSegmentAttribMaxOption(double *val);
extern void SetCSegmentAttribOption(int val,double min,double max);
extern void GetCForegroundOption(int *val);
//extern void GetCPlotColourRange(int *min,int *max);
extern void GetCLblRgnOption(int *val);
extern void GetCDrawingOption(int *val);
extern void GetCOverlayOption(int *val);
#ifdef __cplusplus
}
#endif

class Settings {
    bool logfile,logfileappend, rainbow;
public:

	Settings():logfile(false),logfileappend(false),rainbow(false){ }
    void SetLogFile(bool c) {logfile=c;}
    bool GetLogFile(void) { return logfile;}
    void SetLogAppend(bool c) {logfileappend=c;}
    bool GetLogAppend(void){return logfileappend;}
	void InitOptions(void) { Init_Options();}
	void GetRegionOption(int *val)
	  { GetCRegionOption(val); }
	void GetUpdFreqOption(int *val)
	  { GetCUpdFreqOption(val); }
	void GetUnodePlotOption(int *val)
	  { GetCUnodePlotOption(val); }
	void GetNodeAttribOption(int *val)
	  { GetCNodeAttribOption(val); }
	void SetNodeAttribOption(int id, double min, double max)
	  { SetCNodeAttribOption(id, min, max); }
	void GetNodeAttribMinOption(double *val)
	  { GetCNodeAttribMinOption(val); }
	void GetNodeAttribMaxOption(double *val)
	  { GetCNodeAttribMaxOption(val); }
	void GetFlynnAttribOption(int *val)
	  { GetCFlynnAttribOption(val); }
	void SetFlynnAttribOption(int id, double min, double max)
	  { SetCFlynnAttribOption(id, min, max); }
	void GetFlynnAttribMinOption(double *val)
	  { GetCFlynnAttribMinOption(val); }
	void GetFlynnAttribMaxOption(double *val)
	  { GetCFlynnAttribMaxOption(val); }
	void GetUnodeAttribOption(int *val)
	  { GetCUnodeAttribOption(val); }
	void GetUnodeAttribMinOption(double *val)
	  { GetCUnodeAttribMinOption(val); }
	void GetUnodeAttribMaxOption(double *val)
	  { GetCUnodeAttribMaxOption(val); }
	void GetSegmentAttribOption(int *val)
	  { GetCSegmentAttribOption(val); }
	void GetSegmentAttribMinOption(double *val)
	  { GetCSegmentAttribMinOption(val); }
	void GetSegmentAttribMaxOption(double *val)
	  { GetCSegmentAttribMaxOption(val); }
	void GetBndAttribOption(int *val)
	  { GetCBndAttribOption(val); }
	void GetBndLevelOption(int *val)
	  { GetCBndLevelOption(val); }
	void SetBndLevelOption(int val)
	  { SetCBndLevelOption(val); }
	void GetLblRgnOption(int *val)
	  { GetCLblRgnOption(val); }
	void GetDrawingOption(int *val)
	  { GetCDrawingOption(val); }
	int GetPlotColourRange(int *min,int *max)
	  { return ::GetCPlotColourRange(min,max); }
        void SetRainbow(bool s) { rainbow = s; }
        bool GetRainbow() { return rainbow; }
	void GetOverlayOption(int *val)
	  { GetCOverlayOption(val); }
	void GetBndAttribMinOption(double *val)
	  { GetCBndAttribMinOption(val); }
	void GetBndAttribMaxOption(double *val)
	  { GetCBndAttribMaxOption(val); }
	void SetBndAttribOption(int val,double min,double max)
	  { SetCBndAttribOption(val,min,max); }
	void SetUnodeAttribOption(int val,double min,double max)
	  { SetCUnodeAttribOption(val,min,max); }
	void SetSegmentAttribOption(int val,double min,double max)
	  { SetCSegmentAttribOption(val,min,max); }
	void GetForegroundOption(int *val)
	  { GetCForegroundOption(val); }
};

int FillPolygon(int,float *,float *,int *,int *,int *);
int DrawNode(int,float *,float *,float *,int *,int *);
int DrawUnodes(int col_range,int col_min,int attr,
                double min,double max);
int DrawSegments(int col_range,int col_min,int attr,
                double min,double max);
void DrawGradient( double list[] [3], Coords p1, Coords gvector );
void Error();
//Settings *GetUserOptions();

#ifdef __cplusplus
extern "C" {
#endif
void Log( int level,char msg[4096] );
extern void update_();
extern void clearbg_();
extern int SetStage(int stage),
           drawlabel_(float *,float *,int *,int *,int *,char *,int *);
extern int RescaleBox(float,float,float,float);
extern int drawsegment_(float *,float *,float *,float *,int *);
extern int plotu_(float *,float *,int *);
extern int setpencolor_(int *);
#ifdef __cplusplus
}
#endif
#endif //_settings_h

