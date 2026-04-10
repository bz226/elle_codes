
/*----------------------------------------------------------------
 *    Elle:   settings.h  2.0  2 October 2000
 *
 *    Copyright (c) 2000 by L.A. Evans
 *----------------------------------------------------------------*/
/*
 * $Id$
 */
#ifndef _settings_h
#define _settings_h

#ifndef _cmndefs_h
#include "cmndefs.h"
#endif

#include "attrib.h"
#include "file.h"

#ifndef _E_string_utils_h
#include "string_utils.h"
#endif

#define LABEL_MRK  '"'
#define COMMENT_MRK  #
#define MAX_OPTION_NAME 20


#define SYB_BOOL   201
#define SYB_INT    202
#define SYB_FLOAT  203
#define SYB_DOUBLE 204


/*
 * options - used when reading and writing options
 */
#define O_ORIENT      101
#define O_MARKCELL    102
#define O_ROWS        103
#define O_COLS        104
#define O_FONTHGT     105
#define O_ZOOM        108
#define O_PGSCL       109
#define O_LABEL       130
#define O_FOREGRND    133
#define O_LINESTYLE   134
#define O_LINEWDTH    135
#define O_COLMAP      136
#define O_VERBOSE     137
#define O_PG_XMARG    138
#define O_PG_YMARG    139
#define O_PAPERSZ     140
#define O_COLBAR      141
#define O_RESCALE     145 /* options written only if < O_RESCALE */
#define O_TEXTBG      147
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
#define O_DRWSEGMENTTYP  319
#define O_SEGMENTATTRIB  320
#define O_SEGMENTATTRMIN 321
#define O_SEGMENTATTRMAX 322
/*
 * valid colourmap settings
 */
#define GREY_MAP         100
#define GRAY_MAP         100
#define STD_MAP          101
#define ABS_MAP          102

#define E_DFLT_DNODE_COL  -3
#define E_DFLT_TNODE_COL  -4
#define E_COL_MAX        255 
#define E_COL_MIN          0 

/* orientation of colourbar */
#define CB_VERT   42
#define CB_HORIZ  41
#define CB_NONE   40
#ifndef _defines_h
#define _defines_h
#ifdef XSYB
#define fontdat XFontStruct *
#define colourdat XColor
#define E_BLACK_PEN  0
#else
#define fontdat int
#define colourdat int
#define E_BLACK_PEN  1



#endif
#endif
//war int*
#define XtPointer double*
//#include "cmndefs.h"
#include "filedat.h"

#define CNTRS     5
#define MAXCNTRVALS  11
#define MAXPRFLVALS   12
#define EULER_RGB   699

/*!
  Labels for the attribute preference menus and the corresponding
  attribute id for the elle objects (flynns, nodes, etc)
 */
static valid_terms boundary_terms[] = {
                         { "NONE", NONE },
                         { "STANDARD", STANDARD },
                         { "MISORIENTATION", MISORIENTATION },
                         { "RAINBOW", RAINBOW },
                                 NULL };
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
                         { "N_ATTRIB_D",N_ATTRIB_D },
                         { "N_ATTRIB_E",N_ATTRIB_E },
                         { "N_ATTRIB_F",N_ATTRIB_F },
                                 NULL };

static valid_terms unode_terms[] = {
                         { "NONE", NONE },
                         { "LOCATION", U_LOCATION },
                         { "CONC_A",CONC_A },
                         { "CAXIS_X", CAXIS_X },
                         { "CAXIS_Y", CAXIS_Y },
                         { "CAXIS_Z", CAXIS_Z },
                         { "CAXIS", CAXIS },
                         { "EULER_ALPHA", E3_ALPHA },
                         { "EULER_BETA", E3_BETA },
                         { "EULER_GAMMA", E3_GAMMA },
                         { "EULER_ALPHA", EULER_3 },
                         { "EULER_RGB", EULER_RGB },
                         { "ENERGY",U_ENERGY },
                         { "VISCOSITY", U_VISCOSITY },
                         { "DISLOCATION_DENSITY", U_DISLOCDEN },
                         { "STRAIN/STRESS EXP", U_S_EXPONENT },
                         { "STRAIN_E_XX", E_XX },
                         { "STRAIN_E_XY", E_XY },
                         { "STRAIN_E_YX", E_YX },
                         { "STRAIN_E_YY", E_YY },
                         { "STRAIN_INCR", INCR_S },
                         { "STRAIN_BULK", BULK_S },
                         { "U_ATTRIB_A",U_ATTRIB_A },
                         { "U_ATTRIB_B",U_ATTRIB_B },
                         { "U_ATTRIB_C",U_ATTRIB_C },
                         { "U_ATTRIB_D",U_ATTRIB_D },
                         { "U_ATTRIB_E",U_ATTRIB_E },
                         { "U_ATTRIB_F",U_ATTRIB_F },
                         { "U_FRACTURES",U_FRACTURES},
                         { "U_PHASE",U_PHASE},
                         { "U_TEMPERATURE",U_TEMPERATURE},
                         { "U_DIF_STRESS",U_DIF_STRESS},
                         { "U_MEAN_STRESS",U_MEAN_STRESS},
                         { "U_DENSITY",U_DENSITY},
                         { "U_YOUNGSMODULUS",U_YOUNGSMODULUS},
                                 NULL };

static valid_terms segment_terms[] = {
                         { "LOCATION", S_LOCATION },
                         { "S_PERMEABILITY",S_PERMEABILITY},
                         { "S_FLUID_FLUX",S_FLUID_FLUX},
                           NULL };

static valid_terms region_terms[] = {
                         { "NONE", NONE },
                         { "CAXIS_X", CAXIS_X },
                         { "CAXIS_Y", CAXIS_Y },
                         { "CAXIS_Z", CAXIS_Z },
                         { "CAXIS", CAXIS },
                         { "EULER_ALPHA", E3_ALPHA },
                         { "EULER_BETA", E3_BETA },
                         { "EULER_GAMMA", E3_GAMMA },
                         { "EULER_ALPHA", EULER_3 },
                         { "EULER_RGB", EULER_RGB },
                         { "ENERGY", ENERGY },
                         { "VISCOSITY", VISCOSITY },
                         { "STRAIN/STRESS EXP", S_EXPONENT },
                         { "MINERAL", MINERAL },
                         { "STRAIN_E_XX", E_XX },
                         { "STRAIN_E_XY", E_XY },
                         { "STRAIN_E_YX", E_YX },
                         { "STRAIN_E_YY", E_YY },
                         { "STRAIN_INCR", INCR_S },
                         { "STRAIN_BULK", BULK_S },
                         { "STRAIN_INCR", F_INCR_S },
                         { "STRAIN_BULK", F_BULK_S },
                         { "TAU_XX", TAU_XX },
                         { "TAU_YY", TAU_YY },
                         { "TAU_ZZ", TAU_ZZ },
                         { "TAU_XY", TAU_XY },
                         { "TAU_1", TAU_1 },
                         { "PRESSURE", PRESSURE },
                         { "SPLIT", SPLIT },
                         { "GRAIN", GRAIN },
                         { "EXPAND", EXPAND },
                         { "AGE", AGE },
                         { "CYCLE", CYCLE },
                         { "DISLOCATION_DENSITY", DISLOCDEN },
                         { "ATTRIB_A", F_ATTRIB_A },
                         { "ATTRIB_B", F_ATTRIB_B },
                         { "ATTRIB_C", F_ATTRIB_C },
                         { "ATTRIB_D", F_ATTRIB_D },
                         { "ATTRIB_E", F_ATTRIB_E },
                         { "ATTRIB_F", F_ATTRIB_F },
                         { "ATTRIB_I", F_ATTRIB_I },
                         { "ATTRIB_J", F_ATTRIB_J },
                         { "ATTRIB_K", F_ATTRIB_K },
                                 NULL };

static valid_terms option_terms[] = {
                              { "Update_Interval",O_UPDFREQ },
                              { "Overlay_Plot",O_OVERLAY },
                              { "Unode_triangulation",O_DRWUNODETYP },
                              { "Flynn_Labels",O_LBLREGION },
                              { "Flynn_Attribute",O_FLYNNATTRIB },
                              { "Node_Attribute",O_NODEATTRIB },
                              { "Unode_Attribute",O_UNODEATTRIB },
                              { "Segment_Attribute",O_SEGMENTATTRIB },
                              { "Boundary_Attribute",O_BNDATTRIB },
                              { "Boundary_Level",O_BNDLEVEL },
                                 NULL };

static valid_terms display_terms[] = {
                              { "orientation",O_ORIENT },
                              { "paper_size",O_PAPERSZ },
                              { "font_height",O_FONTHGT },
                              { "page_xmargin",O_PG_XMARG },
                              { "page_ymargin",O_PG_YMARG },
                              { "page_scale",O_PGSCL },
                              { "foreground",O_FOREGRND },
                              { "linestyle",O_LINESTYLE },
                              { "linewidth",O_LINEWDTH },
                              { "colour",O_COLMAP },
                                 NULL };
static valid_terms paper_terms[] = {
                                     { "A4",A4_PAPER },
                                     { "USLetter", US_PAPER },
                                        NULL };
static valid_terms colmap_terms[] = {
                                     {"greyscale",GREY_MAP },
                                     {"grayscale",GREY_MAP },
                                     { "standard", STD_MAP },
                                     { "absolute", ABS_MAP },
                                        NULL };
static valid_terms linestyle_terms[] = {
                                     {"solid",SOLID },
                                     { "dash", DASH },
                                     { "dot", DOT },
                                        NULL };
 /*
  * equivalent to Xlib.h XRectangle
  */
typedef struct {
    short x, y;
    unsigned short width, height;
} rectangle;

 /* data for each cell */
typedef struct {
  	 char drawn;
  	 int row,col;
  	 rectangle rect;
  	 /*save_node *plots;*/
} cell_data;

class Settings
{


public:
    int update_freq;
    int region;
    int overlay;
    int fill;
    int outline;
    int draw_nodes;
    int node_attrib;
    double node_limits[2];
    int segment_attrib;
    double segment_limits[2];
    int unode_attrib;
    double unode_limits[2];
    int flynn_attrib;
    double flynn_limits[2];
    int boundary_attrib;
    double bnd_limits[2];
    int boundary_level;
    int label_region;
    bool logfile,logfileappend,rainbow;
    	 unsigned char orient;
    	 int paper_size;
    	 int dfltfonthgt;
    	 int fg; /* index in Colours[] */
    	 int linestyle; /* -1=solid, 1=dashed */
    	 float page_scale;
    	 float linewidth;
    	 float xmargin,ymargin;
    	 float page_xmargin,page_ymargin;
    	 int colourmap;
    	 int max_colours;
	Settings(void);
        void SetLogFile(bool c);
        bool GetLogFile(void);
        void SetLogAppend(bool c);
        bool GetLogAppend(void);
	void InitOptions();
	int ReadSettings(FILE *fp, char *str, unsigned char initial);
	int WriteSettingsFile(char *filename);
	int WriteSettings(FILE *fp);
	void GetRegionOption(int *val);
	void GetUpdFreqOption(int *val);
	void GetSegmentPlotOption(int *val);
	void GetUnodePlotOption(int *val);
	void GetNodeAttribOption(int *val);
	void SetNodeAttribOption(int id, double min, double max);
	void GetNodeAttribMinOption(double *val);
	void GetNodeAttribMaxOption(double *val);
	void GetFlynnAttribOption(int *val);
	void SetFlynnAttribOption(int id, double min, double max);
	void GetFlynnAttribMinOption(double *val);
	void GetFlynnAttribMaxOption(double *val);
	void GetUnodeAttribOption(int *val);
	void GetUnodeAttribMinOption(double *val);
	void GetUnodeAttribMaxOption(double *val);
	void GetSegmentAttribOption(int *val);
	void GetSegmentAttribMinOption(double *val);
	void GetSegmentAttribMaxOption(double *val);
	void GetBndAttribOption(int *val);
	void GetBndLevelOption(int *val);
	void SetBndLevelOption(int val);
	void GetLblRgnOption(int *val);
	void GetDrawingOption(int *val);
	int GetPlotColourRange(int *min,int *max);
    void SetRainbow(bool s);
    bool GetRainbow();
	void GetOverlayOption(int *val);
	void GetBndAttribMinOption(double *val);
	void GetBndAttribMaxOption(double *val);
	void SetBndAttribOption(int val,double min,double max);
	void SetUnodeAttribOption(int val,double min,double max);
	void SetSegmentAttribOption(int val,double min,double max);
	void GetForegroundOption(int *val);
	//void GetPlotColourRange(int *min,int *max);
	void SetOptionValue(int id,XtPointer val);
	void SetOverlayOption(int p);
	void GetOptionValue(int id,XtPointer val);
	void GetColourmapOption(int *val);
	void SetColourmapOption(int val);
	void SetMaxColoursOption(int val);
	int ValidColmapOption(char *str);
	int ReadMinMax(char *str, double *min, double *max);
	void Update(void);
};

int FillPolygon(int,float *,float *,int *,int *,int *);
int DrawNode(int,float *,float *,float *,int *,int *);
void DrawUnodes(int,int,int,double,double );
void DrawSegments(int,int,int,double,double );
void Error();
Settings *GetUserOptions();

#ifdef __cplusplus
extern "C" {
#endif
int SetupApp(int argc, char **argv);
int Run_App(FILE *);
void Log( int f, const char *msg );

int ElleFindNodeAttributeRange(int id,double *min,double *max);
int ElleFindFlynnAttributeRange(int id,double *min,double *max);
int ElleFindBndAttributeRange(int id,double *min,double *max);
int ElleFindUnodeAttributeRange(int id,double *min,double *max);
int ElleFindSegmentAttributeRange(int id,double *min,double *max);
    //kommt weg, nur wegen backwards compability
void SetStage(int);
void RescaleBox(float, float, float, float);
int plotu_(float *x,float *y,int *w);
int setpencolor_(int *w);
int drawlabel_(float*, float*, int*, int*, int*, char*, int*);
int drawcircle_(int,float*, float*, float*, int*, int*);
int drawsegment_(float*, float*, float*, float*, int*);
void clearbg_();
void update_();
#ifdef __cplusplus
}
#endif
#endif //_settings_h

