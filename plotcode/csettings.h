/*----------------------------------------------------------------
 *    Elle:   csettings.h  2.0  2 October 2000
 *
 *    Copyright (c) 2000 by L.A. Evans
 *----------------------------------------------------------------*/
#ifndef _csettings_h
#define _csettings_h

#ifndef _cmndefs_h
#include "cmndefs.h"
#endif

#ifndef _E_string_utils_h
#include "string_utils.h"
#endif
#include "types.h"

#define LABEL_MRK  '"'
#define COMMENT_MRK  #
#define MAX_OPTION_NAME 20
/*
 * allowed option types
#define SYB_BOOL   201
#define SYB_INT    202
#define SYB_FLOAT  203
#define SYB_DOUBLE 204
 */

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

/*
 * valid colourmap settings
 */
#define GREY_MAP         100
#define GRAY_MAP         100
#define STD_MAP          101
#define ABS_MAP          102

/* orientation of colourbar */
#define CB_VERT   42
#define CB_HORIZ  41
#define CB_NONE   40

static valid_terms display_terms[] = {
                              { "orientation",O_ORIENT },
                              { "paper_size",O_PAPERSZ },
                              { "font_height",O_FONTHGT },
                              { "page_xmargin",O_PG_XMARG },
                              { "page_ymargin",O_PG_YMARG },
                              { "page_scale",O_PGSCL },
                              /*{ "zoom",O_ZOOM },*/
                              /*{ "label",O_LABEL },*/
                              { "foreground",O_FOREGRND },
                              { "linestyle",O_LINESTYLE },
                              { "linewidth",O_LINEWDTH },
                              { "colour",O_COLMAP },
                              /*{ "bar",O_COLBAR },*/
                              { "rescale",O_RESCALE },
                              { "verbose",O_VERBOSE },
                              { "text_bg",O_TEXTBG },
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
static valid_terms colourbar_terms[] = {
                                     {"vertical",CB_VERT },
                                     { "horizontal", CB_HORIZ },
                                     { "none", CB_NONE },
                                        NULL };
#ifdef __cplusplus
extern "C" {
#endif
void InitSettings(void);
void InitOptions(input_options *opts);
int ReadSettings(FILE *fp, char *str, unsigned char initial);
int WriteSettingsFile(char *filename);
int WriteSettings(FILE *fp);
void GetCRegionOption(int *val);
void GetCUpdFreqOption(int *val);
void GetCUnodePlotOption(int *val);
void GetCNodeAttribOption(int *val);
void SetCNodeAttribOption(int id, double min, double max);
void GetCNodeAttribMinOption(double *val);
void GetCNodeAttribMaxOption(double *val);
void GetCFlynnAttribOption(int *val);
void SetCFlynnAttribOption(int id, double min, double max);
void GetCFlynnAttribMinOption(double *val);
void GetCFlynnAttribMaxOption(double *val);
void GetCUnodeAttribOption(int *val);
void GetCUnodeAttribMinOption(double *val);
void GetCUnodeAttribMaxOption(double *val);
void GetCBndAttribOption(int *val);
void GetCBndLevelOption(int *val);
void SetCBndLevelOption(int val);
void GetCLblRgnOption(int *val);
void GetCDrawingOption(int *val);
int GetCPlotColourRange(int *min,int *max);

extern input_options Settings, Initial_Settings;

#ifdef __cplusplus
}
#endif

#endif
