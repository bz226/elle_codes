/*----------------------------------------------------------------
 *    Elle:   types.h  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/
#ifndef _types_h
#define _types_h
/*#include "defines.h"*/

#ifndef _defines_h
#define _defines_h
#ifdef XSYB
#include <X11/Xlib.h>  /* for type XFontStruct in Plot_info */
#include <X11/Intrinsic.h>  /* for type XtPointer */
#define fontdat XFontStruct *
#define colourdat XColor
#define BLACK_PEN  0
#else
#define fontdat int
#define colourdat int
#define BLACK_PEN  1
#define XtPointer int*
#endif
#endif
#include "cmndefs.h"
#include "filedat.h"

#define CNTRS     5
#define MAXCNTRVALS  11
#define MAXPRFLVALS   12

struct plot_params {
    int update_freq;
    int region;
    int overlay;
    int fill;
    int outline;
    int draw_nodes;
    int node_attrib;
    double node_limits[2];
    int unode_attrib;
    double unode_limits[2];
    int flynn_attrib;
    double flynn_limits[2];
    int boundary_attrib;
    double bnd_limits[2];
    int boundary_level;
    int label_region;
};

typedef struct {
    unsigned char orient;
    unsigned char rescale;
    unsigned char verbose;
    int paper_size;
    int rows;
    int columns;
    int dfltfonthgt;
    int fg; /* index in Colours[] */
    int linestyle; /* -1=solid, 1=dashed */
    float page_scale;
    float zoom;
    float linewidth;
    float xmargin,ymargin;
    float page_xmargin,page_ymargin;
    int colourmap;
    int max_colours;
    int text_bg;
    struct plot_params plot_opts;
} input_options;

typedef struct {
    int curr_cell;
    int max_row, max_col;
    float curr_x, curr_y;
    float origin_x, origin_y;
    float pixels_per_cm;
    fontdat font;
    fontdat dflt_font;
    fontdat large_font;
    int title_offset;
    int max_colours;
    int plot_type; /* arrows,cntrs,profile,xy_plot */
    int plot_description; /* lines,shading */
    int var_num;
    char *title;
    char fontname[MAXNAME];
    char variable[MAXNAME*2];
    char dflt_label1[MAXNAME];
    char dflt_label2[MAXNAME];
    char curr_label[MAX_LABEL_LEN];
    unsigned char display_log;
    FILE *log_file_ptr;
    file_data *inp_file;
    int stage;
} plot_data;

/*
 * equivalent to Xlib.h XRectangle
 */
typedef struct {
    short x, y;
    unsigned short width, height;
} rectangle;

/* data for each label drawn in a cell */
typedef struct {
    char label_text[MAX_LABEL_LEN];
    char fontname[MAXNAME];
    unsigned font_pointsize; /* GetFontProperty XA_POINT_SIZE */
    int fg;
    float x,y; /* in page cms, 0,0 at bottom left */
    rectangle box; /* label bounding box, in win pixels */
} label_data;

struct l_node{
    label_data label;
    struct l_node *next;
} ;
typedef struct l_node label_node;

typedef struct {
    int id;
    int type;
    int int_val;
    float float_val;
    unsigned char bool_val;
} opt_data;

struct o_node{
    opt_data opt;
    struct o_node *next;
} ;
typedef struct o_node opt_node;

typedef struct {
    char filename[FILENAME_MAX];
    char plot_param[MAXNAME];
    char dflt_label1[MAXNAME];
    char dflt_label2[MAXNAME];
    int cell_num;
    int record_num;
    int reference_num;
    int plot_type;
    int plot_description;
    int option_num;
    int stipple_type;
    int colour_bar;
    float contour_vals[MAXCNTRVALS];
    float profile_vals[MAXPRFLVALS];
} save_data;

struct s_node{
    save_data data;
    opt_node *changed_opts;
    struct s_node *next;
};
typedef struct s_node save_node;

/* data for each cell */
typedef struct {
    char drawn;
    int row,col;
    rectangle rect;
    float rect_cm_x;
    float rect_cm_y;
    save_node *plots;
} cell_data;
 
#endif

extern colourdat Colours[];
extern plot_data Plot_info;
