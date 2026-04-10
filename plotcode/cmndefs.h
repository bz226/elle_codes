/*----------------------------------------------------------------
 *    Elle:   cmndefs.h  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/
#ifdef SUN
#define FILENAME_MAX   1024
#endif
#define SYB_FILENAME_MAX    255
#define MAX_LINE_LEN        256
#define USEFL_SIZE     8
/*#define USEFL_SIZE     0*/
#define FOREGROUND      1
#define BACKGROUND      0
#define PLOT_COLOURS  128 /* <=256-USEFL_SIZE */
/*#define PLOT_COLOURS  256*/ /*to generate cmap for wx */
#define MAX_COLOURS   (PLOT_COLOURS+USEFL_SIZE)

#define GREY_MAP	   100
#define STD_MAP 	   101

#define DEFAULT_OPTIONS_EXT ".in"
#define DEFAULT_OPTIONS_FILE "showelle.in"
#define DEFAULT_PS_FILE "showelle.ps"
#define DEFAULT_PS_EXT  ".ps"
#define DEFAULT_LOG_EXT ".log"
#define DEFAULT_COLUMNS    0  /* in cells */
#define DEFAULT_ROWS       0  /* in cells */
#define DEFAULT_STAGES     20  /* in cells */
#define DEFAULT_FONTHGT   12
/*#define DEFAULT_LGE_FONTHGT   24*/
#define DEFAULT_LGE_FONTHGT   0

/* window defaults to A4 (29.7x21.0cm) */
/* reduce to 28.4x19.7cm to allow for printer clipping */
#define DEFAULTWIDTH 197.0
#define DEFAULTHEIGHT 197.0
#define USLETTERWIDTH 202.9
#define USLETTERHEIGHT 266.4
#define A4_PAPER           0
#define US_PAPER           1

#define PORTRAIT           0
#define LANDSCAPE          1
 
#define MINWIDTH          1
#define MAXWIDTH         10    /* max no. cells/row */
#define MINHEIGHT         1
#define MAXHEIGHT        10    /* max no. cells/col */

#define NNX3_SIZE		115
#define MAX_LABEL_LEN 80
#define MAXNAME 30
#define PEN_UP 3
#define PEN_DN 2
#define J_CENTRE 0
#define J_BASE   1
#define J_TOP    -1
#define J_LEFT   1
#define J_RIGHT  -1

#define XLMARG    0.0
#define XRMARG    0.0
#define YMARG    0.0
/*
#define XLMARG    0.05
#define XRMARG    0.05
#define YMARG    0.05
*/
 
/* plot parameters */
#define SYB_VELOCITY          0
#define SYB_STRAIN        1
#define SYB_STRESS        2
#define STRN_MRKR         3
#define THICKNESS         4
#define DEFORM            5
#define MESH              6
#define BBOX              7
#define LGMESH            8

/* fix these when dashln() coordinated in xapk and lppak */
#define SOLID             8
#define DASH              1
#define DOT               7

#define FILE_OPT          0
#define DATA_IN           1
#define XY_PLOT           2
#define LOCATE            6
#define OPTIONS           7
#define HELP              8

#define TITLE             20
#define OPTS              21
#define COMMENT           22
#define LABEL             23

/* pwindo indices */
#define XINIT   0
#define YINIT   1
#define XMIN    2
#define XMAX    3
#define YMIN    4
#define YMAX    5
#define SCLX    6
#define SCLY    7
#define SIZE    8
#define XCMIN   9
#define XCMAX   10
#define YCMIN   11
#define YCMAX   12
#define ULEFT   13
#define URGHT   14
#define UBASE   15
#define UTOP    16

#define PWINDO_ENTRIES   17

/* mesh indices */
#define NX0    0
#define NX1    1
#define NX2    2
#define NX3    3
#define NY0    4
#define NY1    5
#define NY2    6
#define NY3    7
#define NXY    8
#define NNX3   9
#define NNY3  10
#define NP3	  11
#define NWK	  12 /* not used */
#define ICON  13

#define MESH_ENTRIES   14


#ifndef MIN
#define MIN(x,y)    ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x,y)    ((x) > (y) ? (x) : (y))
#endif

