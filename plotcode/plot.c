/*----------------------------------------------------------------
 *    Elle:   plot.c  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/
#include <Xm/Xm.h>
#if XmVersion>1001
#include <Xm/XmP.h>
#endif
#include <X11/StringDefs.h>
#include <X11/Intrinsic.h>
#include <X11/Xatom.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cmndefs.h"
#include "plotP.h"
#include "types.h"
#include "csettings.h"
#include "globals.h"
#include "save.h"
/*#include "log.h"*/
#include "error.h"

#define XY 0

#define DEFAULT_CELL_WIDTH   200  /* in pixels */
#define DEFAULT_CELL_HEIGHT  100  /* in pixels */

#define DEFAULT_CELL_WIDTH_CM   148.5  /* in cm */
#define DEFAULT_CELL_HEIGHT_CM  DEFAULTHEIGHT/DEFAULT_PIXMAP_HEIGHT/* in cm */

/* values for instance variable is_drawn */
#define DRAWN 1
#define UNDRAWN 0

#define MAXLINES 10   /* max of horiz or vertical cells */
#define SCROLLBARWIDTH 15

#define offset(field) XtOffsetOf(PlotRec, field)

static XtResource resources[] = {
/* not needed in Motif - already defined by Primitive.
 *    {
 *   XtNforeground, 
 *   XtCForeground, 
 *   XtRPixel, 
 *   sizeof(Pixel),
 *   offset(plot.foreground), 
 *   XtRString, 
 *   XtDefaultForeground
 *    },
 */
     {
    XtNtoggleCallback, 
    XtCToggleCallback, 
    XtRCallback, 
    sizeof(XtPointer),
    offset(plot.callback), 
    XtRCallback, 
    NULL
     },
     {
    XtNpixmapWidthInCells, 
    XtCPixmapWidthInCells, 
    XtRDimension, 
    sizeof(Dimension),
    offset(plot.pixmap_width_in_cells), 
    XtRImmediate, 
    (XtPointer)DEFAULT_COLUMNS
     },
     {
    XtNpixmapHeightInCells, 
    XtCPixmapHeightInCells, 
    XtRDimension, 
    sizeof(Dimension),
    offset(plot.pixmap_height_in_cells), 
    XtRImmediate, 
    (XtPointer)DEFAULT_ROWS
     },
     {
    XtNcellWidthInPixels, 
    XtCCellWidthInPixels, 
    XtRDimension, sizeof(Dimension),
    offset(plot.cell_width_in_pixels), 
    XtRImmediate, 
    (XtPointer)DEFAULT_CELL_WIDTH
     },
     {
    XtNcellHeightInPixels, 
    XtCCellHeightInPixels, 
    XtRDimension, sizeof(Dimension),
    offset(plot.cell_height_in_pixels), 
    XtRImmediate, 
    (XtPointer)DEFAULT_CELL_HEIGHT
     },
/*
     {
    XtNcellWidthInCm, 
    XtCCellWidthInCm, 
    XtRFloat, sizeof(float),
    offset(plot.cell_width_in_cm), 
    XtRImmediate, 
    (XtPointer)DEFAULT_CELL_WIDTH_CM
     },
     {
    XtNcellHeightInCm, 
    XtCCellHeightInCm, 
    XtRFloat, sizeof(float),
    offset(plot.cell_height_in_cm), 
    XtRImmediate, 
    (XtPointer)DEFAULT_CELL_HEIGHT_CM
     },
*/
     {
    XtNcurX, 
    XtCCurX, 
    XtRInt, 
    sizeof(int),
    offset(plot.cur_x), 
    XtRImmediate, 
    (XtPointer) 0
     },
     {
    XtNcurY, 
    XtCCurY, 
    XtRInt, 
    sizeof(int),
    offset(plot.cur_y), 
    XtRImmediate, 
    (XtPointer) 0
     },
     {
    XtNplotList, 
    XtCPlotList, 
    XtRString, 
    sizeof(String),
    offset(plot.plots), 
    XtRImmediate, 
    (XtPointer) 0
     },
     {
    XtNlabelList, 
    XtCLabelList, 
    XtRString, 
    sizeof(String),
    offset(plot.labels), 
    XtRImmediate, 
    (XtPointer) 0
     },
     {
    XtNcellArray, 
    XtCCellArray, 
    XtRString, 
    sizeof(String),
    offset(plot.cell), 
    XtRImmediate, 
    (XtPointer) 0
     },
     {
    XtNshowEntireBitmap, 
    XtCShowEntireBitmap, 
    XtRBoolean, 
    sizeof(Boolean),
    offset(plot.show_all), 
    XtRImmediate, 
    (XtPointer) TRUE
     },
};

/* Declaration of methods */

static void Initialize();
static void Reinitialize();
static void Redisplay();
static void Destroy();
static void Resize();
static Boolean SetValues();
static XtGeometryResult QueryGeometry();

/* these Core methods not needed by BitmapEdit:
 *
 * static void ClassInitialize();
 * static void Realize();
 */

/* the following are private functions unique to plotArea */
static void DrawBorder(), DoCell();
void ChangeCellSize();
void ConfigureCells(), ChangeCellConfig();

/* the following are actions of plotArea */
static void UndrawCell(), OptionalLabel();

static char defaultTranslations[] =
    "<Btn1Down>:    OptionalLabel()        \n\
    <Btn2Down>:    UndrawCell()            \n\
    <Btn2Motion>:  UndrawCell()";

static XtActionsRec actions[] = {
        {"OptionalLabel", OptionalLabel},
        {"UndrawCell", UndrawCell},
};

/* definition in plot.h */
static PlotPointInfo info;

PlotClassRec plotClassRec = {
    {
    /* core_class fields */
#if XmVersion>1001
    /* superclass        */ (WidgetClass) &xmPrimitiveClassRec,
#else
    /* superclass        */ (WidgetClass) &coreClassRec,
#endif
    /* class_name        */ "Plot",
    /* widget_size       */ sizeof(PlotRec),
    /* class_initialize      */ NULL,
    /* class_part_initialize     */ NULL,
    /* class_inited          */ FALSE,
    /* initialize        */ Initialize,
    /* initialize_hook       */ NULL,
    /* realize           */ XtInheritRealize,
    /* actions           */ actions,
    /* num_actions       */ XtNumber(actions),
    /* resources         */ resources,
    /* num_resources         */ XtNumber(resources),
    /* xrm_class         */ NULLQUARK,
    /* compress_motion       */ TRUE,
    /* compress_exposure     */ XtExposeCompressMultiple,
    /* compress_enterleave   */ TRUE,
    /* visible_interest      */ FALSE,
    /* destroy           */ Destroy,
    /* resize            */ Resize,
    /* expose            */ Redisplay,
    /* set_values        */ SetValues,
    /* set_values_hook       */ NULL,
    /* set_values_almost     */ XtInheritSetValuesAlmost,
    /* get_values_hook       */ NULL,
    /* accept_focus      */ NULL,
    /* version           */ XtVersion,
    /* callback_private      */ NULL,
    /* tm_table          */ defaultTranslations,
    /* query_geometry      */ QueryGeometry,
    /* display_accelerator       */ XtInheritDisplayAccelerator,
    /* extension                 */ NULL
    },
#if XmVersion>1001
    {  /* Primitive class fields */
    /* border_highlight   */        _XtInherit,       
    /* border_unhighlight */        _XtInherit,    
    /* translations       */        XtInheritTranslations,      
    /* arm_and_activate   */        NULL,             
    /* syn resources      */        NULL,           
    /* num_syn_resources  */        0, 
    /* extension          */        NULL,             
    },
#endif
    {
    /* extension          */        0,
    },
};

WidgetClass plotWidgetClass = (WidgetClass) & plotClassRec;

float Screen_scale_factor;
extern struct {float psca_x, psca_y, ixo, iyo, iox, ioy;} a00000_;

void UndrawCellBorder();
void DrawCellBorder();
void UpdateCell();
void set_box();
int Write_plot_data();
void LocateOptionalLabel();
void clear_X(), DeleteAllCells(), DeleteCell(), clearbg_();
extern void set_box_dimensions();
extern void Read_label();

static void GetBorderGC(w)
Widget w;
{
    PlotWidget cw = (PlotWidget) w;
    XGCValues values;
    XtGCMask mask = GCForeground | GCBackground | GCDashOffset | 
            GCDashList | GCLineStyle;
    extern GC Border_gc;

    /* 
     * Set foreground and background to those of widget
     * border is drawn as dotted line
     */
#if XmVersion<1002
    values.foreground = 1;
    values.background = 0;
#else
    values.foreground = cw->primitive.foreground;
    values.background = cw->core.background_pixel;
#endif
    values.dashes = 1;
    values.dash_offset = 0;
    values.line_style = LineOnOffDash;

    Border_gc = cw->plot.border_gc = XCreateGC(XtDisplay(cw), 
             cw->plot.big_picture, mask, &values);
}

static void GetDrawGC(w)
Widget w;
{
    PlotWidget cw = (PlotWidget) w;
    XGCValues values;
    XtGCMask mask = GCForeground | GCBackground;
    extern GC Draw_gc;

    /* 
     * Set foreground and background
     */
#if XmVersion<1002
    values.foreground = 1;
    values.background = 0;
#else
    values.foreground = cw->primitive.foreground;
    values.background = cw->core.background_pixel;
#endif

    Draw_gc = cw->plot.draw_gc = XCreateGC(XtDisplay(cw), 
             cw->plot.big_picture, mask, &values);
}

static void GetUndrawGC(w)
Widget w;
{
    PlotWidget cw = (PlotWidget) w;
    XGCValues values;
    XtGCMask mask = GCForeground | GCBackground;
    extern GC Undraw_gc;

    /* 
     * Set foreground and background to bg and fg of widget
     */
#if XmVersion<1002
    values.foreground = 0;
    values.background = 1;
#else
    values.foreground = cw->core.background_pixel;
    values.background = cw->primitive.foreground;
#endif

    Undraw_gc = cw->plot.undraw_gc = XCreateGC(XtDisplay(cw), 
              cw->plot.big_picture, mask, &values);
}

/*
static void GetCopyGC(w)
Widget w;
{
    PlotWidget cw = (PlotWidget) w;
    XGCValues values;
    XtGCMask mask = GCForeground | GCBackground;
    extern GC Copy_gc;

    values.foreground = cw->primitive.foreground;
    values.background = cw->core.background_pixel;

    Copy_gc = cw->plot.copy_gc = XtGetGC(cw, mask, &values);
}
*/

/* ARGSUSED */
static void Initialize(treq, tnew, args, num_args)
Widget treq, tnew;
ArgList args;
Cardinal *num_args;
{
    char msg[100];
    int i,j,indx;
    int dumiro,dumlpl,dumils;
    float pixel_hgt,pixel_wdth,width,height;
    Screen *scr;


    unsigned char warn=0;
    PlotWidget new = (PlotWidget) tnew;
    new->plot.cur_x = 0;
    new->plot.cur_y = 0;

    /* 
     *  Check instance values set by resources that may be invalid. 
     */

    if (new->plot.pixmap_width_in_cells==0) /* not set by command line */
         new->plot.pixmap_width_in_cells = Settings.columns;
    if (new->plot.pixmap_width_in_cells < 1) {
        warn=1;
        new->plot.pixmap_width_in_cells = Settings.columns = 1;
        Initial_Settings.columns = Settings.columns;
    }
    if (new->plot.pixmap_height_in_cells==0) /* not set by command line */
         new->plot.pixmap_height_in_cells = Settings.rows;
    if (new->plot.pixmap_height_in_cells < 1) {
        warn=1;
        new->plot.pixmap_height_in_cells = Settings.rows = 1;
        Initial_Settings.rows = Settings.rows;
    }
    if (warn) {
        sprintf(msg,
        "sybil: columns and/or rows invalid or not set (using %d x %d).",
        new->plot.pixmap_width_in_cells,new->plot.pixmap_height_in_cells); 
        XtWarning(msg);
    }
    ConfigureCells(new);

    GetDrawGC(new);
    GetUndrawGC(new);
    GetBorderGC(new);
			    
    InitBigPixmap(new);
}


void ConfigureCells(PlotWidget new)
{
    int i,j,indx;
    int dumiro,dumlpl,dumils;
    float pixel_hgt,pixel_wdth,width,height;
    float x_offset,y_offset,tmp;
	Dimension wdth,hgt;
    Screen *scr;

    Screen_scale_factor = 1.0;
    scr = XtScreen( new );
    if (Settings.orient==PORTRAIT) {
        height = DEFAULTHEIGHT*(Pwindo[YCMAX]-Pwindo[YCMIN]);
        width = DEFAULTWIDTH*(Pwindo[XCMAX]-Pwindo[XCMIN]);
        pixel_hgt = (float)HeightOfScreen(scr)/HeightMMOfScreen(scr)*height;
        if ((int)pixel_hgt>(HeightOfScreen(scr)-100)) {
            Screen_scale_factor=(float)(HeightOfScreen(scr)-100)/pixel_hgt;
            pixel_hgt = (float)HeightOfScreen(scr)-100;
        }
        pixel_wdth = (float)WidthOfScreen(scr)/WidthMMOfScreen(scr)*width
                                                * Screen_scale_factor;
    }
    else {
        height = DEFAULTWIDTH*(Pwindo[YCMAX]-Pwindo[YCMIN]);
        width = DEFAULTHEIGHT*(Pwindo[XCMAX]-Pwindo[XCMIN]);
        pixel_wdth = (float)WidthOfScreen(scr)/WidthMMOfScreen(scr)*width;
        if ((int)pixel_wdth>(WidthOfScreen(scr)-100)) {
            Screen_scale_factor=(float)(WidthOfScreen(scr)-100)/pixel_wdth;
            pixel_wdth = (float)WidthOfScreen(scr)-100;
        }
        pixel_hgt = (float)HeightOfScreen(scr)/HeightMMOfScreen(scr)*height
                                                * Screen_scale_factor;
    }
    /*
     * allow for main title at top of page
     */
    new->plot.cell_height_in_pixels = (int)(
                                    (pixel_hgt-(float)Plot_info.title_offset)/
                                    new->plot.pixmap_height_in_cells);
    new->plot.cell_width_in_pixels = (int)(pixel_wdth/
                                    new->plot.pixmap_width_in_cells);
    /*
     * cell dimensions in cm, before scaling to fit on screen
     */
    new->plot.cell_height_in_cm = (float)new->plot.cell_height_in_pixels/
                     ((float)HeightOfScreen(scr)/HeightMMOfScreen(scr)*10);
    new->plot.cell_width_in_cm = (float)new->plot.cell_width_in_pixels/
                     ((float)WidthOfScreen(scr)/WidthMMOfScreen(scr)*10);

    if ((new->plot.cur_x < 0) ||  (new->plot.cur_y < 0)) {
        XtWarning("sybil: cur_x and cur_y must be non-negative (using 0, 0).");
        new->plot.cur_x = 0;
        new->plot.cur_y = 0;
    }

    if (new->plot.cell == NULL) {
        new->plot.cell = (cell_data *)XtCalloc( 
                new->plot.pixmap_width_in_cells * 
                new->plot.pixmap_height_in_cells, sizeof(cell_data));
        new->plot.user_allocated = False;
    }
    else
        new->plot.user_allocated = True;  /* user supplied cell array */

    new->plot.pixmap_width_in_pixels = pixel_wdth;
    /*       new->plot.pixmap_width_in_cells * 
           new->plot.cell_width_in_pixels;*/

    new->plot.pixmap_height_in_pixels = pixel_hgt;
     /*      new->plot.pixmap_height_in_cells * 
           new->plot.cell_height_in_pixels + Plot_info.title_offset;*/

    /* 
     * Motif Primitive sets width and height to provide enough room for
     * the highlight and shadow around a widget.  Newsyb
     * doesn't use these features.  A widget that did use these
     * features would *add* its desired dimensions to those set
     * by Primitive.  To use this widget with another widget set, remove
     * the following two lines. 
     */
    new->core.width = 0;
    new->core.height = 0;

    if (new->core.width == 0) {
        if (new->plot.show_all == False)
            new->core.width = (new->plot.pixmap_width_in_pixels 
                    > pixel_wdth) ? pixel_wdth : 
                    (new->plot.pixmap_width_in_pixels);
        else
            new->core.width = new->plot.pixmap_width_in_pixels;
    }

    if (new->core.height == 0) {
        if (new->plot.show_all == False)
            new->core.height = 
                    (new->plot.pixmap_height_in_pixels > 
                    pixel_hgt) ? pixel_hgt : 
                    (new->plot.pixmap_height_in_pixels);
        else
            new->core.height = new->plot.pixmap_height_in_pixels;
    }

    /* index cells 0 -> columns(rows) -1 */
    Plot_info.max_row=new->plot.pixmap_height_in_cells-1;
    Plot_info.max_col=new->plot.pixmap_width_in_cells-1;
    Plot_info.pixels_per_cm=(float)HeightOfScreen(scr)/
                                            HeightMMOfScreen(scr)*10;

dumiro=(int)Settings.orient; dumlpl=11; dumils=2;
xpak_init( &dumiro,&dumlpl,&dumils,XtDisplay(new) );

a00000_.psca_x=(float)WidthOfScreen(scr)/WidthMMOfScreen(scr)*10;
a00000_.psca_y=Plot_info.pixels_per_cm;
a00000_.iyo=new->plot.pixmap_height_in_pixels;
/***FIX this when data read from saved file *****/
        /* set values for each cell */
    for (i=0,indx=0;i<new->plot.pixmap_height_in_cells;i++) {
        for (j=0;j<new->plot.pixmap_width_in_cells;j++,indx++) {
            new->plot.cell[indx].drawn=0;
            new->plot.cell[indx].row=i;
            new->plot.cell[indx].col=j;
            new->plot.cell[indx].rect.x=j*new->plot.cell_width_in_pixels;
            new->plot.cell[indx].rect.y=i*new->plot.cell_height_in_pixels +
                                        Plot_info.title_offset;
            new->plot.cell[indx].rect_cm_x = new->plot.cell[indx].rect.x/
                                             a00000_.psca_x;
            new->plot.cell[indx].rect_cm_y =
               (new->plot.pixmap_height_in_pixels-new->plot.cell[indx].rect.y)/
                                                 a00000_.psca_y;
            new->plot.cell[indx].rect.width=new->plot.cell_width_in_pixels-1;
            new->plot.cell[indx].rect.height=new->plot.cell_height_in_pixels-1;
        }
    }

	/* this assumes only one cell */
    set_box_dimensions( Pwindo,
                        new->plot.cell[Plot_info.curr_cell].rect_cm_x,
						new->plot.cell[Plot_info.curr_cell].rect_cm_y
						 - new->plot.cell_height_in_cm,
                        new->plot.cell_width_in_cm,
                        new->plot.cell_height_in_cm );
    CreateBigPixmap(new);
}

void ChangeCellConfig(cw)
PlotWidget cw;
{

    if (cw->plot.big_picture)
        XFreePixmap(XtDisplay(cw), cw->plot.big_picture);
    /*
     * Free memory allocated with Calloc.  This was done
     * only if application didn't supply cell array.
     */
    if (!cw->plot.user_allocated) {
        XtFree((XtPointer)cw->plot.cell);
        cw->plot.cell = NULL;
    }

    if (Plot_info.title) {
        free(Plot_info.title);
        Plot_info.title = NULL;
    }
    Plot_info.title_offset = DEFAULT_LGE_FONTHGT * 1.5;
    Plot_info.curr_cell = 0;
    Reinitialize(cw);

    InitBigPixmap(cw);
}


static void Reinitialize( tnew )
Widget tnew;
{
    char msg[100];

    unsigned char warn=0;
    PlotWidget new = (PlotWidget) tnew;
    new->plot.cur_x = 0;
    new->plot.cur_y = 0;

    /* 
     *  Check instance values set by resources that may be invalid. 
     */

    new->plot.pixmap_width_in_cells = (Dimension)Settings.columns;
    new->plot.pixmap_height_in_cells = (Dimension)Settings.rows;

    if (warn) {
        sprintf(msg,
        "sybil: columns and/or rows invalid or not set (using %d x %d).",
        new->plot.pixmap_width_in_cells,new->plot.pixmap_height_in_cells); 
        XtWarning(msg);
    }
    ConfigureCells( new );

    InitBigPixmap(new);
}


/* ARGSUSED */
static void Redisplay(w, event)
Widget w;
XExposeEvent *event;
{
    PlotWidget cw = (PlotWidget) w;
    register int x, y;
    unsigned int width, height;
    if (!XtIsRealized(w))
    return;

    if (event) {  /* called from btn-event or expose */
        x = event->x;
        y = event->y; 
        width = event->width;
        height =  event->height;
    } 
    else {        /* called because complete redraw */
        x = 0;
        y = 0; 
        width = cw->core.width;
        height = cw->core.height;
        width = cw->plot.pixmap_width_in_pixels;
        height = cw->plot.pixmap_height_in_pixels;
    }

/*
    XCopyArea(XtDisplay(cw), cw->plot.big_picture, 
                XtWindow(cw), cw->plot.draw_gc, x + 
                cw->plot.cur_x, y + cw->plot.cur_y, 
                width, height, x, y);
*/
    XCopyArea(XtDisplay(cw), cw->plot.big_picture, 
                XtWindow(cw), cw->plot.draw_gc, x, y, 
                width, height, x, y);

    /*if (Settings.mark_cell) DrawBorder(cw->plot.draw_gc, DRAW,*/
                                            /*cw, Plot_info.curr_cell);*/
}

/* ARGSUSED */
static Boolean SetValues(current, request, new, args, num_args)
Widget current, request, new;
ArgList args;
Cardinal *num_args;
{
    PlotWidget curcw = (PlotWidget) current;
    PlotWidget newcw = (PlotWidget) new;
    Boolean do_redisplay = False;

/*
    if (curcw->primitive.foreground != newcw->primitive.foreground) {
        XtReleaseGC(curcw, curcw->plot.copy_gc);
        GetCopyGC(newcw);
        do_redisplay = True;
    }
*/

    if ((curcw->plot.cur_x != newcw->plot.cur_x) || 
            (curcw->plot.cur_y != newcw->plot.cur_y))
        do_redisplay = True;

    if (curcw->plot.cell_height_in_pixels != 
            newcw->plot.cell_height_in_pixels) {
        ChangeCellSize(curcw, newcw->plot.cell_width_in_pixels,
                        newcw->plot.cell_height_in_pixels);
        do_redisplay = True;
    }

    if (curcw->plot.pixmap_width_in_cells != 
            newcw->plot.pixmap_width_in_cells)  {
        newcw->plot.pixmap_width_in_cells = 
                curcw->plot.pixmap_width_in_cells;
        XtWarning("BitmapEdit: pixmap_width_in_cells cannot be set by XtSetValues.\n");
    }

    if (curcw->plot.pixmap_height_in_cells != 
            newcw->plot.pixmap_height_in_cells) {
        newcw->plot.pixmap_height_in_cells = 
                curcw->plot.pixmap_height_in_cells;
        XtWarning("BitmapEdit: pixmap_height_in_cells cannot be set by XtSetValues.\n");
    }

    return do_redisplay;
}

void clear_X(w)
Widget w;
{
    Destroy(w);
}

static void Destroy(w)
Widget w;
{
    int i,num;
    PlotWidget cw = (PlotWidget) w;
    if (cw->plot.big_picture)
        XFreePixmap(XtDisplay(cw), cw->plot.big_picture);

    if (cw->plot.draw_gc)
        XFreeGC(XtDisplay(cw), cw->plot.draw_gc);

    if (cw->plot.undraw_gc)
        XFreeGC(XtDisplay(cw), cw->plot.undraw_gc);

    if (cw->plot.border_gc)
        XFreeGC(XtDisplay(cw), cw->plot.border_gc);

    /*
     * Free memory allocated to save plots drawn to each cell.
     */
    Clear_all_saved_data(w);
    /*
     * Free memory allocated with Calloc.  This was done
     * only if application didn't supply cell array.
     */
    if (!cw->plot.user_allocated)
        XtFree((XtPointer)cw->plot.cell);
}

/*
 * clears outline of previous cell, outlines current cell
 */
void ChangeCell( prev,new )
int prev,new;
{
    extern PlotWidget plotArea;
    PlotWidget cw = plotArea;

    /*if (Settings.mark_cell && XtIsRealized(cw)) {*/
        /*UndrawCellBorder(cw,prev);*/
        /*DrawCellBorder(cw,new);*/
    /*}*/
	/* this assumes only one cell */
    set_box_dimensions( Pwindo,
                        cw->plot.cell[Plot_info.curr_cell].rect_cm_x,
						cw->plot.cell[Plot_info.curr_cell].rect_cm_y
						 - cw->plot.cell_height_in_cm,
                        cw->plot.cell_width_in_cm,
                        cw->plot.cell_height_in_cm );
    init_box(Pwindo,0);
}

void set_box(pwin,plot)
float *pwin;
PlotWidget plot;
{

	/* this assumes only one cell */
    set_box_dimensions( pwin,
					    plot->plot.cell[Plot_info.curr_cell].rect_cm_x,
					    plot->plot.cell[Plot_info.curr_cell].rect_cm_y
						 - plot->plot.cell_height_in_cm,
                        plot->plot.cell_width_in_cm,
                        plot->plot.cell_height_in_cm );
}

void DrawCellBorder(w, number)
Widget w;
int number;
{
    PlotWidget cw = (PlotWidget) w;
    /* update Plot_info */
    Plot_info.curr_x = Plot_info.origin_x =
                        (float) cw->plot.cell[number].rect.x;
    Plot_info.curr_y = Plot_info.origin_y =
                        (float) cw->plot.cell[number].rect.y;

    DrawBorder(cw->plot.draw_gc, DRAW, cw, number);
}

void UndrawCellBorder(w, number)
Widget w;
int number;
{
    PlotWidget cw = (PlotWidget) w;
    DrawBorder(cw->plot.undraw_gc, UNDRAW, cw, number);
}

void UpdateCell(w, number, mode)
Widget w;
int number,mode;
{
    XExposeEvent fake_event;
    PlotWidget cw = (PlotWidget) w;

/*
    fake_event.x = cw->plot.cell[number].rect.x+1;
    fake_event.y = cw->plot.cell[number].rect.y+1;
    fake_event.width = cw->plot.cell[number].rect.width-2;
    fake_event.height = cw->plot.cell[number].rect.height-2;
*/
    Redisplay(cw, NULL);
}

void DeleteCell(number)
int number;
{
	XButtonEvent fake_event;
    extern PlotWidget plotArea;
    PlotWidget cw = plotArea;

    fake_event.x = cw->plot.cell[number].rect.x+1;
    fake_event.y = cw->plot.cell[number].rect.y+1;
    UndrawCell(plotArea,&fake_event);
}

void clearbg_()
{
    extern PlotWidget plotArea;
    PlotWidget cw = plotArea;

    XFillRectangle(XtDisplay(cw), cw->plot.big_picture,
        cw->plot.undraw_gc, 0, Plot_info.title_offset, 
        cw->plot.pixmap_width_in_pixels + 2,
        cw->plot.pixmap_height_in_pixels + 2);
}

void DeleteAllCells()
{
    XExposeEvent fake_event;
    extern PlotWidget plotArea;
    PlotWidget cw = plotArea;

    /* doesn't clear the title area */
    Clear_all_saved_data(cw);
    XFillRectangle(XtDisplay(cw), cw->plot.big_picture,
        cw->plot.undraw_gc, 0, Plot_info.title_offset, 
        cw->plot.pixmap_width_in_pixels + 2,
        cw->plot.pixmap_height_in_pixels + 2);
    Redisplay(cw, NULL);
}

/* BLANK cell*/
static void UndrawCell(w, event)
Widget w;
XEvent *event;
{
    PlotWidget cw = (PlotWidget) w;
    XExposeEvent fake_event;

    int newx = (cw->plot.cur_x + ((XButtonEvent *)event)->x) / 
             cw->plot.cell_width_in_pixels;
    int newy = (cw->plot.cur_y + ((XButtonEvent *)event)->y) / 
             cw->plot.cell_height_in_pixels;
    if (((XButtonEvent *)event)->y>Plot_info.title_offset) {
        if ((cw->plot.pixmap_width_in_cells*newy+newx)==Plot_info.curr_cell) {
            Clear_saved_data( cw,Plot_info.curr_cell );
            XFillRectangle(XtDisplay(cw),
                        cw->plot.big_picture, cw->plot.undraw_gc,
                        cw->plot.cell_width_in_pixels*newx + 1,
                        cw->plot.cell_height_in_pixels*newy + 1 + 
                            Plot_info.title_offset, 
                        (unsigned int)cw->plot.cell_width_in_pixels - 2, 
                        (unsigned int)cw->plot.cell_height_in_pixels - 2);
            fake_event.x = cw->plot.cell[Plot_info.curr_cell].rect.x-1;
            fake_event.y = cw->plot.cell[Plot_info.curr_cell].rect.y-1;
            fake_event.width = cw->plot.cell[Plot_info.curr_cell].rect.width+2;
            fake_event.height = cw->plot.cell[Plot_info.curr_cell].rect.height+2;
            Redisplay(cw, &fake_event);
        }
    }
    else {
        free(Plot_info.title);
        Plot_info.title = NULL;
        XFillRectangle(XtDisplay(cw), cw->plot.big_picture, cw->plot.undraw_gc,
                        1, 1, 
                        (unsigned int)cw->plot.pixmap_width_in_pixels - 1, 
                        (unsigned int)Plot_info.title_offset - 1);
        fake_event.x = 1;
        fake_event.y = 1;
        fake_event.width = cw->plot.pixmap_width_in_pixels;
        fake_event.height = Plot_info.title_offset;
        Redisplay(cw, &fake_event);
    }
}

#if XY
static void ToggleCell(w, event)
Widget w;
XEvent *event;
{
    PlotWidget cw = (PlotWidget) w;
    static int oldx = -1, oldy = -1;
    GC gc;
    int mode;
    int newx, newy;

    /* This is strictly correct, but doesn't
     * seem to be necessary */
    if (event->type == ButtonPress) {
        newx = (cw->plot.cur_x + ((XButtonEvent *)event)->x) / 
        cw->plot.cell_width_in_pixels;
        newy = (cw->plot.cur_y + ((XButtonEvent *)event)->y) / 
        cw->plot.cell_height_in_pixels;
    }
    else  {
        newx = (cw->plot.cur_x + ((XMotionEvent *)event)->x) / 
        cw->plot.cell_width_in_pixels;
        newy = (cw->plot.cur_y + ((XMotionEvent *)event)->y) / 
        cw->plot.cell_height_in_pixels;
    }


    if ((mode = cw->plot.cell[newx+newy*cw->plot.pixmap_width_in_cells].drawn)
                                                            == DRAWN) {
        gc = cw->plot.undraw_gc;
        mode = UNDRAW;
    }
    else {
        gc = cw->plot.draw_gc;
        mode = DRAW;
    }

    if (oldx != newx || oldy != newy) {
        oldx = newx;
        oldy = newy;
        DrawPixmaps(gc, mode, cw, event);
    } 
}
#endif

static void DrawBorder(gc, mode, w, number)
GC gc;
int mode;
Widget w;
int number;
{
    XGCValues valuesold, valuesnew;
    XtGCMask mask = GCLineStyle|GCForeground;
    int newx, newy;

    PlotWidget cw = (PlotWidget) w;
    XExposeEvent fake_event;

    /* otherwise, draw or undraw */
    if (mode==UNDRAW) {
        /* erase outline for old cell */
        XGetGCValues(XtDisplay(cw),cw->plot.border_gc,mask,&valuesold);
        valuesnew.line_style = LineSolid;
        valuesnew.foreground = cw->core.background_pixel;
        XChangeGC(XtDisplay(cw),cw->plot.border_gc,mask,&valuesnew);
        XDrawRectangle(XtDisplay(cw), XtWindow(cw),
            cw->plot.border_gc,
            cw->plot.cell[number].rect.x,
            cw->plot.cell[number].rect.y,
            (unsigned int)cw->plot.cell[number].rect.width,
            (unsigned int)cw->plot.cell[number].rect.height);
        XChangeGC(XtDisplay(cw),cw->plot.border_gc,mask,&valuesold);

/*
        fake_event.x = cw->plot.cell[number].rect.x;
        fake_event.y = cw->plot.cell[number].rect.y;
        fake_event.width = cw->plot.cell[number].rect.width+1;
        fake_event.height = cw->plot.cell[number].rect.height+1;
        Redisplay(cw, &fake_event);
*/
    }
    else {
        /* draw outline for new cell */
        XDrawRectangle(XtDisplay(cw), XtWindow(cw),
            cw->plot.border_gc,
            cw->plot.cell[number].rect.x,
            cw->plot.cell[number].rect.y,
            (unsigned int)cw->plot.cell[number].rect.width,
            (unsigned int)cw->plot.cell[number].rect.height);

    }


    newx = cw->plot.cell[number].col;
    newy = cw->plot.cell[number].row;
    cw->plot.cell[number].drawn = mode;
    info.mode = mode;
    info.newx = newx;
    info.newy = newy;
}

/*
 * draws label at x,y (user co-ords)
 * y_offset<0:  draw label below this y value
 * y_offset=0:  draw label centred on y value
 * y_offset>0:  draw label above this y value
 * x_offset>0:  draw label at this x value
 * x_offset=0:  draw label centred on this x value
 * x_offset<0:  draw label ending on this x value
 */
drawlabel_(x, y, x_offset, y_offset, mode, label, len)
float *x,*y;
int *mode,*x_offset,*y_offset,*len;
char *label;
{
    XGCValues valuesold, valuesnew;
    XtGCMask mask = GCLineStyle|GCForeground;
    XCharStruct overall;
    extern PlotWidget plotArea;

    int width=0,ascent,descent,direction,state;
    float xtmp, ytmp;
    PlotWidget cw = plotArea;

/*
    if (mode==UNDRAW)
        XFillRectangle(XtDisplay(cw), cw->plot.big_picture, gc,
                        cw->plot.cell_width_in_pixels*newx + 1,
                        cw->plot.cell_height_in_pixels*newy + 1,
                        (unsigned int)cw->plot.cell_width_in_pixels - 2,
                       (unsigned int)cw->plot.cell_height_in_pixels -
2);
    else {
    }
*/

    width = 8;
    descent = ascent = 8;
    if (Plot_info.font!=NULL) {
        XSetFont(XtDisplay(cw),cw->plot.draw_gc,Plot_info.font->fid);
        XTextExtents(Plot_info.font,
                        label,strlen(label),
                        &direction,&ascent,&descent,&overall);
        width = overall.width;
    }
    state=PEN_UP;
    xtmp= *x; ytmp= *y;
    plotu_(&xtmp,&ytmp,&state);
    xtmp=a00000_.iox; ytmp=a00000_.ioy;
    if (*y_offset==J_CENTRE) ytmp += (ascent+descent)/2;
    else if (*y_offset==J_TOP) ytmp += ascent+1;
    else if (*y_offset==J_BASE) ytmp -= descent+1;
    if (*x_offset==J_RIGHT) xtmp -= width+1;
    else if (*x_offset==J_CENTRE) {
        width /= 2;
        xtmp -= width;
    }
    /*
     * keep the label on the page
     */
    if ((int)xtmp+width > cw->plot.pixmap_width_in_pixels)
        xtmp = cw->plot.pixmap_width_in_pixels-width-1;
    if ((int)ytmp > cw->plot.pixmap_height_in_pixels)
        ytmp = cw->plot.pixmap_height_in_pixels-1;

    XDrawImageString(XtDisplay(cw),cw->plot.big_picture,cw->plot.draw_gc,
                        (int)xtmp, (int)ytmp,
                        label,strlen(label));

    a00000_.iox = xtmp+width;
    a00000_.ioy = ytmp;
}
 
/*
 * label max and min on the colour bar
 */
labelcolourbar_(min,max,scale,barlen,xlevel,ylevel,vertical)
float *min,*max,*scale,*barlen,*xlevel,*ylevel;
int *vertical;
{
    char number[12];
    int current_pen;
    int format,len,mode,x_offset,y_offset;
    float scale_min, scale_max,x,y;
    XFontStruct *curr_font;

    current_pen=FOREGROUND;
    setpencolor_(&current_pen);
    curr_font = Plot_info.font;
    Plot_info.font = Plot_info.dflt_font;
    scale_min = *min * *scale;
    scale_max = *max * *scale;
    format = MAX(fabs((double)scale_min),fabs((double)scale_max))
                        >10.0 ? 60 : 62;
    format_number_(number,&scale_min,&format);
    if (*vertical) {
        x_offset = J_CENTRE;
        y_offset = J_TOP;
    }
    else {
        x_offset = J_RIGHT;
        y_offset = J_TOP;
    }
    mode = 1; len = format/10;
    drawlabel_(xlevel,ylevel,&x_offset,&y_offset,&mode,number,&len);
    format_number_(number,&scale_max,&format);
    if (*vertical) {
        x_offset = J_CENTRE;
        y_offset = J_BASE;
        x = *xlevel;
        y = *ylevel + *barlen;
    }
    else {
        x_offset = J_LEFT;
        y_offset = J_TOP;
        x = *xlevel + *barlen;
        y = *ylevel;
    }
    mode = 1; len = format/10;
    drawlabel_(&x,&y,&x_offset,&y_offset,&mode,number,&len);
    Plot_info.font = curr_font;
}

drawautolabels_(label,labellen,value,format,pwindo,current_pen)
char *label;
int *format, *labellen;
float *value,*pwindo;
int *current_pen;
{
    char number[12];
    XFontStruct *curr_font;
    int old_pen,x_offset,y_offset,mode,len;

    strcpy(number,"");
    old_pen = *current_pen;
    *current_pen=FOREGROUND;
    setpencolor_(current_pen);
    curr_font = Plot_info.font;
    Plot_info.font = Plot_info.dflt_font;
    x_offset = J_LEFT; y_offset = J_TOP; mode = 1;
    drawlabel_(&pwindo[XCMIN],&pwindo[YCMIN],&x_offset,&y_offset,&mode,
                                                       label,labellen);
    if (*format) {
        format_number_(number,value,format);
        /* draw label at centre
        x_offset = J_LEFT; y_offset = J_TOP; mode = 1; len = strlen(number);
        drawlabel_((pwindo[YCMAX]-pwindo[YCMIN])/2,pwindo[XCMIN],
                                   &x_offset,&y_offset,&mode,number,&len);
        */
        x_offset = J_RIGHT; y_offset = J_TOP; mode = 1; len = strlen(number);
        drawlabel_(&pwindo[XCMAX],&pwindo[YCMIN],
                                  &x_offset,&y_offset,&mode,number,&len);
    }
    *current_pen=old_pen;
    setpencolor_(current_pen);
    Plot_info.font = curr_font;
}

int drawlabelcm_(x,y,offset_x,offset_y,mode,label,nchar)
char *label;
int *offset_x,*offset_y,*mode,*nchar;
float *x,*y;
{
    XButtonEvent fake_event;
    extern PlotWidget plotArea;

    cms_to_pixels(*x,*y,&(fake_event.x),&(fake_event.y),
                      a00000_.psca_x,a00000_.psca_y, (int)a00000_.iyo);
    LocateOptionalLabel(plotArea,&fake_event);
    return(0);
}

void DrawTitle( title )
char *title;
{
    int width=0,ascent,descent,direction;
    extern PlotWidget plotArea;
    PlotWidget cw = plotArea;
    XFontStruct *tmp;
    XExposeEvent fake_event;
    XCharStruct overall;
/*
 * replace with call to OptionalLabel
 */

    tmp = Plot_info.font;
    Plot_info.font = Plot_info.large_font;
    XSetFont(XtDisplay(cw),cw->plot.draw_gc,Plot_info.font->fid);
    XTextExtents(Plot_info.font,
                        title,strlen(title),
                        &direction,&ascent,&descent,&overall);
    width = overall.width;
    XDrawImageString(XtDisplay(cw),cw->plot.big_picture,cw->plot.draw_gc,
                        Plot_info.title_offset,
                        Plot_info.title_offset*2/3,
                        title,strlen(title));
    fake_event.x = Plot_info.title_offset;
    fake_event.width = width;
    fake_event.height = Plot_info.title_offset;
    fake_event.y = 0;
    Plot_info.font = tmp;
    Redisplay(cw, &fake_event);
}

static void OptionalLabel(w, event)
Widget w;
XButtonEvent *event;
{
    extern Widget LabelPrompt;

    if (!(LabelPrompt && XtIsManaged(LabelPrompt))) return;
}

void LocateOptionalLabel(w, event)
Widget w;
XButtonEvent *event;
{
    XGCValues valuesold, valuesnew;
    XtGCMask mask = GCLineStyle|GCForeground;
    XCharStruct overall;

    int width=0,ascent,descent,direction;
    unsigned long size;
    float xcm,ycm;
    PlotWidget cw = (PlotWidget) w;
    int newx = (cw->plot.cur_x + event->x) / 
             cw->plot.cell_width_in_pixels;
    int newy = (cw->plot.cur_y + event->y) / 
             cw->plot.cell_height_in_pixels;
    XExposeEvent fake_event;


    /* this fn should call other fns if allowing draw or undraw */
    int mode=DRAW;

    if (strlen(Plot_info.curr_label)==0) return;
    width = cw->plot.pixmap_width_in_pixels;
    descent = ascent = Plot_info.title_offset;
    if (Plot_info.font!=NULL) {
        XSetFont(XtDisplay(cw),cw->plot.draw_gc,Plot_info.font->fid);
        XTextExtents(Plot_info.font,
                        Plot_info.curr_label,strlen(Plot_info.curr_label),
                        &direction,&ascent,&descent,&overall);
        width = overall.width;
    }
    XDrawImageString(XtDisplay(cw),cw->plot.big_picture,cw->plot.draw_gc,
                        event->x, event->y,
                        Plot_info.curr_label,strlen(Plot_info.curr_label));
    /* store the label info */
    XGetFontProperty(Plot_info.font,XA_POINT_SIZE,&size);
    store_label(w,Plot_info.curr_label,Plot_info.fontname,size,
                       event->x,event->y,width,ascent+descent,Settings.fg);

    cw->plot.cell[newx + newy * cw->plot.pixmap_width_in_cells].drawn = mode;
    info.mode = mode;
    info.newx = newx;
    info.newy = newy;

    fake_event.x = event->x - cw->plot.cur_x;
    fake_event.y = event->y - cw->plot.cur_y - ascent;
    fake_event.width = width;
    fake_event.height = ascent + descent;
    Redisplay(cw, &fake_event);
/*
    XtCallCallbacks(cw, XtNtoggleCallback, &info);
*/
}

/*
 * calculates x,y in cms
 */
pixels_to_cms(x,y,cm_x,cm_y,cnvrt_x,cnvrt_y,hgt)
int x,y,hgt;
float cnvrt_x,cnvrt_y,*cm_x,*cm_y;
{

    *cm_x = (float)x / cnvrt_x / Screen_scale_factor;
    *cm_y = (float)(hgt-y) / cnvrt_y / Screen_scale_factor;
}

/*
 * calculates x,y (cms) in pixels
 */
cms_to_pixels(x,y,pix_x,pix_y,cnvrt_x,cnvrt_y,hgt)
int hgt,*pix_x,*pix_y;
float x,y,cnvrt_x,cnvrt_y;
{

    *pix_x = (int)(x * cnvrt_x * Screen_scale_factor);
    *pix_y = hgt - (int)(y * cnvrt_y * Screen_scale_factor);
}

/*
 * Save label data -
 * font and ptsize
 * x,y as fraction of page width and height
 * colour as index
 */
int store_label( w,label,fontname,size,x,y,wdth,hgt,fg_colour )
Widget w;
char *label,*fontname;
int fg_colour;
int x, y, wdth, hgt;
unsigned size;
{
    int j;
    float xsave, ysave;
    PlotWidget cw = (PlotWidget) w;
    label_node *node,**p;

    pixels_to_cms(x,y,&xsave,&ysave, a00000_.psca_x,a00000_.psca_y,
                           cw->plot.pixmap_height_in_pixels);
/* UPD label coords in fraction of page dim
    if (Settings.orient==PORTRAIT) {
        xsave /= DEFAULTWIDTH/10.0;
        ysave /= DEFAULTHEIGHT/10.0;
    }
    else if (Settings.orient==LANDSCAPE) {
        xsave /= DEFAULTHEIGHT/10.0;
        ysave /= DEFAULTWIDTH/10.0;
    }
*/
    if  ((node = (label_node *)malloc(sizeof(label_node)))!=NULL) {
        strcpy(node->label.label_text,label);
        strcpy(node->label.fontname,fontname);
        node->label.font_pointsize = size/10;
        node->label.fg = fg_colour;
        node->label.x = xsave;
        node->label.y = ysave;
        node->label.box.x = (short)x;
        node->label.box.y = (short)(y - hgt);
        node->label.box.width = (unsigned short)(wdth);
        node->label.box.height = (unsigned short)(hgt);
        node->next = NULL;
    }
    else return(MALLOC_ERR);

    /* append to the plot's list */
    p = &(cw->plot.labels); 
    while (*p) p = &((*p)->next);
    *p = node;
    return(0);
}

CreateBigPixmap(w)
Widget w;
{
    PlotWidget cw = (PlotWidget) w;
    extern Drawable DrawArea;

    cw->plot.big_picture = XCreatePixmap(XtDisplay(cw),
            RootWindow(XtDisplay(cw), DefaultScreen(XtDisplay(cw))),
            cw->plot.pixmap_width_in_pixels + 2, 
            cw->plot.pixmap_height_in_pixels + 2,
            XDefaultDepth(XtDisplay(cw),DefaultScreen(XtDisplay(cw))));
    DrawArea = (Drawable)(cw->plot.big_picture);
}

InitBigPixmap(w)
Widget w;
{
    PlotWidget cw = (PlotWidget) w;

    XFillRectangle(XtDisplay(cw), cw->plot.big_picture,
        cw->plot.undraw_gc, 0, 0, 
        cw->plot.pixmap_width_in_pixels + 2,
        cw->plot.pixmap_height_in_pixels + 2);
}

#if XY
/* A Public function, not static */
char * PlotGetArray(w, width_in_cells, height_in_cells)
Widget w;
int *width_in_cells, *height_in_cells;
{
    PlotWidget cw = (PlotWidget) w;

    *width_in_cells = cw->plot.pixmap_width_in_cells;
    *height_in_cells = cw->plot.pixmap_height_in_cells;
    return (cw->plot.cell);
}
#endif

/* ARGSUSED */
static void Resize(w)
Widget w;
{
#if XY
#endif
    Dimension new_cell_width_in_pixels;
    Dimension new_cell_height_in_pixels;
    
    PlotWidget cw = (PlotWidget) w;
    /* resize does nothing unless new size is bigger than entire pixmap */
    if ((cw->core.width < cw->plot.pixmap_width_in_pixels) ||
            (cw->core.height < cw->plot.pixmap_height_in_pixels)) {
        /* 
         * Calculate the maximum cell size that will allow the
         * entire bitmap to be displayed.
         */
        new_cell_width_in_pixels = cw->core.width / cw->plot.pixmap_width_in_cells;
        new_cell_height_in_pixels = cw->core.height / cw->plot.pixmap_height_in_cells;
        /* if size change mandates a new pixmap, make one */
        ChangeCellSize(cw, new_cell_width_in_pixels,
                                    new_cell_width_in_pixels);
    }
    Redisplay(w,NULL);
}

void ChangeCellSize(w, new_cell_width, new_cell_height)
Widget w;
int new_cell_width;
int new_cell_height;
{
    PlotWidget cw = (PlotWidget) w;
    int x, y;
#if XY

    cw->plot.cell_width_in_pixels = new_cell_width;
    cw->plot.cell_height_in_pixels = new_cell_height;

    /* recalculate variables based on cell size */
    cw->plot.pixmap_width_in_pixels = 
             cw->plot.pixmap_width_in_cells * 
             cw->plot.cell_width_in_pixels;

    cw->plot.pixmap_height_in_pixels = 
             cw->plot.pixmap_height_in_cells * 
             cw->plot.cell_height_in_pixels;
    
    /* destroy old and create new pixmap of correct size */
    XFreePixmap(XtDisplay(cw), cw->plot.big_picture);
    CreateBigPixmap(cw);
    
    /* draw lines into new pixmap */
    DrawIntoBigPixmap(cw);
    
    /* draw current cell array into pixmap */
    for (x = 0; x < cw->plot.pixmap_width_in_cells; x++) {
        for (y = 0; y < cw->plot.pixmap_height_in_cells; y++) {
            if ((int)cw->plot.cell[x + (y * cw->plot.pixmap_width_in_cells)].drawn == DRAWN)
                DoCell(cw, x, y, cw->plot.draw_gc);
            else
                DoCell(cw, x, y, cw->plot.undraw_gc);
        }
    }
#endif
}

static void DoCell(w, x, y, gc)
Widget w;
int x, y;
GC gc;
{
    PlotWidget cw = (PlotWidget) w;
        /* otherwise, draw or undraw */
    XFillRectangle(XtDisplay(cw), cw->plot.big_picture, gc,
             cw->plot.cell_width_in_pixels * x + 2,
             cw->plot.cell_height_in_pixels * y + 2,
             (unsigned int)cw->plot.cell_height_in_pixels - 3,
             (unsigned int)cw->plot.cell_height_in_pixels - 3);

}

/*
void DrawLine( w,gc,x1,y1,x2,y2 )
Widget w;
GC gc;
int x1,y1,x2,y2;
{
    PlotWidget cw = (PlotWidget) w;

    XDrawLine(XtDisplay(cw), cw->plot.big_picture, gc,
                x1,y1,x2,y2);
}
*/

void DrawPreviewLine( w,x1,y1,x2,y2 )
Widget w;
int x1,y1,x2,y2;
{
    extern PlotWidget plotArea;
    extern GC Draw_gc;

    XDrawLine(XtDisplay(w), XtWindow(plotArea), Draw_gc,
                x1,y1,x2,y2);
}

/*
void DrawFilledPolygon( w,gc,points,numpts)
Widget w;
GC gc;
int numpts;
XPoint *points;
{
    int shape,mode;
    extern GC Draw_gc;
    PlotWidget cw = (PlotWidget) w;

    shape = Complex;
    mode = CoordModeOrigin;

    XFillPolygon(XtDisplay(cw),cw->plot.big_picture,Draw_gc,points,
                                    numpts,shape,mode);
}
*/

void DrawRectangle(w,xp1,yp1,xp2,yp2)
Widget w;
int xp1,yp1,xp2,yp2;
{
    extern GC Draw_gc;
    PlotWidget cw = (PlotWidget) w;

    XDrawRectangle(XtDisplay(cw),cw->plot.big_picture,Draw_gc,
                                    xp1, yp1, xp2-xp1, yp2-yp1);
}
 
static XtGeometryResult QueryGeometry(w, proposed, answer)
Widget w;
XtWidgetGeometry *proposed, *answer;
{
    PlotWidget cw = (PlotWidget) w;
    int width, height;

        /***FIX these assignments if not keeping showall=True ***/
    width = cw->plot.pixmap_width_in_pixels;
    height = cw->plot.pixmap_height_in_pixels;

    /* set fields we care about */
    answer->request_mode = CWWidth | CWHeight;

    /* initial width and height */
    if (cw->plot.show_all == True)
        answer->width = cw->plot.pixmap_width_in_pixels;
    else
        answer->width = (cw->plot.pixmap_width_in_pixels > 
                width) ? width : 
                cw->plot.pixmap_width_in_pixels;

    if (cw->plot.show_all == True)
        answer->height = cw->plot.pixmap_height_in_pixels;
    else
        answer->height = (cw->plot.pixmap_height_in_pixels > 
                height) ? height : 
                cw->plot.pixmap_height_in_pixels;

    if (  ((proposed->request_mode & (CWWidth | CWHeight))
            == (CWWidth | CWHeight)) &&
            proposed->width == answer->width &&
            proposed->height == answer->height)
        return XtGeometryYes;
    else if (answer->width == cw->core.width &&
            answer->height == cw->core.height)
        return XtGeometryNo;
    else
        return XtGeometryAlmost;
}

int Save_plot_data( cntr_vals,prfl_vals )
float *cntr_vals, *prfl_vals;
{
    char *fname;
    save_node *node,**p;
    extern PlotWidget plotArea;
    PlotWidget cw = plotArea;

    if  ((node = (save_node *)malloc(sizeof(save_node)))!=NULL) {
        strcpy(node->data.filename,Plot_info.inp_file->fname);
        strcpy(node->data.plot_param,Plot_info.variable);
        strcpy(node->data.dflt_label1,Plot_info.dflt_label1);
        strcpy(node->data.dflt_label2,Plot_info.dflt_label2);
        node->data.cell_num = Plot_info.curr_cell;
        node->data.record_num = Plot_info.inp_file->rec_curr;
        node->data.reference_num = Plot_info.inp_file->ref_curr;
        node->data.plot_type = Plot_info.plot_type;
        node->data.plot_description = Plot_info.plot_description;
        node->data.option_num = Plot_info.var_num;
        /*node->data.colour_bar = Settings.plot_opts.colour_bar;*/
        node->changed_opts = NULL;
        node->next = NULL;
    }
    else return(MALLOC_ERR);

    /* append to list of plots */
    p = &(cw->plot.plots); 
    while (*p) p = &((*p)->next);
/* if inserting, use extra condition in while then
    node->next = *p;
*/
    *p = node;
    return(0);
}

int Clear_all_saved_data( cw )
PlotWidget cw;
{
    save_node *s_node, *s_tmp;
    opt_node *o_node, *o_tmp;

    s_node = cw->plot.plots;
    while (s_node) {
        if (o_node=s_node->changed_opts) {
            while (o_node) {
                o_tmp = o_node->next;
                free(o_node);
                o_node = o_tmp;
            }
        }
        s_tmp = s_node->next;
        free(s_node);
        s_node = s_tmp;
    }
    cw->plot.plots = NULL;
}

int Clear_saved_data( cw,cell_num )
int cell_num;
PlotWidget cw;
{
    XtPointer val;
    save_node **s_addr, *s_tmp;
    opt_node *o_node, *o_tmp, **o_addr;

    s_addr = &(cw->plot.plots);
    while (*s_addr) {
        if ((*s_addr)->data.cell_num==cell_num) {
            if (o_node=(*s_addr)->changed_opts) {
                if (s_addr!=&(cw->plot.plots)) { /* append to previous */
                    while (*o_addr) o_addr = &((*o_addr)->next);
                    *o_addr = o_node;
                }
                else do { /* first plot - change initial settings */
                    switch(o_node->opt.type) {
                    case SYB_INT  : val=(XtPointer)&(o_node->opt.int_val);
                                    break;
                    case SYB_FLOAT: val=(XtPointer)&(o_node->opt.float_val);
                                    break;
                    case SYB_BOOL : val=(XtPointer)&(o_node->opt.bool_val);
                                    break;
                    }
                    o_tmp = o_node;
                    o_node = o_node->next;
                    free(o_tmp);
                } while (o_node!=NULL);
            }
            s_tmp = *s_addr;
            *s_addr = (*s_addr)->next;
            free(s_tmp);
        }
        else {
            o_addr = &((*s_addr)->changed_opts);
            s_addr = &((*s_addr)->next);
        }
    }
}
