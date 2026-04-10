/*--------------------------------------------------------------------
 *    Basil / Sybil:   page.c  1.1  1 October 1998
 *
 *    Copyright (c) 1997 by G.A. Houseman, T.D. Barr, & L.A. Evans
 *    See README file for copying and redistribution conditions.
 *--------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "types.h"
#include "globals.h"
#include "mainps.h"
#include "errnum.h"
#include "csettings.h"

#define CM_PER_INCH    2.54
#define PTS_PER_CM    72.0/CM_PER_INCH 
/* 0.25in margin around A4 page not used by printer */
#define POINT_OFFSET    72/4
#define CM_OFFSET       CM_PER_INCH/4
#define PS_COL_MIN      500
#define PS_COL_MAX     1000
/*
#define PS_COL_MIN      8
#define PS_COL_MAX     71
*/

char OptsFile[FILENAME_MAX+1],PsFile[FILENAME_MAX+1];
int File_no;
page_data Page;
void ChangeCell(),Syntax();
extern void set_box_dimensions();
extern void EllePlotRegions();
extern char *ElleDspOptsFile(), *ElleFile(), *ElleOutFile();

int Set_Colours(red,green,blue,irgbv,steps)
int *red,*green,*blue,*irgbv,steps;
{
    int ncolors, *limits, start;
/*
    Colour_range[0] = PS_COL_MIN;
    Colour_range[1] = PS_COL_MAX;
*/
    if ((limits = (int *)malloc(steps*sizeof(int)))==NULL) return(1);
    ncolors = PLOT_COLOURS;
#if XY
    Colour_range[0] = USEFL_SIZE;
    Colour_range[1] = MAX_COLOURS-1;
    Arrow_colours[0] = 4; /* blue */
    Arrow_colours[1] = 2; /* red */
#endif
    start = USEFL_SIZE;
    Plot_info.max_colours = MAX_COLOURS-1;
    createcolormap_(red,green,blue,irgbv,limits,&steps,&start,
                            &ncolors );
    free(limits);
    return(0);
}

Init_App(argc,argv,optsfp)
int argc;
char **argv;
FILE **optsfp;
{
    char input_str[SYB_FILENAME_MAX+1], *tmpstr, *dir;
    int i,j,indx,file_no,err=0;
    int hgt,len;
    int iro,init=1,ils=1;
    float pt_wdth, pt_hgt, width, height, factor, tmp;
    float x_offset=0.0, y_offset=0.0;
    FILE *fp;

    OptsFile[0] = PsFile[0] = '\0';
    tmpstr=ElleOutFile();
    if (strlen(tmpstr)==0) {
        tmpstr=ElleFile();
        strcpy(PsFile,tmpstr);
        if ((tmpstr=strstr(PsFile,".elle"))!=NULL)
            strcpy(tmpstr,DEFAULT_PS_EXT);
        else strcat(PsFile,DEFAULT_PS_EXT);
    }
    else strcpy(PsFile,tmpstr);

        /* open files */
    tmpstr=ElleDspOptsFile();
    if (strlen(tmpstr)!=0) {
        strncpy(OptsFile,tmpstr,FILENAME_MAX-1);
       fp=fopen(OptsFile,"r");
       if (fp==NULL) error_msg(OPEN_ERR,OptsFile);
    }
    else {
        strcpy(OptsFile,DEFAULT_OPTIONS_FILE);
        /* check in current dir for options file */
        if ((fp = fopen(OptsFile,"r"))==NULL) {
            /* check in user's home dir for options file */
            if ((dir = getenv("HOME"))!=NULL) {
                strcpy(OptsFile,dir);
                strcat(OptsFile,"/");
                strcat(OptsFile,DEFAULT_OPTIONS_FILE);
                fp = fopen(OptsFile,"r");
            }
        }
    }
    if (fp!=NULL) {
        if (err=ReadSettings(fp,input_str,1))
            error_msg(err,input_str);
        fclose(fp);;
    }
    if (err) return(err);
    i = strlen(PsFile);
    fileopen_(PsFile,&i,&File_no);
    if (!File_no) error_msg(OPEN_ERR,PsFile);

        /* initialise the postscript file */
    iro = (int)Settings.orient;
    hplots_(&init,&iro,&File_no,&ils);
        /* initialise the font */
    hgt = Settings.dfltfonthgt;
    len = strlen("Helvetica");
    setfont_("Helvetica",&len,&hgt);

        /* set up the page */
    if (Settings.paper_size==US_PAPER) {
        height = USLETTERHEIGHT;
        width = USLETTERWIDTH;
    }
    else {
        height = DEFAULTHEIGHT;
        width = DEFAULTWIDTH;
    }
    if (Settings.orient!=PORTRAIT) {
        tmp = height;
        height = width;
        width = tmp;
    }
    pt_wdth = width/10.0*PTS_PER_CM;
    pt_hgt = height/10.0*PTS_PER_CM;
    Page.width_in_pts = pt_wdth;
    Page.height_in_pts = pt_hgt;
    Page.width_in_cells = Settings.columns = 1;
    Page.height_in_cells = Settings.rows;
    x_offset = Page.width_in_pts * Settings.page_xmargin;
    y_offset = Page.height_in_pts * Settings.page_ymargin;
   /*
    * allow for main title at top of page
    */
    Plot_info.title_offset += y_offset;
    Page.cell_height_in_pts = (int)( (pt_hgt - y_offset)/
                                    (float)Page.height_in_cells);
    Page.cell_width_in_pts = (int)((pt_wdth - x_offset*2)/
                                    (float)Page.width_in_cells);
    /*
     * cell dimensions in cm
     */
    factor = PTS_PER_CM;
    Page.cell_height_in_cm = (float)Page.cell_height_in_pts/factor;
    Page.cell_width_in_cm = (float)Page.cell_width_in_pts/factor;
 
    if (Page.cell == NULL)
        if ((Page.cell = (cell_data *)calloc(
                Page.width_in_cells *
                Page.height_in_cells, sizeof(cell_data))) == NULL)
            return(MALLOC_ERR);

    for (i=0,indx=0;i<Page.height_in_cells;i++) {
        for (j=0;j<Page.width_in_cells;j++,indx++) {
            Page.cell[indx].drawn=0;
            Page.cell[indx].row=i;
            Page.cell[indx].col=j;
            Page.cell[indx].rect.x=j*Page.cell_width_in_pts + x_offset;
            Page.cell[indx].rect.y=i*Page.cell_height_in_pts +
                                       Plot_info.title_offset;
            /*Page.cell[indx].rect_cm_x=(float)Page.cell[indx].rect.x/factor;*/
            /*Page.cell[indx].rect_cm_y=(float)(Page.height_in_pts -*/
                                          /*Page.cell[indx].rect.y)/factor;*/
            Page.cell[indx].rect.width=Page.cell_width_in_pts-1;
            Page.cell[indx].rect.height=Page.cell_height_in_pts-1;
        }
    }
    indx = Plot_info.curr_cell;
    set_box_dimensions(Pwindo, 0.0, 0.0,
                       Page.cell_width_in_cm,Page.cell_height_in_cm);
    /*
     * set the clipping region to the unit cell (match X window?)
     */
    /*x_offset = y_offset = 0;*/
    width = Page.width_in_cells*Page.cell_width_in_cm;
    height = Page.height_in_cells*Page.cell_height_in_cm;
    clipregionu_(&x_offset, &y_offset, &width, &height);
    return(0); 
}

void Syntax(argc,argv)
int argc;
char **argv;
{
    fprintf(stderr, "options:\n" );
    fprintf(stderr, "   -i   name of log file\n" );
    fprintf(stderr, "   -o   name for Postscript output file\n" );
    fprintf(stderr, "   -h   Print this message\n" );
    fprintf(stderr, "Example:  sybilps -i logfile -o psfile\n" );
}

void warning_msg(err_num,str)
int err_num;
char *str;
{
    char message[SYB_FILENAME_MAX],buf[20];

    strncpy(message,str,sizeof(message)-1);
    append_error_message(message,err_num);
    fprintf(stderr,"%s\n",message);
}

void Exit_App(err)
int err;
{
    int init=0,iro=0,ils=1;

    if (!err) hplots_(&init,&iro,&File_no,&ils);
    if (Plot_info.inp_file!=0)
        if (Plot_info.inp_file->fp!=0) fclose(Plot_info.inp_file->fp);
    exit(0);
}

Run_App(logfp)
FILE *logfp;
{
    int err=0, dum=1;
    float width=0;

    setpencolor_(&Settings.fg);
    setlinewidth_(&Settings.linewidth);
    dashln_(&Settings.linestyle,&dum);
    EllePlotRegions(1);
    Exit_App(0);
}

int SetStage(int stage)
{
    return(0);
}
