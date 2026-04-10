/*----------------------------------------------------------------
 *    Elle:   settings.c  1.0  2 October 2000
 *
 *    Copyright (c) 2000 by L.A. Evans
 *----------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "types.h"
#include "globals.h"
#include "error.h"
#include "string_utils.h"
#include "runopts.h"
#include "csettings.h"
#include "file.h"
#include "options.h"

int ValidColmapOption(char *str);
int ReadMinMax(char *str, double *min, double *max);

extern int ElleFindNodeAttributeRange(int id,double *min,double *max);
extern int ElleFindFlynnAttributeRange(int id,double *min,double *max);
extern void ElleFindBndAttributeRange(int id,double *min,double *max);
extern int ElleFindUnodeAttributeRange(int id,double *min,double *max);
extern int ElleFindSegmentAttributeRange(int id,double *min,double *max);

void InitSettings()
{
    InitOptions(&Settings);
}

/*
 * initialise an input_options structure
 */
void InitOptions(input_options *opts)
{
    opts->orient = PORTRAIT;
    opts->rescale = 1;
    opts->verbose = 0;
    opts->paper_size = A4_PAPER;
    opts->rows = 1;
    opts->columns = 1;
    opts->dfltfonthgt = DEFAULT_FONTHGT;
    opts->fg = FOREGROUND;
    opts->linestyle = SOLID;
    opts->page_scale = 1.0;
    opts->zoom = 1.0;
    opts->linewidth = 1.0;
    opts->xmargin = XLMARG;
    opts->ymargin = YMARG;
    opts->page_xmargin = XLMARG;
    opts->page_ymargin = YMARG;
    opts->colourmap = STD_MAP;
    opts->max_colours = MAX_COLOURS;
    opts->text_bg = 0;

    opts->plot_opts.update_freq = 1;
    opts->plot_opts.overlay = 0;
    opts->plot_opts.region=GRAINS;
    opts->plot_opts.draw_nodes=1;
    opts->plot_opts.node_attrib=NUM_NB;
    opts->plot_opts.node_limits[0]=0;
    opts->plot_opts.node_limits[1]=1;
    opts->plot_opts.boundary_attrib=STANDARD;
    opts->plot_opts.bnd_limits[0]=0;
    opts->plot_opts.bnd_limits[1]=0;
    opts->plot_opts.boundary_level=1;
    opts->plot_opts.flynn_attrib=0;
    opts->plot_opts.flynn_limits[0]=0;
    opts->plot_opts.flynn_limits[1]=0;
    opts->plot_opts.unode_attrib=0;
    opts->plot_opts.unode_limits[0]=0;
    opts->plot_opts.unode_limits[1]=0;
    opts->plot_opts.label_region=1;
}

/*
 * set a field in an input_options structure
 */
void SetOptionValue(struct plot_params *opts,int id,XtPointer val)
{
    double amin=0, amax=0;
    switch(id) {
    case O_REGION   : opts->region = *((int*)val);
                      break;
    case O_OVERLAY   : opts->overlay = *((int*)val);
                      break;
    case O_DRWUNODETYP: opts->draw_nodes = *((int*)val);
                      break;
    case O_NODEATTRIB: opts->node_attrib = *((int*)val);
                       ElleFindNodeAttributeRange(opts->node_attrib,
                                                     &amin, &amax);
                       opts->node_limits[0]=amin;
                       opts->node_limits[1]=amax;
                      break;
    case O_NODEATTRMIN : opts->node_limits[0] = *((double*)val);
                      break;
    case O_NODEATTRMAX : opts->node_limits[1] = *((double*)val);
                      break;
    case O_UNODEATTRIB: opts->unode_attrib = *((int*)val);
                        ElleFindUnodeAttributeRange(opts->unode_attrib,
                                                     &amin, &amax);
                        opts->unode_limits[0]=amin;
                        opts->unode_limits[1]=amax;
                      break;
    case O_UNODEATTRMIN : opts->unode_limits[0] = *((double*)val);
                      break;
    case O_UNODEATTRMAX : opts->unode_limits[1] = *((double*)val);
                      break;
    case O_LBLREGION : opts->label_region = *((int*)val);
                      break;
    case O_FLYNNATTRIB : opts->flynn_attrib = *((int*)val);
                         ElleFindFlynnAttributeRange(opts->flynn_attrib,
                                                     &amin, &amax);
                         opts->flynn_limits[0]=amin;
                         opts->flynn_limits[1]=amax;
                      break;
    case O_FLYNNATTRMIN : opts->flynn_limits[0] = *((double*)val);
                      break;
    case O_FLYNNATTRMAX : opts->flynn_limits[1] = *((double*)val);
                      break;
    case O_BNDATTRIB : opts->boundary_attrib = *((int*)val);
                       if (opts->boundary_attrib==MISORIENTATION)
                           ElleFindBndAttributeRange(opts->boundary_attrib,
                                                     &amin, &amax);
                       opts->bnd_limits[0]=amin;
                       opts->bnd_limits[1]=amax;
                      break;
    case O_BNDATTRMIN : opts->bnd_limits[0] = *((double*)val);
                      break;
    case O_BNDATTRMAX : opts->bnd_limits[1] = *((double*)val);
                      break;
    case O_BNDLEVEL  : opts->boundary_level = *((int*)val);
                      break;
    case O_UPDFREQ  : opts->update_freq = *((int*)val);
                      break;
    default:          break;
    }
}

/*
 * get a field value in an input_options structure
 */
void GetOptionValue(struct plot_params *opts,int id,XtPointer val)
{
    switch(id) {
    case O_REGION  : *((int*)val) = opts->region;
                     break;
    case O_OVERLAY  : *((int*)val) = opts->overlay;
                     break;
    case O_DRWUNODETYP: *((int*)val) = opts->draw_nodes;
                     break;
    case O_NODEATTRIB: *((int*)val) = opts->node_attrib;
                     break;
    case O_NODEATTRMIN: *((double*)val) = opts->node_limits[0];
                     break;
    case O_NODEATTRMAX: *((double*)val) = opts->node_limits[1];
                     break;
    case O_UNODEATTRIB: *((int*)val) = opts->unode_attrib;
                     break;
    case O_UNODEATTRMIN: *((double*)val) = opts->unode_limits[0];
                     break;
    case O_UNODEATTRMAX: *((double*)val) = opts->unode_limits[1];
                     break;
    case O_LBLREGION: *((int*)val) = opts->label_region;
                     break;
    case O_BNDATTRIB : *((int*)val) = opts->boundary_attrib;
                      break;
    case O_BNDATTRMIN: *((double*)val) = opts->bnd_limits[0];
                     break;
    case O_BNDATTRMAX: *((double*)val) = opts->bnd_limits[1];
                     break;
    case O_BNDLEVEL  : *((int*)val) = opts->boundary_level;
                      break;
    case O_FLYNNATTRIB: *((int*)val) = opts->flynn_attrib;
                     break;
    case O_FLYNNATTRMIN: *((double*)val) = opts->flynn_limits[0];
                     break;
    case O_FLYNNATTRMAX: *((double*)val) = opts->flynn_limits[1];
                     break;
    case O_UPDFREQ  : *((int*)val) = opts->update_freq;
                     break;
    default:         break;
    }
}

void GetCForegroundOption(int *val)
{
    *val=Settings.fg;
}

void GetColourmapOption(int *val)
{
    *val=Settings.colourmap;
}

void GetCRegionOption(int *val)
{
    GetOptionValue(&Settings.plot_opts,O_REGION,(XtPointer)val);
}

void GetCUpdFreqOption(int *val)
{
    GetOptionValue(&Settings.plot_opts,O_UPDFREQ,(XtPointer)val);
}

/*
 * combine fill, bnd, overlay etc into bits of one int?
 */
void GetCDrawingOption(int *val)
{
    int fill, outline;

    *val = -1;
    GetCFlynnAttribOption(&fill);
    GetCBndAttribOption(&outline);
    if (outline && !fill) *val = 2;
    else if (!outline && fill) *val = 0;
    else if (outline && fill) *val = 1;
}

void GetCOverlayOption(int *val)
{
    GetOptionValue(&Settings.plot_opts,O_OVERLAY,(XtPointer)val);
}

void GetCUnodePlotOption(int *val)
{
    GetOptionValue(&Settings.plot_opts,O_DRWUNODETYP,(XtPointer)val);
}

void GetCSegmentPlotOption(int *val)
{
}

void GetCNodeAttribOption(int *val)
{
    GetOptionValue(&Settings.plot_opts,O_NODEATTRIB,(XtPointer)val);
}

void GetCNodeAttribMinOption(double *val)
{
    GetOptionValue(&Settings.plot_opts,O_NODEATTRMIN,(XtPointer)val);
}

void GetCNodeAttribMaxOption(double *val)
{
    GetOptionValue(&Settings.plot_opts,O_NODEATTRMAX,(XtPointer)val);
}

void GetCUnodeAttribOption(int *val)
{
    GetOptionValue(&Settings.plot_opts,O_UNODEATTRIB,(XtPointer)val);
}

void GetCSegmentAttribOption(int *val)
{
}

void GetCUnodeAttribMinOption(double *val)
{
    GetOptionValue(&Settings.plot_opts,O_UNODEATTRMIN,(XtPointer)val);
}

void GetCUnodeAttribMaxOption(double *val)
{
    GetOptionValue(&Settings.plot_opts,O_UNODEATTRMAX,(XtPointer)val);
}

void GetCSegmentAttribMinOption(double *val)
{
}

void GetCSegmentAttribMaxOption(double *val)
{
}

void GetCLblRgnOption(int *val)
{
    GetOptionValue(&Settings.plot_opts,O_LBLREGION,(XtPointer)val);
}

void GetCBndAttribOption(int *val)
{
    GetOptionValue(&Settings.plot_opts,O_BNDATTRIB,(XtPointer)val);
}

void GetCBndLevelOption(int *val)
{
    GetOptionValue(&Settings.plot_opts,O_BNDLEVEL,(XtPointer)val);
}

void SetCBndLevelOption(int val)
{
    SetOptionValue(&Settings.plot_opts,O_BNDLEVEL,(XtPointer)(&val));
}

void GetCBndAttribMinOption(double *val)
{
    GetOptionValue(&Settings.plot_opts,O_BNDATTRMIN,(XtPointer)val);
}

void GetCBndAttribMaxOption(double *val)
{
    GetOptionValue(&Settings.plot_opts,O_BNDATTRMAX,(XtPointer)val);
}

void GetCFlynnAttribOption(int *val)
{
    GetOptionValue(&Settings.plot_opts,O_FLYNNATTRIB,(XtPointer)val);
}

void GetCFlynnAttribMinOption(double *val)
{
    GetOptionValue(&Settings.plot_opts,O_FLYNNATTRMIN,(XtPointer)val);
}

void GetCFlynnAttribMaxOption(double *val)
{
    GetOptionValue(&Settings.plot_opts,O_FLYNNATTRMAX,(XtPointer)val);
}

void SetColourmapOption(int val)
{
    Settings.colourmap=val;
}

void SetMaxColoursOption(int val)
{
    Settings.max_colours=val;
}

void SetCFlynnAttribOption(int id, double min, double max)
{
    SetOptionValue(&Settings.plot_opts,O_FLYNNATTRIB,(XtPointer)(&id));
    SetOptionValue(&Settings.plot_opts,O_FLYNNATTRMIN,(XtPointer)(&min));
    SetOptionValue(&Settings.plot_opts,O_FLYNNATTRMAX,(XtPointer)(&max));
}

void SetCNodeAttribOption(int id, double min, double max)
{
    SetOptionValue(&Settings.plot_opts,O_NODEATTRIB,(XtPointer)(&id));
    SetOptionValue(&Settings.plot_opts,O_NODEATTRMIN,(XtPointer)(&min));
    SetOptionValue(&Settings.plot_opts,O_NODEATTRMAX,(XtPointer)(&max));
}

void SetCUnodeAttribOption(int id, double min, double max)
{
    SetOptionValue(&Settings.plot_opts,O_UNODEATTRIB,(XtPointer)(&id));
    SetOptionValue(&Settings.plot_opts,O_UNODEATTRMIN,(XtPointer)(&min));
    SetOptionValue(&Settings.plot_opts,O_UNODEATTRMAX,(XtPointer)(&max));
}

void SetCSegmentAttribOption(int id, double min, double max)
{
}

void SetCBndAttribOption(int id, double min, double max)
{
    SetOptionValue(&Settings.plot_opts,O_BNDATTRIB,(XtPointer)(&id));
    SetOptionValue(&Settings.plot_opts,O_BNDATTRMIN,(XtPointer)(&min));
    SetOptionValue(&Settings.plot_opts,O_BNDATTRMAX,(XtPointer)(&max));
}

int ValidColmapOption(char *str)
{
    return(name_match(str,colmap_terms));
}

int ReadSettings(FILE *fp, char *str, unsigned char initial)
{
    unsigned char finished=0, bool_val=0;
    int indx,len,i,dum,err=0,opt_type=0;
    float tmp;
    double max=0, min=0;
    input_options *opts = &Settings;
    struct plot_params *plot = &Settings.plot_opts;

    while (!finished) {
      fscanf(fp,"%[^A-Za-z#]",str);
      if ((str[0] = getc(fp))=='#') dump_comments(fp);
      else {
        fscanf(fp,"%[^= \n\t]",&str[1]);
        /*
         * if an option is not recognized, read over it and continue
         */
       indx = name_match(str,display_terms);
       if (indx==-1) indx = name_match(str,option_terms);
        if (indx==-1) {
            if (!feof(fp)) {
                fprintf(stderr,"Keyword not understood %s\n",str);
                err=KEY_ERR;
            }
            finished=1;
        }
        else {
          len = strlen(str);
          if (fscanf(fp,"%1s",&str[len])!=1) return(STR_ERR);
          if (str[len] != '=') return(STR_ERR);
          opt_type = 0;
          switch( indx ) {
            case O_ORIENT: fscanf(fp,"%s",str);
                    if (!strcmp(str,"PORTRAIT")) opts->orient=PORTRAIT;
                    else if (!strcmp(str,"LANDSCAPE"))
                                       opts->orient=LANDSCAPE;
                    else return(RANGE_ERR);
                    break;
            case O_PAPERSZ: fscanf(fp,"%s",str);
                    if ((i = name_match(str,paper_terms))==-1)
                         return(RANGE_ERR);
                    opts->paper_size = i;
                    break;
            case O_ROWS: if (fscanf(fp,"%d",&i)!=1) return(NUM_ERR);
                    if (i<MINHEIGHT || i>MAXHEIGHT ) return(RANGE_ERR);
                   opts->rows = i;
                    Plot_info.max_row = i-1;
                    break;
            case O_COLS: if (fscanf(fp,"%d",&i)!=1) return(NUM_ERR);
                    if (i<MINWIDTH || i>MAXWIDTH ) return(RANGE_ERR);
                    opts->columns = i;
                    Plot_info.max_col = i-1;
                    break;
            case O_FONTHGT: if (fscanf(fp,"%d",&i)!=1) return(NUM_ERR);
                   if (i<0) return(RANGE_ERR);
                    opts->dfltfonthgt = i;
                    opt_type = SYB_INT;
                    break;
            case O_PG_XMARG: if (fscanf(fp,"%f",&tmp)!=1) return(NUM_ERR);
                    opts->page_xmargin = tmp;
                    break;
            case O_PG_YMARG: if (fscanf(fp,"%f",&tmp)!=1) return(NUM_ERR);
                    opts->page_ymargin = tmp;
                    break;
            case O_ZOOM: if (fscanf(fp,"%f",&tmp)!=1) return(NUM_ERR);
                    opts->zoom = tmp;
                    opt_type = SYB_FLOAT;
                    break;
            case O_LINEWDTH: if (fscanf(fp,"%f",&tmp)!=1) return(NUM_ERR);
                    opts->linewidth = tmp;
                    if (!initial) setlinewidth_(&tmp);
                    opt_type = SYB_FLOAT;
                    break;
            case O_PGSCL: if (fscanf(fp,"%f",&tmp)!=1) return(NUM_ERR);
                   opts->page_scale = tmp;
                    break;
            /*case O_COLBAR:if (fscanf(fp,"%s",str)!=1) return(NUM_ERR);
                    if ((i = name_match(str,colourbar_terms))==-1)
                         return(RANGE_ERR);
                    opts->colour_bar = i;
                    opt_type = SYB_INT;
                    break;*/
            case O_FOREGRND:if (fscanf(fp,"%d",&i)!=1) return(NUM_ERR);
                   opts->fg = i;
                    if (!initial) setpencolor_(&i);
                    opt_type = SYB_INT;
                    break;
            case O_LINESTYLE:if (fscanf(fp,"%s",str)!=1) return(NUM_ERR);
                    if ((i = name_match(str,linestyle_terms))==-1)
                         return(RANGE_ERR);
                    opts->linestyle = i;
                    /* X window may not exist yet */
                    dum=1;
                    if (!initial) dashln_(&opts->linestyle,&dum);
                    opt_type = SYB_INT;
                    break;
            case O_COLMAP:if (fscanf(fp,"%s",str)!=1) return(NUM_ERR);
                    if ((i = name_match(str,colmap_terms))==-1)
                         return(RANGE_ERR);
                    opts->colourmap = i;
                    opt_type = SYB_INT;
                    break;
            case O_RESCALE: if (fscanf(fp,"%d",&i)!=1) return(NUM_ERR);
                   opts->rescale = i;
                    opt_type = SYB_INT;
                    break;
            case O_VERBOSE: if (fscanf(fp,"%d",&i)!=1) return(NUM_ERR);
                   opts->verbose = i;
                    opt_type = SYB_BOOL;
                    bool_val = (i==1);
                    break;
            case O_TEXTBG: if (fscanf(fp,"%d",&i)!=1) return(NUM_ERR);
                    opts->text_bg = i;
                    opt_type = SYB_BOOL;
                    bool_val = (i==1);
                    break;
/* plot options - set plot_params struct */
            case O_UPDFREQ:
            case O_OVERLAY:
            case O_DRWUNODETYP:
            case O_LBLREGION:
            case O_BNDLEVEL:  if (fscanf(fp,"%d",&i)!=1) return(NUM_ERR);
                    SetOptionValue(plot,indx,&i);
                    break;
            case O_FLYNNATTRIB:if (fscanf(fp,"%s",str)!=1) return(NUM_ERR);
                    if ((i = name_match(str,region_terms))==-1)
                         return(RANGE_ERR);
                    SetOptionValue(plot,indx,&i);
                    if (i!=NONE) {
                        fscanf(fp,"%[^\n]",str);
                        if ((err=ReadMinMax(str,&min,&max)) ==0) {
                            SetOptionValue(plot,
                                    O_FLYNNATTRMIN,(XtPointer)(&min));
                            SetOptionValue(plot,
                                    O_FLYNNATTRMAX,(XtPointer)(&max));
                        }
                        else {
                            dump_comments(fp);
                            err=0;
                        }
                    }
                    break;
            case O_NODEATTRIB:
                    if (fscanf(fp,"%s",str)!=1)
                        fprintf(stderr,"ignoring node attribute setting\n");
                    else {
                      if ((i = name_match(str,node_terms))==-1)
                           return(RANGE_ERR);
                      SetOptionValue(plot,indx,&i);
                      if (i!=NONE && i!=NUM_NB && i!=TRPL_ONLY) {
                          fscanf(fp,"%[^\n]",str);
                          if ((err=ReadMinMax(str,&min,&max)) ==0) {
                              SetOptionValue(plot,
                                    O_NODEATTRMIN,(XtPointer)(&min));
                              SetOptionValue(plot,
                                    O_NODEATTRMAX,(XtPointer)(&max));
                          }
                          else {
                              dump_comments(fp);
                              err=0;
                          }
                      }
                    }
                    break;
            case O_UNODEATTRIB:
                    if (fscanf(fp,"%s",str)!=1)
                        fprintf(stderr,"ignoring unode attribute setting\n");
                    else {
                      if ((i = name_match(str,unode_terms))==-1)
                           return(RANGE_ERR);
                      SetOptionValue(plot,indx,&i);
                      if (i!=NONE && i!=U_LOCATION) {
                          fscanf(fp,"%[^\n]",str);
                          if ((err=ReadMinMax(str,&min,&max)) ==0) {
                              SetOptionValue(plot,
                                    O_UNODEATTRMIN,(XtPointer)(&min));
                              SetOptionValue(plot,
                                    O_UNODEATTRMAX,(XtPointer)(&max));
                          }
                          else {
                              dump_comments(fp);
                              err=0;
                          }
                      }
                    }
                    break;
            case O_BNDATTRIB:
                    if (fscanf(fp,"%s",str)!=1)
                        fprintf(stderr,"ignoring boundary attribute setting");
                    else {
                      if ((i = name_match(str,boundary_terms))==-1)
                         return(RANGE_ERR);
                      SetOptionValue(plot,indx,&i);
                      if (i!=NONE && i!=STANDARD) {
                          fscanf(fp,"%[^\n]",str);
                          if ((err=ReadMinMax(str,&min,&max)) ==0) {
                              SetOptionValue(plot,
                                    O_BNDATTRMIN,(XtPointer)(&min));
                              SetOptionValue(plot,
                                    O_BNDATTRMAX,(XtPointer)(&max));
                          }
                          else {
                              dump_comments(fp);
                              err=0;
                          }
                      }
                    }
                    break;
            default:warning_msg(VAR_ERR,str);
                    fscanf(fp,"%[^A-Za-z0-9]",str);
                    fscanf(fp,"%[^ \n\t]",str);
                    break;
          }
        }
      }
    }
    return( err );
}

#if XY
/*
 * if input file is divided by keywords into SETTING and PLOT sections
 */
int ReadOptions(FILE *fp, char *str)
{
    unsigned char finished=0, bool_val=0;
    int indx,len,i,dum,err=0,opt_type=0;
    double max=0, min=0;
    struct plot_params *opts = &Settings.plot_opts;

    while (!finished) {
      fscanf(fp,"%[^A-Za-z#]",str);
      if ((str[0] = getc(fp))=='#') dump_comments(fp);
      else {
        /*
         * options terms are all 2 words
         */
        len=fscanf(fp,"%[^= \n\t]",&str[1]);
        fscanf(fp,"%c",&str[len]);
        fscanf(fp,"%[^= \n\t]",&str[len+1]);
        /*
         * if an option is not recognized, read over it and continue
         * on next line
         */
       indx = name_match(str,option_terms);
        if (indx==-1) {
            if (!feof(fp)) {
                fprintf(stderr,"Keyword not understood %s\n",str);
                err=KEY_ERR;
                dump_comments(fp);
            }
            else finished=1;
        }
        else {
          len = strlen(str);
          if (fscanf(fp,"%1s",&str[len])!=1) return(STR_ERR);
          if (str[len] != '=') return(STR_ERR);
          opt_type = 0;
          switch( indx ) {
            case O_UPDFREQ:
            case O_DRWUNODETYP:
            case O_LBLREGION: fscanf(fp,"%d",&i);
                    SetOptionValue(opts,indx,&i);
                    break;
            case O_BNDLEVEL: fscanf(fp,"%d",&i);
                    SetOptionValue(opts,indx,&i);
                    break;
            case O_FLYNNATTRIB:if (fscanf(fp,"%s",str)!=1) return(NUM_ERR);
                    if ((i = name_match(str,region_terms))==-1)
                         return(RANGE_ERR);
                    SetOptionValue(opts,indx,&i);
                    if (i!=NONE) {
                        if ((err=ReadMinMax(fp,str,&min,&max)) ==0) {
                            SetOptionValue(&Settings.plot_opts,
                                    O_FLYNNATTRMIN,(XtPointer)(&min));
                            SetOptionValue(&Settings.plot_opts,
                                    O_FLYNNATTRMAX,(XtPointer)(&max));
                        }
                        else dump_comments(fp);
                    }
                    break;
            case O_NODEATTRIB:if (fscanf(fp,"%s",str)!=1) return(NUM_ERR);
                    if ((i = name_match(str,node_terms))==-1)
                         return(RANGE_ERR);
                    SetOptionValue(opts,indx,&i);
                    if (i!=NONE) {
                        if ((err=ReadMinMax(fp,str,&min,&max)) ==0) {
                            SetOptionValue(&Settings.plot_opts,
                                    O_NODEATTRMIN,(XtPointer)(&min));
                            SetOptionValue(&Settings.plot_opts,
                                    O_NODEATTRMAX,(XtPointer)(&max));
                        }
                        else dump_comments(fp);
                    }
                    break;
            case O_BNDATTRIB:if (fscanf(fp,"%s",str)!=1) return(NUM_ERR);
                    if ((i = name_match(str,boundary_terms))==-1)
                         return(RANGE_ERR);
                    SetOptionValue(opts,indx,&i);
                    if (i!=NONE && i!=STANDARD) {
                        if ((err=ReadMinMax(fp,str,&min,&max)) ==0) {
                            SetOptionValue(&Settings.plot_opts,
                                    O_BNDATTRMIN,(XtPointer)(&min));
                            SetOptionValue(&Settings.plot_opts,
                                    O_BNDATTRMAX,(XtPointer)(&max));
                        }
                        else dump_comments(fp);
                    }
                    break;
            default:warning_msg(VAR_ERR,str);
                    fscanf(fp,"%[^A-Za-z0-9]",str);
                    fscanf(fp,"%[^ \n\t]",str);
                    break;
          }
        }
      }
    }
    return( err );
}
#endif

/*
 * expects Min= or min= or Max= or max=
 */
int ReadMinMax(char *str, double *min, double *max)
{
    char buf[MAX_LINE_LEN], *start, *ptr;
    int len=0, err=0;
    double val=0.0;

    buf[0] = '\0';
    
    start = strtok(str," =,\t");
    if (start && (!strcmp(start,"Min") || !strcmp(start,"min"))) {
        ptr = strtok(0," =,\t");
        if (ptr) *min = atof(ptr);
        ptr = strtok(0," =,\t");
        if (ptr && (!strcmp(ptr,"Max") || !strcmp(ptr,"max"))) {
            ptr = strtok(0," =,\t");
            if (ptr) *max = atof(ptr);
        }
        else err = STR_ERR;
    }
    else if (start && (!strcmp(start,"Max") || !strcmp(start,"max"))) {
        ptr = strtok(0," =,\t");
        if (ptr) *max = atof(ptr);
        ptr = strtok(0," =,\t");
        if (ptr && (!strcmp(ptr,"Min") || !strcmp(ptr,"min"))) {
            ptr = strtok(0," =,\t");
            if (ptr) *min = atof(ptr);
        }
        else err = STR_ERR;
    }
    else err = STR_ERR;
    return(err);
}

int WriteSettingsFile(char *filename)
{
    int err=0;
    FILE *fp;

    if ((fp=fopen(filename,"w"))==NULL) err=OPEN_ERR;
    else {
        err=WriteSettings(fp);
        fclose(fp);
    }
    return(err);
}

int WriteSettings(FILE *fp)
{
    char format[20], buf[MAX_OPTION_NAME+1];
    int val=0, i=0, err=0;
    double fval=0.0;
    input_options *opts = &Settings;

    strcpy(format,"");
    strcpy(buf,"");
    sprintf(format,"%s%d%s= ","%-",MAX_OPTION_NAME,"s");
    while (display_terms[i].name!=0 && !err) {
        fprintf(fp,format,display_terms[i].name);
        switch( display_terms[i].id ) {
            case O_ORIENT:
                    if (opts->orient==PORTRAIT)
                        strcpy(buf,"PORTRAIT");
                    else strcpy(buf,"LANDSCAPE");
                    fprintf(fp,"%s",buf);
                    break;
            case O_PAPERSZ:
                    id_match(paper_terms,opts->paper_size,buf);
                    fprintf(fp,"%s",buf);
                    break;
            case O_FONTHGT: fprintf(fp,"%d",opts->dfltfonthgt);
                    break;
            case O_PG_XMARG: fprintf(fp,"%.3f",opts->page_xmargin);
                    break;
            case O_PG_YMARG: fprintf(fp,"%.3f",opts->page_ymargin);
                    break;
            case O_ZOOM: fprintf(fp,"%f",opts->zoom);
                    break;
            case O_LINEWDTH: fprintf(fp,"%.3f",opts->linewidth);
                    break;
            case O_PGSCL: fprintf(fp,"%.3f",opts->linewidth);
                    break;
            case O_FOREGRND: fprintf(fp,"%d",opts->fg);
                    break;
            case O_LINESTYLE:
                    id_match(linestyle_terms,opts->linestyle,buf);
                    fprintf(fp,"%s",buf);
                    break;
            case O_COLMAP:
                    id_match(colmap_terms,opts->colourmap,buf);
                    fprintf(fp,"%s",buf);
                    break;
            case O_RESCALE: fprintf(fp,"%d",opts->rescale);
                    break;
            case O_VERBOSE: fprintf(fp,"%d",opts->verbose);
                    break;
            case O_TEXTBG: fprintf(fp,"%d",opts->text_bg);
                    break;
        }
        if (fprintf(fp,"\n")<0) err=WRITE_ERR;
        i++;
    }
    i=0;
    while (option_terms[i].name!=0 && !err) {
        fprintf(fp,format,option_terms[i].name);
        switch(option_terms[i].id) {
        case O_UPDFREQ: GetCUpdFreqOption(&val);
                        fprintf(fp," %d",val);
                        break;
        case O_DRWUNODETYP: GetCUnodePlotOption(&val);
                        fprintf(fp," %d",val);
                        break;
        case O_LBLREGION: GetCLblRgnOption(&val);
                        fprintf(fp," %d",val);
                        break;
        case O_OVERLAY: GetCOverlayOption(&val);
                        fprintf(fp," %d",val);
                        break;
        case O_FLYNNATTRIB: GetCFlynnAttribOption(&val);
                        id_match(region_terms,val,buf);
                        fprintf(fp," %s",buf);
                        if (val!=NONE) {
                            GetCFlynnAttribMinOption(&fval);
                            fprintf(fp," min= %.6e",fval);
                            GetCFlynnAttribMaxOption(&fval);
                            fprintf(fp," max= %.6e",fval);
                        }
                        break;
        case O_BNDLEVEL: GetCBndLevelOption(&val);
                        fprintf(fp," %d",val);
                        break;
        case O_BNDATTRIB: GetCBndAttribOption(&val);
                        id_match(boundary_terms,val,buf);
                        fprintf(fp," %s",buf);
                        if (val!=NONE && val!=STANDARD) {
                            GetCBndAttribMinOption(&fval);
                            fprintf(fp," min= %.6e",fval);
                            GetCBndAttribMaxOption(&fval);
                            fprintf(fp," max= %.6e",fval);
                        }
                        break;
        case O_NODEATTRIB: GetCNodeAttribOption(&val);
                        id_match(node_terms,val,buf);
                        fprintf(fp," %s",buf);
                        if (val!=NONE && val!=NUM_NB && val!=TRPL_ONLY) {
                            GetCNodeAttribMinOption(&fval);
                            fprintf(fp," min= %.6e",fval);
                            GetCNodeAttribMaxOption(&fval);
                            fprintf(fp," max= %.6e",fval);
                        }
                        break;
        case O_UNODEATTRIB: GetCUnodeAttribOption(&val);
                        id_match(unode_terms,val,buf);
                        fprintf(fp," %s",buf);
                        if (val!=NONE && val!=U_LOCATION) {
                            GetCUnodeAttribMinOption(&fval);
                            fprintf(fp," min= %.6e",fval);
                            GetCUnodeAttribMaxOption(&fval);
                            fprintf(fp," max= %.6e",fval);
                        }
                        break;
        default:        break;
        }
        if (fprintf(fp,"\n")<0) err=WRITE_ERR;
        i++;
    }
    return(err);
}

/*
 * horrible fudge, using Plot_data.max_colours (set in menus.c)
 * fix when colour choice implemented properly
 */
int GetCPlotColourRange(int *min,int *max)
{
    *min = USEFL_SIZE;
    /* cut out pinks */
    *max = USEFL_SIZE+(Plot_info.max_colours-USEFL_SIZE)*0.92;
    return(0);
}
