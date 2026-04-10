/*----------------------------------------------------------------
 *    Elle:   settings.c  1.0  2 October 2000
 *
 *    Copyright (c) 2000 by L.A. Evans
 *
 * $Id$
 *----------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include "types.h"
//#include "globals.h"
#include "error.h"
#include "string_utils.h"
#include "runopts.h"
#include "settings.h"
#include "file.h"
//#include "options.h"
#include "display.h"
#include "general.h"


int ValidColmapOption(char *str);
int ReadMinMax(char *str, double *min, double *max);

void warning_msg(int,const char*);//is declared in page.c...
/*
 * initialise an input_options structure
 */

Settings::Settings(void)
{
}

void Settings::InitOptions()
{
    orient = PORTRAIT;
    paper_size = A4_PAPER;
    dfltfonthgt = DEFAULT_FONTHGT;
    fg = FOREGROUND;
    linestyle = SOLID;
    page_scale = 1.0;
    linewidth = 1.0;
    xmargin = XLMARG;
    ymargin = YMARG;
    page_xmargin = XLMARG;
    page_ymargin = YMARG;
    colourmap = STD_MAP;
    max_colours = MAX_COLOURS;
    update_freq = 1;
    overlay = 0;
    region=GRAINS;
    draw_nodes=0; //was soll das hier sein? hat was mit unodes zu tun...
    node_attrib=NUM_NB;
    node_limits[0]=0;
    node_limits[1]=1;
    boundary_attrib=STANDARD;
    bnd_limits[0]=0;
    bnd_limits[1]=0;
    boundary_level=1;
    flynn_attrib=NONE;
    flynn_limits[0]=0;
    flynn_limits[1]=0;
    unode_attrib=NONE;
    unode_limits[0]=0;
    unode_limits[1]=0;
    segment_attrib=NONE;
    segment_limits[0]=0;
    segment_limits[1]=0;
    label_region=1;
    logfile=false;
    logfileappend=false;
    rainbow=false;
}
void Settings::SetLogFile(bool c)
{
    logfile=c;
}
bool Settings::GetLogFile()
{ return logfile;}
void Settings::SetLogAppend(bool c)
{logfileappend=c;}
bool Settings::GetLogAppend()
{return logfileappend;}
void Settings::SetRainbow(bool s)
{
    rainbow=s;
}
bool Settings::GetRainbow()
{
    return rainbow;
}
/*
 * set a field in an input_options structure
 */
void Settings::SetOptionValue(int id,XtPointer val)
{
    double amin=0, amax=0;
    switch(id) {
    case O_REGION   : region = *((int*)val);
        break;
    case O_OVERLAY   : overlay = *((int*)val);
        break;
    case O_DRWUNODETYP: draw_nodes = *((int*)val);
        break;
    case O_NODEATTRIB: node_attrib = *((int*)val);
        ElleFindNodeAttributeRange(node_attrib,
                                   &amin, &amax);
        node_limits[0]=amin;
        node_limits[1]=amax;
        break;
    case O_NODEATTRMIN : node_limits[0] = *((double*)val);
        break;
    case O_NODEATTRMAX : node_limits[1] = *((double*)val);
        break;
    case O_UNODEATTRIB: unode_attrib = *((int*)val);
        ElleFindUnodeAttributeRange(unode_attrib,
                                    &amin, &amax);
        unode_limits[0]=amin;
        unode_limits[1]=amax;
        break;
    case O_UNODEATTRMIN : unode_limits[0] = *((double*)val);
        break;
    case O_UNODEATTRMAX : unode_limits[1] = *((double*)val);
        break;
    case O_SEGMENTATTRIB: segment_attrib = *((int*)val);
        ElleFindSegmentAttributeRange(segment_attrib,
                                    &amin, &amax);
        segment_limits[0]=amin;
        segment_limits[1]=amax;
        break;
    case O_SEGMENTATTRMIN : segment_limits[0] = *((double*)val);
        break;
    case O_SEGMENTATTRMAX : segment_limits[1] = *((double*)val);
        break;
    case O_LBLREGION : label_region = *((int*)val);
        break;
    case O_FLYNNATTRIB : flynn_attrib = *((int*)val);
        ElleFindFlynnAttributeRange(flynn_attrib,
                                    &amin, &amax);
        flynn_limits[0]=amin;
        flynn_limits[1]=amax;
        break;
    case O_FLYNNATTRMIN : flynn_limits[0] = *((double*)val);
        break;
    case O_FLYNNATTRMAX : flynn_limits[1] = *((double*)val);
        break;
    case O_BNDATTRIB : boundary_attrib = *((int*)val);
        if (boundary_attrib==MISORIENTATION)
            ElleFindBndAttributeRange(boundary_attrib,
                                      &amin, &amax);
        bnd_limits[0]=amin;
        bnd_limits[1]=amax;
        break;
    case O_BNDATTRMIN : bnd_limits[0] = *((double*)val);
        break;
    case O_BNDATTRMAX : bnd_limits[1] = *((double*)val);
        break;
    case O_BNDLEVEL  : boundary_level = *((int*)val);
        break;
    case O_UPDFREQ  : update_freq = *((int*)val);
        break;
    default:          break;
    }
}

/*
 * get a field value in an input_options structure
 */
void Settings::GetOptionValue(int id,XtPointer val)
{
    switch(id) {
    case O_REGION  : *((int*)val) = region;
        break;
    case O_OVERLAY  : *((int*)val) = overlay;
        break;
    case O_DRWUNODETYP: *((int*)val) = draw_nodes;
        break;
    case O_NODEATTRIB: *((int*)val) = node_attrib;
        break;
    case O_NODEATTRMIN: *((double*)val) = node_limits[0];
        break;
    case O_NODEATTRMAX: *((double*)val) = node_limits[1];
        break;
    case O_UNODEATTRIB: *((int*)val) = unode_attrib;
        break;
    case O_UNODEATTRMIN: *((double*)val) = unode_limits[0];
        break;
    case O_UNODEATTRMAX: *((double*)val) = unode_limits[1];
        break;
    case O_SEGMENTATTRIB: *((int*)val) = segment_attrib;
        break;
    case O_SEGMENTATTRMIN: *((double*)val) = segment_limits[0];
        break;
    case O_SEGMENTATTRMAX: *((double*)val) = segment_limits[1];
        break;
    case O_LBLREGION: *((int*)val) = label_region;
        break;
    case O_BNDATTRIB : *((int*)val) = boundary_attrib;
        break;
    case O_BNDATTRMIN: *((double*)val) = bnd_limits[0];
        break;
    case O_BNDATTRMAX: *((double*)val) = bnd_limits[1];
        break;
    case O_BNDLEVEL  : *((int*)val) = boundary_level;
        break;
    case O_FLYNNATTRIB: *((int*)val) = flynn_attrib;
        break;
    case O_FLYNNATTRMIN: *((double*)val) = flynn_limits[0];
        break;
    case O_FLYNNATTRMAX: *((double*)val) = flynn_limits[1];
        break;
    case O_UPDFREQ  : *((int*)val) = update_freq;
        break;
    default:         break;
    }
}

void Settings::GetForegroundOption(int *val)
{
    *val=fg;
}

void Settings::GetColourmapOption(int *val)
{
    *val=colourmap;
}

void Settings::GetRegionOption(int *val)
{
    GetOptionValue(O_REGION,(XtPointer)val);
}

void Settings::GetUpdFreqOption(int *val)
{
    GetOptionValue(O_UPDFREQ,(XtPointer)val);
}

/*
 * combine fill, bnd, overlay etc into bits of one int?
 */
void Settings::GetDrawingOption(int *val)
{
    int fill, outline;

    *val = -1;
    GetFlynnAttribOption(&fill);
    GetBndAttribOption(&outline);
    if (outline && !fill) *val = 2;
    else if (!outline && fill) *val = 0;
    else if (outline && fill) *val = 1;
}

void Settings::GetOverlayOption(int *val)
{
    GetOptionValue(O_OVERLAY,(XtPointer)val);
}

void Settings::SetOverlayOption( int p )
{
    overlay = p;
}

void Settings::GetSegmentPlotOption(int *val)
{
    GetOptionValue(O_DRWSEGMENTTYP,(XtPointer)val);
}

void Settings::GetUnodePlotOption(int *val)
{
    GetOptionValue(O_DRWUNODETYP,(XtPointer)val);
}

void Settings::GetNodeAttribOption(int *val)
{
    GetOptionValue(O_NODEATTRIB,(XtPointer)val);
}

void Settings::GetNodeAttribMinOption(double *val)
{
    GetOptionValue(O_NODEATTRMIN,(XtPointer)val);
}

void Settings::GetNodeAttribMaxOption(double *val)
{
    GetOptionValue(O_NODEATTRMAX,(XtPointer)val);
}

void Settings::GetUnodeAttribOption(int *val)
{
    GetOptionValue(O_UNODEATTRIB,(XtPointer)val);
}

void Settings::GetUnodeAttribMinOption(double *val)
{
    GetOptionValue(O_UNODEATTRMIN,(XtPointer)val);
}

void Settings::GetUnodeAttribMaxOption(double *val)
{
    GetOptionValue(O_UNODEATTRMAX,(XtPointer)val);
}

void Settings::GetSegmentAttribOption(int *val)
{
    GetOptionValue(O_SEGMENTATTRIB,(XtPointer)val);
}

void Settings::GetSegmentAttribMinOption(double *val)
{
    GetOptionValue(O_SEGMENTATTRMIN,(XtPointer)val);
}

void Settings::GetSegmentAttribMaxOption(double *val)
{
    GetOptionValue(O_SEGMENTATTRMAX,(XtPointer)val);
}

void Settings::GetLblRgnOption(int *val)
{
    GetOptionValue(O_LBLREGION,(XtPointer)val);
}

void Settings::GetBndAttribOption(int *val)
{
    GetOptionValue(O_BNDATTRIB,(XtPointer)val);
}

void Settings::GetBndLevelOption(int *val)
{
    GetOptionValue(O_BNDLEVEL,(XtPointer)val);
}

void Settings::SetBndLevelOption(int val)
{
    SetOptionValue(O_BNDLEVEL,(XtPointer)(&val));
}

void Settings::GetBndAttribMinOption(double *val)
{
    GetOptionValue(O_BNDATTRMIN,(XtPointer)val);
}

void Settings::GetBndAttribMaxOption(double *val)
{
    GetOptionValue(O_BNDATTRMAX,(XtPointer)val);
}

void Settings::GetFlynnAttribOption(int *val)
{
    GetOptionValue(O_FLYNNATTRIB,(XtPointer)val);
}

void Settings::GetFlynnAttribMinOption(double *val)
{
    GetOptionValue(O_FLYNNATTRMIN,(XtPointer)val);
}

void Settings::GetFlynnAttribMaxOption(double *val)
{
    GetOptionValue(O_FLYNNATTRMAX,(XtPointer)val);
}

void Settings::SetColourmapOption(int val)
{
    colourmap=val;
}

void Settings::SetMaxColoursOption(int val)
{
    max_colours=val;
}

void Settings::SetFlynnAttribOption(int id, double min, double max)
{
    SetOptionValue(O_FLYNNATTRIB,(XtPointer)(&id));
    SetOptionValue(O_FLYNNATTRMIN,(XtPointer)(&min));
    SetOptionValue(O_FLYNNATTRMAX,(XtPointer)(&max));
}

void Settings::SetNodeAttribOption(int id, double min, double max)
{
    SetOptionValue(O_NODEATTRIB,(XtPointer)(&id));
    SetOptionValue(O_NODEATTRMIN,(XtPointer)(&min));
    SetOptionValue(O_NODEATTRMAX,(XtPointer)(&max));
}

void Settings::SetUnodeAttribOption(int id, double min, double max)
{
    SetOptionValue(O_UNODEATTRIB,(XtPointer)(&id));
    SetOptionValue(O_UNODEATTRMIN,(XtPointer)(&min));
    SetOptionValue(O_UNODEATTRMAX,(XtPointer)(&max));
}

void Settings::SetSegmentAttribOption(int id, double min, double max)
{
    SetOptionValue(O_SEGMENTATTRIB,(XtPointer)(&id));
    SetOptionValue(O_SEGMENTATTRMIN,(XtPointer)(&min));
    SetOptionValue(O_SEGMENTATTRMAX,(XtPointer)(&max));
}

void Settings::SetBndAttribOption(int id, double min, double max)
{
    SetOptionValue(O_BNDATTRIB,(XtPointer)(&id));
    SetOptionValue(O_BNDATTRMIN,(XtPointer)(&min));
    SetOptionValue(O_BNDATTRMAX,(XtPointer)(&max));
}

int Settings::ValidColmapOption(char *str)
{
    return(name_match(str,colmap_terms));
}

int Settings::ReadSettings(FILE *fp, char *str, unsigned char initial)
{
    unsigned char finished=0, bool_val=0;
    int indx,len,i,dum,err=0,opt_type=0;
    float tmp;
    double max=0, min=0;

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
                    if (!strcmp(str,"PORTRAIT")) orient=PORTRAIT;
                    else if (!strcmp(str,"LANDSCAPE"))
                        orient=LANDSCAPE;
                    else return(RANGE_ERR);
                    break;
                case O_PAPERSZ: fscanf(fp,"%s",str);
                    if ((i = name_match(str,paper_terms))==-1)
                        return(RANGE_ERR);
                    paper_size = i;
                    break;
                case O_FONTHGT: if (fscanf(fp,"%d",&i)!=1) return(NUM_ERR);
                    if (i<0) return(RANGE_ERR);
                    dfltfonthgt = i;
                    break;
                case O_PG_XMARG: if (fscanf(fp,"%f",&tmp)!=1) return(NUM_ERR);
                    page_xmargin = tmp;
                    break;
                case O_PG_YMARG: if (fscanf(fp,"%f",&tmp)!=1) return(NUM_ERR);
                    page_ymargin = tmp;
                    break;
                case O_LINEWDTH: if (fscanf(fp,"%f",&tmp)!=1) return(NUM_ERR);
                    linewidth = tmp;
                    //if (!initial) setlinewidth_(&tmp); disabled weil die von mir gespeichert wird...
                    break;
                case O_PGSCL: if (fscanf(fp,"%f",&tmp)!=1) return(NUM_ERR);
                    page_scale = tmp;
                    break;
                    /*case O_FOREGRND:if (fscanf(fp,"%d",&i)!=1) return(NUM_ERR);
                           fg = i;
                            if (!initial) setpencolor_(&i);
                            opt_type = SYB_INT;
                            break;
                    case O_LINESTYLE:if (fscanf(fp,"%s",str)!=1) return(NUM_ERR);
                            if ((i = name_match(str,linestyle_terms))==-1)
                                 return(RANGE_ERR);
                            linestyle = i;

                            dum=1;
                            if (!initial) dashln_(&linestyle,&dum);
                            opt_type = SYB_INT;
                            break;*/
                case O_COLMAP:if (fscanf(fp,"%s",str)!=1) return(NUM_ERR);
                    if ((i = name_match(str,colmap_terms))==-1)
                        return(RANGE_ERR);
                    colourmap = i;
                    break;
                    /* plot options - set plot_params struct */
                case O_UPDFREQ:
                case O_OVERLAY:
                case O_DRWUNODETYP:
                case O_LBLREGION:
                case O_BNDLEVEL:  if (fscanf(fp,"%d",&i)!=1) return(NUM_ERR);
                    SetOptionValue(indx,(XtPointer)&i);
                    break;
                case O_FLYNNATTRIB:if (fscanf(fp,"%s",str)!=1) return(NUM_ERR);
                    if ((i = name_match(str,region_terms))==-1)
                        return(RANGE_ERR);
                    SetOptionValue(indx,(XtPointer)&i);
                    if (i!=NONE) {
                        fscanf(fp,"%[^\n]",str);
                        if ((err=ReadMinMax(str,&min,&max)) ==0) {
                            SetOptionValue(O_FLYNNATTRMIN,(XtPointer)(&min));
                            SetOptionValue(O_FLYNNATTRMAX,(XtPointer)(&max));
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
                        SetOptionValue(indx,(XtPointer)&i);
                        if (i!=NONE && i!=NUM_NB && i!=TRPL_ONLY) {
                            fscanf(fp,"%[^\n]",str);
                            if ((err=ReadMinMax(str,&min,&max)) ==0) {
                                SetOptionValue(O_NODEATTRMIN,(XtPointer)(&min));
                                SetOptionValue(O_NODEATTRMAX,(XtPointer)(&max));
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
                        SetOptionValue(indx,(XtPointer)&i);
                        if (i!=NONE && i!=U_LOCATION) {
                            fscanf(fp,"%[^\n]",str);
                            if ((err=ReadMinMax(str,&min,&max)) ==0) {
                                SetOptionValue(O_UNODEATTRMIN,(XtPointer)(&min));
                                SetOptionValue(O_UNODEATTRMAX,(XtPointer)(&max));
                            }
                            else {
                                dump_comments(fp);
                                err=0;
                            }
                        }
                    }
                    break;
                case O_SEGMENTATTRIB:
                    if (fscanf(fp,"%s",str)!=1)
                        fprintf(stderr,"ignoring segment attribute setting\n");
                    else {
                        if ((i = name_match(str,segment_terms))==-1)
                            return(RANGE_ERR);
                        SetOptionValue(indx,(XtPointer)&i);
                        if (i!=NONE && i!=S_LOCATION) {
                            fscanf(fp,"%[^\n]",str);
                            if ((err=ReadMinMax(str,&min,&max)) ==0) {
                                SetOptionValue(O_SEGMENTATTRMIN,(XtPointer)(&min));
                                SetOptionValue(O_SEGMENTATTRMAX,(XtPointer)(&max));
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
                        SetOptionValue(indx,(XtPointer)&i);
                        if (i!=NONE && i!=STANDARD) {
                            fscanf(fp,"%[^\n]",str);
                            if ((err=ReadMinMax(str,&min,&max)) ==0) {
                                SetOptionValue(O_BNDATTRMIN,(XtPointer)(&min));
                                SetOptionValue(O_BNDATTRMAX,(XtPointer)(&max));
                            }
                            else {
                                dump_comments(fp);
                                err=0;
                            }
                        }
                    }
                    break;
                default:warning_msg(VAR_ERR,(const char *)str);
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
int Settings::ReadOptions(FILE *fp, char *str)
{
    unsigned char finished=0, bool_val=0;
    int indx,len,i,dum,err=0,opt_type=0;
    double max=0, min=0;
    struct plot_params *opts = &plot_opts;

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
                    SetOptionValue(indx,&i);
                    break;
                case O_BNDLEVEL: fscanf(fp,"%d",&i);
                    SetOptionValue(indx,&i);
                    break;
                case O_FLYNNATTRIB:if (fscanf(fp,"%s",str)!=1) return(NUM_ERR);
                    if ((i = name_match(str,region_terms))==-1)
                        return(RANGE_ERR);
                    SetOptionValue(indx,&i);
                    if (i!=NONE) {
                        if ((err=ReadMinMax(fp,str,&min,&max)) ==0) {
                            SetOptionValue(O_FLYNNATTRMIN,(XtPointer)(&min));
                            SetOptionValue(O_FLYNNATTRMAX,(XtPointer)(&max));
                        }
                        else dump_comments(fp);
                    }
                    break;
                case O_NODEATTRIB:if (fscanf(fp,"%s",str)!=1) return(NUM_ERR);
                    if ((i = name_match(str,node_terms))==-1)
                        return(RANGE_ERR);
                    SetOptionValue(indx,&i);
                    if (i!=NONE) {
                        if ((err=ReadMinMax(fp,str,&min,&max)) ==0) {
                            SetOptionValue(
                                O_NODEATTRMIN,(XtPointer)(&min));
                            SetOptionValue(
                                O_NODEATTRMAX,(XtPointer)(&max));
                        }
                        else dump_comments(fp);
                    }
                    break;
                case O_BNDATTRIB:if (fscanf(fp,"%s",str)!=1) return(NUM_ERR);
                    if ((i = name_match(str,boundary_terms))==-1)
                        return(RANGE_ERR);
                    SetOptionValue(indx,&i);
                    if (i!=NONE && i!=STANDARD) {
                        if ((err=ReadMinMax(fp,str,&min,&max)) ==0) {
                            SetOptionValue(
                                O_BNDATTRMIN,(XtPointer)(&min));
                            SetOptionValue(
                                O_BNDATTRMAX,(XtPointer)(&max));
                        }
                        else dump_comments(fp);
                    }
                    break;
                default:warning_msg(VAR_ERR,(const char *)str);
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
int Settings::ReadMinMax(char *str, double *min, double *max)
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

int Settings::WriteSettingsFile(char *filename)
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

int Settings::WriteSettings(FILE *fp)
{
    char format[20], buf[MAX_OPTION_NAME+1];
    int val=0, i=0, err=0;
    double fval=0.0;

    strcpy(format,"");
    strcpy(buf,"");
    sprintf(format,"%s%d%s= ","%-",MAX_OPTION_NAME,"s");
    while (display_terms[i].name!=0 && !err) {
        fprintf(fp,format,display_terms[i].name);
        switch( display_terms[i].id ) {
        case O_ORIENT:
            if (orient==PORTRAIT)
                strcpy(buf,"PORTRAIT");
            else strcpy(buf,"LANDSCAPE");
            fprintf(fp,"%s",buf);
            break;
        case O_PAPERSZ:
            id_match(paper_terms,paper_size,buf);
            fprintf(fp,"%s",buf);
            break;
        case O_FONTHGT: fprintf(fp,"%d",dfltfonthgt);
            break;
        case O_PG_XMARG: fprintf(fp,"%.3f",page_xmargin);
            break;
        case O_PG_YMARG: fprintf(fp,"%.3f",page_ymargin);
            break;
        case O_LINEWDTH: fprintf(fp,"%.3f",linewidth);
            break;
        case O_PGSCL: fprintf(fp,"%.3f",linewidth);
            break;
        case O_FOREGRND: fprintf(fp,"%d",fg);
            break;
        case O_LINESTYLE:
            id_match(linestyle_terms,linestyle,buf);
            fprintf(fp,"%s",buf);
            break;
        case O_COLMAP:
            id_match(colmap_terms,colourmap,buf);
            fprintf(fp,"%s",buf);
            break;
        }
        if (fprintf(fp,"\n")<0) err=WRITE_ERR;
        i++;
    }
    i=0;
    while (option_terms[i].name!=0 && !err) {
        fprintf(fp,format,option_terms[i].name);
        switch(option_terms[i].id) {
        case O_UPDFREQ: GetUpdFreqOption(&val);
            fprintf(fp," %d",val);
            break;
        case O_DRWUNODETYP: GetUnodePlotOption(&val);
            fprintf(fp," %d",val);
            break;
        case O_DRWSEGMENTTYP: GetSegmentPlotOption(&val);
            fprintf(fp," %d",val);
            break;
        case O_LBLREGION: GetLblRgnOption(&val);
            fprintf(fp," %d",val);
            break;
        case O_OVERLAY: GetOverlayOption(&val);
            fprintf(fp," %d",val);
            break;
        case O_FLYNNATTRIB: GetFlynnAttribOption(&val);
            id_match(region_terms,val,buf);
            fprintf(fp," %s",buf);
            if (val!=NONE) {
                GetFlynnAttribMinOption(&fval);
                fprintf(fp," min= %.6e",fval);
                GetFlynnAttribMaxOption(&fval);
                fprintf(fp," max= %.6e",fval);
            }
            break;
        case O_BNDLEVEL: GetBndLevelOption(&val);
            fprintf(fp," %d",val);
            break;
        case O_BNDATTRIB: GetBndAttribOption(&val);
            id_match(boundary_terms,val,buf);
            fprintf(fp," %s",buf);
            if (val!=NONE && val!=STANDARD) {
                GetBndAttribMinOption(&fval);
                fprintf(fp," min= %.6e",fval);
                GetBndAttribMaxOption(&fval);
                fprintf(fp," max= %.6e",fval);
            }
            break;
        case O_NODEATTRIB: GetNodeAttribOption(&val);
            id_match(node_terms,val,buf);
            fprintf(fp," %s",buf);
            if (val!=NONE && val!=NUM_NB && val!=TRPL_ONLY) {
                GetNodeAttribMinOption(&fval);
                fprintf(fp," min= %.6e",fval);
                GetNodeAttribMaxOption(&fval);
                fprintf(fp," max= %.6e",fval);
            }
            break;
        case O_UNODEATTRIB: GetUnodeAttribOption(&val);
            id_match(unode_terms,val,buf);
            fprintf(fp," %s",buf);
            if (val!=NONE && val!=U_LOCATION) {
                GetUnodeAttribMinOption(&fval);
                fprintf(fp," min= %.6e",fval);
                GetUnodeAttribMaxOption(&fval);
                fprintf(fp," max= %.6e",fval);
            }
            break;
        case O_SEGMENTATTRIB: GetSegmentAttribOption(&val);
            id_match(segment_terms,val,buf);
            fprintf(fp," %s",buf);
            if (val!=NONE && val!=S_LOCATION) {
                GetSegmentAttribMinOption(&fval);
                fprintf(fp," min= %.6e",fval);
                GetSegmentAttribMaxOption(&fval);
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
int Settings::GetPlotColourRange(int *min,int *max)
{
    *min = 0;//J USEFL_SIZE;

    *max = 255;//J USEFL_SIZE+(max_colours-USEFL_SIZE)*0.92;
    return(0);
}
