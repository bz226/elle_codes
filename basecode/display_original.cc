/*****************************************************
* Copyright: (c) L. A. Evans
* File:      $RCSfile$
* Revision:  $Revision$
* Date:      $Date$
* Author:    $Author$
*
******************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "attrib.h"
#include "bflynns.h"
#include "nodes.h"
#include "display.h"
#include "runopts.h"
#include "general.h"
#include "file.h"
#include "interface.h"
#include "error.h"
#include "convert.h"
#include "check.h"
#include "polygon.h"
#include "misorient.h"
#include "unodes.h"
#include "segments.h"
#include "setup.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/
#include "triangle.h"
#include "triattrib.h"

#ifndef NODISP

#include "../plotcode/settings.h" //includes decl for Log() - sort this out
#include "../wxplotcode/settings.h" //includes decl for Log() - sort this out


int ElleFillFlynns(int,int);
int EllePlotFlynns(int outline,int draw_bndlevel,int draw_bndattrib);
int ElleMarkNode(int dd,int *col,float x,float y,float size);
void EllePlotNodes();
void EllePlotUnodes();
void EllePlotSegments();
void PlotUnodes(int col_range, int col_min,
                int attr, double min, double max);
void EllePlotPolygon(int index,float *xpts, float *ypts,
                     int *id, int num_nodes,
                     int col, int outline, int bndattrib, char *label);
void EllePlotBoundaryAttribute(int *id, int num, float *xpts, float *ypts);
Settings *options;


void Init_Data( void ) {
    options = GetUserOptions();
options->InitOptions ();
}

void ReportError()
{
    Error();
}

// the dummy fn below is only necessary for diffusion
void GetUpdateFreq(int *val)
{
    options->GetUpdFreqOption(val);
    if (*val==0) *val = 50;
}

void ElleUpdateDisplay()
{
    int count = ElleCount();
    if (ElleDisplay()) {
        ElleShowStages(count);
        // PlotRegions refreshes screen so it should be called last
        EllePlotRegions(count);
    }

}

void EllePlotRegions(int count)
{
    int choice,outline,drawnodes,draw_bndattrib,draw_bndlevel,upd,olay=0;
    int draw_unodes, triangles;
    int draw_segments;
    int max;

    options->GetUpdFreqOption(&upd);
    if(upd==0)
    	upd=1;
    if (count && count%upd) return;
    
    options->GetOverlayOption(&olay);
    if (!olay) clearbg_();

    options->GetRegionOption(&choice);
    options->GetDrawingOption(&outline);
    /*
     * outline = -1 no fill, no outline
     * outline = 0  fill only
     * outline = 1  fill + outline polygon
     * outline = 2  outline only
     */
    options->GetNodeAttribOption(&drawnodes);
    options->GetUnodeAttribOption(&draw_unodes);
    options->GetUnodePlotOption(&triangles);
    options->GetSegmentAttribOption(&draw_segments);
    options->GetBndAttribOption(&draw_bndattrib);
    if (draw_bndattrib==MISORIENTATION)
        CalculateBoundaryAttribute(MISORIENTATION);
    if(draw_bndattrib==RAINBOW)
        options->SetRainbow(true);
    else
        options->SetRainbow(false);
    options->GetBndLevelOption(&draw_bndlevel);
    /*
     * should flynnarray class store and maintain max level field?
        if (draw_bndlevel > (max=ElleMaxLevels())) draw_bndlevel=max;
     */
    choice=GRAINS;
    switch(choice) {
        /*
         * the drawing order needs to be right to avoid overwriting
         */
    case GRAINS:
        if (draw_segments!=NONE )
            EllePlotSegments();               /* fill segments */
        if ((draw_unodes!=NONE && triangles ) ||
              (draw_unodes!=NONE &&
                (!triangles||draw_unodes==U_LOCATION)))
            EllePlotUnodes();               /* fill unodes */
        else if (outline>=0 && outline!=2)
            ElleFillFlynns(0,draw_bndlevel); /* fill flynns */
        if (outline>0) {                 /* draw boundaries */
            if (draw_bndattrib!=STANDARD) outline = 3;
            else outline=2;
            if(draw_bndattrib==RAINBOW) outline = 4;
            EllePlotFlynns(outline,draw_bndlevel,draw_bndattrib);
        }
        if (drawnodes!=NONE) EllePlotNodes();
#if XY
        if (draw_unodes!=NONE &&
                (!triangles||draw_unodes==U_LOCATION))
            EllePlotUnodes();               /* plot unodes */
#endif
        break;
    case TRIANGLES:EllePlotTriangles();
        break;
    default:       break;
    }

    update_();
}

/*
 * this should become flynns.plot() ??
 */
int ElleFillFlynns(int outline, int draw_bndlevel)
{
    int col=0, i, j, num_nodes, max, label;
    int *id=0, draw_attrib, draw_bndattrib;
    int col_range, col_max = E_COL_MAX, col_min = E_COL_MIN;
    float *xpts=0, *ypts=0;
    double val,valb,valc, amin, amax, arange;
    Coords current,prev;
    Coords_3D coordval;
    double theta,phi;

    options->GetPlotColourRange(&col_min,&col_max);
    options->GetLblRgnOption(&label);
    options->GetBndAttribOption(&draw_bndattrib);
    options->GetFlynnAttribOption(&draw_attrib);
    options->GetFlynnAttribMinOption(&amin);
    options->GetFlynnAttribMaxOption(&amax);
    arange = amax-amin;
    if (arange==0)
        if (ElleFindFlynnAttributeRange(draw_attrib,&amin,&amax)==0)
            arange = amax-amin;
    max = ElleMaxFlynns();
    col_range = col_max - col_min;
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i) && !isParent(i)) {
            ElleFlynnNodes(i,&id,&num_nodes);
            ElleNodePosition(id[0],&current);
            prev = current;
            if ((xpts = (float *)malloc(num_nodes*sizeof(float)))==0 ||
                (ypts = (float *)malloc(num_nodes*sizeof(float)))==0)
                OnError("ElleFillFlynns",MALLOC_ERR);
            j=0;
            do {
                ElleNodePlotXY(id[j],&current,&prev);
                xpts[j] = (float)current.x;
                ypts[j] = (float)current.y;
                prev = current;
                j++;
            } while (j<num_nodes);
            col=0; j=0;
            if (ElleFlynnAttributeActive(COLOUR))
                ElleGetFlynnIntAttribute(i,&col,COLOUR);
            else
                switch(draw_attrib) {
                case GRAIN: ElleGetFlynnIntAttribute(i,&j,GRAIN);
                    if (j>=0) col = (j%col_range)+col_min;
                    break;
                case EXPAND:ElleGetFlynnIntAttribute(i,&j,EXPAND);
                    if (j==0) col = col_min+(int)(col_range*0.24);
                    else col = col_min+(int)(col_range*0.82);
                    break;
                case MINERAL:ElleGetFlynnIntAttribute(i,&j,MINERAL);
                    if (j==QUARTZ)
                        col=col_min+(int)(col_range*0.35);
                    else if (j==CALCITE)
                        col = col_min+(int)(col_range*0.25);
                    else if (j==FELDSPAR)
                        col = col_min+(int)(col_range*0.65);
                    else if (j==MICA)
                        col = col_min+(int)(col_range*0.74);
                    else if (j==GARNET)
                        col = col_min+(int)(col_range*0.91);
                    else if (j==MINERAL_A) col = 4;
                    else if (j==MINERAL_B) col = 5;
                    else if (j==MINERAL_C) col = 7;
                    break;
                case VISCOSITY:
                case S_EXPONENT:
                case ENERGY:
                case AGE:
                case CYCLE:
                case DISLOCDEN:
                case F_ATTRIB_A:
                case F_ATTRIB_B:
                case F_ATTRIB_C:
                case F_ATTRIB_D:
                case F_ATTRIB_E:
                case F_ATTRIB_F:
                case F_ATTRIB_I:
                case F_ATTRIB_J:
                case F_ATTRIB_K:
                case TAU_XX:
                case TAU_YY:
                case TAU_ZZ:
                case TAU_XY:
                case TAU_1:
                case PRESSURE:
                    ElleGetFlynnRealAttribute(i,&val,draw_attrib);
                    if (arange>0) {
                        col = (int)((val-amin)/arange*col_range) + col_min;
                        if (col<col_min) col=col_min;
                        if (col>col_max) col=col_max;
                    }
                    break;
                case CAXIS: ElleGetFlynnCAxis(i,&coordval);
                    CartesianToPolar(coordval.x,coordval.y,
                                     coordval.z,&theta,&phi);
                    if (theta<0.0) theta = PI-theta;
                    theta *= RTOD;
                    val = theta;
                    if (arange>0) {
                        col = (int)((val-amin)/arange*col_range) + col_min;
                        if (col<col_min) col=col_min;
                        if (col>col_max) col=col_max;
                    }
                    break;
                case E3_ALPHA:
                case EULER_3:ElleGetFlynnEuler3(i,&val,&valb,&valc);
                    if (arange>0) {
                        //val=fmod((val+90.0),30.0)/15;
                        //if(val > 1.0) val=2.0-val;
                        col = (int)((val-amin)/arange*col_range) + col_min;
                        if (col<col_min) col=col_min;
                        if (col>col_max) col=col_max;
                    }
                    break;
                case INCR_S:
                case BULK_S:
                case F_INCR_S:
                case F_BULK_S:
                case E_XX:
                case E_XY:
                case E_YX:
                case E_YY:
                case E_ZZ:  ElleGetFlynnRealAttribute(i,&val,draw_attrib);
                    if (arange>0) {
                        col = (int)((val-amin)/arange*col_range) + col_min;
                        if (col<col_min) col=col_min;
                        if (col>col_max) col=col_max;
                    }
                    else col=0;
                    break;
                default:    break;
                }

            EllePlotPolygon(i,xpts,ypts,id,num_nodes,
                            col,outline,draw_bndattrib,0);
            if (id) free(id); id=0;
            if (xpts) free(xpts); xpts=0;
            if (ypts) free(ypts); ypts=0;
        }
    }
    return( 0 );
}

/*
 * this should become flynns.plot() ??
 */
int EllePlotFlynns(int outline, int draw_bndlevel, int draw_bndattrib)
{
    int col=0, oldcol=0, i, j, num_nodes, max, label;
    int *id=0, level;
    float *xpts=0, *ypts=0;
    Coords current,prev;
    char buf[15];

    options->GetLblRgnOption(&label);
    options->GetForegroundOption(&col);
    /*if (draw_bndattrib!=STANDARD && outline>0) */
    /*CalculateBoundaryAttribute(draw_bndattrib);*/
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i) &&
                (((level=ElleFlynnLevel(i))==draw_bndlevel) ||
                 (!isParent(i) && level < draw_bndlevel))) {
            ElleFlynnNodes(i,&id,&num_nodes);
            ElleNodePosition(id[0],&current);
            prev = current;
            if ((xpts = (float *)malloc(num_nodes*sizeof(float)))==0 ||
                (ypts = (float *)malloc(num_nodes*sizeof(float)))==0)
                OnError("EllePlotFlynns",MALLOC_ERR);
            j=0;
            do {
                ElleNodePlotXY(id[j],&current,&prev);
                xpts[j] = (float)current.x;
                ypts[j] = (float)current.y;
                prev = current;
                j++;
            } while (j<num_nodes);
            if (label) {
                sprintf(buf,"%d",i);
                EllePlotPolygon(i,xpts,ypts,id,num_nodes,
                                col,outline,draw_bndattrib,buf);
            }
            else
                EllePlotPolygon(i,xpts,ypts,id,num_nodes,
                                col,outline,draw_bndattrib,0);
            if (id) free(id); id=0;
            if (xpts) free(xpts); xpts=0;
            if (ypts) free(ypts); ypts=0;
        }
    }
    return( 0 );
}


void EllePlotTriangles()
{
    int col, i, j, max, outline;
    int id[3], num=3;
    int col_range, col_max = E_COL_MAX, col_min = E_COL_MIN;
    double val, attr_max, attr_min, range, tmpx, tmpy;
    float xpts[3], ypts[3];

    /*
     * temporarily using region settings -
     * fill can't be set so turn it on
     */
    options->GetPlotColourRange(&col_min,&col_max);
    options->GetDrawingOption(&outline);
    /*if (outline) outline = 1;*/
    /*else outline = 0;*/
    outline = 0;
    col_range = col_max-col_min;
    attr_max = ElleTriAttributeMax(0);
    attr_min = ElleTriAttributeMin(0);
    range = attr_max - attr_min;
    if (range==0.0) range = col_range;
    max = ElleMaxTriangles();
    for (i=0;i<max;i++) {
        ElleTrianglePoints(i,id);
        for (j=0;j<3;j++) {
            ElleTriPointPosition(id[j],&tmpx,&tmpy);
            xpts[j] = tmpx;
            ypts[j] = tmpy;
        }
        val = ElleTriAttribute(i,0);
        col = (int)((val-attr_min)/range * col_range + col_min);
        if (col < col_min) col = col_min;
        if (col > col_max) col = col_max;
        EllePlotPolygon(-1,xpts,ypts,0,3,col,outline,0,0);
        /*fillpoly_(xpts,ypts,&num,&col,&outline);*/
    }
    update_();
}

void EllePlotPolygon(int index,float *xptsorig, float *yptsorig,
                     int *id, int num_nodes,
                     int col, int outline, int bndattrib, char *label)
{
    int xflags[2], yflags[2];
    int j,i,k,start=0;
    /*float eps=5e-5;*/
    float eps=1.5e-4;
    float incrx, incry;
    float xoffset=0.0;
    float *xpts=0, *ypts=0;

    int len, x_off=0, y_off=0, mode=1;
    int xflagslabel[2], yflagslabel[2];
    double xcentre, ycentre, area;
    float centre[2];
    Coords bl,tr,lbl,ltr,tmpbl,tmptr;
    CellData unitcell;

    if ((xpts = (float *)malloc(num_nodes*sizeof(float)))==0 ||
            (ypts = (float *)malloc(num_nodes*sizeof(float)))==0)
        OnError("EllePlotPolygon",MALLOC_ERR);
    for (j=0;j<num_nodes;j++) {
        xpts[j] = xptsorig[j];
        ypts[j] = yptsorig[j];
    }
    if (label!=0) len = strlen(label);
    /*
     * do checks on xpts,ypts to decide whether
     * to replot on unit edges
     */

    ElleCellBBox(&unitcell);
    incrx = unitcell.xlength;
    incry = unitcell.ylength;
    xoffset = unitcell.xoffset;
    if ((unitcell.cum_xoffset<-eps || unitcell.cum_xoffset>eps) &&
            xoffset>unitcell.xlength-eps && xoffset<unitcell.xlength+eps)
        xoffset=0;
    ElleCheckUnit(xpts,ypts,num_nodes,xflags,yflags,&bl,&tr);
    /*
     * if xoffset >= incrx/2 may need to draw offset back a unit
     */
    start=0;
    if (yflags[0]||yflags[1]) start=-1;
    for (i=start;i<xflags[0]+1;i++) {
        for (j=0;j<num_nodes;j++) {
            xpts[j] = xptsorig[j] + i*incrx;
            ypts[j] = yptsorig[j];
        }
        tmpbl = bl;
        tmptr = tr;
        tmpbl.x += i*incrx;
        tmptr.x += i*incrx;
        if (tmpbl.y<unitcell.cellBBox[TOPRIGHT].y &&
                tmptr.y>unitcell.cellBBox[BASELEFT].y) {
            if (outline!=3 || outline==4)
                FillPolygon(index,xpts,ypts,&num_nodes,&col,&outline);
            else
                EllePlotBoundaryAttribute(id,num_nodes,xpts,ypts);
            if (label!=0) {
                if (polyCentroid(xpts,ypts,num_nodes,
                                 &xcentre,&ycentre,&area) !=2) {
                    centre[0] = (float)xcentre; centre[1] = (float)ycentre;
                    ElleCheckUnit(&centre[0],&centre[1],1,xflagslabel,yflagslabel,
                                  &lbl,&ltr);
                    if (!xflagslabel[0] && !xflagslabel[1] &&
                            !yflagslabel[0] && !yflagslabel[1])
                        drawlabel_(&centre[0],&centre[1],&x_off,&y_off,&mode,label,&len);
                }
            }
        }
        for (k=0; k< yflags[0]; k++) {
            for (j=0;j<num_nodes;j++) xpts[j] += (k+1)*xoffset;
            for (j=0;j<num_nodes;j++) ypts[j] += (k+1)*incry;
            tmpbl.x += (k+1)*xoffset;
            tmpbl.y += (k+1)*incry;
            tmptr.x += (k+1)*xoffset;
            tmptr.y += (k+1)*incry;
            if (tmpbl.x<unitcell.cellBBox[TOPRIGHT].x &&
                    tmptr.x>unitcell.cellBBox[BASELEFT].x) {
                if (outline!=3)
                    FillPolygon(index,xpts,ypts,&num_nodes,&col,&outline);
                else
                    EllePlotBoundaryAttribute(id,num_nodes,xpts,ypts);
                if (label!=0) {
                    if (polyCentroid(xpts,ypts,num_nodes,
                                     &xcentre,&ycentre,&area) !=2) {
                        centre[0] = (float)xcentre; centre[1] = (float)ycentre;
                        ElleCheckUnit(&centre[0],&centre[1],1,xflagslabel,yflagslabel,
                                      &lbl,&ltr);
                        if (!xflagslabel[0] && !xflagslabel[1] &&
                                !yflagslabel[0] && !yflagslabel[1])
                            drawlabel_(&centre[0],&centre[1],&x_off,&y_off,&mode,label,&len);
                    }
                }
            }
        }
        for (j=0;j<num_nodes;j++) {
            xpts[j] = xptsorig[j] + i*incrx;
            ypts[j] = yptsorig[j];
        }
        tmpbl.x = bl.x + i*incrx;
        tmpbl.y = bl.y;
        tmptr.x = tr.x + i*incrx;
        tmptr.y = tr.y;
        for (k=0; k< yflags[1]; k++) {
            for (j=0;j<num_nodes;j++) xpts[j] -= (k+1)*xoffset;
            for (j=0;j<num_nodes;j++) ypts[j] -= (k+1)*incry;
            tmpbl.x -= (k+1)*xoffset;
            tmpbl.y -= (k+1)*incry;
            tmptr.x -= (k+1)*xoffset;
            tmptr.y -= (k+1)*incry;
            if (tmpbl.x<unitcell.cellBBox[TOPRIGHT].x &&
                    tmptr.x>unitcell.cellBBox[BASELEFT].x) {
                if (outline!=3)
                    FillPolygon(index,xpts,ypts,&num_nodes,&col,&outline);
                else
                    EllePlotBoundaryAttribute(id,num_nodes,xpts,ypts);
                if (label!=0) {
                    if (polyCentroid(xpts,ypts,num_nodes,
                                     &xcentre,&ycentre,&area) !=2) {
                        centre[0] = (float)xcentre; centre[1] = (float)ycentre;
                        ElleCheckUnit(&centre[0],&centre[1],1,xflagslabel,yflagslabel,
                                      &lbl,&ltr);
                        if (!xflagslabel[0] && !xflagslabel[1] &&
                                !yflagslabel[0] && !yflagslabel[1])
                            drawlabel_(&centre[0],&centre[1],&x_off,&y_off,&mode,label,&len);
                    }
                }
            }
        }
        if (start==0) start=1;
    }
    for (i=start;i<xflags[1]+1;i++) {
        for (j=0;j<num_nodes;j++) {
            xpts[j] = xptsorig[j] - i*incrx;
            ypts[j] = yptsorig[j];
        }
        tmpbl.x = bl.x - i*incrx;
        tmpbl.y = bl.y;
        tmptr.x = tr.x - i*incrx;
        tmptr.y = tr.y;
        if (tmpbl.y<unitcell.cellBBox[TOPRIGHT].y &&
                tmptr.y>unitcell.cellBBox[BASELEFT].y) {
            if (outline!=3)
                FillPolygon(index,xpts,ypts,&num_nodes,&col,&outline);
            else
                EllePlotBoundaryAttribute(id,num_nodes,xpts,ypts);
            if (label!=0) {
                if (polyCentroid(xpts,ypts,num_nodes,
                                 &xcentre,&ycentre,&area) !=2) {
                    centre[0] = (float)xcentre; centre[1] = (float)ycentre;
                    ElleCheckUnit(&centre[0],&centre[1],1,xflagslabel,yflagslabel,
                                  &lbl,&ltr);
                    if (!xflagslabel[0] && !xflagslabel[1] &&
                            !yflagslabel[0] && !yflagslabel[1])
                        drawlabel_(&centre[0],&centre[1],&x_off,&y_off,&mode,label,&len);
                }
            }
        }
        for (k=0; k< yflags[0]; k++) {
            for (j=0;j<num_nodes;j++) xpts[j] += (k+1)*xoffset;
            for (j=0;j<num_nodes;j++) ypts[j] += (k+1)*incry;
            tmpbl.x += (k+1)*xoffset;
            tmpbl.y += (k+1)*incry;
            tmptr.x += (k+1)*xoffset;
            tmptr.y += (k+1)*incry;
            if (tmpbl.x<unitcell.cellBBox[TOPRIGHT].x &&
                    tmptr.x>unitcell.cellBBox[BASELEFT].x) {
                if (outline!=3)
                    FillPolygon(index,xpts,ypts,&num_nodes,&col,&outline);
                else
                    EllePlotBoundaryAttribute(id,num_nodes,xpts,ypts);
                if (label!=0) {
                    if (polyCentroid(xpts,ypts,num_nodes,
                                     &xcentre,&ycentre,&area) !=2) {
                        centre[0] = (float)xcentre; centre[1] = (float)ycentre;
                        ElleCheckUnit(&centre[0],&centre[1],1,xflagslabel,yflagslabel,
                                      &lbl,&ltr);
                        if (!xflagslabel[0] && !xflagslabel[1] &&
                                !yflagslabel[0] && !yflagslabel[1])
                            drawlabel_(&centre[0],&centre[1],&x_off,&y_off,&mode,label,&len);
                    }
                }
            }
        }
        for (j=0;j<num_nodes;j++) {
            xpts[j] = xptsorig[j] - i*incrx;
            ypts[j] = yptsorig[j];
        }
        tmpbl.x = bl.x - i*incrx;
        tmpbl.y = bl.y;
        tmptr.x = tr.x - i*incrx;
        tmptr.y = tr.y;
        for (k=0; k< yflags[1]; k++) {
            for (j=0;j<num_nodes;j++) xpts[j] -= (k+1)*xoffset;
            for (j=0;j<num_nodes;j++) ypts[j] -= (k+1)*incry;
            tmpbl.x -= (k+1)*xoffset;
            tmpbl.y -= (k+1)*incry;
            tmptr.x -= (k+1)*xoffset;
            tmptr.y -= (k+1)*incry;
            if (tmpbl.x<unitcell.cellBBox[TOPRIGHT].x &&
                    tmptr.x>unitcell.cellBBox[BASELEFT].x) {
                if (outline!=3)
                    FillPolygon(index,xpts,ypts,&num_nodes,&col,&outline);
                else
                    EllePlotBoundaryAttribute(id,num_nodes,xpts,ypts);
                if (label!=0) {
                    if (polyCentroid(xpts,ypts,num_nodes,
                                     &xcentre,&ycentre,&area) !=2) {
                        centre[0] = (float)xcentre; centre[1] = (float)ycentre;
                        ElleCheckUnit(&centre[0],&centre[1],1,xflagslabel,yflagslabel,
                                      &lbl,&ltr);
                        if (!xflagslabel[0] && !xflagslabel[1] &&
                                !yflagslabel[0] && !yflagslabel[1])
                            drawlabel_(&centre[0],&centre[1],&x_off,&y_off,&mode,label,&len);
                    }
                }
            }
        }
        if (start==0) start=1;
    }
    for (i=start;i<yflags[1]+1;i++) {
        for (j=0;j<num_nodes;j++) xpts[j] = xptsorig[j]+i*xoffset;
        for (j=0;j<num_nodes;j++) ypts[j] = yptsorig[j]+i*incry;
        tmpbl.x = bl.x+i*xoffset;
        tmpbl.y = bl.y+i*incry;
        tmptr.x = tr.x+i*xoffset;
        tmptr.y = tr.y+i*incry;
        if (tmpbl.x<unitcell.cellBBox[TOPRIGHT].x &&
                tmptr.x>unitcell.cellBBox[BASELEFT].x) {
            if (outline!=3)
                FillPolygon(index,xpts,ypts,&num_nodes,&col,&outline);
            else
                EllePlotBoundaryAttribute(id,num_nodes,xpts,ypts);
            if (label!=0) {
                if (polyCentroid(xpts,ypts,num_nodes,
                                 &xcentre,&ycentre,&area) !=2) {
                    centre[0] = (float)xcentre; centre[1] = (float)ycentre;
                    ElleCheckUnit(&centre[0],&centre[1],1,xflagslabel,yflagslabel,
                                  &lbl,&ltr);
                    if (!xflagslabel[0] && !xflagslabel[1] &&
                            !yflagslabel[0] && !yflagslabel[1])
                        drawlabel_(&centre[0],&centre[1],&x_off,&y_off,&mode,label,&len);
                }
            }
        }
        if (start==0) start=1;
    }
    for (i=start;i<yflags[0]+1;i++) {
        for (j=0;j<num_nodes;j++) xpts[j] = xptsorig[j]+i*xoffset;
        for (j=0;j<num_nodes;j++) ypts[j] = yptsorig[j]+i*incry;
        tmpbl.x = bl.x+i*xoffset;
        tmpbl.y = bl.y+i*incry;
        tmptr.x = tr.x+i*xoffset;
        tmptr.y = tr.y+i*incry;
        if (tmpbl.x<unitcell.cellBBox[TOPRIGHT].x &&
                tmptr.x>unitcell.cellBBox[BASELEFT].x) {
            if (outline!=3)
                FillPolygon(index,xpts,ypts,&num_nodes,&col,&outline);
            else
                EllePlotBoundaryAttribute(id,num_nodes,xpts,ypts);
            if (label!=0) {
                if (polyCentroid(xpts,ypts,num_nodes,
                                 &xcentre,&ycentre,&area) !=2) {
                    centre[0] = (float)xcentre; centre[1] = (float)ycentre;
                    ElleCheckUnit(&centre[0],&centre[1],1,xflagslabel,yflagslabel,
                                  &lbl,&ltr);
                    if (!xflagslabel[0] && !xflagslabel[1] &&
                            !yflagslabel[0] && !yflagslabel[1])
                        drawlabel_(&centre[0],&centre[1],&x_off,&y_off,&mode,label,&len);
                }
            }
        }
    }
    if (xpts) free(xpts); xpts=0;
    if (ypts) free(ypts); ypts=0;
}

void ElleShowStages(int stages)
{
    SetStage( stages );
}
void EllePlotBoundaryAttribute(int *id, int num, float *xpts, float *ypts)
{
    int i,j,col;
    int col_range, col_max = E_COL_MAX, col_min = E_COL_MIN;
    int state = 3, oldcol;
    double val, amin, amax, arange;

    options->GetForegroundOption(&oldcol);
    options->GetPlotColourRange(&col_min,&col_max);
    options->GetBndAttribMinOption(&amin);
    options->GetBndAttribMaxOption(&amax);
    arange = amax - amin;
    if (arange==0) FindBndAttributeRange(MISORIENTATION,&amin,&amax);
    col_range = col_max-col_min;
    int step = col_range/6;

    for (i=0; i<num; i++) {
        j = (i+1)%num;
        if (id[j] < id[i]) {
            col=1;
            // val will be default (-1) if segment is phase boundary
            val=ElleGetBoundaryAttribute(id[i],id[j]);
            if (val>=0 && val<6) col = col_min;
            else if (val>5 && val<11) col = col_min + step;
            else if (val>10 && val<16) col = col_min + step*2;
            else if (val>15 && val<21) col = col_min + step*3;
            else if (val>20 && val<26) col = col_min + step*4;
            else if (val>25 && val<31) col = col_min + step*5;
            else if (val>30 && val<181) col = col_min + step*6;
            if (val>=amax) col = col_max;
            if (val<=amin) col = col_min;
            drawsegment_(&xpts[i],&ypts[i],&xpts[j],&ypts[j],&col);
        }
    }
    /* reset pen color */
    setpencolor_(&oldcol);
}

void EllePlotNodes()
{
    int i,col,max;
    float ptrad;
    int col_range, col_max = E_COL_MAX, col_min = E_COL_MIN, node_colour;
    double attr_max, attr_min, range=0, val;
    Coords current;

    max = ElleMaxNodes();
    ptrad=ElleGetPointRadius(max);
    options->GetNodeAttribOption(&node_colour);
    options->GetPlotColourRange(&col_min,&col_max);
    col_range = col_max-col_min;
    if (node_colour!=NUM_NB && node_colour!=TRPL_ONLY &&
            ElleFindNodeAttribIndex(node_colour)==NO_INDX)
        node_colour = NUM_NB;
    if (node_colour != NUM_NB && node_colour!=TRPL_ONLY) {
        /*ElleNodeAttributeRange(node_colour,&attr_max,&attr_min);*/
        options->GetNodeAttribMinOption(&attr_min);
        options->GetNodeAttribMaxOption(&attr_max);
        range = attr_max - attr_min;
        if (range==0.0) range = col_range;
    }
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            ElleNodePosition(i,&current);
            /* need to have ElleAttributeActive(int id) */
            if (node_colour == NUM_NB||node_colour == TRPL_ONLY) {
                if (ElleNodeIsDouble(i)) col=E_DFLT_DNODE_COL;
                else col=E_DFLT_TNODE_COL;
            }
            else {
                val = ElleNodeAttribute(i,node_colour);
                col = (int)((val-attr_min)/range * col_range) + col_min;
                if (col < col_min) col = col_min;
                if (col > col_max) col = col_max;
            }
            if (node_colour!=TRPL_ONLY || ElleNodeIsTriple(i))
                ElleMarkNode(i,&col,current.x,current.y,ptrad);
        }
    }
}


void EllePlotUnodes()
{
    int i, maxf;
    int col=7, attr=U_LOCATION, triangles=0;
    int col_range, col_max = E_COL_MAX, col_min = E_COL_MIN;
    double max=0.0,min=0.0;

    options->GetPlotColourRange(&col_min,&col_max);
    col_range = col_max-col_min;
    if (ElleUnodesActive()) {
        options->GetUnodeAttribOption(&attr);
        options->GetUnodeAttribMinOption(&min);
        options->GetUnodeAttribMaxOption(&max);
        options->GetUnodePlotOption(&triangles);
        /*
         * triangulation not done if only unode locations requested
         */
        if (attr!=U_LOCATION && triangles==1) {
            maxf = ElleMaxFlynns();
            /*
             * the line below was put in to stop drawing of the
             * clockwise grain in a "single" grain file
             */
            if (ElleNumberOfGrains()<3) maxf=1;
            for (i=0;i<maxf;i++) {
                if (ElleFlynnIsActive(i)) {
                    ElleClearTriAttributes();
                    if (ElleNodeMarginAttributeActive(attr))
                        ElleSetTriReferenceId(i);
                    /*TriangulateUnodes(i,MeshData.tri);*/
                    TriangulateUnodes(i,MeshData.tri,attr);
                    ElleSetTriAttributeFromNodes(0,attr);
                    ElleSetTriAttributeMin(0,min);
                    ElleSetTriAttributeMax(0,max);
                    EllePlotTriangles();
                }
            }
            ElleClearTriAttributes();
        }
        else DrawUnodes(col_range,col_min,attr,min,max);
    }
}

void PlotUnodes(int col_range, int col_min,
                int attr, double min, double max)
{
    int col=E_DFLT_DNODE_COL, err=0, umax=0, i;
    double val=0.0, val2=0.0, range=0.0;
    Coords xy;

    umax=ElleMaxUnodes();
    float ptrad=ElleGetPointRadius(umax);
    range = max-min;
    if (attr!=U_LOCATION && range==0)
        UnodeAttributeRange(attr,&min,&max);
    /*for (std::vector<Unode>::iterator it=Unodes->begin();*/
    /*it!=Unodes->end();it++) {*/
    for (i=0; i<umax; i++) {
        if (range!=0) {
            switch(attr) {
            case CAXIS: err= ElleGetUnodeAttribute(i,&val,&val2,attr );
                break;
            case E3_ALPHA:
            case EULER_3: err= ElleGetUnodeAttribute(i, &val,attr);
                        //val=fmod((val+90.0),30.0)/15;
                        //val=fmod(val,180.0)/90;
                        //if(val > 1.0) val=2.0-val;
                break;
            default: err= ElleGetUnodeAttribute(i, &val,attr);
                break;
            }
            col = (int)((val-min)/range * col_range) + col_min;
            if (col<col_min) col=col_min;
            else if (col>col_min+col_range) col=col_min+col_range;
        }
        /*it->getPosition(&xy);*/
        ElleGetUnodePosition(i,&xy);
        ElleMarkNode(i,&col,(float)xy.x,(float)xy.y,ptrad);
    }
}

void PlotSegments(int col_range, int col_min,
                int attr, double min, double max)
{
    int col=E_DFLT_DNODE_COL, err=0, smax=0, i;
    double val=0.0, val2=0.0, range=0.0;
    Coords xy;

    smax=ElleMaxSegments();
    float ptrad=ElleGetPointRadius(smax);
    range = max-min;
    if (attr!=S_LOCATION && range==0)
        SegmentAttributeRange(attr,&min,&max);
    for (i=0; i<smax; i++) {
        if (range!=0) {
            err= ElleGetSegmentAttribute(i, &val,attr);
            col = (int)((val-min)/range * col_range) + col_min;
            if (col<col_min) col=col_min;
            else if (col>col_min+col_range) col=col_min+col_range;
        }
        ElleGetSegmentPosition(i,&xy);
        ElleMarkNode(i,&col,(float)xy.x,(float)xy.y,ptrad);
    }
}

void EllePlotSegments()
{
    int i, maxf;
    int col=7, attr=S_LOCATION, triangles=0;
    int col_range, col_max = E_COL_MAX, col_min = E_COL_MIN;
    double max=0.0,min=0.0;

    options->GetPlotColourRange(&col_min,&col_max);
    col_range = col_max-col_min;
    if (ElleMaxSegments()>0) {
        options->GetSegmentAttribOption(&attr);
        options->GetSegmentAttribMinOption(&min);
        options->GetSegmentAttribMaxOption(&max);
        DrawSegments(col_range,col_min,attr,min,max);
    }
}

float ElleGetPointRadius(int max)
{
    float ptrad;

    if (max > 10000) ptrad = 0.002;
    else if (max > 5000) ptrad = 0.003;
    else if (max > 1000) ptrad = 0.004;
    else ptrad = 0.005;
    return(ptrad);
}

int ElleMarkNode(int dd,int *col,float x,float y,float size)
{
    int outline=0;
    float xpts[4],ypts[4];

    xpts[0] = x;
    ypts[0] = y;
    DrawNode(dd,&xpts[0],&ypts[0],&size,col,&outline);
    return(0);
}

void ElleRescaleWin()
{
    float xmin, xmax, ymin, ymax;
    CellData unitcell;

    ElleCellBBox(&unitcell);
    xmin = (float)unitcell.cellBBox[BASELEFT].x;
    xmax = (float)unitcell.cellBBox[TOPRIGHT].x;
    if (unitcell.cellBBox[3].x<xmin) {
        xmin = (float)unitcell.cellBBox[TOPLEFT].x;
        xmax = (float)unitcell.cellBBox[BASERIGHT].x;
    }
    ymin = (float)unitcell.cellBBox[BASELEFT].y;
    ymax = (float)unitcell.cellBBox[TOPLEFT].y;

    RescaleBox(xmin,xmax,ymin,ymax);
}

/*
 * display stuff to be updated when a new file has been opened
 */
void ElleUpdateSettings()
{
    int draw_attrib,choice,max;

    ElleRescaleWin();
    options->GetFlynnAttribOption(&draw_attrib);
    // This needs a rethink - display attribs may differ from file attribs
    if (draw_attrib==EULER_RGB) draw_attrib=EULER_3;
    if (draw_attrib!=NONE && !ElleFlynnAttributeActive(draw_attrib))
        options->SetFlynnAttribOption(NONE,0.0,0.0);
    options->GetNodeAttribOption(&draw_attrib);
    if (draw_attrib!=NONE &&
            draw_attrib!=NUM_NB && draw_attrib!=TRPL_ONLY &&
            !ElleNodeAttributeActive(draw_attrib))
        options->SetNodeAttribOption(NUM_NB,0.0,0.0);
    options->GetUnodeAttribOption(&draw_attrib);
    // This needs a rethink - display attribs may differ from file attribs
    if (draw_attrib==EULER_RGB) draw_attrib=E3_ALPHA;
    if (draw_attrib!=NONE && draw_attrib!=U_LOCATION &&
            !ElleUnodeAttributeActive(draw_attrib))
        options->SetUnodeAttribOption(NONE,0.0,0.0);
    options->GetBndAttribOption(&draw_attrib);
    if (draw_attrib==MISORIENTATION && !ElleFlynnAttributeActive(E3_ALPHA))
        options->SetBndAttribOption(STANDARD,0.0,0.0);
    options->GetBndLevelOption(&choice);
    max=ElleMaxLevels();
    if (max<choice || (!choice && max))
        options->SetBndLevelOption(max);
    ElleSetRescale(0);
}

#else
#include "log.h"

void ElleUpdateDisplay()
{
}

void Log( int loglevel, const char *msg )
{
    fprintf(stderr, "%s\n",msg);
}

void ElleUpdateSettings()
{
}

void ElleShowStages(int stages)
{
}

void Init_Data(void)
{
}

void EllePlotRegions(int count)
{
}

void ElleRescaleWin()
{
}

// the 2 dummy fns below are only necessary for diffusion
void GetUpdateFreq(int *val)
{
    *val=0;
}

void EllePlotTriangles()
{
}

#endif

