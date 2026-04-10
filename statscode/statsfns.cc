 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "attrib.h"
#include "bflynns.h"
#include "nodes.h"
#include "polygon.h"
#include "interface.h"
#include "errnum.h"
#include "error.h"
#include "stats.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/
#define STATS_EXT ".stats"

struct stats_params Stats;

#ifdef __cplusplus
extern "C" {
#endif
double ES_AverageGrainArea();
float ES_SecondMomentGrainSides();
float ES_SecondMomentGrainArea();
int PanozzoAnalysis(FoliationData *data);
#ifdef __cplusplus
}
#endif

/*!
	\brief		[Initialise statistics]
	\par		Description:
	Initialises the default name for the statistics output file
	to "tmp" 
	Initialises the struct stats_data
	\remarks
		This function is called by Elle_Init()
*/
void ES_Init_Stats()
{

    ES_SetStatsFileRoot("tmp");
    Stats.frequency=0;
    Stats.nodes=1;
    Stats.grains=1;
    Stats.av_area=1;
    Stats.bnd_length=1;
    Stats.sides_m2=1;
    Stats.areas_m2=1;
}

void ES_SetStatsFileRoot(const char *fname)
{
    strcpy(Stats.file,fname);
    strcat(Stats.file,STATS_EXT);
}

char *ES_StatsFilename()
{
    return(Stats.file);
}

void ES_SetstatsInterval(int val)
{
    Stats.frequency= val;
}

int ES_statsInterval()
{
    return(Stats.frequency);
}

void ES_SetSaveNodes(int val)
{
    Stats.nodes = val;
}

int ES_SaveNodes()
{
    return(Stats.nodes);
}

void ES_SetSaveGrains(int val)
{
    Stats.grains = val;
}

int ES_SaveGrains()
{
    return(Stats.grains);
}

void ES_SetSaveavArea(int val)
{
    Stats.av_area = val;
}

int ES_SaveavArea()
{
    return(Stats.av_area);
}

void ES_SetSaveBLength(int val)
{
    Stats.bnd_length= val;
}

int ES_SaveBLength()
{
    return(Stats.bnd_length);
}

void ES_SetSavem2Sides(int val)
{
    Stats.sides_m2 = val;
}

int ES_Savem2Sides()
{
    return(Stats.sides_m2);
}

void ES_SetSavem2Areas(int val)
{
    Stats.areas_m2 = val;
}

int ES_Savem2Areas()
{
    return(Stats.areas_m2);
}

int ES_WriteStatistics(int count)
{
    int djs, tjs;
    int err=0;
    FILE *fp;
    FoliationData fol_data;

    /*
     * this routine needs expanding to access the stats flags
     * and only calculate and write required values.
     * Should also check for write errors
     * change to iostream calls
     */
    if (count<ES_statsInterval()) {
        if ((fp=fopen(ES_StatsFilename(),"w"))==0)
            OnError(ES_StatsFilename(),OPEN_ERR);
        fprintf(fp,"*\n"); /* for cricket graph */
        fprintf(fp,"%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s",
                "time","Grains","DJs","TJs","BLength","Area","m2_sides","m2_area");
        fprintf(fp,"\t%s\t%s\t%s\t%s\t%s",
                "minang","maxang","ratio","blen","acc");
        fprintf(fp,"\n");
    }
    else if ((fp=fopen(ES_StatsFilename(),"a"))==0)
        OnError(ES_StatsFilename(),OPEN_ERR);
    ElleNumberOfNodes(&djs, &tjs);
    fprintf(fp,"%6d\t%8d\t%8d\t%8d\t%.2f\t%12.10f\t%.3f\t%.3f",
            count,ElleNumberOfGrains(),djs,tjs,ES_BoundaryLength(),
            ES_AverageGrainArea(),
            ES_SecondMomentGrainSides(),ES_SecondMomentGrainArea());
    err = PanozzoAnalysis(&fol_data);
    fprintf(fp,"\t%.1f\t%.1f\t%.3f\t%.2f\t%.2f",fol_data.minAng,
            fol_data.maxAng,fol_data.ratio,fol_data.gblength,
            fol_data.accuracy);
    fprintf(fp,"\n");
    fclose(fp);
    return(err);
}

double ES_AverageGrainArea()
{
    /*
     * assumes unit cell area is 1.0
     */
    int numgrains;

    numgrains = ElleNumberOfGrains();
    return ((float)(1.0/numgrains));
}

float ES_BoundaryLength()
{
    int max, nbnodes[3], i, j;
    double total;
    
    total = 0.0;
    max = ElleMaxNodes();
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            ElleNeighbourNodes(i,nbnodes);
            for (j=0;j<3;j++) {
                if (nbnodes[j]!=NO_NB && nbnodes[j]>i)
                    total += ElleNodeSeparation(i,nbnodes[j]);
            }
        }
    }
/*
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            ElleNeighbourNodes(i,nbnodes);
            for (j=0;j<3;j++) {
                if (nbnodes[j]!=NO_NB)
                    total += ElleNodeSeparation(i,nbnodes[j]);
            }
        }
    }
    total /= 2.0;
*/
    return((float)total);
}

#define MAXSIDES 25

float ES_SecondMomentGrainSides()
{
    int i, j, max, numsides[MAXSIDES], *ids, numnodes;
    int total, numdj;

    for (i=0;i<MAXSIDES;i++) numsides[i] = 0;
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i)) {
            ElleFlynnNodes(i,&ids,&numnodes);
            /* only include triple junctions */
            for (j=0,numdj=0;j<numnodes;j++)
                if (ElleNodeIsDouble(ids[j])) numdj++;
            numnodes -= numdj;
            if (numnodes>MAXSIDES) numnodes=MAXSIDES;
            numsides[numnodes-1] += 1;
            free(ids);
        }
    }
    for (i=0,total=0;i<MAXSIDES;i++) {
        j = i+1;
        total += (j-6) * (j-6) * numsides[i];
    }
    return((float)total/ElleNumberOfGrains());
}

float ES_SecondMomentGrainArea()
{
    int i, max;
    double total, area, av_area;
    ERegion reg;

    av_area = ES_AverageGrainArea();
    max = ElleMaxFlynns();
    for (i=0,total=0.0;i<max;i++) {
        if (ElleFlynnIsActive(i)) {
            reg = i;
            area = ElleRegionArea(reg);
            total += ((area/av_area - 1) *(area/av_area - 1));
        }
    }
    return((float)(total/ElleNumberOfGrains()));
}


int PanozzoAnalysis(FoliationData *data)
{
    int max, i, j, k, nbnodes[3];
    double len, totlen, dx, dy;
    double rosedata[181], pabs, p,alfa, cosalfa;
    double minL, maxL, minA, maxA;
    Coords xy, xynb;

    totlen = 0.0;
    for (i=0; i<181; i++) rosedata[i] = 0.0;

    max = ElleMaxNodes();
    for (i=0; i<max; i++) {
        if (ElleNodeIsActive(i)) {
            ElleNeighbourNodes(i,nbnodes);
            for (j=0;j<3;j++) {
                if (nbnodes[j]!=NO_NB && nbnodes[j]>i) {
                    ElleNodePosition(i,&xy);
                    ElleNodePlotXY(nbnodes[j],&xynb,&xy);
                    dx = (xy.x - xynb.x);
                    dy = (xy.y - xynb.y);
                    len = sqrt(dx*dx + dy*dy);
                    if (len > 0.0) {
                        totlen += len;
                        cosalfa = dy/len;
                        alfa = acos(cosalfa);
                        if (dx < 0.0) alfa = M_PI-alfa;
                        for (k=0;k<180;k++) {
                            pabs = cos(alfa - (double)k/180.0*M_PI);
                            p = len * fabs(pabs);
                            rosedata[k] += p;
                        }
                    }
                    else
                        printf("PanozzoAnalysis-len 0",0);
                }
            }
        }
    }
    minL = maxL = rosedata[0];
    minA = maxA = 0;
    for (i=0; i<180; i++) {
        if (rosedata[i] < minL) {
            minL = rosedata[i];
            minA = (double)i;
        }
        if (rosedata[i] > maxL) {
            maxL = rosedata[i];
            maxA = (double)i;
        }
    }
    if (maxL > 0.0)
        for (i=0; i<181; i++) rosedata[i] /= maxL;
    if (minL > 0.0) data->ratio = (float)(maxL/minL);
    data->maxAng = (float)maxA;
    data->minAng = (float)minA;
    data->gblength = totlen;
    if (maxA > minA) data->accuracy = (float)fabs(maxA-minA-90.0);
    else data->accuracy = (float)fabs(minA-maxA-90.0);
}

int ES_PanozzoAnalysis(int id,FoliationData *data)
{
    int max, i, j, k, nbnodes[3];
    double len, totlen, dx, dy;
    double rosedata[181], pabs, p,alfa, cosalfa;
    double minL, maxL, minA, maxA;
    Coords xy, xynb;
    std::vector<int>flynn_nodes;
                                                                                
    totlen = 0.0;
    for (i=0; i<181; i++) rosedata[i] = 0.0;
                                                                                
    ElleFlynnNodes(id,flynn_nodes);
    max = flynn_nodes.size();
    for (i=0; i<max; i++) {
        if (ElleNodeIsActive(flynn_nodes[i])) {
            ElleNeighbourNodes(flynn_nodes[i],nbnodes);
            for (j=0;j<3;j++) {
                if (nbnodes[j]!=NO_NB && nbnodes[j]!=flynn_nodes[i]) {
                    ElleNodePosition(flynn_nodes[i],&xy);
                    ElleNodePlotXY(nbnodes[j],&xynb,&xy);
                    dx = (xy.x - xynb.x);
                    dy = (xy.y - xynb.y);
                    len = sqrt(dx*dx + dy*dy);
                    if (len > 0.0) {
                        totlen += len;
                        cosalfa = dy/len;
                        alfa = acos(cosalfa);
                        if (dx < 0.0) alfa = M_PI-alfa;
                        for (k=0;k<180;k++) {
                            pabs = cos(alfa - (double)k/180.0*M_PI);
                            p = len * fabs(pabs);
                            rosedata[k] += p;
                        }
                    }
                    else
                        OnError("PanozzoAnalysis-len 0",0);
                }
            }
        }
    }
    minL = maxL = rosedata[0];
    minA = maxA = 0;
    for (i=0; i<180; i++) {
        if (rosedata[i] < minL) {
            minL = rosedata[i];
            minA = (double)i;
        }
        if (rosedata[i] > maxL) {
            maxL = rosedata[i];
            maxA = (double)i;
        }
    }
    if (maxL > 0.0)
        for (i=0; i<181; i++) rosedata[i] /= maxL;
    if (minL > 0.0) data->ratio = (float)(maxL/minL);
    data->maxAng = (float)maxA;
    data->minAng = (float)minA;
    data->gblength = totlen;
    if (maxA > minA) data->accuracy = (float)fabs(maxA-minA-90.0);
    else data->accuracy = (float)fabs(minA-maxA-90.0);
}


int ES_NodeAttributeStats(int attrib_index,NodeAttributeStats *n_stats)
{
    unsigned char first=1;
    int err=0;
    int max, index, i, num_active=0;
    double total=0, val_min=0, val_max=0, val;

    index = ElleFindNodeAttribIndex(attrib_index);
    if (index==NO_NB) err=ATTRIBID_ERR;
    else {
        max = ElleMaxNodes();
        for (i=0; i<max; i++) {
            if (ElleNodeIsActive(i)) {
                val = ElleNodeAttribute(i,attrib_index);
                total += val;
                if (first) {
                    val_min = val_max = val;
                    first = 0;
                }
                else {
                    if (val<val_min) val_min = val;
                    if (val>val_max) val_max = val;
                }
                num_active++;
            }
        }
        if (num_active>0) n_stats->average = total/num_active;
        n_stats->min = val_min;
        n_stats->max = val_max;
    }
    return(err);
}
