 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef ES_stats_h
#define ES_stats_h
/*!
 	\file		stats.h
 	\brief		Header for statistics functions
 	\par		Description:
  					declares types used in statistics functions:
  					struct stats_params
  					NodeAttributeStats
  					FoliationData
 
  					Statistic Function Declarations
 */
struct stats_params {
    char file[FILENAME_MAX];
    int frequency;
    unsigned char nodes;
    unsigned char grains;
    unsigned char av_area;
    unsigned char bnd_length;
    unsigned char sides_m2;
    unsigned char areas_m2;
};

typedef struct {
    double average;
    double min;
    double max;
} NodeAttributeStats;

typedef struct {
    float maxAng, minAng;
    float accuracy;
    float gblength;
    float ratio;
} FoliationData;

int ES_PanozzoAnalysis(int fid, FoliationData *data);
#ifdef __cplusplus
extern "C" {
#endif
void ES_Init_Stats();
int ES_statsInterval();
int ES_WriteStatistics(int count);
void ES_SetStatsFileRoot(const char *fname);
void ES_SetstatsInterval(int val);
int ES_statsInterval();
void ES_SetSaveNodes(int val);
int ES_SaveNodes();
void ES_SetSaveGrains(int val);
int ES_SaveGrains();
void ES_SetSaveavArea(int val);
int ES_SaveavArea();
void ES_SetSaveBLength(int val);
int ES_SaveBLength();
void ES_SetSavem2Sides(int val);
int ES_Savem2Sides();
void ES_SetSavem2Areas(int val);
int ES_Savem2Areas();
int ES_NodeAttributeStats(int attrib_index,
                          NodeAttributeStats *n_stats);
float ES_BoundaryLength();
#ifdef __cplusplus
}
#endif

#endif
