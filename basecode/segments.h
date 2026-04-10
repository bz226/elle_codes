#ifndef _E_segments_h
#define _E_segments_h

#include "segmentsP.h"

int ElleWriteSegmentData(const char *fname);
int ElleReadSegmentData(FILE *fp, char str[]);

void ElleInitSegmentAttribute(int id);
int SegmentAttributeRange(int attr,double *min,double *max);
int ElleReadSegmentData(FILE *fp, char str[]);
int ElleReadSegmentAttribData(FILE *fp, char str[], int attr_id);

int ElleMaxSegments();
bool ElleGetSegmentPosition(int id, Coords *segment_pos);
int ElleGetSegmentAttribute(int id, double *val, const int attrib_id);
void SegmentAttributeList(int **attr_ids, int *maxa);


#endif
