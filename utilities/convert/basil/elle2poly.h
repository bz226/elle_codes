#ifndef C_elle2poly_h
#define C_elle2poly_h

#define MAX_INTERIOR_PTS 40

typedef struct {
    int active;
    int segment[2];
    int num_interior_pts;
    Coords interior_pt[MAX_INTERIOR_PTS];
    double value;
} RegionAttribute;

typedef struct {
    int maxgrains;
    RegionAttribute *elems;
} RegionAttribArray;

#ifdef __cplusplus
extern "C" {
#endif
int CnvtCountNodes();
void CnvtCheckNodes();
int CnvtReadData( FILE *fp, char *str);
int CnvtReadSegmentData(FILE *fp, char str[]);
int CnvtWriteNodeData(FILE *fpout,int numnodes,char *str,int *polynum);
int CnvtWriteSegmentData(FILE *fpout,int numsegs,char *str,
                         int *polynum, int numnodes);
void CnvtSetLimits(Coords *limits);
void CnvtInitCorners();
void CnvtAddCorners(Coords*limits);
int CnvtNodeOnCorner(int node);
void CnvtJoinCorners();
int CnvtNearestCorner(int node);
int CnvtCheckSegments(int node, Coords *limits);
int CnvtFindIntersection( Coords *prev, Coords *xy, Coords *newxy,
                          int *found);
void CnvtInitRegionAttribArray();
int CnvtReallocateRegionAttribArray(int size);
void CnvtSetInteriorPts();
void CnvtSetRegionAttributeValue(int index, double value);
void CnvtSetRegionInteriorPt(int index, Coords *pos);
void CnvtSetRegionSegment(int index, int start, int end);
void CnvtRegionAttributeValue(int index, double *val);
int CnvtRegionNumInteriorPts(int index);
int CnvtRegionInteriorPt(int index, int ptindex, Coords *pos);
void CnvtRegionSegment(int index, int *seg);
int CnvtAttributeEntryIsActive(int index);
int CnvtMaxAttributeEntries();
void CnvtFindCorners(int node);
int CnvtNodeCheck(int node);
void CnvtInsertBoundaryNodes();
void CnvtClearRgnAttribArray(RegionAttribArray *attr);
void CnvtFindInteriorPt(int node1, int node2, Coords *pt);
int CnvtBoundaryCheck(Coords *xy);
int CnvtSegmentWrap(int node,int nb,
                    Coords *xy,Coords *prev);
int CnvtOutsideLimits(Coords *xy);
void CnvtRenumber(int numnodes, int **polynum);
int CnvtLocationMatch(Coords *loc1, Coords *loc2);
int CnvtWhichBoundary(Coords *xy);
void CnvtFindMirrorPosition(Coords *xy,Coords *newxy);
void CnvtInitMirrorNodes();
int CnvtSetMirrorNode(int node, int mirror);
int CnvtGetMirrorNode(int node);
int CnvtCorner(int index);
void CnvtAddBoundaryNode(int nb, Coords*newxy, int *newn);
int CnvtNodeCount();
int CnvtSegmentCount();
#ifdef __cplusplus
}
#endif
#endif
