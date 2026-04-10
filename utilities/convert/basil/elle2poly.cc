#include <stdio.h>
#include <stdlib.h>
#include "nodes.h"
#include "version.h"
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include "general.h"
#include "file.h"
#include "file_utils.h"
#ifdef FLYNN_VER
#include "interface.h"
#include "bflynns.h"
#else
#include "grains.h"
#endif
#include "regions.h"
#include "runopts.h"
#include "unodes.h"
#include "flags.h"
#include "error.h"
#include "poly.h"
#include "polygon.h"
#include "crossings.h"
#include "elle2poly.h"
#include "bndlist.h"
#include "gpcclip.h"
#include "check.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;

#define POLY_FILE_EXT  ".poly"

RegionAttribArray RgnAttributes;

#define PBOPT       "pb"  // periodic boundaries
#define V2FOPT      "v2f"  // convert unode voronoi regions to flynns
#define NOTOPO      "notopo"  // no topology checks

#define MIRROR_MAX  25000
int Mirror_nodes[MIRROR_MAX][2];
int Corners[4], CornersRepos[4];
double Vertices[4][2];

#ifdef __cplusplus
extern "C" {
#endif
int ElleToPoly( char *,FILE *fpout,int perbnd,bool unode2flynn,bool notopo,
                int *maxorig,int **originalnodes,
                int *numnodes,int **polynumber );
void CnvtCreateBoundaryList(int perbnd);
void CnvtJoinBoundarySegments();
int CnvtSegments();
int CnvtPointOnCorner(Coords *ptxy, int *cnrnode);
int CnvtWriteRegionAttributeData(FILE *fpout, char *str);
int CnvtWriteAttributeData(FILE *fpout, char *str);
int CnvtWriteNodeNumberReference(FILE *fpout,
                                  int maxorig, int *originalnodes,
                                  int numnodes,int *polynumber);
int CnvtWriteUnodeViscosity(FILE *fpout);
void CnvtMarkOriginalNodes(int *maxnodes, int **orignodes);
int CnvtNodeOnCornerRepos(int node);
void CnvtFindCornersRepos();

extern int ElleInit();
#ifdef __cplusplus
}
#endif
void ElleCheckUnit(vector<Coords> &pts, int *xflags,int *yflags,
                   Coords *bl,Coords *tr);


int main(int argc,char **argv)
{
    char infile[FILENAME_MAX], outfile[FILENAME_MAX];
    int err=0, perbnd=0;
    bool unode2flynn=false;
    bool notopo=false;
    int *originalnodes=0, maxorig;
    int *polynumber=0, numnodes;
    double tmp;
    FILE *fpin, *fpout;
    const char *cp;

    if (argc==1) {
        fprintf(stderr,"No input file specified\n");
        fprintf(stderr,"Syntax is \n");
        fprintf(stderr,"\t%s input_file [output_file]\n",argv[0]);
        fprintf(stderr,"(output_file defaults to input_file%s)\n",
                                                   POLY_FILE_EXT);
        return(1);
    }
    /*
     * only infile specified so outfile defaults to
     * infile concatenated with POLY_FILE_EXT
     */
    if (argc==2) {
        strcpy(infile,argv[1]);
        strcpy(outfile,argv[1]);
        strcat(outfile,POLY_FILE_EXT);
    }
    if (argc>2) {
        strcpy(infile,argv[1]);
        strcpy(outfile,argv[2]);
    }
    if ((fpin=fopen(infile,"r"))==NULL) {
        fprintf(stderr,"Could not open input file %s\n",infile);
        return(1);
    }
    fclose(fpin);
    if ((fpout=fopen(outfile,"w"))==NULL) {
        fprintf(stderr,"Could not open output file %s\n",outfile);
        return(1);
    }
    if (argc>3) {
        if (!strcmp(argv[3],PBOPT)) perbnd = 1;
        else if (!strcmp(argv[3],V2FOPT)) unode2flynn = true;
        else if (!strcmp(argv[3],NOTOPO)) notopo = true;
    }
    if (argc>4) {
        if (!strcmp(argv[4],PBOPT)) perbnd = 1;
        else if (!strcmp(argv[4],V2FOPT)) unode2flynn = true;
        else if (!strcmp(argv[4],NOTOPO)) notopo = true;
    }

if (notopo)
printf("notopo is true\n");
else
printf("notopo is false\n");
    ElleInit();
    ElleSetDisplay(0);
    ElleSetAppName(argv[0]);

    CnvtInitBoundaryFlagArray();
    /*CnvtInitCheckFlagArray();*/
    CnvtInitRegionAttribArray();

    err = ElleToPoly(infile,fpout,perbnd,unode2flynn,notopo,
                     &maxorig,&originalnodes,&numnodes,&polynumber);
    if (err) OnError("",err);

    /* write elle filename in poly file */
    if (fprintf(fpout,"%s ",ELLE_KEY) < 0) OnError(outfile,WRITE_ERR);
    if (fprintf(fpout,"%s\n",infile) < 0) OnError(outfile,WRITE_ERR);

    /* write an elle/poly node number cross-reference in poly file */
    if (err=CnvtWriteNodeNumberReference(fpout,maxorig,originalnodes,
                                       numnodes,polynumber))
        OnError(outfile,err);
    /* write unode positions and viscosity, if present
	if (ElleUnodesActive() && ElleUnodeAttributeActive(U_VISCOSITY))
		if (err=CnvtWriteUnodeViscosity(fpout)) OnError(outfile,err);
    */
    /* write a suggested max triangle area in poly file */
    if (fprintf(fpout,"%s ",AREA_KEY) < 0) OnError(outfile,WRITE_ERR);
    if (unode2flynn) {
        tmp = ElleminNodeSep()*ElleminNodeSep();
        tmp = EllemaxNodeSep()*EllemaxNodeSep();
    }
    else {
        tmp = EllemaxNodeSep()*EllemaxNodeSep();
    }
    if (tmp < AREA_MIN) tmp = AREA_MIN;
	/////NOTE large tmp
	/////tmp = 0.005;
    if (fprintf(fpout,"%f\n",tmp) < 0)
        OnError(outfile,WRITE_ERR);

    /* write a suggested triangle quality in poly file */
	/////NOTE 5
    /////if (fprintf(fpout,"%s %d\n",QUALITY_KEY,5) < 0)
    if (unode2flynn) {
      if (fprintf(fpout,"%s %d\n",QUALITY_KEY,15) < 0)
		OnError(outfile,WRITE_ERR);
    }
    else {
      if (fprintf(fpout,"%s %d\n",QUALITY_KEY,20) < 0)
		OnError(outfile,WRITE_ERR);
    }
    /*
     * need to change the io in this programme to iostream
     * syntax. Til then, need to use pointer from string object
     */
    string outversion = ElleGetCreationString();
    if (fprintf(fpout,"%s\n",outversion.c_str()) < 0)
        OnError(outfile,WRITE_ERR);

    fclose(fpout);

    CnvtClearBoundaryFlagArray();
    /*CnvtClearCheckFlagArray();*/
    CnvtClearRgnAttribArray(&RgnAttributes);
    delete[] polynumber;
    delete[] originalnodes;
    if (err) OnError("",err);

    return(0);
}

int ElleToPoly( char *infile,FILE *fpout,
                int perbnd, bool unode2flynn, bool notopo,
                int *maxorig,int **originalnodes,
                int *numnodes,int **polynumber )
{
    char str[80];
    int err=0, end=0;
    int numsegs;

    /*if (err = CnvtReadData(fpin, str)) return(err);*/
    if (err = ElleReadData(infile)) return(err);
    if ((ElleNumberOfGrains() > 0) && (notopo == false)) {
        if (err = ElleFlynnTopologyCheck()) return(err);
        ElleWriteData(infile); //in case topology check changed data
    }
    else if (unode2flynn) {
        if ((err=Voronoi2Flynns(false))!=0)
            return(err);
        /*if (err = ElleFlynnTopologyCheck()) return(err);*/
/*
    int k;
    int numnodes=ElleMaxNodes();
    for (k=0;k<numnodes;k++) {
        if (ElleNodeIsActive(k)) {
            if (ElleNodeIsDouble(k)) {
                ElleCheckDoubleJ(k);
            }
            else if (ElleNodeIsTriple(k)) {
                ElleCheckTripleJ(k);
            }
        }
    }
*/
        ElleWriteData("vor2flynn.elle");
        strcpy(infile,"vor2flynn.elle");
    }

    CnvtMarkOriginalNodes(maxorig,originalnodes);
    CnvtCheckNodes();
    CnvtSetInteriorPts();

    CnvtSegments();

    /*
     * should this routine check for and remove bnd nodes which
     * are too close together? esp close to corners
     */
    CnvtCreateBoundaryList(perbnd);
    CnvtJoinBoundarySegments();
    *numnodes = CnvtNodeCount();
    numsegs = CnvtSegmentCount();

    CnvtRenumber(*numnodes,polynumber);

    /* nodes */
    if (err = CnvtWriteNodeData(fpout,*numnodes,str,*polynumber))
        return(err);
    /* segments */
    if (err = CnvtWriteSegmentData(fpout,numsegs,str,
                                   *polynumber,*numnodes)) return(err);
    /* holes */
    if (fprintf(fpout,"0\n")<0) return(WRITE_ERR);
    /* attributes */
    if (err = CnvtWriteRegionAttributeData(fpout,str)) return(err);

    return(err);
}

int CnvtReadData( FILE *fp, char *str)
{
    char *start, *ptr;
    int num, key, keys[20], finished=0, err=0;
    int c;
    int i, j, max;
    int nbnodes[3];
    ERegion full_id[3];

    if ((num = fscanf(fp,"%s", str))!=1) return(READ_ERR);
    while (!feof(fp) && !finished) {
        /*
         * find keywords
         */
        validate(str,&key,FileKeys);
        switch( key ) {
        case E_OPTIONS : if (err = ElleReadOptionData(fp,str))
                           finished = 1;
                       break;
        case FLYNNS  :
        case REGIONS :
                       if (err = CnvtReadSegmentData(fp,str))
                           finished = 1;
                       break;
        case LOCATION: if (err = ElleReadLocationData(fp,str))
                           finished = 1;
                       break;
        case UNODES  : 
                       /*
                        * read the order of attribute vals
                        * old code (pre 2.3.4) had attributes on 
                        * same line as position data
                        */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          start = str;
                          while ((ptr=strtok(start," \t\r\n")) && 
                                  i<20 && !finished) {
                              validate(ptr,&keys[i],FileKeys);
                              if (keys[i]<0) finished = 1;
                              else {
                                  start = 0;
                                  i++;
                              }
                           }
                       }
                       if (err = ElleReadUnodeData(fp,str,keys,i))
                           finished = 1;
                       break;
        case VISCOSITY:
        case S_EXPONENT:
        case F_ATTRIB_A:if (err = ElleReadFlynnRealAttribData(fp,str,key))
                           finished = 1;
                       break;
        case EXPAND:   if (err = ElleReadFlynnIntAttribData(fp,str,key))
                           finished = 1;
                       break;
        case U_VISCOSITY:
                       if (err=ElleReadUnodeRealAttribData(fp,str,&key,1))
                            finished = 1;
                       break;
        case COMMENT : dump_comments(fp);
                       num = fscanf(fp,"%[^A-Za-z#]",str);
                       num = fscanf(fp,"%s",str);
                       break;
        default:       ElleSkipSection(fp,str,FileKeys);
                       break;
        }
    }
    /*fclose(fp);*/
    return(err);
}

int CnvtReadSegmentData(FILE *fp, char str[])
{
    int err=0;

#ifdef FLYNN_VER
    if (err = ElleReadFlynnData(fp,str)) return(err);
#else
    if (err = ElleReadGrainData(fp,str)) return(err);
    ElleSetRegionFirstNodes();
#endif
    return(err);
}

int CnvtWriteNodeData(FILE *fpout,int numnodes,char *str,int *polynum)
{
    int max,i,k,err=0,n=0;
    Coords xy;

    if (fprintf(fpout,"%d 2 0 1\n",numnodes)<0) return(WRITE_ERR);
    for (k=0;k<numnodes;k++) {
        i = polynum[k];
        if (ElleNodeIsActive(i)) {
            ElleNodePosition(i,&xy);
            if (fprintf(fpout,"%d %.8f %.8f %d\n",k+1,xy.x,xy.y,
                                 CnvtNodeOnBoundary(i))<0)
                return(WRITE_ERR);
            n++;
        }
    }
    if (n!=numnodes) err = NODECNT_ERR;
printf("no. of nodes written = %d\n",numnodes);
    return(err);
}

int CnvtWriteSegmentData(FILE *fpout,int numsegs,char *str,
                         int *polynum, int numnodes)
{
    /*
     * write segment data in the form expected by triangle code
     */
    unsigned char valid, one_crnr;
    int max,i,j,k,err=0,n=0,nbnodes[3],bnd,nbi;
    int bndn,bndnb;
    int n_on_crnr=0, nb_on_crnr=0;
    Coords xy1,xy2;
    CellData unit;

    ElleCellBBox(&unit);

    /* no. of segments + boundary mark flag */
    if (fprintf(fpout,"%d 1\n",numsegs)<0) return(WRITE_ERR);
    for (k=0,n=1;k<numnodes;k++) {
        i = polynum[k];
        if (ElleNodeIsActive(i)) {
            ElleNeighbourNodes(i,nbnodes);
            for (j=0;j<3;j++) {
                if (nbnodes[j]!=NO_NB) {
                    bnd = ((bndn=CnvtNodeOnBoundary(i)) &&
                              (bndnb=CnvtNodeOnBoundary(nbnodes[j])));
    /*
     * if the nodes are on different boundaries then 
     * one node should be a corner
     * if nodes are joined across a crnr, segment is not bnd segment
     * Triangle marks nodes inserted on bnd segment as bnd nodes
     */
                    if (bnd) {
                        bnd=bndn;
                        if (bndn!=bndnb) {
                            bnd = ((n_on_crnr=CnvtNodeOnCorner(i))||
                                (nb_on_crnr=CnvtNodeOnCorner(nbnodes[j])));
                            if (bnd && n_on_crnr) bnd=bndnb;
                            else if (bnd && nb_on_crnr) bnd=bndn;
                        }
                    }
    /*
     * segment no. + 2 node nos. + boundary flag
     * node and segment nos. start at 1, so increment before writing
     */
                    nbi=0;
                    while (polynum[nbi]!=nbnodes[j] && nbi<numnodes) nbi++;
                    if (fprintf(fpout,"%d %d %d %d\n",n,k+1,
                                            nbi+1,bnd)<0)
                        return(WRITE_ERR);
    /*
     * test for any remaining wrapped segments
     */
                    ElleNodePosition(i,&xy1);
                    ElleNodePosition(nbnodes[j],&xy2);
                    if ((fabs(xy1.x-xy2.x)>(unit.xlength*0.5) ||
                        fabs(xy1.y-xy2.y)>(unit.ylength*0.5)) && 
                        !CnvtNodeOnCorner(i) &&
                        !CnvtNodeOnCorner(nbnodes[j]))
                      printf("\nnodes too far apart %d %d\n",i,nbnodes[j]);

                    n++;
                }
            }
        }
    }
    if (n!=numsegs+1) err = SEGCNT_ERR;
printf("Segments written = %d\n",n-1);
    return(err);
}

int CnvtWriteRegionAttributeData(FILE *fpout,char *str)
{
    int max,i,j,k,numpts,err=0,n=0;
    float val;
    Coords pos;

    /*
     * write the no. of regions (flynns may wrap)
     * write a line consisting of:
     * entry no.(1..n), xy coords for a point within the region
     *   and the flynn id for all triangles within the region
     */
    max = ElleMaxFlynns();
    for (j=0,n=0;j<max;j++)
        if (ElleFlynnIsActive(j) &&
                                 !isParent(j)) {
            numpts=CnvtRegionNumInteriorPts(j);
            n+=numpts;
        }

    fprintf(fpout,"%d\n",n);
    for (j=0,n=1;j<max;j++) {
        if (ElleFlynnIsActive(j) &&
                                 !isParent(j)) {
            numpts=CnvtRegionNumInteriorPts(j);
            for(k=0;k<numpts;k++) {
                CnvtRegionInteriorPt(j,k,&pos);
                val = (float)j;
                if (fprintf(fpout, "%d %.8lf %.8lf %f -1.0\n",
                                        n++,pos.x,pos.y, val)<0)
                    return(WRITE_ERR);
            }
        }
    }
    err = CnvtWriteAttributeData(fpout,str);
    return(err);
}

int CnvtWriteAttributeData(FILE *fpout,char *str)
{
    char *label;
    int max,i,j,k,numpts,err=0;
    int *active,maxa;
    double val, dflt;

   /*
    * This only writes the flynn viscosity
	* strain-rate v stress exp or
	* dummy attribute A
	* need to put attrib/keyword arrays in sep header and generalise
    */
    /*
     * write the keyword
     * write the default value
     * for each lowest level child (cover the whole unit cell),
     * write a line consisting of:
     *   flynn id and the attribute value for all triangles
     *     with this id as an attribute
     */
    ElleFlynnDfltAttributeList(&active,&maxa);
    max = ElleMaxFlynns();
    for (i=0;i<maxa;i++) {
        switch(active[i]) {
/*
        case VISCOSITY: dflt = (float)ElleDefaultFlynnRealAttribute();
                         fprintf(fpout,"%s\n",VISCOSITY_KEY);
                         fprintf(fpout,"%s %f\n",DEFAULT,dflt);
                         for (j=0;j<max;j++) {
                             if (ElleFlynnIsActive(j) &&
                                     !isParent(j)) {
                                 ElleGetFlynnViscosity(j,&val);
                                 if (val!=dflt) 
                                     if (fprintf(fpout,
                                                 "%d %f\n",j,val)<0)
                                             return(WRITE_ERR);
                             }
                         }
                     break;
*/
        case VISCOSITY:
        case S_EXPONENT:
        case F_ATTRIB_A: dflt = ElleDefaultFlynnRealAttribute(active[i]);
                         if (active[i]==VISCOSITY) label=VISCOSITY_KEY;
						 else if (active[i]==S_EXPONENT) label=SE_KEY;
                         else if (active[i]==F_ATTRIB_A)
                                          label=FLYNN_REAL_ATTRIB_KEY;
                         fprintf(fpout,"%s\n",label);
                         fprintf(fpout,"%s %.8e\n",DEFAULT,dflt);
                         for (j=0;j<max;j++) {
                             if (ElleFlynnIsActive(j) &&
                                     ElleFlynnHasAttribute(j,active[i])) {
                                 ElleGetFlynnRealAttribute(j,&val,active[i]);
                                 if (val!=dflt) 
                                     if (fprintf(fpout,
                                                 "%d %.8e\n",j,val)<0)
                                             return(WRITE_ERR);
                             }
                         }
                     break;
        default:     break;
        }
    }
    if (active) free(active);
    return(err);
}

/*
 * the array in the basil solution (Data_arrays_int[1..NELLEP])
 * is the basil index (1..NUP) for an elle node
 */
int CnvtWriteNodeNumberReference(FILE *fpout,
                                  int maxorig, int *originalnodes,
                                  int numnodes,int *polynumber)
{
    int i,j,k,num=0;

    for (i=0;i<maxorig;i++) if (originalnodes[i]==1) num++;
    if (fprintf(fpout,"%d %d\n",num,maxorig)<0)
                        return(WRITE_ERR);
    for (k=0;k<numnodes;k++) {
        i=polynumber[k];
        if (i<maxorig && originalnodes[i]==1) 
            if (fprintf(fpout,"%d %d\n",i,k+1)<0)
                        return(WRITE_ERR);
    }
    return(0);
}

void CnvtMarkOriginalNodes(int *maxnodes, int **orignodes)
{
    /*
     * keep a record of the nodes read from the elle file
     */
    int i, j, max, maxindex, count, *p;

    max = ElleMaxNodes();
    /* index of last active node in array */
    for (maxindex=max-1;maxindex>=0 && !ElleNodeIsActive(maxindex);
                                                     maxindex-- ) ;
    *maxnodes = maxindex+1;
    /* allocate and initialise the array */
    if ((*orignodes = p = (int *)malloc(sizeof(int)*(maxindex+1)))==0)
        OnError("CnvtMarkOriginalNodes",MALLOC_ERR);
    for (j=0;j<=maxindex;j++) p[j] = 0;

    for (j=0;j<*maxnodes;j++) if (ElleNodeIsActive(j)) p[j]=1;
}

void CnvtRenumber(int numnodes, int **polynum)
{
    /*
     * triangle code expects node numbers to be consecutive
     * ie 1..numnodes, no gaps
     */
    int i, j, max, maxindex, count, *p;

    max = ElleMaxNodes();
    /* index of last active node in array */
    for (maxindex=max-1;maxindex>=0 && !ElleNodeIsActive(maxindex);
                                                     maxindex-- ) ;
    /* allocate and initialise node number cross-reference */
    if ((*polynum = new int[maxindex+1])==0)
        OnError("CnvtRenumber",MALLOC_ERR);
    p = *polynum;
    for (j=0;j<=maxindex;j++) p[j] = j;

    /* how many gaps? */
    for (j=0,count=0;j<numnodes;j++) if (!ElleNodeIsActive(j)) count++;
    if (count == 0) return;
    for (j=0,i=0;j<count;j++) {
        while (i<max && ElleNodeIsActive(i)) i++;
        p[i++] = maxindex--;
        while (!ElleNodeIsActive(maxindex)) maxindex--;
    }
}

void CnvtSetLimits(Coords *limits)
{
    int i;
    CellData unitcell;

    ElleCellBBox(&unitcell);
    for (i=0;i<4;i++) {
        limits[i] = unitcell.cellBBox[i];
        Vertices[i][0] = unitcell.cellBBox[i].x;
        Vertices[i][1] = unitcell.cellBBox[i].y;
    }
}

void CnvtCheckNodes()
{
    int i, j, k, max;
    int nbnodes[3],found,newn,bnd;
    Coords limits[4], xy, xy2;
    double eps=1e-6, deformx, deformy;

    /*
     * check for nodes close to boundaries (causes a cluster
     *     of small triangles, so move node to boundary)
     * add nodes at bounding box corners, if necessary
     */
    CnvtInitCorners();

    CnvtSetLimits(limits);

    max = ElleMaxNodes();
    /*
     * find boundary nodes and check for corner nodes
     */
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            /*CnvtSetCheckFlag(i);*/
            ElleNodePosition(i,&xy);
            if (bnd=CnvtBoundaryCheck(&xy)) {
                ElleCopyToPosition(i,&xy);
                CnvtSetBoundaryFlag(i,bnd);
                CnvtFindCorners(i);
            }
        }
    }

    CnvtAddCorners(limits);
    /*
     * if not a repositioned elle file
     */
    deformx = limits[0].x-limits[3].x;
    deformy = limits[3].y-limits[2].y;
    if (deformx>eps || deformx<-eps || deformy>eps || deformy<-eps ||
         ElleSSOffset()>eps|| ElleSSOffset()<-eps)
        CnvtFindCornersRepos();

    /*
     * mirror boundary nodes
     */
    CnvtInitMirrorNodes();
    CnvtSetMirrorNode(Corners[0],Corners[1]);
    CnvtSetMirrorNode(Corners[3],Corners[2]);
    CnvtSetMirrorNode(CornersRepos[0],CornersRepos[2]);
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i) && CnvtNodeOnBoundary(i)) {
            ElleNodePosition(i,&xy);
            if ((newn=CnvtGetMirrorNode(i))==NO_NB) {
                CnvtFindMirrorPosition(&xy,&xy2);
                if (!CnvtPointOnCorner(&xy2,&newn)) {
                    newn = ElleFindSpareNode();
                    ElleCopyToPosition(newn,&xy2);
                    CnvtSetBoundaryFlag(newn,CnvtWhichBoundary(&xy2));
                }
                CnvtSetMirrorNode(i,newn);
            }
        }
    }
#if XY
#endif
}

int CnvtSegments()
{
    int i, j, k, max;
    int nbnodes[3],found,newn,bnd;
    ERegion full_id[3];
    Coords limits[4], xy, xy2;

    /*
     * only record each segment once
     * add boundary nodes and new segments, if a segment wraps
     */
    max = ElleMaxNodes();
#ifdef FLYNN_VER
    full_id[0]=NO_NB;
#else
    full_id[0].grain=full_id[0].subgrain=full_id[0].unit=NO_NB;
#endif
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            ElleNeighbourNodes(i,nbnodes);
            for (j=0;j<3;j++)
                if (nbnodes[j]!=NO_NB && nbnodes[j]<i)
                    ElleSetNeighbour(i,j,NO_NB,&full_id[0]);
        }
    }
    /*
     * remove invalid (wrapping) segments
     */
    CnvtSetLimits(limits);

    max = ElleMaxNodes();
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            CnvtCheckSegments(i,limits);
        }
    }

    /*
     * debugging check for errors from CnvtCheckSegments
     */
    max = ElleMaxNodes();
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            ElleNodePosition(i,&xy);
            for (j=i+1;j<max;j++) {
                if (ElleNodeIsActive(j)) {
                    ElleNodePosition(j,&xy2);
                    if (xy.x==xy2.x && xy.y==xy2.y) {
printf("matching %d %d\n",i,j);
                        /*OnError("Matching nodes",0);*/
                    }
                }
            }
            /* check for orphan nodes (needed for ideal PSLG)
             * Should only find corner nodes
            ElleNeighbourNodes(i,nbnodes);
            for (j=0,found=0;j<3;j++)
                if (nbnodes[j]!=NO_NB) found = 1;
            if (!found) {
                for (k=0,found=0;k<max && !found;k++) {
                    if (ElleNodeIsActive(k)) {
                        ElleNeighbourNodes(k,nbnodes);
                        for (j=0;j<3;j++)
                            if (nbnodes[j]==i) found = 1;
                    }
                }
            }
            if (!found) {
                printf("orphan node %d\n",i);
            } */
        }
    }
    /*
     * check for boundary nodes without a mirror node
     * which may happen when segments wrap diagonally
     * (see CnvtCheckSegments - could be fixed there)
     * probably only matters when setting up periodic bnds
     */
    max = ElleMaxNodes();
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i) && CnvtNodeOnBoundary(i)) {
            if ((newn=CnvtGetMirrorNode(i))==NO_NB) {
                ElleNodePosition(i,&xy);
                CnvtFindMirrorPosition(&xy,&xy2);
                if (!CnvtPointOnCorner(&xy2,&newn)) {
                    newn = ElleFindSpareNode();
                    ElleCopyToPosition(newn,&xy2);
                    CnvtSetBoundaryFlag(newn,CnvtWhichBoundary(&xy2));
                }
                CnvtSetMirrorNode(i,newn);
            }
        }
    }
}

int CnvtCheckSegments(int node, Coords *limits)
{
    int i,j;
    int nbnodes[3], newn=NO_NB, newn2=NO_NB, bndnode, nb, cnrnb;
    int  err=0, found=0, wrap=0;
    int node_on_bnd=0, nb_on_bnd=0;
    int bnddiff, bnd;
    double xlen, ylen, xdiff, ydiff;
    double xoffset,yoffset;
    double eps = 1.0e-6;
    Coords xy,prev,xy0,newxy[2];
    ERegion rgn;
    CellData unitcell;
    /*
     * add 1 node if node on boundary
     * remove segment if both nodes on boundary
     * add 2 nodes (and 2 segments) if a segment crosses a
     *            boundary
     */

    ElleNeighbourNodes(node,nbnodes);

#ifdef FLYNN_VER
    rgn=NO_NB;
#else
    rgn.grain = rgn.subgrain = rgn.unit = NO_NB;
#endif
    xlen = limits[1].x - limits[0].x;
    ylen = limits[2].y - limits[1].y;
    node_on_bnd = CnvtNodeOnBoundary(node);
    for (i=0;i<3;i++) {
        ElleNodePosition(node,&prev);
        if (nbnodes[i]!=NO_NB) {
            ElleNodePosition(nbnodes[i],&xy);
            nb_on_bnd = CnvtNodeOnBoundary(nbnodes[i]);
            /*
             * both are boundary nodes
             */
            if (node_on_bnd && nb_on_bnd) {
                ElleNodePosition(nbnodes[i],&xy);
                bnddiff = abs(node_on_bnd - nb_on_bnd);
                /*
                 * remove the segment if nodes on opp. boundaries
                 * or the same boundary
                 * or both are corner nodes
                 */
                if (bnddiff==0 || bnddiff==2 ||
                    (CnvtNodeOnCorner(node) /*&&*/||
                     CnvtNodeOnCorner(nbnodes[i])))
                    ElleSetNeighbour(node,i,NO_NB,&rgn);
                else {
                    xdiff = fabs((xy.x - prev.x));
                    ydiff = fabs((xy.y - prev.y));
                    if (xdiff > xlen*0.5 || ydiff > ylen*0.5 ) {
                /*
                 * remove the segment
                 * and create a new segment with mirror node 
                 */
                        ElleSetNeighbour(node,i,NO_NB,&rgn);
                        if (ydiff > ylen*0.5) {
                            ElleCellBBox(&unitcell);
                            xoffset = unitcell.cellBBox[TOPLEFT].x -
                                       unitcell.cellBBox[BASELEFT].x;
                            xy.x += (prev.y-xy.y)*unitcell.ylength *
                                   unitcell.xoffset;
                            xdiff = fabs((xy.x - prev.x));
                            ElleNodePosition(nbnodes[i],&xy);
                        }
                        if (!(xdiff > xlen*0.5 && ydiff > ylen*0.5)) {
                            if ((newn=CnvtGetMirrorNode(node))==NO_NB)
                                OnError("Boundary node with no mirror node",0);
                            ElleNodePosition(newn,&prev);
                            xdiff = fabs((xy.x - prev.x));
                            ydiff = fabs((xy.y - prev.y));
                            if (xdiff > xlen*0.5 || ydiff > ylen*0.5 ) {
                                ElleNodePosition(node,&prev);
                                if ((newn2=CnvtGetMirrorNode(nbnodes[i]))==NO_NB)
                                    OnError("Boundary node with no mirror node",0);
                                ElleNodePosition(newn2,&xy);
                                xdiff = fabs((xy.x - prev.x));
                                ydiff = fabs((xy.y - prev.y));
                                if (xdiff > xlen*0.5 || ydiff > ylen*0.5 ) {
                                    ElleNodePosition(newn,&prev);
                                    xdiff = fabs((xy.x - prev.x));
                                    ydiff = fabs((xy.y - prev.y));
                                    if (xdiff > xlen*0.5 || ydiff > ylen*0.5 )
                                        OnError("I give up",0);
                                    ElleSetNeighbour(newn,NO_NB,newn2,&rgn);
                                }
                                else {
                                    ElleSetNeighbour(node,NO_NB,newn2,&rgn);
                                }
                            }
                            else
                                ElleSetNeighbour(newn,NO_NB,nbnodes[i],&rgn);
                        }
/*
                        else
                            ElleSetNeighbour(newn,NO_NB,nbnodes[i],&rgn);
*/
                    }
                }
            }
            else if (CnvtSegmentWrap(nbnodes[i],node,&xy,&prev)) {
                /*
                 * one node on a boundary
                 */
#ifdef FLYNN_VER
                rgn=NO_NB;
#else
                rgn.grain = rgn.subgrain = rgn.unit = NO_NB;
#endif
                if (node_on_bnd || nb_on_bnd) {
                    if (node_on_bnd) { bndnode = node; nb = nbnodes[i]; }
                    else /*nb_on_bnd*/ { bndnode = nbnodes[i]; nb = node; }
                    ElleSetNeighbour(node,i,NO_NB,&rgn);
                    ElleNodePosition(bndnode,&xy0);
                    ElleNodePosition(nb,&prev);
                    ElleNodePlotXY(bndnode,&xy,&prev);
                    if (CnvtNodeOnCornerRepos(bndnode)) {
                        newn = CnvtNearestCorner(nb);
                        ElleSetNeighbour(newn,NO_NB,nb,0);
                    }
                    else if (CnvtNodeOnCorner(bndnode)) {
                        if (ElleSSOffset()==0||
                            fabs((xy.y - xy0.y))<(ylen*0.5) ||
                             (fabs(fabs(xy.y - xy0.y)-ylen)<eps &&
                               fabs(xy.x - xy0.x)-fabs(ElleSSOffset())>=eps)) {
                                 //fabs((xy.x - xy0.x))<(xlen*0.5))) {
                            newn = CnvtNearestCorner(nb);
                        }
                        else {
                            if (bndnode==node) bnd = node_on_bnd;
                            else bnd = nb_on_bnd;
                            if (bnd<3) newn = CornersRepos[2];
                            else newn = CornersRepos[0];
                        }
                        ElleSetNeighbour(newn,NO_NB,nb,0);
                    }
                    else if (!CnvtOutsideLimits(&xy)) {
                        if ((newn=CnvtGetMirrorNode(bndnode))==NO_NB) {
                            CnvtAddBoundaryNode(nb,&xy,&newn);
                            CnvtSetMirrorNode(bndnode,newn);
                        }
                        ElleSetNeighbour(newn,NO_NB,nb,0);
                    }
                    else {
                        if (err = CnvtFindIntersection(&prev,&xy,&newxy[0],
                                               &found))
                            return(err);
                        CnvtAddBoundaryNode(nb,&newxy[0],&newn);
                        ElleNodePosition(bndnode,&prev);
                        ElleNodePlotXY(nb,&xy,&prev);
                        if ((err = CnvtFindIntersection(&prev,&xy,&newxy[1],
                                               &found))==0 ) {
                            if (found && !CnvtLocationMatch(&newxy[1],&prev)) {
                                CnvtAddBoundaryNode(bndnode,&newxy[1],&newn2);
                                CnvtSetMirrorNode(newn,newn2);
                            }
                        }
                    }
                }
                else {
                    /*
                     * add nodes at the segment-boundary intersections
                     */
                    ElleNodePosition(node,&prev);
                    ElleNodePlotXY(nbnodes[i],&xy,&prev);
                    if (err = CnvtFindIntersection(&prev,&xy,&newxy[0],
                                           &found))
                        return(err);
                    if (found)  {
#ifdef FLYNN_VER
                        rgn=NO_NB;
#else
                        rgn.grain = rgn.subgrain = rgn.unit = NO_NB;
#endif
                        ElleSetNeighbour(node,i,NO_NB,&rgn);
                        if (CnvtPointOnCorner(&newxy[0],&cnrnb))
                            ElleSetNeighbour(node,NO_NB,cnrnb,0);
                        else
                            CnvtAddBoundaryNode(node,&newxy[0],&newn);
                    }
                    ElleNodePosition(nbnodes[i],&prev);
                    ElleNodePlotXY(node,&xy,&prev);
                    if (err = CnvtFindIntersection(&prev,&xy,&newxy[1],
                                               &found))
                        return(err);
                    if (found) {
                        if (CnvtPointOnCorner(&newxy[1],&cnrnb))
                            ElleSetNeighbour(nbnodes[i],NO_NB,cnrnb,0);
                        else {
                            CnvtAddBoundaryNode(nbnodes[i],&newxy[1],&newn2);
                            if (abs((CnvtNodeOnBoundary(newn)-
                                    CnvtNodeOnBoundary(newn2)))==2)
                                CnvtSetMirrorNode(newn,newn2);
                        }
                    }
                }
            }
        }
    }
}

void CnvtCreateBoundaryList(int perbnd)
{
    int i, j, k, max, count=0;
    int start, end_x, end_y, bnd_x, bnd_y, bnd;
    Coords xy, xy2, tmp;
    CellData unitcell;

    /*
     * add boundary nodes to anticlockwise-ordered list
     */

    for (i=0;i<4;i++) {
        ElleNodePosition((k=CnvtCorner(i)),&xy);
        AddToList(xy.x,xy.y,k,i+1);
    }

    /*
     * find boundary nodes
     */
    max = ElleMaxNodes();
    for (i=0;i<max;i++) {
        bnd = CnvtNodeOnBoundary(i);
        if (ElleNodeIsActive(i) && bnd && !CnvtNodeOnCorner(i)) {
            ElleNodePosition(i,&xy);
            AddToList(xy.x,xy.y,i,bnd);
        }
    }
    if (perbnd) CnvtInsertBoundaryNodes();
/*PrintList();*/
}

#define BNDGAP 0.02
/*#define BNDGAP 0.005*/

void CnvtInsertBoundaryNodes()
{
    int i, j, k, max, count=0;
    int start, end, end_y, bnd_x, bnd_y, bnd, newn, newn2;
    int next;
    double diff, intervals;
    double ratiox, ratioy, setstep, step;
    double xoffset, deformx, deformy;
    double eps = 1.0e-6;
    double incr=0, minsep=20*eps;
    Coords xy, xy2, tmp, nextxy;
    CellData unitcell;

    /*
     * add boundary nodes to this and the right (xmax) boundary
     * at intervals of ???
     */

    ElleCellBBox(&unitcell);
    setstep = ElleminNodeSep();
    if (setstep < BNDGAP) setstep = BNDGAP;
    ratiox = ratioy = 0.0;
    deformx = unitcell.cellBBox[TOPLEFT].x -
                      unitcell.cellBBox[BASELEFT].x;
    if (fabs(deformx) > eps &&
        fabs(unitcell.xoffset) > eps)
        ratiox = deformx/unitcell.ylength;
    deformy = unitcell.cellBBox[TOPLEFT].y -
                      unitcell.cellBBox[TOPRIGHT].y;
    if (fabs(deformy) > eps &&
        fabs(unitcell.yoffset) > eps)
        ratioy = deformy/unitcell.xlength;
    /*
     * add nodes on Ymin, Ymax boundaries
     */
    start = CnvtCorner(BASELEFT);
    j=newn=start;
    end = CnvtCorner(BASERIGHT);
    ElleNodePosition(start,&xy);
    NextInList(j,&xy2.x,&xy2.y,&i);
    while (j != end ) {
        diff = xy2.x-xy.x;
        if ((intervals=diff/setstep) > 2.0) {
            NextInList(j,&nextxy.x,&nextxy.y,&next);
            intervals += 0.5;
            /*step = diff/truncf(intervals);*/
            step = diff/floor(intervals);
            max = (int)intervals - 1;
            for (k=0;k<max;k++) {
                xy.x += step;
                if (ratioy)
                  xy.y = (ratioy*(xy.x-unitcell.cellBBox[1].x) +
                                                 unitcell.cellBBox[1].y);
                /*if (pointSeparation(&xy,&nextxy)>(10*eps)) {*/
                tmp = xy;
                newn = ElleFindSpareNode();
                ElleCopyToPosition(newn,&xy);
                CnvtSetBoundaryFlag(newn,1);
                AddToList(xy.x,xy.y,newn,1);
                xy.y += unitcell.ylength;
                xy.x += unitcell.xoffset;
                /*xy.x += unitcell.xoffset+deformx;*/
                newn = ElleFindSpareNode();
                /*xy.x += xoffset;*/
                /*ElleCopyToPosition(newn,&xy);*/
                ElleNodeUnitXY(&xy);
                ElleSetPosition(newn,&xy);
                CnvtSetBoundaryFlag(newn,3);
                AddToList(xy.x,xy.y,newn,3);
                xy = tmp;
                count++;
                /*}*/
            }
        }
        else if (i!=end && diff<=minsep) {
            NextInList(i,&nextxy.x,&nextxy.y,&next);
            diff = nextxy.x-xy2.x;
            if (diff >= minsep) {
                incr = diff/2.0;
                if (incr>minsep) incr=minsep;
                xy2.y += incr;
                ElleSetPosition(i,&xy2);
                next = CnvtGetMirrorNode(i);
                ElleNodePosition(next,&nextxy);
                nextxy.x=xy2.x;
                ElleSetPosition(next,&nextxy);
            }
            else {
                char msg[256] = "";
                sprintf(msg,"Too many points near %f %f\n",xy2.x,xy2.y);
                OnError(msg,0);
            }
                
        }
        xy.x = xy2.x; xy.y = xy2.y; j=i;
        NextInList(j,&xy2.x,&xy2.y,&i);
    }
    /*
     * add nodes on Xmin, Xmax boundaries
     */
    end = CnvtCorner(TOPRIGHT);
    NextInList(j,&xy2.x,&xy2.y,&i);
    while (j != end ) {
        diff = xy2.y-xy.y;
        if ((intervals=diff/setstep) > 2.0) {
            intervals += 0.5;
            /*step = diff/truncf(intervals);*/
            step = diff/floor(intervals);
            max = (int)intervals - 1;
            for (k=0;k<max;k++) {
                xy.y += step;
                if (ratiox)
                  xy.x = (ratiox*(xy.y-unitcell.cellBBox[1].y) +
                                                 unitcell.cellBBox[1].x);
                  /*xy.x += (float)(ratio*step);*/
                tmp = xy;
                newn = ElleFindSpareNode();
                ElleCopyToPosition(newn,&xy);
                CnvtSetBoundaryFlag(newn,2);
                AddToList(xy.x,xy.y,newn,2);
                xy.x -= unitcell.xlength;
                newn = ElleFindSpareNode();
                ElleCopyToPosition(newn,&xy);
                CnvtSetBoundaryFlag(newn,4);
                AddToList(xy.x,xy.y,newn,4);
                xy = tmp;
                count++;
            }
        }
        else if (i!=end && diff<=minsep) {
            NextInList(i,&nextxy.x,&nextxy.y,&next);
            diff = nextxy.y-xy2.y;
            if (diff >= minsep) {
                incr = diff/2.0;
                if (incr>minsep) incr=minsep;
                xy2.y += incr;
                ElleSetPosition(i,&xy2);
                next = CnvtGetMirrorNode(i);
                ElleNodePosition(next,&nextxy);
                nextxy.y=xy2.y;
                ElleSetPosition(next,&nextxy);
            }
            else {
                char msg[256] = "";
                sprintf(msg,"Too many points near %f %f\n",xy2.x,xy2.y);
                OnError(msg,0);
            }
        }
        xy.x = xy2.x; xy.y = xy2.y; j=i;
        NextInList(j,&xy2.x,&xy2.y,&i);
    }
}

void CnvtJoinBoundarySegments()
{
    int i,j,max,id,nbnodes[3];
    int bnd1,bnd2,start;
    int err=0;
int count=0,k;
    Coords xy;
    ERegion rgn;

#ifdef FLYNN_VER
    rgn=NO_NB;
#else
    rgn.grain = rgn.subgrain = rgn.unit = NO_NB;
#endif
    /*
     * clear existing boundary segments
     */
    max = ElleMaxNodes();
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i) && (bnd1=CnvtNodeOnBoundary(i))) {
            ElleNeighbourNodes(i,nbnodes);
            for (j=0;j<3;j++) {
                if (bnd2=CnvtNodeOnBoundary(nbnodes[j])) 
                    if (bnd1==bnd2 || ((CnvtNodeOnCorner(i) ||
                                       CnvtNodeOnCorner(nbnodes[j])) &&
                                       abs(bnd1-bnd2)==1) )
                        ElleSetNeighbour(i,j,NO_NB,&rgn);
            }
        }
    }
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            ElleNeighbourNodes(i,nbnodes);
            for (j=0;j<3;j++) {
                if (nbnodes[j]!=NO_NB) {
                    if    (CnvtNodeOnBoundary(i) &&
                              CnvtNodeOnBoundary(nbnodes[j]))
                        printf("%d %d\n",i,nbnodes[j]);
                }
            }
        }
    }
    /*
     * join the nodes in the boundary list
     */
    j = start = CnvtCorner(BASELEFT);
    NextInList(j,&xy.x,&xy.y,&i);
    while (i != start ) {
        /*
         * boundary nodes may end up with 4 neighbours
         */
        if ((err=ElleSetNeighbour(j,NO_NB,i,0))==LIMIT_ERR)
            if ((err=ElleSetNeighbour(i,NO_NB,j,0))==LIMIT_ERR)
                OnError("CnvtJoinBoundarySegments",err);
count++;
        j=i;
        NextInList(j,&xy.x,&xy.y,&i);
    }
    if ((err=ElleSetNeighbour(j,NO_NB,i,0))==LIMIT_ERR)
        if ((err=ElleSetNeighbour(i,NO_NB,j,0))==LIMIT_ERR)
            OnError("CnvtJoinBoundarySegments",err);
count++;
printf("bnd seg count %d\n",count);
}

void CnvtAddBoundaryNode(int nb, Coords*newxy, int *newn)
{
    int found=0, err=0;

    *newn = ElleFindSpareNode();
    ElleSetPosition(*newn,newxy);
    CnvtSetBoundaryFlag(*newn,CnvtWhichBoundary(newxy));
    ElleSetNeighbour(*newn,NO_NB,nb,0);
}

int CnvtFindIntersection( Coords *prev, Coords *xy, Coords *newxy,
                          int *found)
{
    double xlen, ylen, eps, xp, xoffset;
    Coords chk1,chk2;
    double x1, y1, x2, y2, x3, y3, x4, y4, x, y;
    CellData unit;

    eps = 1e-6;
    ElleCellBBox(&unit);
    xoffset = unit.cellBBox[TOPLEFT].x - unit.cellBBox[BASELEFT].x;
    *found = 0;
    x1 = prev->x;
    y1 = prev->y;
    x2 = xy->x;
    y2 = xy->y;
    xp = (xy->y - unit.cellBBox[BASELEFT].y)/unit.ylength * xoffset
                          + unit.cellBBox[BASELEFT].x;
    /* minimum x */
    if ( xy->x <= xp+eps ) {
        x3 = unit.cellBBox[BASELEFT].x;
        y3 = unit.cellBBox[BASELEFT].y;
        x4 = unit.cellBBox[TOPLEFT].x;
        y4 = unit.cellBBox[TOPLEFT].y;
        *found = lines_intersect(x1,y1,x2,y2,x3,y3,x4,y4,&x,&y);
        if (*found==1) {
            chk1.x = x; chk1.y = y;
            if (CnvtLocationMatch(xy,&chk1)) *found = 0;
            else {
                newxy[0].x = x; newxy[0].y = y;
            }
        }
    }
    /* maximum x */
    xp += unit.xlength;
    if ( xy->x >= xp-eps && !(*found)) {
        x3 = unit.cellBBox[BASERIGHT].x;
        y3 = unit.cellBBox[BASERIGHT].y;
        x4 = unit.cellBBox[TOPRIGHT].x;
        y4 = unit.cellBBox[TOPRIGHT].y;
        *found = lines_intersect(x1,y1,x2,y2,x3,y3,x4,y4,&x,&y);
        if (*found==1) {
            chk1.x = x; chk1.y = y;
            if (CnvtLocationMatch(xy,&chk1)) *found = 0;
            else {
                newxy[0].x = x; newxy[0].y = y;
            }
        }
    }
    /* minimum y */
    if ( xy->y <= unit.cellBBox[BASELEFT].y+eps && !(*found)) {
        x3 = unit.cellBBox[BASELEFT].x;
        y3 = unit.cellBBox[BASELEFT].y;
        x4 = unit.cellBBox[BASERIGHT].x;
        y4 = unit.cellBBox[BASERIGHT].y;
        *found = lines_intersect(x1,y1,x2,y2,x3,y3,x4,y4,&x,&y);
        if (*found==1) {
            chk1.x = x; chk1.y = y;
            if (CnvtLocationMatch(xy,&chk1)) *found = 0;
            else {
                newxy[0].x = x; newxy[0].y = y;
            }
        }
    }
    /* maximum y */
    if ( xy->y >= unit.cellBBox[TOPLEFT].y-eps && !(*found)) {
        x3 = unit.cellBBox[TOPRIGHT].x;
        y3 = unit.cellBBox[TOPRIGHT].y;
        x4 = unit.cellBBox[TOPLEFT].x;
        y4 = unit.cellBBox[TOPLEFT].y;
        *found = lines_intersect(x1,y1,x2,y2,x3,y3,x4,y4,&x,&y);
        if (*found==1) {
            chk1.x = x; chk1.y = y;
            if (CnvtLocationMatch(xy,&chk1)) *found = 0;
            else {
                newxy[0].x = x; newxy[0].y = y;
            }
        }
    }
    return(0);
}

int CnvtSegmentWrap(int node,int nb,
                    Coords *xy,Coords *prev)
{
    int found=0;
    Coords tmp1,tmp2;

    ElleNodePosition(nb,xy);
    ElleNodePlotXY(node,prev,xy);
    if ( CnvtOutsideLimits(prev) ) {
        found = 1;
    }
    else {
        ElleNodePosition(node,prev);
        ElleNodePlotXY(nb,xy,prev);
        if ( CnvtOutsideLimits(xy) ) {
            found = 1;
        }
    }
    return(found);
}

int CnvtOutsideLimits(Coords *xy)
{
    double eps;
    double xp1,xp2;
    CellData unit;

    eps = 1e-6;
    double pnt[2];

    if (CnvtWhichBoundary(xy)) return(0);
    pnt[0] = xy->x;
    pnt[1] = xy->y;
    return(!CrossingsTest(&Vertices[0][0],4,pnt));
}

int CnvtCorner(int index)
{
    return(Corners[index]);
}

void CnvtInitCorners()
{
    int i;

    for (i=0;i<4;i++) Corners[i] = CornersRepos[i] = NO_NB;
}

void CnvtAddCorners(Coords *limits)
{
    int i, j, max;
    Coords xy, newxy;

    max = ElleMaxNodes();
    for (i=0;i<4;i++) {
        if (Corners[i]==NO_NB) {
            Corners[i] = ElleFindSpareNode();
            xy = limits[i];
            ElleSetPosition(Corners[i],&xy);
            CnvtSetBoundaryFlag(Corners[i],i+1);
        }
    }
}

int CnvtNodeOnCorner(int node)
{
    int i, found;

    for (i=0,found=0;i<4 && !found;i++) if (Corners[i]==node) found=1;
    return(found);
}

int CnvtNodeOnCornerRepos(int node)
{
    int i, found;

    for (i=0,found=0;i<4 && !found;i++)
        if (CornersRepos[i]==node) found=1;
    return(found);
}

int CnvtNearestCorner(int node)
{
    int i, index;
    double min;
    double d;
    Coords xy1, xy2;

    index = 0;
    ElleNodePosition(node,&xy1);
    ElleNodePosition(Corners[0],&xy2);
    min = (xy2.x-xy1.x)*(xy2.x-xy1.x) + (xy2.y-xy1.y)*(xy2.y-xy1.y);
    for (i=1;i<4;i++) {
        ElleNodePosition(Corners[i],&xy2);
        d = (xy2.x-xy1.x)*(xy2.x-xy1.x) + (xy2.y-xy1.y)*(xy2.y-xy1.y);
        if (d < min) {
            index = i;
            min = d;
        }
    }
    return(Corners[index]);
}

void CnvtJoinCorners()
{
    int i, j, max;

    for (i=0;i<4;i++) {
        j = (i+1)%4;
        if (!ElleNodeIsTriple(Corners[i])) {
            if (ElleFindNbIndex(Corners[j],Corners[i])==NO_NB)
                ElleSetNeighbour(Corners[i],NO_NB,Corners[j],0);
        }
        else if (!ElleNodeIsTriple(Corners[j])) {
            if (ElleFindNbIndex(Corners[i],Corners[j])==NO_NB)
                ElleSetNeighbour(Corners[j],NO_NB,Corners[i],0);
        }
        else OnError("CnvtJoinCorners ",LIMIT_ERR);
    }
}

void CnvtFindMirrorPosition(Coords *xy,Coords *newxy)
{
    int bnd;
    double xoffset;
    CellData unit;

    ElleCellBBox(&unit);
    xoffset = unit.cellBBox[TOPLEFT].x - unit.cellBBox[BASELEFT].x;
    bnd = CnvtWhichBoundary(xy);
    *newxy = *xy;
    switch(bnd) {
    case 1: newxy->y += unit.ylength;
            newxy->x += unit.xoffset;
            break;
    /*case 2: newxy->x -= unit.xlength;*/
    case 2: newxy->x = (xy->y - unit.cellBBox[BASELEFT].y)/unit.ylength
                       * xoffset + unit.cellBBox[BASELEFT].x;
            break;
    case 3: newxy->y -= unit.ylength;
            newxy->x -= unit.xoffset;
            break;
    /*case 4: newxy->x += unit.xlength;*/
    case 4: newxy->x = (xy->y - unit.cellBBox[BASELEFT].y)/unit.ylength
                       * xoffset + unit.cellBBox[BASERIGHT].x;
            break;
    }
    if (xoffset != unit.xoffset) ElleNodeUnitXY(newxy);
}

int CnvtWhichBoundary(Coords *xy)
{
    int i,ip1;
    CellData unitcell;

    ElleCellBBox(&unitcell);
    for (i=0;i<4;i++) {
        ip1 = (i+1)%4;
        if (PointOnSegment(unitcell.cellBBox[i].x,
                           unitcell.cellBBox[i].y,
                           unitcell.cellBBox[ip1].x,
                           unitcell.cellBBox[ip1].y,
                           xy->x,xy->y))
            return(i+1);
    }
    return(0);
}

int CnvtNodeCount()
{
    /*
     * returns the number of active nodes
     */
    int i, num=0, max;

    max = ElleMaxNodes();
    for (i=0;i<max;i++)
        if (ElleNodeIsActive(i)) num++;
    return(num);
}

int CnvtSegmentCount()
{
    int i, j, nbnodes[3], max, num=0;

    /*
     * returns the number of segments
     */
    max = ElleMaxNodes();

    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            ElleNeighbourNodes(i,nbnodes);
            for (j=0;j<3;j++) if (nbnodes[j]!=NO_NB) num++;
        }
    }
    return(num);
}

int CnvtBoundaryCheck(Coords *xy)
{
    /*
     * accepts a point in the unit cell
     * checks whether it is inside a smaller cell
     * if not, it is too close to a boundary so move it
     * NOTE: this depends on the node sep set in the elle file
     *       if the separation is large, nodes may be moved to the bnd
     *       and end up coincident with a bnd node....
     */
    int i, on_bnd=0, bnd=0, cnr1, cnr2;
    double mingap, xp, deltax, deltay, xoff, tanang, tmpx, tmpy;
    double xoffset;
    double innerlimits[4][2],pnt[2];
    double theta;
    CellData unit;

    /*
     * 0.087=sin(5deg)
     * 0.342=sin(20deg)
     * check basil2elle.c (FindBasilIndex) if this value is changed
     */
    mingap = EllemaxNodeSep()*0.087;
    ElleCellBBox(&unit);
    xoffset = unit.cellBBox[TOPLEFT].x - unit.cellBBox[BASELEFT].x;
    if (xoffset == 0) theta=0;
    else theta = atan(unit.ylength/unit.xoffset);
    if (theta) xoff = mingap/tan(theta);
    tanang = xoffset/unit.ylength;
    xoff = mingap * tanang;
    pnt[0] = xy->x;
    pnt[1] = xy->y;
    innerlimits[0][0] = unit.cellBBox[0].x + mingap + xoff;
    innerlimits[0][1] = unit.cellBBox[0].y + mingap;
    innerlimits[1][0] = unit.cellBBox[1].x - mingap + xoff;
    innerlimits[1][1] = unit.cellBBox[1].y + mingap;
    innerlimits[2][0] = unit.cellBBox[2].x - mingap - xoff;
    innerlimits[2][1] = unit.cellBBox[2].y - mingap;
    innerlimits[3][0] = unit.cellBBox[3].x + mingap - xoff;
    innerlimits[3][1] = unit.cellBBox[3].y - mingap;
        /* on a bnd or too close */
    if (bnd=CnvtWhichBoundary(xy)) {
        on_bnd=1;
        if (bnd==1 && (xy->y != unit.cellBBox[BASELEFT].y))
                xy->y = unit.cellBBox[BASELEFT].y;
        if (bnd==2 && xoffset==0 &&
             (xy->x != unit.cellBBox[BASERIGHT].x))
                xy->x = unit.cellBBox[BASERIGHT].x;
        if (bnd==3 && (xy->y != unit.cellBBox[TOPLEFT].y))
                xy->y = unit.cellBBox[TOPLEFT].y;
        if (bnd==4 && xoffset==0 &&
             (xy->x != unit.cellBBox[BASELEFT].x))
                xy->x = unit.cellBBox[BASELEFT].x;
    }
    else {
        if (!CrossingsTest(&innerlimits[0][0],4,pnt)) {
            /* move it to the boundary */
            deltay = xy->y-unit.cellBBox[0].y;
            if (deltay <= mingap) {
                bnd = 1;
            }
            else {
                deltay = unit.cellBBox[2].y - xy->y;
                if (deltay <= mingap) {
                    bnd = 3;
                }
                else {
                    xp = (xy->y - unit.cellBBox[BASELEFT].y)/
                          unit.ylength * xoffset
                              + unit.cellBBox[BASELEFT].x;
                    if ((deltax=xy->x-xp) <= mingap) {
                        bnd = 4;
                    }
                    else {
                        xp += unit.xlength;
                        if ((deltax=xp - xy->x) <= mingap) bnd = 2;
                    }
                }
            }
            switch(bnd) {
            case 1: xy->y = unit.cellBBox[0].y;
                    /*if (theta) xy->x += deltay/tan(theta);*/
                    if (unit.xoffset) xy->x -= deltay*tanang;
                    on_bnd = 1;
                    break;
            case 4:
            case 2: xy->x = xp;
                    on_bnd = 1;
                    break;
            case 3: xy->y = unit.cellBBox[2].y;
                    /*if (theta) xy->x += deltay/tan(theta);*/
                    if (unit.xoffset) xy->x += deltay*tanang;
                    on_bnd = 1;
                    break;
            }
        }
    }
    if (on_bnd) {
    /* 
     * is it close to a corner
     */
        tmpx = xy->x-unit.cellBBox[bnd-1].x;
        tmpy = xy->y-unit.cellBBox[bnd-1].y;
        if ((tmpx*tmpx + tmpy*tmpy) < (mingap*mingap)) {
            xy->x = unit.cellBBox[bnd-1].x;
            xy->y = unit.cellBBox[bnd-1].y;
        }
        else {
            tmpx = xy->x-unit.cellBBox[bnd%4].x;
            tmpy = xy->y-unit.cellBBox[bnd%4].y;
            if ((tmpx*tmpx + tmpy*tmpy) < (mingap*mingap)) {
                xy->x = unit.cellBBox[bnd%4].x;
                xy->y = unit.cellBBox[bnd%4].y;
            }
        }
    }

    return((on_bnd==1)?bnd:0);
}

int CnvtPointOnCorner(Coords *ptxy, int *cnrnode)
{
    int i, on_bnd=0;
    double eps;
    Coords xy;

    eps = 1e-6;
    *cnrnode = NO_NB;
    for (i=0;i<4;i++) {
        ElleNodePosition(Corners[i],&xy);
        /* is it on a corner? */
        if (ptxy->x > xy.x-eps &&
            ptxy->y > xy.y-eps &&
            ptxy->x < xy.x+eps &&
            ptxy->y < xy.y+eps)
            *cnrnode=Corners[i];
    }
    return( (*cnrnode==NO_NB) ? 0 : 1);
}

void CnvtFindCorners(int node)
{
    int i, on_bnd=0;
    double eps;
    Coords xy;
    CellData unit;

    eps = 1e-6;
    ElleCellBBox(&unit);
    ElleNodePosition(node,&xy);
    for (i=0;i<4;i++) {
        /* is it on a corner? */
        if (xy.x>unit.cellBBox[i].x-eps &&
            xy.y>unit.cellBBox[i].y-eps &&
            xy.x<unit.cellBBox[i].x+eps &&
            xy.y<unit.cellBBox[i].y+eps)
            Corners[i]=node;
    }
}

void CnvtFindCornersRepos()
{
    int max,i,n;
    Coords xy, reposxy0, reposxy2;
    CellData unit;

    double eps = 1e-6;
    ElleCellBBox(&unit);
    xy.x = (unit.cellBBox[0].x + unit.cellBBox[1].x)*0.5;
    xy.y = unit.cellBBox[0].y;
    ElleNodePlotXY(Corners[2],&reposxy0,&xy);
    xy.x = (unit.cellBBox[2].x + unit.cellBBox[3].x)*0.5;
    xy.y = unit.cellBBox[3].y;
    ElleNodePlotXY(Corners[0],&reposxy2,&xy);
    max = ElleMaxNodes();
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            n=CnvtNodeOnBoundary(i);
            if (n==1 && CornersRepos[0]==NO_NB) {
                ElleNodePosition(i,&xy);
                if (xy.x>reposxy0.x-eps &&
                    xy.y>reposxy0.y-eps &&
                    xy.x<reposxy0.x+eps &&
                    xy.y<reposxy0.y+eps)
                    { CornersRepos[0] = CornersRepos[1] = i;}
            }
            else if (n==3 && CornersRepos[2]==NO_NB) {
                ElleNodePosition(i,&xy);
                if (xy.x>reposxy2.x-eps &&
                    xy.y>reposxy2.y-eps &&
                    xy.x<reposxy2.x+eps &&
                    xy.y<reposxy2.y+eps)
                    { CornersRepos[2] = CornersRepos[3] = i;}
            }
        }
    }
    if (CornersRepos[0]==NO_NB) {
        CornersRepos[0] = CornersRepos[1] = ElleFindSpareNode();
        ElleSetPosition(CornersRepos[0],&reposxy0);
        CnvtSetBoundaryFlag(CornersRepos[0],1);
    }
    if (CornersRepos[2]==NO_NB) {
        CornersRepos[2] = CornersRepos[3] = ElleFindSpareNode();
        ElleSetPosition(CornersRepos[2],&reposxy2);
        CnvtSetBoundaryFlag(CornersRepos[2],3);
    }
}

void CnvtClearRgnAttribArray(RegionAttribArray *attr)
{
    if (attr->elems) free(attr->elems);
}

void CnvtInitRegionAttribArray()
{
    int max,err=0;

    RgnAttributes.maxgrains = 0;
    RgnAttributes.elems = 0;
    max = ElleMaxNodes();
    if (err = CnvtReallocateRegionAttribArray(max))
        OnError( "Reallocating Attribute array",err );
}

int CnvtMaxAttributeEntries()
{
    return(RgnAttributes.maxgrains);
}

int CnvtReallocateRegionAttribArray(int size)
{
    int oldsize, i, j;
    int err=0;
    Coords init;
    /*
     * reallocate memory for the attribute array
     */
    oldsize = RgnAttributes.maxgrains;
    if ((RgnAttributes.elems = (RegionAttribute *)
                                   realloc(RgnAttributes.elems,
                                        size*sizeof(RegionAttribute))) ==0)
         return( MALLOC_ERR );
    RgnAttributes.maxgrains = size;
    init.x = init.y = 0.0;
    for (i=oldsize;i<size;i++) {
        RgnAttributes.elems[i].active=0;
        RgnAttributes.elems[i].segment[0]=NO_NB;
        RgnAttributes.elems[i].segment[1]=NO_NB;
        RgnAttributes.elems[i].num_interior_pts=0;
        for (j=0;j<MAX_INTERIOR_PTS;j++)
            RgnAttributes.elems[i].interior_pt[j]=init;
        RgnAttributes.elems[i].value=0.0;
    }
    return( 0 );
}

int CnvtLocationMatch(Coords *loc1, Coords *loc2)
{
    double eps;

    eps = 1e-5;
    if (loc1->x > loc2->x-eps && loc1->x < loc2->x+eps &&
        loc1->y > loc2->y-eps && loc1->y < loc2->y+eps)
        return(1);
    return(0);
}

void CnvtFindInteriorPt(Coords *pos1, Coords *pos2, Coords *pt)
{
    double x1, y1, x2, y2, xnew, ynew;

    x1 = pos1->x;
    y1 = pos1->y;
    x2 = pos2->x;
    y2 = pos2->y;
    // find point near beginning of segment
    x2 = x1 + (x2-x1)*0.1;
    y2 = y1 + (y2-y1)*0.1;
    rotate_coords_ac(x2,y2,x1,y1,&xnew,&ynew,0.2);
    /*rotate_coords_ac(x2,y2,x1,y1,&xnew,&ynew,0.05);*/
    pt->x = xnew;
    pt->y = ynew;
}

void CnvtSetRegionSegment(int index, int start, int end)
{
    int max,err;

    max = CnvtMaxAttributeEntries();
    if (index >= max)  {
        max = (int)(1.5*index);
        if (err = CnvtReallocateRegionAttribArray(max))
            OnError( "Reallocating region attrib array",err );
    }
    RgnAttributes.elems[index].segment[0] = start;
    RgnAttributes.elems[index].segment[1] = end;
}

void CnvtRegionSegment(int index, int *seg)
{
    seg[0] = RgnAttributes.elems[index].segment[0];
    seg[1] = RgnAttributes.elems[index].segment[1];
}

void CnvtSetInteriorPts()
{
    int i, j, k, m, err=0, max;
    int num_nodes, *ids;
    int num_poly, scnt, start, next, found;
    int clipped;
    int xflags[2], yflags[2];
    double xoffset;
    Coords current, curr, prev, bl, tr;
    Coords pos, pos_a, pos_b;
    double *coords, pnt[2], *ptr, minarea=1e-9;
    double *flynncoords;
    CellData unit;
	vector<Coords> pv1, pu, checkcell;
	vector<vector<Coords> > res, unitcells;

    /*
     * create poly for cell
     */
    ElleCellBBox(&unit);
    /*xoffset = unit.cellBBox[TOPLEFT].x - unit.cellBBox[BASELEFT].x;*/
    xoffset = unit.xoffset;
#if XY
    for (i=0;i<4;i++) pu.push_back(unit.cellBBox[i]);
    /* index 0 is basic unitcell */
    unitcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    /* index 1 is  +y */
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x + xoffset;
        pos.y = unit.cellBBox[i].y + unit.ylength;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    checkcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    /* index 2 is +x +y */
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x + unit.xlength + xoffset;
        pos.y = unit.cellBBox[i].y + unit.ylength;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    checkcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    /* index 3 is  +x */
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x + unit.xlength;
        pos.y = unit.cellBBox[i].y;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    checkcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    /* index 4 is  +x -y */
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x + unit.xlength - xoffset;
        pos.y = unit.cellBBox[i].y - unit.ylength;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    checkcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    /* index 5 is   -y */
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x - xoffset;
        pos.y = unit.cellBBox[i].y - unit.ylength;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    checkcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    /* index 6 is  -x -y */
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x - xoffset - unit.xlength;
        pos.y = unit.cellBBox[i].y - unit.ylength;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    checkcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    /* index 7 is  -x */
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x - unit.xlength;
        pos.y = unit.cellBBox[i].y;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    checkcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    /* index 8 is  -x +y */
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x - unit.xlength + xoffset;
        pos.y = unit.cellBBox[i].y + unit.ylength;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    checkcells.push_back(pu);
#endif
    
#ifdef FLYNN_VER
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i)) {
            ElleFlynnNodes(i,&ids,&num_nodes);
#else
    max = ElleMaxGrains();
    for (i=0;i<max;i++) {
        if (ElleGrainIsActive(i)) {
            ElleGrainNodes(i,&ids,&num_nodes);
#endif
    /*
     * set the first point then check for wrapping grain
     */
            flynncoords = new double[num_nodes*2];
            ElleNodePosition(ids[0],&current);
            prev = current;
            j=0;
            do {
                ElleNodePlotXY(ids[j],&current,&prev);
                pv1.push_back(current);
                flynncoords[j]=current.x;
                flynncoords[j+num_nodes]=current.y;
                j++;
                prev = current;
            } while (j<num_nodes);

            ElleCheckUnit(&flynncoords[0],&flynncoords[num_nodes],num_nodes,
                          xflags,yflags,&bl,&tr);
            clipped = 0;
            unitcells.clear();
            scnt=0;
            for (j=scnt;j<=xflags[0]+1;j++) {
                checkcell.clear();
                pu.clear();
                for (m=0;m<4;m++) {
                    pos.x = unit.cellBBox[m].x - j*unit.xlength;
                    pos.y = unit.cellBBox[m].y;
                    pu.push_back(pos);
                    checkcell.push_back(pos);
                }
                unitcells.push_back(pu);
                for (k=0;k<yflags[0];k++) {
                    pu.clear();
                    for (m=0;m<4;m++) {
                        pos.x = checkcell[m].x-(k+1)*xoffset;
                        pos.y = checkcell[m].y-(k+1)*unit.ylength;
                        pu.push_back(pos);
                    }
                    unitcells.push_back(pu);
                }
                for (k=0;k<yflags[1];k++) {
                    pu.clear();
                    for (m=0;m<4;m++) {
                        pos.x = checkcell[m].x+(k+1)*xoffset;
                        pos.y = checkcell[m].y+(k+1)*unit.ylength;
                        pu.push_back(pos);
                    }
                    unitcells.push_back(pu);
                }
            }
            for (j=scnt;j<xflags[1]+1;j++) {
                checkcell.clear();
                pu.clear();
                for (m=0;m<4;m++) {
                    pos.x = unit.cellBBox[m].x + (j+1)*unit.xlength;
                    pos.y = unit.cellBBox[m].y;
                    pu.push_back(pos);
                    checkcell.push_back(pos);
                }
                unitcells.push_back(pu);
                for (k=0;k<yflags[0];k++) {
                    pu.clear();
                    for (m=0;m<4;m++) {
                        pos.x = checkcell[m].x-(k+1)*xoffset;
                        pos.y = checkcell[m].y-(k+1)*unit.ylength;
                        pu.push_back(pos);
                    }
                    unitcells.push_back(pu);
                }
                for (k=0;k<yflags[1];k++) {
                    pu.clear();
                    for (m=0;m<4;m++) {
                        pos.x = checkcell[m].x+(k+1)*xoffset;
                        pos.y = checkcell[m].y+(k+1)*unit.ylength;
                        pu.push_back(pos);
                    }
                    unitcells.push_back(pu);
                }
            }
           
m=unitcells.size();
num_poly = res.size();
            for (j=0;j<unitcells.size();j++) {
                gpcclip(unitcells[j],pv1,res,GPC_INT);
            }
        /*
            for (j=0;j<9;j++) {
                gpcclip(unitcells[j],pv1,res,GPC_INT);
            }
         * gpc returns the flynn inside the unit cell and the
         * part(s) of the flynn outside the unitcell.
         * Each returned polygon that is outside the unitcell 
         * starts at a boundary pt. (?)
         * REWRITE USING THESE POLYGONS RATHER THAN CHECK NODES
         * AND SEGMENTS
         */
            num_poly = res.size();
            for (int m=0;m<num_poly;m++) {
                int num= res[m].size();
                /*if (num < 3) OnError("clipped less than triangle",0);*/
                if (num >= 3) {
                    coords = new double[num*2];
                    for (k=0, ptr=coords; k<num; k++) {
                        prev = res[m][k];
                        *ptr = (double)prev.x; ptr++;
                        *ptr = (double)prev.y; ptr++;
                    }
                    /*
                     * walk around until valid interior pt found
                     */
                    start=num/2; next = (start+1)%num;
                    found = 0;
                    do {
                        prev = res[m][start];
                        current = res[m][next];
                        CnvtFindInteriorPt(&prev,&current,&pos);
                        pnt[0] = pos.x;
                        pnt[1] = pos.y;
                        found = CrossingsTest(coords,num,pnt);
                        start = next;
                        next = (next+1)%num;
                    } while (start!=0 && !found);
                    if (found) CnvtSetRegionInteriorPt(i,&pos);
                    else {
                        // check area as gcclip can return a polygon
                        // with 3 points the same if an elle node is
                        // right on the cell boundary
                        // ignore, if this is the case
                        double *xpts = new double[num];
                        double *ypts = new double[num];
                        for (k=0, ptr=coords; k<num; k++) {
                            xpts[k] = *ptr; ptr++;
                            ypts[k] = *ptr; ptr++;
                        }
                        double parea = polyArea(xpts,ypts,num);
                        delete[] xpts;
                        delete[] ypts;
                        if (parea > minarea)
                            fprintf(stderr,"Interior point not found");
                    }
                    delete[] coords;
                }
/*
                for (k=0; k<num; k++) {
                    pos = res[m][k];
					cout << pos.x << ',' << pos.y << "   ";
                }
				cout << endl;
*/
            }
            pv1.clear();
            num_poly = res.size();
            for (j=0;j<num_poly;j++) {
                res[j].clear();
            }
            res.clear();
            delete[] flynncoords;
        }
    }
}
#if XY
void CnvtSetInteriorPts()
{
    int i, j, k, err=0, max;
    int num_nodes, *ids;
    int num_poly, start, next, found;
    int clipped;
    int xflags[2],yflags[2];
    float xoffset;
    Coords current, prev;
    Coords pos, bl, tr;
    double *coords, pnt[2], *ptr, minarea=1e-9;
    CellData unit;
	vector<Coords> unitcell, pu;
	vector<vector<Coords> > res, pv1;

    /*
     * create poly for cell
     */
    ElleCellBBox(&unit);
    xoffset = unit.cellBBox[TOPLEFT].x - unit.cellBBox[BASELEFT].x;
    for (i=0;i<4;i++) unitcell.push_back(unit.cellBBox[i]);
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x + xoffset;
        pos.y = unit.cellBBox[i].y + unit.ylength;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x + unit.xlength + xoffset;
        pos.y = unit.cellBBox[i].y + unit.ylength;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x + unit.xlength;
        pos.y = unit.cellBBox[i].y;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x + unit.xlength - xoffset;
        pos.y = unit.cellBBox[i].y - unit.ylength;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x - xoffset;
        pos.y = unit.cellBBox[i].y - unit.ylength;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x - xoffset - unit.xlength;
        pos.y = unit.cellBBox[i].y - unit.ylength;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x - unit.xlength;
        pos.y = unit.cellBBox[i].y;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    for (i=0;i<4;i++) pu.pop_back();
    for (i=0;i<4;i++) {
        pos.x = unit.cellBBox[i].x - unit.xlength + xoffset;
        pos.y = unit.cellBBox[i].y + unit.ylength;
        pu.push_back(pos);
    }
    unitcells.push_back(pu);
    
#ifdef FLYNN_VER
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i)) {
            ElleFlynnNodes(i,&ids,&num_nodes);
#else
    max = ElleMaxGrains();
    for (i=0;i<max;i++) {
        if (ElleGrainIsActive(i)) {
            ElleGrainNodes(i,&ids,&num_nodes);
#endif
    /*
     * set the first point then check for wrapping grain
     */
            ElleNodePosition(ids[0],&current);
            prev = current;
            j=0;
            do {
                ElleNodePlotXY(ids[j],&current,&prev);
                pu.push_back(current);
                j++;
                prev = current;
            } while (j<num_nodes);
            pv1.push_back(pu);
            ElleCheckUnit(pu,xflags,yflags,&bl,&tr);
            for (k=0;k<num_nodes;k++) pu.pop_back();

            clipped = 0;
            for (j=0;j<pv1.size();j++) {
                gpcclip(pv1[j],unitcell,res,GPC_INT);
            }
        /*
         * gpc returns the flynn inside the unit cell and the
         * part(s) of the flynn outside the unitcell.
         * Each returned polygon that is outside the unitcell 
         * starts at a boundary pt. (?)
         * REWRITE USING THESE POLYGONS RATHER THAN CHECK NODES
         * AND SEGMENTS
         */
            num_poly = res.size();
            for (int m=0;m<num_poly;m++) {
                int num= res[m].size();
                /*if (num < 3) OnError("clipped less than triangle",0);*/
                if (num >= 3) {
                    coords = new double[num*2];
                    for (k=0, ptr=coords; k<num; k++) {
                        prev = res[m][k];
                        *ptr = (double)prev.x; ptr++;
                        *ptr = (double)prev.y; ptr++;
                    }
                    /*
                     * walk around until valid interior pt found
                     */
                    start=0; next = 1;
                    found = 0;
                    do {
                        prev = res[m][start];
                        current = res[m][next];
                        CnvtFindInteriorPt(&prev,&current,&pos);
                        pnt[0] = (double)pos.x;
                        pnt[1] = (double)pos.y;
                        found = CrossingsTest(coords,num,pnt);
                        start = next;
                        next = (next+1)%num;
                    } while (start!=0 && !found);
                    if (found) CnvtSetRegionInteriorPt(i,&pos);
                    else {
                        // check area as gcclip can return a polygon
                        // with 3 points the same if an elle node is
                        // right on the cell boundary
                        // ignore, if this is the case
                        double *xpts = new double[num];
                        double *ypts = new double[num];
                        for (k=0, ptr=coords; k<num; k++) {
                            xpts[k] = *ptr; ptr++;
                            ypts[k] = *ptr; ptr++;
                        }
                        double parea = polyArea(xpts,ypts,num);
                        delete[] xpts;
                        delete[] ypts;
                        if (parea > minarea)
                            fprintf(stderr,"Interior point not found");
                    }
                    delete[] coords;
                }
/*
                for (k=0; k<num; k++) {
                    pos = res[m][k];
					cout << pos.x << ',' << pos.y << "   ";
                }
				cout << endl;
*/
            }
            for (j=0;j<num_nodes;j++) pv1.pop_back();
            num_poly = res.size();
            for (j=0;j<num_poly;j++) {
                num_nodes = res[j].size();
                for (k=0; k<num_nodes; k++) res[j].pop_back();
                res.pop_back();
            }
        }
    }
}

int CnvtRegionNumInteriorPts(int index)
{
    int max;

    max = CnvtMaxAttributeEntries();
    if ((index >= max)|| !CnvtAttributeEntryIsActive(index))
         OnError("CnvtRegionNumInteriorPts",GRNNUM_ERR);
    return(RgnAttributes.elems[index].num_interior_pts);
}
#endif

/*
 * small corner/edge flynn may disappear when nodes moved to boundary
 */
int CnvtRegionNumInteriorPts(int index)
{
    int max, num=0;

    max = CnvtMaxAttributeEntries();
    if ((index < max)&& CnvtAttributeEntryIsActive(index))
        num = RgnAttributes.elems[index].num_interior_pts;
    return(num);
}

void CnvtSetRegionInteriorPt(int index, Coords *pos)
{
    int max, err;
    int ptindex;
    Coords unitpos;

    max = CnvtMaxAttributeEntries();
    if (index >= max)  {
        max = (int)(1.5*index);
        if (err = CnvtReallocateRegionAttribArray(max))
            OnError( "Reallocating region attrib array",err );
    }
    RgnAttributes.elems[index].active=1;
    ptindex = RgnAttributes.elems[index].num_interior_pts;
    if (ptindex == MAX_INTERIOR_PTS)
        OnError("Too many interior pts",0);
    unitpos = *pos;
    ElleNodeUnitXY(&unitpos);
    RgnAttributes.elems[index].interior_pt[ptindex] = unitpos;
    RgnAttributes.elems[index].num_interior_pts++;
}

int CnvtRegionInteriorPt(int index, int ptindex, Coords *pos)
{
    int max;

    max = CnvtMaxAttributeEntries();
    if ((index >= max)|| !CnvtAttributeEntryIsActive(index))
         OnError("CnvtRegionInteriorPt",GRNNUM_ERR);
    if (ptindex >= RgnAttributes.elems[index].num_interior_pts)
        return(0);
    *pos = RgnAttributes.elems[index].interior_pt[ptindex];
    return(1);
}

void CnvtSetRegionAttributeValue(int index, float val)
{
    int max, err;

    max = CnvtMaxAttributeEntries();
    if (index >= max)  {
        max = (int)(1.5*index);
        if (err = CnvtReallocateRegionAttribArray(max))
            OnError( "Reallocating region attrib array",err );
    }
    RgnAttributes.elems[index].active=1;
    RgnAttributes.elems[index].value = val;
}

void CnvtRegionAttributeValue(int index, float *val)
{
    int max;

    max = CnvtMaxAttributeEntries();
    if ((index >= max)|| !CnvtAttributeEntryIsActive(index))
         OnError("CnvtRegionAttributeValue",GRNNUM_ERR);
    *val = RgnAttributes.elems[index].value;
}

int CnvtAttributeEntryIsActive(int index)
{
    return(RgnAttributes.elems[index].active);
}

void CnvtInitMirrorNodes()
{
    int i;

    for (i=0;i<MIRROR_MAX;i++)
        Mirror_nodes[i][0] = Mirror_nodes[i][1] = NO_NB;
}

int CnvtSetMirrorNode(int node, int mirror)
{
    int i, found=0;

    for (i=0;i<MIRROR_MAX && Mirror_nodes[i][0]!=NO_NB && !found; i++) {
        if (Mirror_nodes[i][0]==node) found = 1;
    }
    if (i==MIRROR_MAX) OnError("mirror array size exceeded",0);
    else if (Mirror_nodes[i][0]==NO_NB) {
        Mirror_nodes[i][0]=node;
        Mirror_nodes[i][1]=mirror;
    }
    else if (found && Mirror_nodes[i][1]!=mirror)
        OnError("mirror node already set",0);
    return(0);
}

int CnvtGetMirrorNode(int node)
{
    int i, found=0, mirror;

    mirror = NO_NB;
    for (i=0;i<MIRROR_MAX && Mirror_nodes[i][0]!=NO_NB && !found; i++) {
        if (Mirror_nodes[i][0]==node || Mirror_nodes[i][1]==node) {
            if (Mirror_nodes[i][0]==node) mirror = Mirror_nodes[i][1];
            else mirror = Mirror_nodes[i][0];
            found = 1;
        }
    }
    return(mirror);
}

int CnvtWriteUnodeViscosity(FILE *fpout)
{
	int err=0, max, maxf, i, j;
	int *rgn, len;
	double dflt, val;
	Coords pos;

	max = ElleMaxUnodes();
	/*
	 * setup array where index is unode id and
	 * entry is the flynn in which the unode is
	 * located
	 */
	if ((rgn=new int[max])==0) 
		OnError("CnvrtWriteUnodeViscosity",MALLOC_ERR);
	for (i=0;i<max;i++) rgn[i]=NO_VAL;
	maxf = ElleMaxFlynns();
	for (i=0;i<maxf;i++) {
		if (ElleFlynnIsActive(i)) {
			vector <int> l;
			ElleGetFlynnUnodeList(i,l);
			len = l.size();
			for (j=0;j<len;j++) rgn[l[j]] = i;
		}
	}
	/*
	 * Unode Section of poly file
	 * keyword
	 * no.of unodes (assuming all are active)
	 * id x y flynn (one line per unode)
	 */
    if (fprintf(fpout,"%s\n",UNODES_KEY) < 0) err=WRITE_ERR;
    if (fprintf(fpout,"%d\n",max) < 0) err=WRITE_ERR;
	for (i=0;i<max && !err;i++) {
		ElleGetUnodePosition(i,&pos);
		if (fprintf(fpout,"%d %.8lf %.8lf %d\n",i,pos.x,pos.y,rgn[i]) < 0)
				err=WRITE_ERR;
	}
	if (rgn) delete[] rgn;
	/*
	 * Unode Viscosity Section of poly file
	 * keyword
	 * Default keyword and default value
	 * id value (one line per unode)
	 */
	if (!err) {
		if (fprintf(fpout,"%s\n",U_VISCOSITY_KEY) < 0) err=WRITE_ERR;
		ElleGetDefaultUnodeAttribute(&dflt,U_VISCOSITY);
        fprintf(fpout,"%s %.8e\n",DEFAULT,dflt);
		for (i=0;i<max && !err;i++) {
			ElleGetUnodeAttribute(i,&val,U_VISCOSITY);
			if (val!=dflt)
				if (fprintf(fpout,"%d %.8e\n",i,val) < 0) err=WRITE_ERR;
		}
	}
	return(err);
}

