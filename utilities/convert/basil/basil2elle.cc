#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include "nodes.h"
#include "version.h"
#include <string.h>
#include <math.h>
#include "cmndefs.h"
#include "types.h"
#include "data.h"
#include "arrays.h"
#include "sybfile.h"
#include "strain.h"
#include "error.h"
#include "bflynns.h"
#include "regions.h"
#include "interface.h"
#include "file.h"
#include "file_utils.h"
#include "runopts.h"
#include "unodes.h"
#include "general.h"
#include "crossings.h"

#include "polygon.h"

using std::ios;
using std::ofstream;
using std::string;
using std::vector;

plot_data Plot_info;
input_options Settings, Initial_Settings;
float Pwindo[PWINDO_ENTRIES];

int FindBasilIndex(int **basil_index_adr,int bmax,float *ex, float *ey,
                   int *ibc,int bndmax,int *max_index,char *err_str);
int CalcFlynnStrain(float **arraysf,int **arraysi,
               float *fl_vars,int *int_vars, int *basil_index,
               int *keys, int *count);
int CalcStrain(float **arraysf,int **arraysi,
               float *fl_vars,int *int_vars, int *basil_index,
               int *keys, int *count);
int CalcFlynnStress(float **arraysf,int **arraysi,
               float *fl_vars,int *int_vars, int *basil_index,
               int *keys, int *count);
int CalcStress(float **arraysf,int **arraysi,
               float *fl_vars,int *int_vars, int *basil_index,
               int *keys, int *count);
int CalcUnodeStrain(float time_ref,float **arraysf,int **arraysi,
                    float *fl_vars,int *int_vars,Coords *newbox);
int CopyData( FILE *fp, FILE **fpout, char *str, Coords *newbox,
              float time_ref);
int CalcUnodePosition(float time_ref,float **arraysf,int **arraysi,
                    float *fl_vars,int *int_vars,Coords *newbox);
int CopyUnodeData( FILE *fp, FILE **fpout, char *str, Coords *newbox,
              float dt);
void copyFlynnAttributesToUnode(int unodeid, int flynnid);
int ReadNodeData( FILE *fp, char *str);
void FindBBox(float *xvals,float *yvals,int size,
              Coords *bl,Coords *br,Coords *tr,Coords *tl);
int WriteVelocity(FILE *fp,float *uvp,int *int_vars,int *basil_index);
int WriteStress(FILE *fp,float *vals,int *int_vars,int *basil_index);
int UpdatePositions(float **arraysf,int **arraysi,float *fl_vars,int *int_vars,
                    float time_ref,int *basil_index,Coords *newbbox);
int UpdateIntrpPositions(float **arraysf,int **arraysi,float *fl_vars,int *int_vars,
                    float time_ref,Coords *newbbox);
int UpdateUnodePos(float dt);
void Init_Plot();
#ifdef __cplusplus
extern "C" {
#endif
int lgdef_(float *,float *,float *,float *,float *,float *,
           float *,float *,float *,float *,float *,float *,
           int *,int *,
           int *,int *,int *,int *,int *,int *,int *,int *,
           int *,int *);
int strain_(int *,int *,float *,float *,float *,float *,float *,
            float *,float *,float *,float *,float *,float *,
            int *,int *,int *,int *,int *,int *,int *);
int strainavg_(int *,float *,float *,float *,float *,float *,
               int *,int *,int *,float *,float *,
               int *,int *,int *);
int matchmeshnode_(int *,int *,double *,double *,
                               float *, float *,
                               int *, int *, int *,
                               int *,int *,int *);
int strainpnt_(int *,double *,double *,float *,float *,float *,
               float *,int *,int *,int *,float *,float *,
               int *,int *,int *,int *);
int straic_(int *,int *,float *,float *,int *,int *,
            float *,float *,float *,float *,float *,float *,
            float *,float *,
            float *,float *,float *,int *,int *, int *, int *,int *,int *,
            int *,int *,int *,int *);
int stressavg_(int *,int *,float *,float *,float *,float *,
            float *,float *,float *,float *,float *,
            float *,float *,float *,int *,int *, int *, int *,int *,int *,
            int *,int *,int *,int *,int *,int *);
int ntrpltpnt_(float*,float *,int *,double *,double *,
               float *,float *,float *,float *,
               int *,int *,int *,int *,int *,int *);
int ntrpltpres_(float *,float *,int *,int *,int *,int *,int *);
int ntrpln_(float *,float *,int *,float *,float *,float *,float *,float *,
					float *,float *,float *,
					float *,float *,int *, int *,int *,
					int *,int *,int *,int *);
int ParseOptions(int argc,char **argv,char **in,char **out,int *record);
#ifdef __cplusplus
}
#endif

char *Infile, *Outfile;
main(int argc, char *argv[])
{
    char Input_str[1024], *cptr, buf[81];
    bool f2u = false;
    int err=0, i, j, max;
    int record=1;
    float time_ref;
    FILE *ellein, *elleout;
    Coords newbbox[4], oldbbox[4];

    if (argc==1 || ParseOptions(argc,argv,&Infile,&Outfile,
                         &record))
        exit(1);
    /*
     * this is done in the plot lib ParseOptions which
     * needs to be made more flexible so it can be used here
     */
    ElleSetAppName(argv[0]);

    ElleInitNodeArray();
    ElleInitRunOptions();

    ElleSetDisplay(0);

    Init_Plot();
    init_arrays();
    if (err = file_open( &(Plot_info.inp_file),Infile,
                            SYB_FILENAME_MAX,"rb" ))
        OnError(Infile,err);
    /*
     * read the first record
     * and match node locations with initial ex,ey, storing indices
     * Store time of this record - read_ref doesn't read vars
     */
    err=read_data(Plot_info.inp_file->fp,
                    String_vars,
                    Data_vars_int,Data_vars_fl,
                    Data_arrays_int,Data_arrays_fl,
                    &Pwindo[XCMIN], &Pwindo[XCMAX],
                    &Pwindo[YCMIN], &Pwindo[YCMAX],
                    &Plot_info.inp_file->rec_curr,
                    Plot_info.inp_file->rec_req,
                    &Plot_info.inp_file->rec_max,
                    0 /* rotate */
                    );
    if (err) OnError("",err);
    time_ref = Data_vars_fl[TIME];

    /* read elle filename from basil file comments */
    for (i=0,j=COMMENTS_START;i<80;i++,j++)
        Input_str[i]=String_vars[j];
    Input_str[i]='\0';
    sscanf(Input_str,"%s",buf);
    if (strcmp(buf,"ELLE")) OnError("Keyword ELLE not found",0);
    sscanf(&Input_str[strlen(buf)],"%s",buf);
    if ((ellein = fopen( buf, "r" ))==0) OnError(buf,OPEN_ERR);
    if (!strcmp(buf,"vor2flynn.elle")) f2u = true;

    /* elle write filename */
    if ((elleout = fopen( Outfile, "w" ))==0) OnError(Outfile,OPEN_ERR);

    /* read x,y */
    if (err=ReadNodeData(ellein,Input_str)) {
        fclose(ellein);
        OnError(Input_str,err);
    }
    rewind(ellein);

    max = Data_vars_int[NUP];
    if (Data_vars_int[IMSH]==3 && Data_arrays_int[ELLENODE]==0) {
        if (err=FindBasilIndex(&(Data_arrays_int[ELLENODE]),max,
                           Data_arrays_fl[EX], Data_arrays_fl[EY],
                           Data_arrays_int[IBC],Data_vars_int[NBP],
                           &Data_vars_int[NELLEP],Input_str))
        OnError("FindBasilIndex",err);
    }
    /*
     * read the reference timestep ( set at 1 )
     * reads x,y vals into EXREF, EYREF
     */
    Plot_info.inp_file->ref_req = 1;
    if (err=read_reference(Plot_info.inp_file->fp,
                    Data_vars_int,Data_arrays_fl,
                    Plot_info.inp_file->rec_curr,
                    &Plot_info.inp_file->ref_curr,
                    Plot_info.inp_file->ref_req,
                    0 /* rotate */
                    ))
         OnError("reading reference data",err);
    /*
     * read the requested timestep
     */
    if (record!=1) {
        Plot_info.inp_file->rec_req = record;
        err=read_data(Plot_info.inp_file->fp,
                        String_vars,
                        Data_vars_int,Data_vars_fl,
                        Data_arrays_int,Data_arrays_fl,
                        &Pwindo[XCMIN], &Pwindo[XCMAX],
                        &Pwindo[YCMIN], &Pwindo[YCMAX],
                        &Plot_info.inp_file->rec_curr,
                        Plot_info.inp_file->rec_req,
                        &Plot_info.inp_file->rec_max,
                        0 /* rotate */
                        );
    }
    fclose(Plot_info.inp_file->fp);
    /*
     * get the bounding box for the cell from the requested timestep
    FindBBox(Data_arrays_fl[EX],Data_arrays_fl[EY], Data_vars_int[NUP],
                &newbbox[0],&newbbox[1],&newbbox[2],&newbbox[3]);
    ElleUpdateCellBBox(&newbbox[0],&newbbox[1],
                    &newbbox[2],&newbbox[3]);
     */
    /*
     * Set the node positions to the basil positions
     * Write the data (other than velocity or stress)
     * including the new bounding box
     * Write velocity data for stored indices
     * Calc stress data and write it for stored indices
     */

    if (Data_arrays_int[ELLENODE] && Data_vars_int[IMSH]==3)
      UpdatePositions(Data_arrays_fl,Data_arrays_int,Data_vars_fl,Data_vars_int,
                        time_ref,Data_arrays_int[ELLENODE],newbbox);
	else
	  UpdateIntrpPositions(Data_arrays_fl,Data_arrays_int,Data_vars_fl,Data_vars_int,
                          time_ref,newbbox);
    /*
     * need to change the io in this programme to iostream
     * syntax. Til then, need to use pointer from string object
     */
    string outversion = ElleGetCreationString();
    if (fprintf(elleout,"%s\n",outversion.c_str()) < 0)
        OnError(Outfile,WRITE_ERR);
    if (f2u==true)
      err = CopyUnodeData(ellein, &elleout, Input_str, newbbox,
                          Data_vars_fl[TIME]-time_ref);
    else
      err = CopyData(ellein, &elleout, Input_str, newbbox, time_ref);
    fclose(ellein);
    fclose(elleout);
    clear_arrays();
    return(err);
}

int CalcFlynnStrain(float **arraysf,int **arraysi,
               float *fl_vars,int *int_vars, int *basil_index,
               int *keys, int *count)
{
    int err=0;
    int i, j, k, n;
    int offset, *iptr, max, nmax, index, id;
    int verbose=0;
    float straintensor[4],avgstrain;
    double tmp, val;

    if (arraysf[SNTRP]==NULL)
        if ((arraysf[SNTRP]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
                return(MALLOC_ERR);
    *count=0;
    keys[0]=E_XX;
    (*count)++;
    keys[1]=E_XY;
    (*count)++;
    keys[2]=E_YX;
    (*count)++;
    keys[3]=E_YY;
    (*count)++;
    /*
     *****WARNING - dumping these
    keys[4]=F_INCR_S; 
    (*count)++;
    keys[*count]=F_BULK_S;
    (*count)++;
    */
    keys[4]=INCR_S;  /* max principal in-plane strain */
    (*count)++;
    keys[*count]=BULK_S;
    (*count)++;
    for(i=0;i<(*count);i++)
        ElleInitFlynnAttribute(keys[i]);
    max = ElleMaxFlynns();
    /*
     * for each flynn
     */
    for (j=0;j<max;j++) {
        if (ElleFlynnIsActive(j)) {
            strainavg_(&j,arraysf[EX],arraysf[EY],
                       arraysf[EXREF],arraysf[EYREF],arraysf[SNTRP],
                       arraysi[LEM],arraysi[NOR],arraysi[POLYN],
                       straintensor,&avgstrain,
                       &int_vars[NE],&int_vars[NUP],
                       &verbose);
            for (i=0;i<4;i++) 
                ElleSetFlynnRealAttribute(j,(double)straintensor[i],
                                                       keys[i]);
            tmp = (double)avgstrain;
            ElleSetFlynnRealAttribute(j,tmp,INCR_S);
            ElleGetFlynnRealAttribute(j,&val,BULK_S);
            ElleSetFlynnRealAttribute(j,val+tmp,BULK_S);
        }
    }
    if (arraysf[SNTRP]!=NULL) {
        free(arraysf[SNTRP]);
        arraysf[SNTRP]=NULL;
    }
    return(err);
}
#if XY
 /* this just does an average of the flynn boundary node vals */
int CalcFlynnStrain(float **arraysf,int **arraysi,
               float *fl_vars,int *int_vars, int *basil_index,
               int *keys, int count)
{
    int err=0;
    int plot_type, jell=0, opt, i, j, k, n;
    int offset, *iptr, max, nmax, index, id;
    int *node_ids, num_nodes;
    float *ptr,val,total;

    plot_type = CNTRS;
    if (arraysf[AMESH]==NULL)
        if ((arraysf[AMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
               return(MALLOC_ERR);
    if (arraysf[BMESH]==NULL)
        if ((arraysf[BMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
                return(MALLOC_ERR);
    if (arraysf[CMESH]==NULL)
        if ((arraysf[CMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
                return(MALLOC_ERR);
    if (arraysf[DMESH]==NULL)
        if ((arraysf[DMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
                return(MALLOC_ERR);
    if (arraysf[EMESH]==NULL)
        if ((arraysf[EMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
                return(MALLOC_ERR);
    if (arraysi[IHELP]==NULL)
        if ((arraysi[IHELP]=(int *)malloc(int_vars[NUP]*sizeof(int)))
                             ==NULL) return(MALLOC_ERR);
    for(i=0;i<count;i++) {
        ElleInitFlynnAttribute(keys[i]);
        if (keys[i]==F_INCR_S) ElleInitFlynnAttribute(F_BULK_S);
    }
    max = ElleMaxFlynns();
    nmax = ElleMaxNodes();
    for(i=0;i<count;i++) {
        switch(keys[i]) {
        case E_XX: opt = EDXX;
              break;
        case E_XY: opt = EDXY;
              break;
        case E_YY: opt = EDYY;
              break;
        case E_ZZ: opt = EDZZ;
              break;
        case F_INCR_S: opt = PSR1;
              break;
        case INCR_S: opt = PSR1;
              break;
        default: err = ATTRIBID_ERR;
              break;
        }
        if (err) return(err);
        strain_(&jell,&opt,arraysf[AMESH],arraysf[BMESH],
                arraysf[CMESH],arraysf[DMESH],arraysf[EMESH],
                &fl_vars[SE],&fl_vars[BIG],
                arraysf[EX],arraysf[EY],arraysf[VHB],arraysf[UVP],
                arraysi[LEM],arraysi[NOR],
                &int_vars[NE],&int_vars[NUP],&int_vars[NROWS],
                &int_vars[NP3],&int_vars[IVIS]
                );
        ptr = arraysf[AMESH];
        /*
         * for each flynn
         */
        for (j=0;j<max;j++) {
            if (ElleFlynnIsActive(j)) {
                total = 0;
                ElleFlynnNodes(j, &node_ids, &num_nodes);
                for (k=0;k<num_nodes;k++) {
                /*
                 * find the average for this parameter
                 */
                    id = basil_index[node_ids[k]];
                    if (id < 0 || id > nmax)
                        fprintf(stderr,"node out of range %d\n",id);
                    total += ptr[id];
                }
                total /= num_nodes;
                total *= fl_vars[TIME]; /* strain-rate -> strain */
                ElleSetFlynnStrain(j,total,keys[i]);
                if (keys[i]==F_INCR_S) {
                    ElleGetFlynnStrain(j,&val,F_BULK_S);
                    ElleSetFlynnStrain(j,val+total,F_BULK_S);
                }
            }
        }
    }
    return(err);
}
#endif

int CalcStrain(float **arraysf,int **arraysi,
               float *fl_vars,int *int_vars, int *basil_index,
               int *keys, int *count)
{
    int plot_type, jell=0, opt, i, j, k, n;
    int offset, *iptr, emax, index, id;
    float *ptr;
    double tmp, val;

    /* zero unused variables */

    plot_type = CNTRS;
    if (arraysf[AMESH]==NULL)
        if ((arraysf[AMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
               return(MALLOC_ERR);
    if (arraysf[BMESH]==NULL)
        if ((arraysf[BMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
                return(MALLOC_ERR);
    if (arraysf[CMESH]==NULL)
        if ((arraysf[CMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
                return(MALLOC_ERR);
    if (arraysf[DMESH]==NULL)
        if ((arraysf[DMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
                return(MALLOC_ERR);
    if (arraysf[EMESH]==NULL)
        if ((arraysf[EMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
                return(MALLOC_ERR);
    if (arraysi[IHELP]==NULL)
        if ((arraysi[IHELP]=(int *)malloc(int_vars[NUP]*sizeof(int)))
                             ==NULL) return(MALLOC_ERR);
    i=0;
    (*count)=0;
    ElleInitNodeAttribute(INCR_S);
    if (ElleBulkStrainIndex()==NO_INDX) {
        fprintf(stderr,"Initializing bulk strain attribute\n");
        ElleInitNodeAttribute(BULK_S);
    }
    opt = PSR1;
    strain_(&jell,&opt,arraysf[AMESH],arraysf[BMESH],
                arraysf[CMESH],arraysf[DMESH],arraysf[EMESH],
                &fl_vars[SE],&fl_vars[BIG],
                arraysf[EX],arraysf[EY],arraysf[VHB],arraysf[UVP],
                arraysi[LEM],arraysi[NOR],
                &int_vars[NE],&int_vars[NUP],&int_vars[NROWS],
                &int_vars[NP3],&int_vars[IVIS]
                );
    emax = int_vars[NELLEP];
    ptr = arraysf[AMESH];
    for (j=0;j<emax;j++) {
        if (basil_index[j]!=-1) {
            id = basil_index[j];
            val = ElleNodeAttribute(j,BULK_S);
            tmp = (double)(ptr[id]*fl_vars[TIME]);
            ElleSetNodeAttribute(j,tmp,INCR_S);
            ElleSetNodeAttribute(j,val+tmp,BULK_S);
        }
    }
    keys[0]=INCR_S;
    (*count)++;
    keys[1]=BULK_S;
    (*count)++;
    return(0);
}

int CalcFlynnStress(float **arraysf,int **arraysi,
               float *fl_vars,int *int_vars, int *basil_index,
               int *keys, int count)
{
    int plot_type, jell=0, opt, i, j, max, n;
    int verbose=0;
    float avgstress=0;

    /* zero unused variables */

    i=0;
    while (i<NUM_STRESS_VALS && i<count) {
        switch(keys[i]) {
        case TAU_XX: opt = TAXX;
                     break;
        case TAU_YY: opt = TAYY;
                     break;
        case TAU_ZZ: opt = TAZZ;
                     break;
        case TAU_XY: opt = TAXY;
                     break;
        case TAU_1 : opt = TAU1;
                     break;
        case PRESSURE: opt = PRES;
                     break;
        default:     return(ID_ERR);
                     break;
        }
        ElleInitFlynnAttribute(keys[i]);
    max = ElleMaxFlynns();
    /*
     * for each flynn
     */
    for (j=0;j<max;j++) {
        if (ElleFlynnIsActive(j)) {
            stressavg_(&j,&opt,&avgstress,
                       &fl_vars[SE],&fl_vars[ARGAN],&fl_vars[BRGAN],
                       &fl_vars[HLENSC],&fl_vars[BDEPSC],&fl_vars[BIG],
                       arraysf[EX],arraysf[EY],
                       arraysf[VHB],arraysf[UVP],arraysf[SSQ],
                       arraysi[IELFIX],
                       arraysi[LEM],arraysi[NOR],arraysi[POLYN],
                       &int_vars[NE],&int_vars[NUP],&int_vars[NUVP],
                       &int_vars[NFP],&int_vars[IVV],
                       &int_vars[NCOMP],&int_vars[ICR],
                       &verbose);
            ElleSetFlynnRealAttribute(j,(double)avgstress,keys[i]);
        }
    }
    i++;
    }
    return(0);
}

int CalcStress(float **arraysf,int **arraysi,
               float *fl_vars,int *int_vars, int *basil_index,
               int *keys, int *count)
{
    int plot_type, jell=0, opt, i, j, k, n;
    int offset, *iptr, bmax, emax, index, id;
    float *ptr, *pres;

    /* zero unused variables */

    plot_type = CNTRS;
    if (arraysf[AMESH]==NULL)
        if ((arraysf[AMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
               return(MALLOC_ERR);
    if (arraysf[BMESH]==NULL)
        if ((arraysf[BMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
                return(MALLOC_ERR);
    if (arraysf[CMESH]==NULL)
        if ((arraysf[CMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
                return(MALLOC_ERR);
    if (arraysf[DMESH]==NULL)
        if ((arraysf[DMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
                return(MALLOC_ERR);
    if (arraysf[EMESH]==NULL)
        if ((arraysf[EMESH]=(float *)
            malloc(int_vars[NUP]*sizeof(float)))==NULL)
                return(MALLOC_ERR);
    if (arraysi[IHELP]==NULL)
        if ((arraysi[IHELP]=(int *)malloc(int_vars[NUP]*sizeof(int)))
                             ==NULL) return(MALLOC_ERR);
    i=0;
    (*count)=0;
    offset = int_vars[NUP]*2 + int_vars[NFP];
    while (i<NumElements(StressKeys) && i<NUM_STRESS_VALS) {
        ElleInitNodeAttribute(StressKeys[i].id);
        switch(StressKeys[i].id) {
        case TAU_XX: opt = TAXX;
                     break;
        case TAU_YY: opt = TAYY;
                     break;
        case TAU_ZZ: opt = TAZZ;
                     break;
        case TAU_XY: opt = TAXY;
                     break;
        case TAU_1 : opt = TAU1;
                     break;
        case PRESSURE: opt = PRES;
                     break;
        default:     return(ID_ERR);
                     break;
        }
        /* set the starting addr to the next block in AMESH */
        /*ptr = arraysf[AMESH]+i*int_vars[NUP];*/
        ptr = arraysf[AMESH];
        strain_(&jell,&opt,ptr,arraysf[BMESH],
                arraysf[CMESH],arraysf[DMESH],arraysf[EMESH],
                &fl_vars[SE],&fl_vars[BIG],
                arraysf[EX],arraysf[EY],arraysf[VHB],arraysf[UVP],
                arraysi[LEM],arraysi[NOR],
                &int_vars[NE],&int_vars[NUP],&int_vars[NROWS],
                &int_vars[NP3],&int_vars[IVIS]
                );
        if (opt==PRES) {
            pres = arraysf[UVP] + offset;
            ntrpltpres_(ptr,pres,arraysi[NOR],arraysi[LEM],
                        &int_vars[NUP],&int_vars[NE],&int_vars[NN]);
/*
 * index into P part of UVP is NOR(LEM(k,n)) where
 * k=1,3 (only vertices) and n is the element no.
 * index into U,V part of UVP is LEM(k,n)
 * to interpolate onto AMESH sybil would use
 *   DO n = 1,NE
 *     DO k = 1,3
 *       LK = LEM(k,n)
 *       NLK = NOR(LK)
 *       AMESH(LK) = UVP(offset + NLK)
 *       K4 = MOD(k,3) + 3
 *       LK4 = LEM(K4,n)
 *       K3 = MOD(k+1,3)+1
 *       LK3 = LEM(K3,n)
 *       NLK3 = NOR(LK3)
 *       P1 = UVP(offset+NLK)
 *       P2 = UVP(offset+NLK3)
 *       AMESH(LK4) = 0.5*(P1+P2)
 */
        }
        bmax = int_vars[NUP];
        emax = int_vars[NELLEP];
        for (j=0;j<emax;j++) {
            if (basil_index[j]!=-1) {
                id = basil_index[j];
                ElleSetNodeAttribute(j,(double)(ptr[id-1]),StressKeys[i].id);
            }
        }
        keys[i]=StressKeys[i].id;
        (*count)++;
        i++;
    }
    return(0);
}

int CalcUnodePosition(float time_ref,float **arraysf,int **arraysi,
                    float *fl_vars,int *int_vars,Coords *newbox)
{
    int err=0;
    int i, j, k, n;
    int max;
    int mesh_flag=3; // lgdef should only calculate mesh
    int dumi=0, verbose=0;
    float *dumx=0, *dumy=0;
    float dt;
    float straintensor[4],avgstrain;
    double tmp, val, eps=1e-5, maxx,minx,maxy,miny;
    Coords incr, curr, tmpcoords, tol[8];

	tol[3].x=tol[4].x=tol[7].x=eps;
	tol[1].x=tol[5].x=tol[6].x=-eps;
	tol[0].y=tol[4].y=tol[5].y=eps;
	tol[2].y=tol[6].y=tol[7].y=-eps;
	tol[1].y=tol[3].y=0.0;
	tol[0].x=tol[2].x=0.0;
	minx = newbox[0].x;
	miny = newbox[0].y;
	maxx = newbox[2].x;
	maxy = newbox[2].y;
	if (newbox[3].x<minx)  minx = newbox[3].x;
	if (newbox[1].x>maxx)  maxx = newbox[1].x;
	if (newbox[3].y>maxy)  maxy = newbox[3].y;
	if (newbox[1].y<miny)  miny = newbox[1].y;

    max = ElleMaxUnodes();
    float *u_x = new float[max];
    float *u_y = new float[max];
    float *u_velx = new float[max];
    float *u_vely = new float[max];
    if (u_x==0 || u_y==0 || u_velx==0 || u_vely==0) {
        err = MALLOC_ERR;
        if (u_x) delete[] u_x;
        if (u_y) delete[] u_y;
        if (u_velx) delete[] u_velx;
        if (u_vely) delete[] u_vely;
    }
    dt = fl_vars[TIME]-time_ref;
    /*
     * for each unode
     */
    for (j=0;j<max && !err;j++) {
        ElleGetUnodePosition(j,&curr);
        u_x[j]=curr.x;
        u_y[j]=curr.y;
    }
    if (!err)
        lgdef_(&dt,arraysf[UVP],u_velx,u_vely,dumx,dumy,
               arraysf[EXREF],arraysf[EYREF],u_x,u_y,dumx,dumy,
               arraysi[LEM],arraysi[NOR],
               &int_vars[NUP],&int_vars[NE],&int_vars[NUP],&max,
               &dumi,&dumi,&dumi,&dumi,
               &mesh_flag,&int_vars[IFLT]);
    for (j=0;j<max && !err;j++) {
        ElleGetUnodePosition(j,&curr);
        /*
         * unode position in unit cell (deformed)
         */
	    curr.x += (double)(dt*u_velx[j]);
        curr.y += (double)(dt*u_vely[j]);

		if (curr.x>maxx) curr.x=maxx;
		if (curr.x<minx) curr.x=minx;
		if (curr.y>maxy) curr.y=maxy;
		if (curr.y<miny) curr.y=miny;
		/*
		 * this gives problems because the eps is set to float value
        ElleNodeUnitXY(&curr);
		*/
        ElleSetUnodePosition(j,&curr);
    }
    return(err);
}

int CalcUnodeStrain(float time_ref,float **arraysf,int **arraysi,
                    float *fl_vars,int *int_vars,Coords *newbox)
{
    int err=0;
    int i, j, k, n;
    int max;
    int mesh_flag=3; // lgdef should only calculate mesh
    int dumi=0, verbose=0;
    float *dumx=0, *dumy=0;
    float dt;
    float straintensor[4],avgstrain;
    double tmp, val, eps=1e-5, maxx,minx,maxy,miny;
    Coords incr, curr, tmpcoords, tol[8];

	tol[3].x=tol[4].x=tol[7].x=eps;
	tol[1].x=tol[5].x=tol[6].x=-eps;
	tol[0].y=tol[4].y=tol[5].y=eps;
	tol[2].y=tol[6].y=tol[7].y=-eps;
	tol[1].y=tol[3].y=0.0;
	tol[0].x=tol[2].x=0.0;
	minx = newbox[0].x;
	miny = newbox[0].y;
	maxx = newbox[2].x;
	maxy = newbox[2].y;
	if (newbox[3].x<minx)  minx = newbox[3].x;
	if (newbox[1].x>maxx)  maxx = newbox[1].x;
	if (newbox[3].y>maxy)  maxy = newbox[3].y;
	if (newbox[1].y<miny)  miny = newbox[1].y;

    max = ElleMaxUnodes();
    float *u_x = new float[max];
    float *u_y = new float[max];
    float *u_velx = new float[max];
    float *u_vely = new float[max];
    if (u_x==0 || u_y==0 || u_velx==0 || u_vely==0) {
        err = MALLOC_ERR;
        if (u_x) delete[] u_x;
        if (u_y) delete[] u_y;
        if (u_velx) delete[] u_velx;
        if (u_vely) delete[] u_vely;
    }
    dt = fl_vars[TIME]-time_ref;
    /*
     * for each unode
     */
    for (j=0;j<max && !err;j++) {
        ElleGetUnodePosition(j,&curr);
        u_x[j]=curr.x;
        u_y[j]=curr.y;
    }
    if (!err)
        lgdef_(&dt,arraysf[UVP],u_velx,u_vely,dumx,dumy,
               arraysf[EXREF],arraysf[EYREF],u_x,u_y,dumx,dumy,
               arraysi[LEM],arraysi[NOR],
               &int_vars[NUP],&int_vars[NE],&int_vars[NUP],&max,
               &dumi,&dumi,&dumi,&dumi,
               &mesh_flag,&int_vars[IFLT]);
    for (j=0;j<max && !err;j++) {
        ElleGetUnodePosition(j,&curr);
////!!!!! Either this should be done in file.cc when the attribute is
////      or (ElleUnodeAttributeActive(U_STRAIN) should check for any
////      strain value
        if (!ElleUnodeAttributeActive(U_STRAIN)) {
            ElleInitUnodeAttribute(U_STRAIN);
        }
        if (ElleUnodeAttributeActive(U_STRAIN)) {
            strainpnt_(&j,&curr.x,&curr.y,arraysf[EX],arraysf[EY],
                       arraysf[EXREF],arraysf[EYREF],
                       arraysi[LEM],arraysi[NOR],arraysi[POLYN],
                       straintensor,&avgstrain,
                       &int_vars[NE],&int_vars[NUP],
                       &verbose,&err);
			// problems with unodes on external boundaries
			i=0;
    		while (err && i<8) {
				err=0;
				tmpcoords.x = curr.x+tol[i].x; tmpcoords.y = curr.y+tol[i].y;
	            strainpnt_(&j,&tmpcoords.x,&tmpcoords.y,
					   arraysf[EX],arraysf[EY],
                       arraysf[EXREF],arraysf[EYREF],
                       arraysi[LEM],arraysi[NOR],arraysi[POLYN],
                       straintensor,&avgstrain,
                       &int_vars[NE],&int_vars[NUP],
                       &verbose,&err);
				i++;
			}
            for (i=0;i<NUM_FLYNN_STRAIN_VALS && !err;i++) {
			  k=FlynnStrainKeys[i].id;
              if (ElleUnodeAttributeActive(k)) {
				switch (k) {
				case E_XX:
                	ElleSetUnodeAttribute(j,(double)straintensor[0],k);
					break;
				case E_XY:
                	ElleSetUnodeAttribute(j,(double)straintensor[1],k);
					break;
				case E_YX:
                	ElleSetUnodeAttribute(j,(double)straintensor[2],k);
					break;
				case E_YY:
                	ElleSetUnodeAttribute(j,(double)straintensor[3],k);
					break;
				case F_INCR_S:
				case INCR_S:
                	ElleSetUnodeAttribute(j,(double)avgstrain,k);
					break;
				case F_BULK_S:
				case BULK_S:
            		ElleGetUnodeAttribute(j,&val,k);
                	ElleSetUnodeAttribute(j,val+(double)avgstrain,k);
					break;
				default:
					break;
				}
			  }
        	}
        }
        if (ElleUnodeAttributeActive(U_FINITE_STRAIN)) {
            ElleGetUnodeAttribute(j,&curr.x,CURR_S_X);
            ElleGetUnodeAttribute(j,&curr.y,CURR_S_Y);
            ElleSetUnodeAttribute(j,curr.x,PREV_S_X);
            ElleSetUnodeAttribute(j,curr.y,PREV_S_Y);
            /*
             * "unwrapped" unode position
             */
            curr.x += (double)(dt*u_velx[j]);
            curr.y += (double)(dt*u_vely[j]);
            ElleSetUnodeAttribute(j,curr.x,CURR_S_X);
            ElleSetUnodeAttribute(j,curr.y,CURR_S_Y);
        }
		else {
        /*
         * unode position in unit cell (deformed)
         */
	        curr.x += (double)(dt*u_velx[j]);
        	curr.y += (double)(dt*u_vely[j]);
		}
		if (curr.x>maxx) curr.x=maxx;
		if (curr.x<minx) curr.x=minx;
		if (curr.y>maxy) curr.y=maxy;
		if (curr.y<miny) curr.y=miny;
		/*
		 * this gives problems because the eps is set to float value
        ElleNodeUnitXY(&curr);
		*/
        ElleSetUnodePosition(j,&curr);
    }
    return(err);
}

/*
 * JUST READ ALL THE NODE AND FLYNN ATTRIBUTES??
 * or only store necessary data - basil data already there
 */
int ReadNodeData( FILE *fp, char *str)
{
    char *ptr, *start;
    const int MAX_KEYS=20;
    int num, key, finished=0, err=0, keys[NUM_STRESS_VALS];
    int keyss[NUM_FLYNN_STRAIN_VALS];
    int i;

    if ((num = fscanf(fp,"%s", str))!=1) return(READ_ERR);
    while (!feof(fp) && !finished) {
        /*
         * find keywords
         */
        validate(str,&key,FileKeys);
        switch( key ) {
        /*
         * just read node locations and grains
         * read bulk strain for updating
         * option data is read so FindBasilIndex can use EllemaxNodeSep()
         */
        case E_OPTIONS:if (err = ElleReadOptionData(fp,str))
                           finished = 1;
                       break;
        case REGIONS :
        case FLYNNS  : if (err = ElleReadFlynnData(fp,str))
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
                                  i<MAX_VALS && !finished) {
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
        case FLYNN_STRAIN: /*
                        * read the order of strain vals
                        */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          start = str;
                          while ((ptr=strtok(start," \t\r\n")) &&
                                  i<NUM_FLYNN_STRAIN_VALS && !finished) {
                              validate(ptr,&keyss[i],FlynnStrainKeys);
                              if (keyss[i]<0) finished = 1;
                              else {
                                  start = 0;
                                  i++;
                              }
                           }
                           if (i<1) finished = 1;
                       }
                       if (finished) err = READ_ERR;
                       else if (err = ElleReadFlynnStrainData(fp,str,keyss,i))
                           finished = 1;
                       break;
        case FLYNN_STRESS: /*
                        * read the order of stress vals
                        */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          start = str;
                          while ((ptr=strtok(start," \t\r\n")) &&
                                  i<NUM_STRESS_VALS && !finished) {
                              validate(ptr,&keyss[i],StressKeys);
                              if (keyss[i]<0) finished = 1;
                              else {
                                  start = 0;
                                  i++;
                              }
                           }
                           if (i<1) finished = 1;
                       }
                       if (finished) err = READ_ERR;
                       else if (err = ElleReadFlynnStressData(fp,str,keyss,i))
                           finished = 1;
                       break;
        case STRAIN: /*
                      * read the order of strain-rate vals
                      */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          start = str;
                          while ((ptr=strtok(start," \t\r\n")) &&
                                  i<NUM_STRAIN_VALS && !finished) {
                              validate(ptr,&keys[i],StrainKeys);
                              if (keys[i]<0) finished = 1;
                              else {
                                  start = 0;
                                  i++;
                              }
                           }
                           if (i<1) finished = 1;
                       }
                       if (finished) err = READ_ERR;
                       else if (err = ElleReadStrainData(fp,str,keys,i))
                           finished = 1;
                       break;
        case STRESS  : /*
                        * read the order of stress vals
                        */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          start = str;
                          while ((ptr=strtok(start," \t\r\n")) &&
                                  i<NUM_STRESS_VALS && !finished) {
                              validate(ptr,&keys[i],StressKeys);
                              if (keys[i]<0) finished = 1;
                              else {
                                  start = 0;
                                  i++;
                              }
                           }
                           if (i<1) finished = 1;
                       }
                       if (finished) err = READ_ERR;
                       else if (err = ElleReadStressData(fp,str,keys,i))
                           finished = 1;
                       break;
        case VELOCITY: dump_comments( fp ); /* read VEL_X VEL_Y */
                       if (err = ElleReadVelocityData(fp,str))
                           finished = 1;
                       break;
        case U_STRAIN: /*
                        * read the order of strain vals
                        */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          start = str;
                          while ((ptr=strtok(start," \t\r\n")) &&
                                  i<NUM_FLYNN_STRAIN_VALS && !finished) {
                              validate(ptr,&keys[i],FlynnStrainKeys);
                              if (keys[i]<0) finished = 1;
                              else {
                                  start = 0;
                                  i++;
                              }
                           }
                           if (i<1) finished = 1;
                       }
                       if (finished) err = READ_ERR;
                       else if (err =
                               ElleReadUnodeRealAttribData(fp,str,keys,i))
                           finished = 1;
                       break;
        case U_FINITE_STRAIN: /*
                        * read the order of strain vals
                        */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          start = str;
                          while ((ptr=strtok(start," \t\r\n")) &&
                                  i<NUM_FINITE_STRAIN_VALS && !finished) {
                              validate(ptr,&keys[i],FiniteStrainKeys);
                              if (keys[i]<0) finished = 1;
                              else {
                                  start = 0;
                                  i++;
                              }
                           }
                           if (i<1) finished = 1;
                       }
                       if (finished) err = READ_ERR;
                       else if (err =
                               ElleReadUnodeRealAttribData(fp,str,keys,i))
                           finished = 1;
                       break;
        default:       ElleSkipSection(fp,str,FileKeys);
                       break;
        }
    }
    return(err);
}

int CopyData( FILE *fp, FILE **fpout, char *str, Coords *newbbox,
              float time_ref)
{
    char buf[81];
    char *start, *ptr;
    int num, key, finished=0, err=0, i=0;
    int keys[NUM_STRESS_VALS], count;
    long pos;
    double offset;
    double eps = 1e-5; /* basil uses float */
    double tmp;
    CellData data;
    ofstream outf;

    if ((num = fscanf(fp,"%s", str))!=1) return(READ_ERR);
    /*
     * don't copy the first comment line "Created by ..."
     */
    validate(str,&key,FileKeys);
    if (key==COMMENT) {
        ElleSkipLine(fp,str);
        if ((num = fscanf(fp,"%s", str))!=1) return(READ_ERR);
    }
    while (!feof(fp) && !finished) {
        /*
         * find keywords
         */
        validate(str,&key,FileKeys);
        switch( key ) {
        /*
         * copy OPTIONS and FLYNNS sections
         */
        case COMMENT:  if (err = ElleCopyLine(fp,*fpout,str))
                           finished = 1;
                       if ((num = fscanf(fp,"%s", str))!=1)
                           return(READ_ERR);
                       break;
        case E_OPTIONS:if (err = ElleReadOptionData(fp,str))
                           finished = 1;
                       if (!finished) {
                           offset = ElleCumSSOffset();
                       /*
                        * reset the cell dimensions
                        */
                           ElleUpdateCellBBox(&newbbox[0],&newbbox[1],
                                           &newbbox[2],&newbbox[3]);
#if XY
                           ElleCellBBox(&data);
                           ElleSetCumSSOffset(offset+data.xoffset);
                               /* if no pure shear component */
                           if (fabs(data.ylength-1.0)<eps) {
                             offset = modf(ElleCumSSOffset(),&tmp);
                           }
                           else {
                             offset = ElleCumSSOffset();
                             if (data.xlength<offset)
                                 offset = offset-data.xlength;
                           }
                           ElleSetSSOffset(offset);
#endif
                           if (err = ElleWriteOptionData(*fpout))
                               finished = 1;
                       }
                       break;
        case LOCATION: if (err = ElleWriteNodeLocation(*fpout))
                           finished = 1;
                       if (err = ElleSkipSection(fp,str,FileKeys))
                           finished = 1;
                       break;
        case VELOCITY: if (err = ElleSkipSection(fp,str,FileKeys))
                           finished = 1;
                       else if (err=WriteVelocity(*fpout,Data_arrays_fl[UVP],
                                Data_vars_int,Data_arrays_int[ELLENODE])) {
                           fprintf(stdout,"Error writing velocities\n");
                           finished = 1;
                       }
                       break;
        case FLYNN_STRAIN:
	        if (Data_arrays_int[ELLENODE] && Data_vars_int[IMSH]==3) {
               if (err=CalcFlynnStrain(Data_arrays_fl,
                                               Data_arrays_int,
                                               Data_vars_fl,Data_vars_int,
                                               Data_arrays_int[ELLENODE],
                                               keys,&count)) {
               OnError("CalcFlynnStrain",err);
               }
        
               if (err=ElleWriteFlynnStrainData(*fpout,keys,count)) {
                   fprintf(stdout,"Error writing flynn strain\n");
                   finished = 1;
               }
               else if (err = ElleSkipSection(fp,str,FileKeys))
                   finished = 1;
            }
            else {
                fprintf(stderr,"Fixed mesh, not calculating flynn strain\n");
                if (err = ElleSkipSection(fp,str,FileKeys))
                   finished = 1;
            }
                break;
        case FLYNN_STRESS:
	        if (Data_arrays_int[ELLENODE] && Data_vars_int[IMSH]==3) {
                      //fprintf(*fpout,"%s ",str);
                       /*
                        * read the order of stress vals
                        */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          //fputs(str,*fpout);
                          start = str;
                          while ((ptr=strtok(start," \t\r\n")) &&
                                  i<NUM_STRESS_VALS && !finished) {
                              validate(ptr,&keys[i],StressKeys);
                              if (keys[i]<0) finished = 1;
                              else {
                                  start = 0;
                                  i++;
                              }
                           }
                           if (i<1) finished = 1;
                       }
                       if (finished) err = READ_ERR;
               count = i;
               if (err=CalcFlynnStress(Data_arrays_fl,
                                               Data_arrays_int,
                                               Data_vars_fl,Data_vars_int,
                                               Data_arrays_int[ELLENODE],
                                               keys,count)) {
               OnError("CalcFlynnStress",err);
               }
        
               if (err=ElleWriteFlynnStressData(*fpout,keys,count)) {
                   fprintf(stdout,"Error writing flynn stress\n");
                   finished = 1;
               }
               else if (err = ElleSkipSection(fp,str,FileKeys))
                   finished = 1;
            }
            else {
                fprintf(stderr,"Fixed mesh, not calculating flynn stress\n");
                if (err = ElleSkipSection(fp,str,FileKeys))
                   finished = 1;
            }
                break;
        case STRESS:
                       if (err=CalcStress(Data_arrays_fl,
                                          Data_arrays_int,
                                          Data_vars_fl,
                                          Data_vars_int,
                                          Data_arrays_int[ELLENODE],
                                          keys,&count)) {
                           OnError("CalcStress",err);
                       }
                       if (err=ElleWriteStressData(*fpout,keys,count)) {
                           fprintf(stdout,"Error writing node stresses\n");
                           finished = 1;
                       }
                       else if (err = ElleSkipSection(fp,str,FileKeys))
                           finished = 1;
                       break;
        case STRAIN:   if (err = ElleSkipSection(fp,str,FileKeys))
                           finished = 1;
                       break;
        case UNODES:
                       fclose(*fpout);
                       outf.open(Outfile,ios::out|ios::app);
                       if (!outf) return(OPEN_ERR);
                       if (err=CalcUnodeStrain(time_ref,
                                          Data_arrays_fl,
                                          Data_arrays_int,
                                          Data_vars_fl,
                                          Data_vars_int,
										  newbbox
                                          )) {
                           OnError("CalcUnodeStrain",err);
                       }
                       if (err = ElleWriteUnodeLocation(outf))
                           finished = 1;
                       else if (err = ElleSkipSection(fp,str,FileKeys))
                           finished = 1;
                       outf.close();
                       /* reopen fpout for append */
                       if ((*fpout = fopen( Outfile, "a" ))==0)
                           OnError(Outfile,OPEN_ERR);
                       break;
        case U_FINITE_STRAIN:
                      fprintf(*fpout,"%s ",str);
                       /*
                        * read the order of strain vals
                        */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          fputs(str,*fpout);
                          start = str;
                          while ((ptr=strtok(start," \t\r\n")) &&
                                  i<NUM_FINITE_STRAIN_VALS && !finished) {
                              validate(ptr,&keys[i],FiniteStrainKeys);
                              if (keys[i]<0) finished = 1;
                              else {
                                  start = 0;
                                  i++;
                              }
                           }
                           if (i<1) finished = 1;
                       }
                       if (finished) err = READ_ERR;
                       fclose(*fpout);
                       outf.open(Outfile,ios::out|ios::app);
                       outf.precision(8);
                       outf.setf(ios::scientific,ios::floatfield);
                       if (!outf) return(OPEN_ERR);
                       if (err = ElleWriteUnodeAttributeData(outf,
                                                         keys,i))
                           finished = 1;
                       else if (err = ElleSkipSection(fp,str,FileKeys))
                           finished = 1;
                       outf.close();
                       /* reopen fpout for append */
                       if ((*fpout = fopen( Outfile, "a" ))==0)
                           OnError(Outfile,OPEN_ERR);
                       break;
        case U_STRAIN:
                      fprintf(*fpout,"%s ",str);
                       /*
                        * read the order of strain vals
                        */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          fputs(str,*fpout);
                          start = str;
                          while ((ptr=strtok(start," \t\r\n")) &&
                                  i<NUM_FLYNN_STRAIN_VALS && !finished) {
                              validate(ptr,&keys[i],FlynnStrainKeys);
                              if (keys[i]<0) finished = 1;
                              else {
                                  start = 0;
                                  i++;
                              }
                           }
                           if (i<1) finished = 1;
                       }
                       if (finished) err = READ_ERR;
                       fclose(*fpout);
                       outf.open(Outfile,ios::out|ios::app);
                       outf.precision(8);
                       outf.setf(ios::scientific,ios::floatfield);
                       if (!outf) return(OPEN_ERR);
                       if (err = ElleWriteUnodeAttributeData(outf,
                                                         keys,i))
                           finished = 1;
                       else if (err = ElleSkipSection(fp,str,FileKeys))
                           finished = 1;
                       outf.close();
                       /* reopen fpout for append */
                       if ((*fpout = fopen( Outfile, "a" ))==0)
                           OnError(Outfile,OPEN_ERR);
                       break;
        default:       if (err = ElleCopySection(fp,*fpout,str,FileKeys))
                           finished = 1;
                       break;
        }
    }
    return(err);
}

int CopyUnodeData( FILE *fp, FILE **fpout, char *str, Coords *newbbox,
              float dt)
{
    char buf[81];
    char *start, *ptr;
    int num, key, finished=0, err=0, i=0, j=0, k=0;
    int keys[NUM_STRESS_VALS], count;
    int flynnStrainKeys[NUM_STRESS_VALS], fscount;
    double val=0;
    Coords xy;
    float *vel=Data_arrays_fl[UVP];
    float pt_u, pt_v, x, y;
    int max = 0;
    int index = 0;
    double offset, eps=1e-5;
    double tmp;
    CellData data;
    ofstream outf;

    err = UpdateUnodePos(dt);
    /*
     * calculate strain and stress values (check if active)
     */
    if (Data_arrays_int[ELLENODE] && Data_vars_int[IMSH]==3) {
        if (err=CalcFlynnStrain(Data_arrays_fl,
                                Data_arrays_int,
                                Data_vars_fl,Data_vars_int,
                                Data_arrays_int[ELLENODE],
                                flynnStrainKeys,&fscount)) {
             OnError("CalcFlynnStrain",err);
        }
        
    }
    else {
        fprintf(stderr,"Fixed mesh, not calculating flynn strain\n");
        return(1);
    }

    if ((num = fscanf(fp,"%s", str))!=1) return(READ_ERR);
    /*
     * don't copy the first comment line "Created by ..."
     */
    validate(str,&key,FileKeys);
    if (key==COMMENT) {
        ElleSkipLine(fp,str);
        if ((num = fscanf(fp,"%s", str))!=1) return(READ_ERR);
    }
    while (!feof(fp) && !finished) {
        /*
         * find keywords
         */
        validate(str,&key,FileKeys);
        switch( key ) {
        /*
         * copy OPTIONS and FLYNNS sections
         */
        case COMMENT:  if (err = ElleCopyLine(fp,*fpout,str))
                           finished = 1;
                       if ((num = fscanf(fp,"%s", str))!=1)
                           return(READ_ERR);
                       break;
        case E_OPTIONS:if (err = ElleReadOptionData(fp,str))
                           finished = 1;
                       if (!finished) {
                           offset = ElleCumSSOffset();
                           ElleUpdateCellBBox(&newbbox[0],&newbbox[1],
                                           &newbbox[2],&newbbox[3]);
#if XY
                           ElleCellBBox(&data);
                           ElleSetCumSSOffset(offset+data.xoffset);
                               /* if no pure shear component */
                           if (fabs(data.ylength-1.0)<eps) {
                             offset = modf(ElleCumSSOffset(),&tmp);
                           }
                           else {
                             offset = ElleCumSSOffset();
                             if (data.xlength<offset)
                                 offset = offset-data.xlength;
                           }
                           ElleSetSSOffset(offset);


before genshear change
                           offset = ElleCumSSOffset();
                       /*
                        * reset the cell dimensions
                        */
                           ElleSetCellBBox(&newbbox[0],&newbbox[1],
                                           &newbbox[2],&newbbox[3]);
                           ElleCellBBox(&data);
                           ElleSetCumSSOffset(offset+data.xoffset);
                           offset = modf(ElleCumSSOffset(),&tmp);
                           ElleSetSSOffset(offset);
#endif
                           if (err = ElleWriteOptionData(*fpout))
                               finished = 1;
                       }
                       break;
        case UNODES:
					   vel=Data_arrays_fl[UVP];
#if XY
					       ntrpltpnt_(&pt_u,vel,&i,&(xy.x),&(xy.y),
                               Data_arrays_fl[EX],
							   Data_arrays_fl[EY],
                               Data_arrays_fl[EXREF],
							   Data_arrays_fl[EYREF],
                               Data_arrays_int[LEM],
							   Data_arrays_int[NOR],
							   Data_arrays_int[POLYN],
                               &Data_vars_int[NE],&Data_vars_int[NUP],&err);
					       vel=Data_arrays_fl[UVP]+Data_vars_int[NUP];
					       ntrpltpnt_(&pt_v,vel,&i,&(xy.x),&(xy.y),
                               Data_arrays_fl[EX],
							   Data_arrays_fl[EY],
                               Data_arrays_fl[EXREF],
							   Data_arrays_fl[EYREF],
                               Data_arrays_int[LEM],
							   Data_arrays_int[NOR],
							   Data_arrays_int[POLYN],
                               &Data_vars_int[NE],&Data_vars_int[NUP],&err);
                           xy.x += pt_u*dt;
                           xy.y += pt_v*dt;
#endif
                       fclose(*fpout);
                       outf.open(Outfile,ios::out|ios::app);
                       if (!outf) return(OPEN_ERR);
                       if (err = ElleWriteUnodeLocation(outf))
                           finished = 1;
                       else if (err = ElleSkipSection(fp,str,FileKeys))
                           finished = 1;
                       outf.close();
                       /* reopen fpout for append */
                       if ((*fpout = fopen( Outfile, "a" ))==0)
                           OnError(Outfile,OPEN_ERR);
                       break;
#if XY
        case U_FINITE_STRAIN:
                      fprintf(*fpout,"%s ",str);
                       /*
                        * read the order of strain vals
                        */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          fputs(str,*fpout);
                          start = str;
                          while ((ptr=strtok(start," \t\r\n")) &&
                                  i<NUM_FINITE_STRAIN_VALS && !finished) {
                              validate(ptr,&keys[i],FiniteStrainKeys);
                              if (keys[i]<0) finished = 1;
                              else {
                                  start = 0;
                                  i++;
                              }
                           }
                           if (i<1) finished = 1;
                       }
                       if (finished) err = READ_ERR;
                       fclose(*fpout);
                       outf.open(Outfile,ios::out|ios::app);
                       outf.precision(8);
                       outf.setf(ios::scientific,ios::floatfield);
                       if (!outf) return(OPEN_ERR);
                       if (err = ElleWriteUnodeAttributeData(outf,
                                                         keys,i))
                           finished = 1;
                       else if (err = ElleSkipSection(fp,str,FileKeys))
                           finished = 1;
                       outf.close();
                       /* reopen fpout for append */
                       if ((*fpout = fopen( Outfile, "a" ))==0)
                           OnError(Outfile,OPEN_ERR);
                       break;
#endif
        case U_STRAIN:
                      fprintf(*fpout,"%s ",str);
                       /*
                        * read the order of strain vals
                        */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          fputs(str,*fpout);
                          start = str;
                          while ((ptr=strtok(start," \t\r\n")) &&
                                  i<NUM_FLYNN_STRAIN_VALS && !finished) {
                              validate(ptr,&keys[i],FlynnStrainKeys);
                              if (keys[i]<0) finished = 1;
                              else {
                                  ElleInitUnodeAttribute(keys[i]);
                                  tmp=ElleDefaultFlynnRealAttribute(keys[i]);
                                  ElleSetDefaultUnodeAttribute(tmp,keys[i]);
                                  start = 0;
                                  i++;
                              }
                           }
                           if (i<1) finished = 1;
                       }
                       
                       if (finished) err = READ_ERR;
                       fclose(*fpout);
                       for (j=0; j<ElleMaxFlynns(); j++) {
                         if (ElleFlynnIsActive(j))
                           for (k=0; k<i; k++) {
                             ElleGetFlynnRealAttribute(j,&tmp,keys[k]);
                             ElleSetUnodeAttribute(j,tmp,keys[k]);
                           }
                       }
                       outf.open(Outfile,ios::out|ios::app);
                       outf.precision(8);
                       outf.setf(ios::scientific,ios::floatfield);
                       if (!outf) return(OPEN_ERR);
                       if (err = ElleWriteUnodeAttributeData(outf,
                                                         keys,i))
                           finished = 1;
                       else if (err = ElleSkipSection(fp,str,FileKeys))
                           finished = 1;
                       outf.close();
                       /* reopen fpout for append */
                       if ((*fpout = fopen( Outfile, "a" ))==0)
                           OnError(Outfile,OPEN_ERR);
                       break;

        case U_ATTRIB_A:
        case U_VISCOSITY:
                       if (err = ElleCopySection(fp,*fpout,str,FileKeys))
                           finished = 1;
                       break;
        default:       if (err = ElleSkipSection(fp,str,FileKeys))
                           finished = 1;
                       break;
        }
    }
    return(err);
}

int UpdateUnodePos(float dt)
{
    int num, err=0, i=0, j=0;
    int max = 0;
    int index = 0;
    float *ex=Data_arrays_fl[EX], *ey=Data_arrays_fl[EY];
    float *u=Data_arrays_fl[UVP], *v=Data_arrays_fl[UVP]+Data_vars_int[NUP];
    float x, y;
    Coords xy;

    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
      if (ElleFlynnIsActive(i)) {
        ElleGetUnodePosition(i,&xy);
        x = (float)xy.x;
        y = (float)xy.y;
		matchmeshnode_(&index,&i,&(xy.x),&(xy.y),
                               Data_arrays_fl[EXREF],
                               Data_arrays_fl[EYREF],
                               Data_arrays_int[LEM],
                               Data_arrays_int[NOR],
                               Data_arrays_int[POLYN],
                               &Data_vars_int[NE],&Data_vars_int[NUP],&err);
        if (!err && index>0 && index<=NUP) {
          xy.x += u[index-1]*dt;
          xy.y += v[index-1]*dt;
          ElleSetUnodePosition(i,&xy);
        }
      }
    }
    return(err);
}

void FindBBox(float *xvals,float *yvals,int size,
              Coords *bl,Coords *br,Coords *tr,Coords *tl)
{
    int i, setmin1=0, setmax1=0, setmin2=0, setmax2=0;
    float xmin,ymin,xmax,ymax;
    float eps;

    eps = 1e-3;
    bl->y = br->y = tr->y = tl->y = yvals[0];
    bl->x = br->x = tr->x = tl->x = xvals[0];
    xmin = xmax = xvals[0];
    ymin = ymax = yvals[0];
    for (i=1;i<size;i++) {
        if (yvals[i] < br->y) { br->y = bl->y = yvals[i]; }
        if (yvals[i] > tr->y) { tr->y = tl->y = yvals[i]; }
    }
    ymin = br->y + eps;
    ymax = tr->y - eps;
    for (i=0;i<size;i++) {
        if (yvals[i] < ymin) {
            if (!setmin1) {
                bl->x = xvals[i];
                setmin1 = 1;
            }
            else if (xvals[i] < bl->x) bl->x = xvals[i];
            if (!setmax1) {
                br->x = xvals[i];
                setmax1 = 1;
            }
            else if (xvals[i] > br->x) br->x = xvals[i];
        }
        else if (yvals[i] > ymax) {
            if (!setmin2) {
                tl->x = xvals[i];
                setmin2 = 1;
            }
            else if (xvals[i] < tl->x) tl->x = xvals[i];
            if (!setmax2) {
                tr->x = xvals[i];
                setmax2 = 1;
            }
            else if (xvals[i] > tr->x) tr->x = xvals[i];
        }
    }
}

int FindBasilIndex(int **basil_index_adr,int bmax,float *ex, float *ey,
                   int *ibc, int bndmax, int *max_index, char *err_str)
{
    float xmin, xmax, ymin, ymax;
    float xminbnd, xmaxbnd, yminbnd, ymaxbnd;
    double eps, epsbnd;
    int *basil_index;
    int i, nmax, emax, j, found=0;
    int k, bnd, match;
int dum;
    Coords xy;

    eps = 1e-6;
    /*
     * this is slightly larger than the value used in elle2poly
     * (in CnvtBoundaryCheck - moving nodes to boundary)
     */
    epsbnd = EllemaxNodeSep()*0.088;
    nmax = ElleMaxNodes();
    if ((*basil_index_adr = basil_index =
                (int *)malloc(nmax * sizeof(int)))==0)
        OnError("",MALLOC_ERR);
    for (j=0;j<nmax;j++) basil_index[j] = -1;
    for (i=0,emax=0;i<nmax;i++) {
        if (ElleNodeIsActive(i)) {
            ElleNodePosition(i,&xy);
            xmin = (float)(xy.x-eps);
            xmax = (float)(xy.x+eps);
            ymin = (float)(xy.y-eps);
            ymax = (float)(xy.y+eps);
            xminbnd = (float)(xy.x-epsbnd);
            xmaxbnd = (float)(xy.x+epsbnd);
            yminbnd = (float)(xy.y-epsbnd);
            ymaxbnd = (float)(xy.y+epsbnd);
            bnd=0;
            for (k=0;k<bndmax;k++) if (ibc[k]==i+1) bnd=1;
            for (j=0,found=0;j<bmax && !found;j++) {
                if (bnd) {
                    if (ex[j] > xminbnd && ex[j] < xmaxbnd &&
                        ey[j] >yminbnd && ey[j] < ymaxbnd ) found=1;
                }
                else {
                    if (ex[j] > xmin && ex[j] < xmax &&
                        ey[j] >ymin && ey[j] < ymax ) found=1;
                }
                if (found ) {
                    if (i>emax) emax=i;
                    if (basil_index[i] != -1) {
                      sprintf(err_str,"Basil node overwritten %d, Elle node %d",basil_index[j],i);
                      printf("%s\n",err_str);
                      /*return(1);*/
                    }
                    else basil_index[i] = j+1;
                }
            }
            if (!found) {
                sprintf(err_str,"Basil node not found, Elle node %d",i);
                printf("%s\n",err_str);
                /*return(1);*/
            }
        }
    }
    *max_index = emax+1;
    return(0);
}

/*
 * the array in the basil solution (Data_arrays_int[1..NELLEP])
 * is the basil index (1..NUP) for an elle node
 * Recalculate position from velocity to allow for nodes that were 
 *  moved to the boundary by elle2poly and may just oscillate.
 *  Adjust dimensions of newbbox, if necessary
 */
int UpdatePositions(float **arraysf,int **arraysi,float *fl_vars,int *int_vars,
                    float time_ref,int *basil_index,Coords *newbbox)
{
    int i,j,id,bmax,emax,dum=0,*nor, *ibc;
    float *ex, *ey, *ex_ref, *ey_ref, *velx, *vely;
    float dt, xlen, ylen;
	float *calcx, *calcy;
    float eps = 1e-6;
    Coords newxy, oldbbox[4];

    bmax = int_vars[NUP];
    emax = int_vars[NELLEP];
	calcx = new float[bmax];
	calcy = new float[bmax];
	ibc = arraysi[IBC];
	nor = arraysi[NOR];
    newxy.x = newxy.y = 0.0;
    ex = arraysf[EX];
    ey = arraysf[EY];
    ex_ref = arraysf[EXREF];
    ey_ref = arraysf[EYREF];
    velx = arraysf[UVP];
    vely = arraysf[UVP]+int_vars[NUP];
    dt = fl_vars[TIME]-time_ref;

    for (i=0;i<bmax;i++) calcx[i]=ex[i];
    for (i=0;i<bmax;i++) calcy[i]=ey[i];
    FindBBox(calcx,calcy,bmax,
                &oldbbox[0],&oldbbox[1],&oldbbox[2],&oldbbox[3]);
    for (i=0;i<emax;i++) {
        if (basil_index[i]!=-1) {
            id = basil_index[i];
            ElleNodePosition(i,&newxy);
            newxy.x += velx[id-1]*dt;
            newxy.y += vely[id-1]*dt;
            ElleCopyToPosition(i,&newxy);
		    calcx[id]=newxy.x;
		    calcy[id]=newxy.y;
            dum++;
        }
    }
    FindBBox(calcx,calcy,bmax,
                &newbbox[0],&newbbox[1],&newbbox[2],&newbbox[3]);
	if (oldbbox[0].x<newbbox[0].x) newbbox[0].x=oldbbox[0].x;
	if (oldbbox[0].y<newbbox[0].y) newbbox[0].y=oldbbox[0].y;
	if (oldbbox[1].x>newbbox[1].x) newbbox[1].x=oldbbox[1].x;
	if (oldbbox[1].y<newbbox[1].y) newbbox[1].y=oldbbox[1].y;
	if (oldbbox[2].x>newbbox[2].x) newbbox[2].x=oldbbox[2].x;
	if (oldbbox[2].y>newbbox[2].y) newbbox[2].y=oldbbox[2].y;
	if (oldbbox[3].x<newbbox[3].x) newbbox[3].x=oldbbox[3].x;
	if (oldbbox[3].y>newbbox[3].y) newbbox[3].y=oldbbox[3].y;
// Not sure what the previous block is doing but try and remove
// numerical drift
	if (fabs(newbbox[0].x)<eps) newbbox[0].x=0.0;
	if (fabs(newbbox[0].y)<eps) newbbox[0].y=0.0;
	if (fabs(newbbox[1].x)<eps) newbbox[1].x=0.0;
	if (fabs(newbbox[1].y)<eps) newbbox[1].y=0.0;
	if (fabs(newbbox[2].x)<eps) newbbox[2].x=0.0;
	if (fabs(newbbox[2].y)<eps) newbbox[2].y=0.0;
	if (fabs(newbbox[3].x)<eps) newbbox[3].x=0.0;
	if (fabs(newbbox[3].y)<eps) newbbox[3].y=0.0;
	/*
	 * this should create a new reference frame which conserves area
	ylen = newbbox[3].y - newbbox[0].y;
	xlen = 1.0/ylen;
	newbbox[1].x=newbbox[0].x+xlen;
	newbbox[2].x=newbbox[3].x+xlen;
	 */
printf("num nodes matched %d\n",dum);
emax = ElleMaxNodes();
for (i=0,dum=0;i<emax;i++) if (ElleNodeIsActive(i)) dum++;
printf("num elle nodes %d\n",dum);
    return(0);
}

/*
 * This routine is for solutions which do not have Elle nodes as
 * mesh points.
 * Recalculate position from velocity using NTRPLN
 *  Adjust dimensions of newbbox, if necessary
 */
int UpdateIntrpPositions(float **arraysf,int **arraysi,float *fl_vars,int *int_vars,
                    float time_ref,Coords *newbbox)
{
    int i,j,id,bmax,emax,*nor, *ibc;
    int dum=0,ncnt=0,cnt=1,verbose=0;
    float *ex, *ey, *ex_ref, *ey_ref, *velx, *vely;
	float pt_u, pt_v, x, y, ss0=0;
    float dt, xlen, ylen;
	float *calcx, *calcy;
    Coords newxy, oldbbox[4];

    bmax = int_vars[NUP];
	emax = ElleMaxNodes();
	dum = bmax; if (emax>bmax) dum=emax;
	calcx = new float[dum];
	calcy = new float[dum];
	ibc = arraysi[IBC];
	nor = arraysi[NOR];
    newxy.x = newxy.y = 0.0;
    ex_ref = arraysf[EXREF];
    ey_ref = arraysf[EYREF];
    ex = arraysf[EX];
    ey = arraysf[EY];
    velx = arraysf[UVP];
    vely = arraysf[UVP]+int_vars[NUP];
    dt = fl_vars[TIME]-time_ref;

    for (i=0;i<bmax;i++) calcx[i]=ex[i];
    for (i=0;i<bmax;i++) calcy[i]=ey[i];
    FindBBox(calcx,calcy,bmax,
                &oldbbox[0],&oldbbox[1],&oldbbox[2],&oldbbox[3]);
    for (i=0,ncnt=0;i<emax;i++) {
	  if (ElleNodeIsActive(i)) {
        ElleNodePosition(i,&newxy);
		x = newxy.x; y=newxy.y;
		ntrpln_(velx,&pt_u,&cnt,&ss0,&x,&y,&x,&y,
					&fl_vars[TBXOFF],&fl_vars[TBYOFF],&fl_vars[VELXO],
					ex_ref,ey_ref,arraysi[LEM], arraysi[NOR],&dum,
					&int_vars[NE],&int_vars[NUP],&int_vars[NUP],&verbose);
		ntrpln_(vely,&pt_v,&cnt,&ss0,&x,&y,&x,&y,
					&fl_vars[TBXOFF],&fl_vars[TBYOFF],&fl_vars[VELYO],
					ex_ref,ey_ref,arraysi[LEM], arraysi[NOR],&dum,
					&int_vars[NE],&int_vars[NUP],&int_vars[NUP],&verbose);
        newxy.x += pt_u*dt;
        newxy.y += pt_v*dt;
        ElleCopyToPosition(i,&newxy);
		calcx[ncnt]=newxy.x;
		calcy[ncnt]=newxy.y;
        ncnt++;
      }
    }
    if (int_vars[IMSH]==3) {
    FindBBox(calcx,calcy,ncnt,
                &newbbox[0],&newbbox[1],&newbbox[2],&newbbox[3]);
	if (oldbbox[0].x<newbbox[0].x) newbbox[0].x=oldbbox[0].x;
	if (oldbbox[0].y<newbbox[0].y) newbbox[0].y=oldbbox[0].y;
	if (oldbbox[1].x>newbbox[1].x) newbbox[1].x=oldbbox[1].x;
	if (oldbbox[1].y<newbbox[1].y) newbbox[1].y=oldbbox[1].y;
	if (oldbbox[2].x>newbbox[2].x) newbbox[2].x=oldbbox[2].x;
	if (oldbbox[2].y>newbbox[2].y) newbbox[2].y=oldbbox[2].y;
	if (oldbbox[3].x<newbbox[3].x) newbbox[3].x=oldbbox[3].x;
	if (oldbbox[3].y>newbbox[3].y) newbbox[3].y=oldbbox[3].y;
    }
    else {
    for (i=0;i<4;i++) newbbox[i] = oldbbox[i];
    }
	/*
	 * this should create a new reference frame which conserves area
	ylen = newbbox[3].y - newbbox[0].y;
	xlen = 1.0/ylen;
	newbbox[1].x=newbbox[0].x+xlen;
	newbbox[2].x=newbbox[3].x+xlen;
	 */
    return(0);
}

int WriteVelocity(FILE *fp,float *uvp,int *int_vars,int *basil_index)
{
    char label[20];
    int i,id, bmax, emax;

    emax = int_vars[NELLEP];
    bmax = int_vars[NUP];

    if (!id_match(FileKeys,VELOCITY,label)) return(KEY_ERR);
    fprintf(fp,"%s",label);
    if (!id_match(VelocityKeys,VEL_X,label)) return(KEY_ERR);
    fprintf(fp," %s",label);
    if (!id_match(VelocityKeys,VEL_Y,label)) return(KEY_ERR);
    fprintf(fp," %s",label);
    fprintf(fp,"\n");
    for (i=0;i<emax;i++) {
        if (basil_index[i]!=-1) {
            id = basil_index[i];
            fprintf(fp,"%d %.8e %.8e\n",i,uvp[id-1],uvp[id-1+bmax]);
        }
    }
    return(0);
}

int WriteStress(FILE *fp,float *vals,int *int_vars,int *basil_index)
{
    char label[20];
    int i,j,id, bmax, emax;
    float *ptr;

    emax = int_vars[NELLEP];
    bmax = int_vars[NUP];
    if (!id_match(FileKeys,STRESS,label)) return(KEY_ERR);
    fprintf(fp,"%s",label);
    for (i=0;i<NUM_STRESS_VALS;i++) fprintf(fp," %s",StressKeys[i].name);
    fprintf(fp,"\n");
    for (i=0;i<emax;i++) {
        id = basil_index[i];
        if (basil_index[i]!=-1) {
            fprintf(fp,"%d",i);
            for (j=0,ptr=&vals[id-1];j<NUM_STRESS_VALS;j++,ptr+=bmax)
                fprintf(fp," %.8e",*ptr);
            fprintf(fp,"\n");
        }
    }
    return(0);
}


void copyFlynnAttributesToUnode(int unodeid, int flynnid)
{
    int i, j, *attr_id, maxa=0;
    int u_attr = NO_VAL;
    double val=0, dflt=0;
    extern vector<Unode> *Unodes;

    ElleFlynnDfltAttributeList(&attr_id, &maxa);
    for (j=0; j<maxa; j++) {
        dflt = ElleDefaultFlynnRealAttribute(attr_id[j]);
        if (attr_id[j]==VISCOSITY) u_attr = U_VISCOSITY;
        else u_attr = attr_id[j];
        ElleGetFlynnRealAttribute(flynnid, &val, attr_id[j]);
        if (val != dflt)
            ElleSetUnodeAttribute(unodeid, val, u_attr);
    }
    if (attr_id!=0) free(attr_id);
}
/*
 * this is a copy of sybil's initialisation with defines replaced
 */
void Init_Plot()
{
    Plot_info.curr_cell = 0;
    Plot_info.max_col = 1;
    Plot_info.max_row = 1;
    Plot_info.curr_x = Plot_info.curr_y = 0.0;
    Plot_info.origin_x = Plot_info.origin_y = 0.0;
    Plot_info.title_offset = (int)(18 * 1.5);
    Plot_info.plot_type = -1;
    Plot_info.plot_description = 0;
    Plot_info.var_num = 0;
    Plot_info.title = NULL;
    strcpy(Plot_info.fontname,"Helvetica");
    strcpy(Plot_info.variable,"");
    strcpy(Plot_info.dflt_label1,"");
    strcpy(Plot_info.dflt_label2,"");
    strcpy(Plot_info.curr_label,"");
    Plot_info.display_log = 0;
    Plot_info.log_file_ptr = NULL;
    Plot_info.inp_file = NULL;
}
