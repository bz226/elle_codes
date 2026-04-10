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
#include <cmath>
#include "convert.h"
#include "attrib.h"
#include "bflynns.h"
#include "error.h"
#include "file.h"
#include "regions.h"
#include "interface.h"
#include "general.h"
#include "lut.h"
#include "gz_utils.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/
extern int Region_ver;

int ElleWriteRegionData(FILE *fp)
{
    int err=0;

    if (err = ElleWriteFlynnData(fp)) return(err);
    err = ElleWriteRegionAttributeData(fp);
    return(err);
}

int ElleWriteRegionAttributeData(FILE *fp)
{
    char label[20];
    int err=0, i, j;
    int *active, maxa, *keys;

    ElleFlynnDfltAttributeList(&active,&maxa);
    for (i=0;i<maxa && !err;i++) {
        switch(active[i]) {
        case EXPAND:
        case COLOUR:
        case GRAIN :
        case SPLIT : err = ElleWriteFlynnIntAttribData(fp,active[i]);
                     break;
        case MINERAL: err = ElleWriteMineralData(fp);
                     break;
        case CAXIS  :err = ElleWriteCAxisData(fp);
                     if (ElleEnergyLUTSize()>0)
                         err = ElleWriteGBEnergyLUT(fp);
                     while (i<(maxa-1) && (active[i+1]==CAXIS_Z ||
                                       active[i+1]==CAXIS_Y ||
                                       active[i+1]==CAXIS_X ||
                                       active[i+1]==CAXIS     )) i++;
                     break;
        case E3_ALPHA: err = ElleWriteEuler3Data(fp);
                     if (ElleEnergyLUTSize()>0)
                         err = ElleWriteGBEnergyLUT(fp);
                     while (i<(maxa-1) && (active[i+1]==E3_BETA ||
                                       active[i+1]==E3_GAMMA)) i++;
                     break;
        case F_INCR_S:
        case INCR_S:
        case F_BULK_S:
        case BULK_S:
        case E_XX   :
        case E_XY   :
        case E_YX   :
        case E_YY   :
        case E_ZZ   : 
                    if ((keys=(int *)malloc(NUM_FLYNN_STRAIN_VALS
                                                    * sizeof(int)))==0)
                        OnError("ElleWriteRegionAttributeData",MALLOC_ERR);
                    for (j=0;j<NUM_FLYNN_STRAIN_VALS;j++) keys[j] = NO_INDX;
                    j=0;
                    while (i<maxa &&
                           id_match(FlynnStrainKeys,active[i],label)) {
                        keys[j] = active[i];
                        j++; i++;
                    }
                    i--; /* for loop will increment again */
                    err = ElleWriteFlynnStrainData(fp,keys,j);
                    free(keys);
                    break;
        case TAU_XX   :
        case TAU_YY   :
        case TAU_XY   :
        case TAU_ZZ   :
        case TAU_1    : 
        case PRESSURE : 
                    if ((keys=(int *)malloc(NUM_STRESS_VALS
                                                    * sizeof(int)))==0)
                        OnError("ElleWriteRegionAttributeData",MALLOC_ERR);
                    for (j=0;j<NUM_STRESS_VALS;j++) keys[j] = NO_INDX;
                    j=0;
                    while (i<maxa &&
                           id_match(StressKeys,active[i],label)) {
                        keys[j] = active[i];
                        j++; i++;
                    }
                    i--; /* for loop will increment again */
                    err = ElleWriteFlynnStressData(fp,keys,j);
                    free(keys);
                    break;
        case AGE:
        case CYCLE:
                    if ((keys=(int *)malloc(NUM_FLYNN_AGE_VALS
                                                    * sizeof(int)))==0)
                        OnError("ElleWriteRegionAttributeData",MALLOC_ERR);
                    for (j=0;j<NUM_FLYNN_AGE_VALS;j++) keys[j] = NO_INDX;
                    j=0;
                    while (i<maxa &&
                           id_match(FlynnAgeKeys,active[i],label)) {
                        keys[j] = active[i];
                        j++; i++;
                    }
                    i--; /* for loop will increment again */
                    err = ElleWriteAgeData(fp,keys,j);
                    free(keys);
                    break;
        case VISCOSITY:
        case S_EXPONENT:
        case ENERGY :
        case DISLOCDEN:
        case F_ATTRIB_A :
        case F_ATTRIB_B :
        case F_ATTRIB_C :
        case F_ATTRIB_D :
        case F_ATTRIB_E :
        case F_ATTRIB_F :
        case F_ATTRIB_I :
        case F_ATTRIB_J :
        case F_ATTRIB_K : err = ElleWriteFlynnRealAttribData(fp,active[i]);
                     break;
        default:     err = ATTRIBID_ERR;
                     break;
        }
    }
    if (active) free(active);
    return(err);
}

int ElleReadFlynnIntAttribData(FILE *fp, char str[],int attr)
{
    int err=0, num;
    int val;
    int full_id;
    int s, u;

    ElleInitFlynnAttribute(attr);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%d\n", &val)) != 1)
                return(READ_ERR);
                ElleSetDefaultFlynnIntAttribute((int)val,attr);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if (Region_ver)
                if ((num = fscanf(fp,"%d %d",&s,&u))!=2)
                    return(READ_ERR);
            if ((num = fscanf(fp,"%d", &val))!=1) return(READ_ERR);
            /* do range checks on val */
            if (ElleFlynnIsActive(full_id))
                ElleSetFlynnIntAttribute(full_id,val,attr);
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleReadMineralData(FILE *fp, char str[])
{
    int err=0, num;
    char val[MAX_MINERAL_LEN+1];
    int full_id, mineral_id;

    ElleInitFlynnAttribute(MINERAL);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%s\n", val)) != 1)
                return(READ_ERR);
                validate(val,&mineral_id,MineralKeys);
                if (mineral_id<0) return(KEY_ERR);
                ElleSetDefaultFlynnMineral(mineral_id);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if ((num = fscanf(fp,"%s", val))!=1) return(READ_ERR);
            /* do range checks on val */
            validate(val,&mineral_id,MineralKeys);
            if (mineral_id<0) return(KEY_ERR);
            if (ElleFlynnIsActive(full_id))
                ElleSetFlynnMineral(full_id,mineral_id);
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleReadFlynnStrainData(FILE *fp,char str[],int *keys,int count)
{
    /*
     * Read flynn strain data in an elle file.
     * Strain labels then strain data on a separate
     * line for each flynn with the format: %d %f %f ....
     * (flynn id number then strain values in the order read into keys)
     */
    int i;
    int nn, num, err=0;
    double val[NUM_FLYNN_STRAIN_VALS];

    for (i=0;i<count;i++) 
        ElleInitFlynnAttribute(keys[i]);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        else if (str[0] == '#') dump_comments( fp );
        else if (str[0]<'0' || str[0]>'9') return(0);
        /*
         * SET_ALL not valid for strain vals
         */
        else {
            if ((nn = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            for (i=0;i<count;i++) {
                if ((num = fscanf(fp," %lf", &val[i])) != 1)
                    return(READ_ERR);
                ElleSetFlynnRealAttribute(nn,val[i],keys[i]);
            }
        }
        num = fscanf(fp,"\n");
    }
    return(0);
}

int ElleReadFlynnStressData(FILE *fp,char str[],int *keys,int count)
{
    /*
     * Read flynn stress data in an elle file.
     * Strain labels then strain data on a separate
     * line for each flynn with the format: %d %f %f ....
     * (flynn id number then stress values in the order read into keys)
     */
    int i;
    int nn, num, err=0;
    double val[NUM_STRESS_VALS];

    for (i=0;i<count;i++) 
        ElleInitFlynnAttribute(keys[i]);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        else if (str[0] == '#') dump_comments( fp );
        else if (str[0]<'0' || str[0]>'9') return(0);
        /*
         * SET_ALL not valid for stress vals
         */
        else {
            if ((nn = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            for (i=0;i<count;i++) {
                if ((num = fscanf(fp," %lf", &val[i])) != 1)
                    return(READ_ERR);
                ElleSetFlynnRealAttribute(nn,val[i],keys[i]);
            }
            num = fscanf(fp,"\n");
        }
    }
    return(0);
}

int ElleReadAgeData(FILE *fp, char str[], int *keys, int count)
{
    int i, err=0, num;
    int full_id;
    int s, u;

    double val[NUM_FLYNN_AGE_VALS];

    for (i=0;i<count;i++) 
        ElleInitFlynnAttribute(keys[i]);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            for (i=0;i<count;i++) {
                if ((num = fscanf(fp,"%lf\n", &val[i])) != 1)
                    return(READ_ERR);
                ElleSetDefaultFlynnAge(val[i],keys[i]);
            }
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if (Region_ver)
                if ((num = fscanf(fp,"%d %d",&s,&u))!=2)
                    return(READ_ERR);
            for (i=0;i<count;i++) {
                if ((num = fscanf(fp," %lf", &val[i])) != 1)
                    return(READ_ERR);
            /* do range checks on val */
                if (ElleFlynnIsActive(full_id))
                    ElleSetFlynnAge(full_id,val[i],keys[i]);
            }
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleReadFlynnRealAttribData(FILE *fp, char str[],int id)
{
    int err=0, num;
    double val;
    int full_id;
    int s, u;

    ElleInitFlynnAttribute(id);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%lf\n", &val)) != 1)
                return(READ_ERR);
                ElleSetDefaultFlynnRealAttribute(val,id);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if (Region_ver)
                if ((num = fscanf(fp,"%d %d",&s,&u))!=2)
                    return(READ_ERR);
            if ((num = fscanf(fp,"%lf", &val))!=1) return(READ_ERR);
            /* do range checks on val */
            if (ElleFlynnIsActive(full_id))
                ElleSetFlynnRealAttribute(full_id,val,id);
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleReadCAxisData(FILE *fp, char str[])
{
    /*
     * routine to be called after the CAxis keyword has
     * been read in an Elle file
     * Reads lines with the format %d %lf %lf
     *           (node id, angle theta, angle phi; angles in degrees)
     */
    int err=0, num;
    Coords_3D val;
    int full_id;
    int s, u;
    double theta,phi;

    ElleInitFlynnAttribute(CAXIS);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp)) return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%lf %lf\n", &theta,&phi)) != 2)
                return(READ_ERR);
            /* convert from degrees to radians */
            theta *= DTOR;
            phi *= DTOR;
            PolarToCartesian(&val.x,&val.y,&val.z,theta,phi);
            ElleSetDefaultFlynnCAxis(&val);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id= atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if (Region_ver)
                if ((num = fscanf(fp,"%d %d",&s,&u))!=2)
                    return(READ_ERR);
            if ((num = fscanf(fp,"%lf %lf\n", &theta,&phi)) != 2)
                return(READ_ERR);
            theta *= DTOR;
            phi *= DTOR;
            PolarToCartesian(&val.x,&val.y,&val.z,theta,phi);
            /* do range checks on val */
            if (ElleFlynnIsActive(full_id))
                ElleSetFlynnCAxis(full_id, &val);
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleReadEuler3Data(FILE *fp, char str[])
{
    /*
     * routine to be called after the EULER_3 keyword has
     * been read in an Elle file
     * Reads lines with the format %d %f %f %f
     *    (node id, angle alpha, angle beta, angle gamma; angles in degrees)
     */
    int err=0, num;
    int full_id;
    double alpha, beta, gamma;

    ElleInitFlynnAttribute(EULER_3);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp)) return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%lf %lf %lf\n", &alpha,&beta,&gamma)) != 3)
                return(READ_ERR);
            ElleSetDefaultFlynnEuler3(alpha,beta,gamma);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id= atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if ((num = fscanf(fp,"%lf %lf %lf\n", &alpha,&beta,&gamma)) != 3)
                return(READ_ERR);
            /* do range checks on val */
            if (ElleFlynnIsActive(full_id))
                ElleSetFlynnEuler3(full_id, alpha, beta, gamma);
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleWriteMineralData(FILE *fp)
{
    char label[20], val[MAX_MINERAL_LEN+1];
    int i, err=0, mineral_id, dflt, num;
    int max;

    if (!id_match(FileKeys,MINERAL,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    dflt = ElleDefaultFlynnMineral();
    if (!id_match(MineralKeys,dflt,val)) return(KEY_ERR);
    if ((num = fprintf(fp,"%s %s\n",SET_ALL,val))<0)
        return(WRITE_ERR);
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i) && ElleFlynnHasAttribute(i,MINERAL)) {
            ElleGetFlynnMineral(i,&mineral_id);
            if (mineral_id!=dflt) {
                if (!id_match(MineralKeys,mineral_id,val)) return(KEY_ERR);
                if ((num = fprintf(fp,"%d %s\n", i, val))<0)
                    return(WRITE_ERR);
            }
        }
    }
    return(err);
}

int ElleWriteFlynnIntAttribData(FILE *fp, int key)
{
    char label[20];
    int i, err=0, val, dflt, num;
    int max;

    if (!id_match(FileKeys,key,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    dflt = ElleDefaultFlynnIntAttribute(key);
    if ((num = fprintf(fp,"%s %d\n",SET_ALL,dflt))<0)
        return(WRITE_ERR);
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i) && ElleFlynnHasAttribute(i,key)) {
            ElleGetFlynnIntAttribute(i,&val,key);
            if (val!=dflt) {
                if ((num = fprintf(fp,"%d %d\n", i, val))<0)
                    return(WRITE_ERR);
            }
        }
    }
    return(err);
}

int ElleWriteAgeData(FILE *fp,int *keys,int count)
{
    char label[20];
    int i, j, err=0, num=0;
    int max;
    double val,dflt;

    if (!id_match(FileKeys,AGE,label)) return(KEY_ERR);
    fprintf(fp,"%s\t",label);
    for (i=0;i<count;i++) {
        if (!id_match(FlynnAgeKeys,keys[i],label)) return(KEY_ERR);
        fprintf(fp,"%s ",label);
    }
    fprintf(fp,"\n");
    if ((num = fprintf(fp,"%s",SET_ALL))<0)
        return(WRITE_ERR);
    for (j=0;j<count;j++)  {
        dflt = ElleDefaultFlynnRealAttribute(keys[j]);
        if (fprintf(fp," %.8e",dflt) < 0) err = WRITE_ERR;
    }
    fprintf(fp,"\n");
    max = ElleMaxFlynns();
    for (i=0;i<max && !err;i++) {
        if (ElleFlynnIsActive(i) && ElleFlynnHasAttribute(i,AGE)) {
            fprintf(fp,"%d",i);
            for (j=0;j<count;j++)  {
                ElleGetFlynnRealAttribute(i,&val,keys[j]);
                if (fprintf(fp," %.8e",val) < 0) err = WRITE_ERR;
            }
            fprintf(fp,"\n");
        }
    }
    return(err);
}

int ElleWriteFlynnRealAttribData(FILE *fp, int id)
{
    char label[20];
    int i, err=0, num;
    unsigned char int_attrib=0;
    int max;
    double val, dflt;

    if (!id_match(FileKeys,id,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    int_attrib = isFlynnIntegerAttribute(id);
    dflt = ElleDefaultFlynnRealAttribute(id);
    if (int_attrib) {
        if ((num = fprintf(fp,"%s %d\n",SET_ALL,(int)dflt))<0)
            return(WRITE_ERR);
    }
    else {
        if ((num = fprintf(fp,"%s %.8e\n",SET_ALL,dflt))<0)
            return(WRITE_ERR);
    }
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i) && ElleFlynnHasAttribute(i,id)) {
            ElleGetFlynnRealAttribute(i,&val,id);
            if (val!=dflt) {
                if (int_attrib) {
                    if ((num = fprintf(fp,"%d %d\n", i, (int)val))<0)
                        return(WRITE_ERR);
                }
                else {
                    if ((num = fprintf(fp,"%d %.8e\n", i, val))<0)
                        return(WRITE_ERR);
                }
            }
        }
    }
    return(err);
}

int ElleWriteFlynnStrainData(FILE *fp,int *keys,int count)
{
    char label[20];
    int i, j, err=0;
    int max;
    double val;

    if (!id_match(FileKeys,FLYNN_STRAIN,label)) return(KEY_ERR);
    fprintf(fp,"%s\t",label);
    for (i=0;i<count;i++) {
        if (!id_match(FlynnStrainKeys,keys[i],label)) return(KEY_ERR);
        fprintf(fp,"%s ",label);
    }
    fprintf(fp,"\n");
    max = ElleMaxFlynns();
    for (i=0;i<max && !err;i++) {
        if (ElleFlynnIsActive(i)) {
            fprintf(fp,"%d",i);
            for (j=0;j<count;j++)  {
                ElleGetFlynnRealAttribute(i,&val,keys[j]);
                if (fprintf(fp," %.8e",val) < 0) err = WRITE_ERR;
            }
            fprintf(fp,"\n");
        }
    }
    return(err);
}

int ElleWriteFlynnStressData(FILE *fp,int *keys,int count)
{
    char label[20];
    int i, j, err=0;
    int max;
    double val;

    if (!id_match(FileKeys,FLYNN_STRESS,label)) return(KEY_ERR);
    fprintf(fp,"%s\t",label);
    for (i=0;i<count;i++) {
        if (!id_match(StressKeys,keys[i],label)) return(KEY_ERR);
        fprintf(fp,"%s ",label);
    }
    fprintf(fp,"\n");
    max = ElleMaxFlynns();
    for (i=0;i<max && !err;i++) {
        if (ElleFlynnIsActive(i)) {
            fprintf(fp,"%d",i);
            for (j=0;j<count;j++)  {
                ElleGetFlynnRealAttribute(i,&val,keys[j]);
                if (fprintf(fp," %.8e",val) < 0) err = WRITE_ERR;
            }
            fprintf(fp,"\n");
        }
    }
    return(err);
}

int ElleWriteCAxisData(FILE *fp)
{
    char label[20];
    int i, err=0, num;
    int max;
    double theta,phi;
    Coords_3D val, dflt;

    if (!id_match(FileKeys,CAXIS,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    ElleDefaultFlynnCAxis(&dflt);
    CartesianToPolar(dflt.x,dflt.y,dflt.z,&theta,&phi);
    if (theta<0.0) theta = PI-theta;
    /********!!!!!!!!! shouldn't this be
    if (theta<0.0) theta += 2.0*PI;
    ********/
    theta *= RTOD;
    phi *= RTOD;
    if ((num = fprintf(fp,"%s %d %d\n",SET_ALL,
                                  (int)(theta+0.5),(int)(phi+0.5)))<0)
        return(WRITE_ERR);
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i) && ElleFlynnHasAttribute(i,CAXIS)) {
            ElleGetFlynnCAxis(i,&val);
            if (val.x!=dflt.x || val.y!=dflt.y || val.z!=dflt.z) {
                CartesianToPolar(val.x,val.y,val.z,&theta,&phi);
                if (theta<0.0) theta = PI-theta;
    /********!!!!!!!!! shouldn't this be
    if (theta<0.0) theta += 2.0*PI;
    ********/
                theta *= RTOD;
                phi *= RTOD;
                if ((num = fprintf(fp,"%d %d %d\n", i,
                             (int)(theta+0.5), (int)(phi+0.5)))<0)
                    return(WRITE_ERR);
            }
        }
    }
    return(err);
}

int ElleWriteEuler3Data(FILE *fp)
{
    char label[20];
    int i, err=0, num;
    int max;
    double alpha, beta, gamma;
    double dflta, dfltb, dfltc;

    if (!id_match(FileKeys,EULER_3,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    ElleDefaultFlynnEuler3(&dflta,&dfltb,&dfltc);
    if ((num = fprintf(fp,"%s %.8e %.8e %.8e\n",SET_ALL,
                                  dflta,dfltb,dfltc))<0)
        return(WRITE_ERR);
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i) && ElleFlynnHasAttribute(i,EULER_3)) {
            ElleGetFlynnEuler3(i,&alpha, &beta, &gamma);
            if (dflta!=alpha || dfltb!=beta || dfltc!=gamma)
                if ((num = fprintf(fp,"%d %.8e %.8e %.8e\n", i,
                             alpha, beta, gamma))<0)
                    return(WRITE_ERR);
        }
    }
    return(err);
}

int LoadZIPFlynnIntAttribData( gzFile in, char str[], int attr )
{
  int err = 0, num;
  int val;
  int full_id;
  int s, u;
                                                                                
  ElleInitFlynnAttribute( attr );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
                                                                                
    else if ( !strcmp( str, SET_ALL ) )
    {
      val = atoi( ( gzReadSingleString( in ) ).c_str() );
      ElleSetDefaultFlynnIntAttribute( ( int )val, attr );
    }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( full_id = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR
);
      if( Region_ver )
      {
        s = atoi( ( gzReadSingleString( in ) ).c_str() );
        u = atoi( ( gzReadSingleString( in ) ).c_str() );
      }
      val = atoi( ( gzReadSingleString( in ) ).c_str() );
                                                                                
      /* do range checks on val */
      if ( ElleFlynnIsActive( full_id ) )
        ElleSetFlynnIntAttribute( full_id, val, attr );
      //     fscanf( fp, "\n" );
    }
  }
  return ( err );
}

int LoadZIPMineralData( gzFile in, char str[] )
{
  int err = 0, num;
  char *val;//[MAX_MINERAL_LEN + 1];
  int full_id, mineral_id;
                                                                                
  ElleInitFlynnAttribute( MINERAL );
  while ( !gzeof(in) && !err )
{
str=((char*)(gzReadSingleString(in)).c_str());
if ( str[0] == '#' ) gzReadLineSTD(in);
                                                                                
    else if ( !strcmp( str, SET_ALL ) )
    {
     val=((char*)(gzReadSingleString(in)).c_str());
      validate( val, & mineral_id, MineralKeys );
      if ( mineral_id < 0 ) return ( KEY_ERR );
      ElleSetDefaultFlynnMineral( mineral_id );
    }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( full_id = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR
);
     val=((char*)(gzReadSingleString(in)).c_str());
      /* do range checks on val */
      validate( val, & mineral_id, MineralKeys );
      if ( mineral_id < 0 ) return ( KEY_ERR );
      if ( ElleFlynnIsActive( full_id ) )
        ElleSetFlynnMineral( full_id, mineral_id );
    //  fscanf( fp, "\n" );
    }
  }
  return ( err );
}

int LoadZIPFlynnStrainData( gzFile in, char str[], int * keys, int count)
{
  /* * Read flynn strain data in an elle file. * Strain labels then
 * strain data on a separate
  * line for each flynn with the format: %d %f %f .... * (flynn id
  * number then strain values in the order read into keys) */
  int i;
  int nn, num, err = 0;
  double val[NUM_FLYNN_STRAIN_VALS];
                                                                                
  for ( i = 0; i < count; i++ )
    ElleInitFlynnAttribute( keys[i] );
    while ( !gzeof(in) && !err )
  {
  str=((char*)(gzReadSingleString(in)).c_str());
  if ( str[0] == '#' ) gzReadLineSTD(in);
                                                                                
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    /* * SET_ALL not valid for strain vals */
    else
    {
      if ( ( nn = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR );
      for ( i = 0; i < count; i++ )
      {
         val[i]=atof((gzReadSingleString(in)).c_str());
        ElleSetFlynnRealAttribute( nn, val[i], keys[i] );
      }
    }
    //fscanf( fp, "\n" );
  }
  return ( 0 );
}

int LoadZIPAgeData( gzFile in, char str[], int * keys, int count )
{
  int i, err = 0, num;
  int full_id;
  int s, u;
                                                                                
  double val[NUM_FLYNN_AGE_VALS];
                                                                                
  for ( i = 0; i < count; i++ )
    ElleInitFlynnAttribute( keys[i] );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
                                                                                
    else if ( !strcmp( str, SET_ALL ) )
    {
      for ( i = 0; i < count; i++ )
      {
        val[i] = atof( ( gzReadSingleString( in ) ).c_str() );
        ElleSetDefaultFlynnAge( val[i], keys[i] );
      }
    }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( full_id = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR
);
      if ( Region_ver )
        s = atoi( ( gzReadSingleString( in ) ).c_str() );
      u = atoi( ( gzReadSingleString( in ) ).c_str() );
      for ( i = 0; i < count; i++ )
      {
        val[i] = atof( ( gzReadSingleString( in ) ).c_str() );
        /* do range checks on val */
        if ( ElleFlynnIsActive( full_id ) )
          ElleSetFlynnAge( full_id, val[i], keys[i] );
      }
    }
  }
  return ( err );
}

int LoadZIPFlynnRealAttribData( gzFile in, char str[], int id )
{
  int err = 0, num;
  double val;
  int full_id;
  int s, u;
                                                                                
  ElleInitFlynnAttribute( id );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
                                                                                
    else if ( !strcmp( str, SET_ALL ) )
    {
      val = atof( ( gzReadSingleString( in ) ).c_str() );
      ElleSetDefaultFlynnRealAttribute( val, id );
    }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( full_id = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR
);
      if ( Region_ver )
      {
        s = atoi( ( gzReadSingleString( in ) ).c_str() );
        u = atoi( ( gzReadSingleString( in ) ).c_str() );
      }
      val = atof( ( gzReadSingleString( in ) ).c_str() );
                                                                                
      /* do range checks on val */
      if ( ElleFlynnIsActive( full_id ) )
        ElleSetFlynnRealAttribute( full_id, val, id );
      //    fscanf( fp, "\n" );
    }
  }
  return ( err );
}

int LoadZIPCAxisData( gzFile in, char str[] )
{
  /* * routine to be called after the CAxis keyword has * been read in
 * an Elle file * Reads lines with the format %d %lf %lf
  *           (node id, angle theta, angle phi; angles in degrees) */
  int err = 0, num;
  Coords_3D val;
  int full_id;
  int s, u;
  double theta, phi;
                                                                                
  ElleInitFlynnAttribute( CAXIS );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
                                                                                
    else if ( !strcmp( str, SET_ALL ) )
    {
      theta = atof( ( gzReadSingleString( in ) ).c_str() );
      phi = atof( ( gzReadSingleString( in ) ).c_str() );
      /* convert from degrees to radians */
      theta *= DTOR;
      phi *= DTOR;
      PolarToCartesian( & val.x, & val.y, & val.z, theta, phi );
      ElleSetDefaultFlynnCAxis( & val );
    }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( full_id = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR
);
      if ( Region_ver )
      {
        s = atoi( ( gzReadSingleString( in ) ).c_str() );
        u = atoi( ( gzReadSingleString( in ) ).c_str() );
      }
      theta = atof( ( gzReadSingleString( in ) ).c_str() );
      phi = atof( ( gzReadSingleString( in ) ).c_str() );
      /* convert from degrees to radians */
      theta *= DTOR;
      phi *= DTOR;
      PolarToCartesian( & val.x, & val.y, & val.z, theta, phi );
      /* do range checks on val */
      if ( ElleFlynnIsActive( full_id ) )
        ElleSetFlynnCAxis( full_id, & val );
                                                                                     
    }
  }
  return ( err );
}

int LoadZIPEuler3Data( gzFile in, char str[] )
{
  /* * routine to be called after the EULER_3 keyword has * been read in
 * an Elle file
  * Reads lines with the format %d %f %f %f *    (node id, angle alpha,
  * angle beta, angle gamma; angles in degrees) */
  int err = 0, num;
  int full_id;
  double alpha, beta, gamma;
                                                                                
  ElleInitFlynnAttribute( EULER_3 );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
                                                                                
    else if ( !strcmp( str, SET_ALL ) )
    {
      alpha = atof( ( gzReadSingleString( in ) ).c_str() );
      beta = atof( ( gzReadSingleString( in ) ).c_str() );
      gamma = atof( ( gzReadSingleString( in ) ).c_str() );
      ElleSetDefaultFlynnEuler3( alpha, beta, gamma );     }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );     else
    {
      if ( ( full_id = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR
);
      alpha = atof( ( gzReadSingleString( in ) ).c_str() );
      beta = atof( ( gzReadSingleString( in ) ).c_str() );
      gamma = atof( ( gzReadSingleString( in ) ).c_str() );
                                                                                
      /* do range checks on val */
      if ( ElleFlynnIsActive( full_id ) )
        ElleSetFlynnEuler3( full_id, alpha, beta, gamma );
      //      fscanf( fp, "\n" );
    }
  }
  return ( err );
}

int LoadZIPGrainData( gzFile in, char str[] )
{
  int err = 0, num;
  float val;
  int full_id;
                                                                                
  ElleInitFlynnAttribute( GRAIN );
  while ( !gzeof(in) && !err )
{
str=((char*)(gzReadSingleString(in)).c_str());
if ( str[0] == '#' ) gzReadLineSTD(in);
                                                                                
    else if ( !strcmp( str, SET_ALL ) )
    {
       val=atof((gzReadSingleString(in)).c_str());
      ElleSetDefaultFlynnGrain( ( int )val );
    }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( full_id = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR
);
       val=atof((gzReadSingleString(in)).c_str());
      /* do range checks on val */
      if ( ElleFlynnIsActive( full_id ) )
        ElleSetFlynnGrain( full_id, ( int )val );
  //    fscanf( fp, "\n" );
    }
  }
  return ( err );
}

int LoadZIPColourData( gzFile in, char str[] )
{
  int err = 0, num;
  float val;
  int full_id;
  int s, u;
                                                                                
  ElleInitFlynnAttribute( COLOUR );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
    else if ( !strcmp( str, SET_ALL ) )
    {
      val = atof( ( gzReadSingleString( in ) ).c_str() );
      ElleSetDefaultFlynnColour( ( int )val );
    }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( full_id = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR
);
      if ( Region_ver )
      {
        s = atoi( ( gzReadSingleString( in ) ).c_str() );
        u = atoi( ( gzReadSingleString( in ) ).c_str() );
      }
      val = atof( ( gzReadSingleString( in ) ).c_str() );
      /* do range checks on val */
      if ( ElleFlynnIsActive( full_id ) )
        ElleSetFlynnColour( full_id, ( int )val );
      //fscanf( fp, "\n" );
    }
  }
  return ( err );
}

int LoadZIPViscosityData( gzFile in, char str[] )
{
  int err = 0, num;
  float val;
  int full_id;
  int s, u;
                                                                                
  ElleInitFlynnAttribute( VISCOSITY );
  while ( !gzeof(in) && !err )
{
str=((char*)(gzReadSingleString(in)).c_str());
if ( str[0] == '#' ) gzReadLineSTD(in);
                                                                                
    else if ( !strcmp( str, SET_ALL ) )
    {
      val=atof((gzReadSingleString(in)).c_str());
      ElleSetDefaultFlynnViscosity( val );
    }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( full_id = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR
);
      if ( Region_ver )
        {
           s=atoi((gzReadSingleString(in)).c_str());
            u=atoi((gzReadSingleString(in)).c_str());
        }
 val=atof((gzReadSingleString(in)).c_str());
                                                                                
      /* do range checks on val */
      if ( ElleFlynnIsActive( full_id ) )
        ElleSetFlynnViscosity( full_id, val );
  //    fscanf( fp, "\n" );
    }
  }
  return ( err );
}

int LoadZIPDislocDenData( gzFile in, char str[] )
{
  int err = 0, num;
  float val;
  int full_id;
  int s, u;
                                                                                
  ElleInitFlynnAttribute( DISLOCDEN );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
                                                                                
    else if ( !strcmp( str, SET_ALL ) )
    {
      val = atof( ( gzReadSingleString( in ) ).c_str() );
      ElleSetDefaultFlynnDislocDen( val );
    }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( full_id = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR
);
      if ( Region_ver )
      {
        s = atoi( ( gzReadSingleString( in ) ).c_str() );
        u = atoi( ( gzReadSingleString( in ) ).c_str() );
      }
      val = atof( ( gzReadSingleString( in ) ).c_str() );
      /* do range checks on val */
      if ( ElleFlynnIsActive( full_id ) )
        ElleSetFlynnDislocDen( full_id, val );
      //fscanf( fp, "\n" );
    }
  }
  return ( err );
}

int LoadZIPEnergyData( gzFile in, char str[] )
{
  int err = 0, num;
  float val;
  int full_id;
  int s, u;
                                                                                     
  ElleInitFlynnAttribute( ENERGY );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
                                                                                     
    else if ( !strcmp( str, SET_ALL ) )
    {
      val = atof( ( gzReadSingleString( in ) ).c_str() );
      ElleSetDefaultFlynnEnergy( val );
    }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( full_id = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR
);
      if ( Region_ver )
      {
        s = atoi( ( gzReadSingleString( in ) ).c_str() );
        u = atoi( ( gzReadSingleString( in ) ).c_str() );
      }
      val = atof( ( gzReadSingleString( in ) ).c_str() );
                                                                                     
      /* do range checks on val */
      if ( ElleFlynnIsActive( full_id ) )
        ElleSetFlynnEnergy( full_id, val );
      //fscanf( fp, "\n" );
    }
  }
  return ( err );
}

int SaveZIPAgeData( gzFile out, int * keys, int count )
{
  char label[20];
  int i, j, err = 0, num = 0;
  int max;
  double val, dflt;
                                                                                     
  if ( !id_match( FileKeys, AGE, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\t", label );
  for ( i = 0; i < count; i++ )
  {
    if ( !id_match( FlynnAgeKeys, keys[i], label ) ) return ( KEY_ERR );
    gzprintf( out, "%s ", label );
  }
  gzprintf( out, "\n" );
  if ( ( num = gzprintf( out, "%s", SET_ALL ) ) < 0 )
    return ( WRITE_ERR );
  for ( j = 0; j < count; j++ )
  {
    dflt = ElleDefaultFlynnRealAttribute( keys[j] );
    if ( gzprintf( out, " %.8e", dflt ) < 0 ) err = WRITE_ERR;
  }
  gzprintf( out, "\n" );
  max = ElleMaxFlynns();
  for ( i = 0; i < max && !err; i++ )
  {
    if ( ElleFlynnIsActive( i ) && ElleFlynnHasAttribute( i, AGE ) )
    {
      gzprintf( out, "%d", i );
      for ( j = 0; j < count; j++ )
      {
        ElleGetFlynnRealAttribute( i, & val, keys[j] );
        if ( gzprintf( out, " %.8e", val ) < 0 ) err = WRITE_ERR;
      }
      gzprintf( out, "\n" );
    }
  }
  return ( err );
}

int SaveZIPGrainData( gzFile out )
{
  char label[20];
  int i, err = 0, val, dflt, num;
  int max;
                                                                                     
  if ( !id_match( FileKeys, GRAIN, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );
  dflt = ElleDefaultFlynnGrain();
  if ( ( num = gzprintf( out, "%s %d\n", SET_ALL, dflt ) ) < 0 )
    return ( WRITE_ERR );
  max = ElleMaxFlynns();
  for ( i = 0; i < max; i++ )
  {
    if ( ElleFlynnIsActive( i ) && ElleFlynnHasAttribute( i, GRAIN ) )
    {
      ElleGetFlynnGrain( i, & val );
      if ( val != dflt )
      {
        if ( ( num = gzprintf( out, "%d %d\n", i, val ) ) < 0 )
          return ( WRITE_ERR );
      }
    }
  }
  return ( err );
}

int SaveZIPCAxisData( gzFile out )
{
  char label[20];
  int i, err = 0, num;
  int max;
  double theta, phi;
  Coords_3D val, dflt;
                                                                                     
  if ( !id_match( FileKeys, CAXIS, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );
  ElleDefaultFlynnCAxis( & dflt );
  CartesianToPolar( dflt.x, dflt.y, dflt.z, & theta, & phi );
  if ( theta < 0.0 ) theta = PI - theta;
  /** !!!!!!!!! shouldn't this be if (theta<0.0) theta += 2.0*PI; */
  theta *= RTOD;
  phi *= RTOD;
  if ( ( num = gzprintf( out, "%s %d %d\n", SET_ALL, ( int )( theta +
0.5 ), ( int )( phi + 0.5 ) ) ) < 0 )
    return ( WRITE_ERR );
  max = ElleMaxFlynns();
  for ( i = 0; i < max; i++ )
  {
    if ( ElleFlynnIsActive( i ) && ElleFlynnHasAttribute( i, CAXIS ) )
    {
      ElleGetFlynnCAxis( i, & val );
      if ( val.x != dflt.x || val.y != dflt.y || val.z != dflt.z )
      {
        CartesianToPolar( val.x, val.y, val.z, & theta, & phi );
        if ( theta < 0.0 ) theta = PI - theta;
        /** !!!!!!!!! shouldn't this be if (theta<0.0) theta += 2.0*PI;
 * */
        theta *= RTOD;
        phi *= RTOD;
        if ( ( num = gzprintf( out, "%d %d %d\n", i, ( int )( theta +
0.5 ), ( int )( phi + 0.5 ) ) ) < 0 )
          return ( WRITE_ERR );
      }
    }
  }
  return ( err );
}

int SaveZIPEuler3Data( gzFile out )
{
  char label[20];
  int i, err = 0, num;
  int max;
  double alpha, beta, gamma;
  double dflta, dfltb, dfltc;
                                                                                     
  if ( !id_match( FileKeys, EULER_3, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );
  ElleDefaultFlynnEuler3( & dflta, & dfltb, & dfltc );
  if ( ( num = gzprintf( out, "%s %.8e %.8e %.8e\n", SET_ALL, dflta,
dfltb, dfltc ) ) < 0 )
    return ( WRITE_ERR );
  max = ElleMaxFlynns();
  for ( i = 0; i < max; i++ )
  {
    if ( ElleFlynnIsActive( i ) && ElleFlynnHasAttribute( i, EULER_3 ) )
    {
      ElleGetFlynnEuler3( i, & alpha, & beta, & gamma );
      if ( dflta != alpha || dfltb != beta || dfltc != gamma )
        if ( ( num = gzprintf( out, "%d %.8e %.8e %.8e\n", i, alpha,
beta, gamma ) ) < 0 )
          return ( WRITE_ERR );
    }
  }
  return ( err );
}

int SaveZIPColourData( gzFile out )
{
  char label[20];
  int i, err = 0, val, num, dflt;
  int max;
                                                                                     
  if ( !id_match( FileKeys, COLOUR, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );
  dflt = ElleDefaultFlynnColour();
  if ( ( num = gzprintf( out, "%s %d\n", SET_ALL, dflt ) ) < 0 )
    return ( WRITE_ERR );
  max = ElleMaxFlynns();
  for ( i = 0; i < max; i++ )
  {
    if ( ElleFlynnIsActive( i ) && ElleFlynnHasAttribute( i, COLOUR ) )
    {
      ElleGetFlynnColour( i, & val );
      if ( val != dflt )
      {
        if ( ( num = gzprintf( out, "%d %d\n", i, val ) ) < 0 )
          return ( WRITE_ERR );
      }
    }
  }
  return ( err );
}

int SaveZIPDislocDenData( gzFile out )
{
  char label[20];
  int i, err = 0, num;
  int max;
  float val, dflt;
                                                                                     
  if ( !id_match( FileKeys, DISLOCDEN, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );
  dflt = ElleDefaultFlynnDislocDen();
  if ( ( num = gzprintf( out, "%s %f\n", SET_ALL, dflt ) ) < 0 )
    return ( WRITE_ERR );
  max = ElleMaxFlynns();
  for ( i = 0; i < max; i++ )
  {
    if ( ElleFlynnIsActive( i ) && ElleFlynnHasAttribute( i, DISLOCDEN )
)
    {
      ElleGetFlynnDislocDen( i, & val );
      if ( val != dflt )
      {
        if ( ( num = gzprintf( out, "%d %f\n", i, val ) ) < 0 )
          return ( WRITE_ERR );
      }
    }
  }
  return ( err );
}

int SaveZIPEnergyData( gzFile out )
{
  char label[20];
  int i, err = 0, num;
  int max;
  float val, dflt;
  if ( !id_match( FileKeys, ENERGY, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );
  dflt = ElleDefaultFlynnEnergy();
  if ( ( num = gzprintf( out, "%s %f\n", SET_ALL, dflt ) ) < 0 )
    return ( WRITE_ERR );
  max = ElleMaxFlynns();
  for ( i = 0; i < max; i++ )
  {
    if ( ElleFlynnIsActive( i ) && ElleFlynnHasAttribute( i, ENERGY ) )
    {
      ElleGetFlynnEnergy( i, & val );
      if ( val != dflt )
      {
        if ( ( num = gzprintf( out, "%d %f\n", i, val ) ) < 0 )
          return ( WRITE_ERR );
      }
    }
  }
  return ( err );
}

int SaveZIPExpandData( gzFile out )
{
  char label[20];
  int i, err = 0, val, dflt, num;
  int max;
                                                                                     
  if ( !id_match( FileKeys, EXPAND, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );
  dflt = ElleDefaultFlynnExpand();
  if ( ( num = gzprintf( out, "%s %d\n", SET_ALL, dflt ) ) < 0 )
    return ( WRITE_ERR );
  max = ElleMaxFlynns();
  for ( i = 0; i < max; i++ )
  {
    if ( ElleFlynnIsActive( i ) && ElleFlynnHasAttribute( i, EXPAND ) )
    {
      ElleGetFlynnIntAttribute( i, & val, EXPAND );
      if ( val != dflt )
      {
        if ( ( num = gzprintf( out, "%d %d\n", i, val ) ) < 0 )
          return ( WRITE_ERR );
      }
    }
  }
  return ( err );
}
                                                                                     
int SaveZIPFlynnIntAttribData( gzFile out, int key )
{
  char label[20];
  int i, err = 0, val, dflt, num;
  int max;
                                                                                     
  if ( !id_match( FileKeys, key, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );
  dflt = ElleDefaultFlynnIntAttribute( key );
  if ( ( num = gzprintf( out, "%s %d\n", SET_ALL, dflt ) ) < 0 )
    return ( WRITE_ERR );
  max = ElleMaxFlynns();
  for ( i = 0; i < max; i++ )
  {
    if ( ElleFlynnIsActive( i ) && ElleFlynnHasAttribute( i, key ) )
    {
      ElleGetFlynnIntAttribute( i, & val, key );
      if ( val != dflt )
      {
        if ( ( num = gzprintf( out, "%d %d\n", i, val ) ) < 0 )
          return ( WRITE_ERR );
      }
    }
  }
  return ( err );
}

int SaveZIPFlynnRealAttribData( gzFile out, int id )
{
  char label[20];
  int i, err = 0, num;
  unsigned char int_attrib = 0;
  int max;
  double val, dflt;
                                                                                     
  if ( !id_match( FileKeys, id, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );
  int_attrib = isFlynnIntegerAttribute( id );
  dflt = ElleDefaultFlynnRealAttribute( id );
  if ( int_attrib )
  {
    if ( ( num = gzprintf( out, "%s %d\n", SET_ALL, ( int )dflt ) ) < 0
)
      return ( WRITE_ERR );
  }
  else
  {
    if ( ( num = gzprintf( out, "%s %.8e\n", SET_ALL, dflt ) ) < 0 )
      return ( WRITE_ERR );
  }
  max = ElleMaxFlynns();
  for ( i = 0; i < max; i++ )
  {
    if ( ElleFlynnIsActive( i ) && ElleFlynnHasAttribute( i, id ) )
    {
      ElleGetFlynnRealAttribute( i, & val, id );
      if ( val != dflt )
      {
        if ( int_attrib )
        {
          if ( ( num = gzprintf( out, "%d %d\n", i, ( int )val ) ) < 0 )
            return ( WRITE_ERR );
        }
        else
        {
          if ( ( num = gzprintf( out, "%d %.8e\n", i, val ) ) < 0 )
            return ( WRITE_ERR );
        }
      }
    }
  }
  return ( err );
}

int SaveZIPFlynnStrainData( gzFile out, int * keys, int count )
{
  char label[20];
  int i, j, err = 0;
  int max;
  double val;
                                                                                     
  if ( !id_match( FileKeys, FLYNN_STRAIN, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\t", label );
  for ( i = 0; i < count; i++ )
  {
    if ( !id_match( FlynnStrainKeys, keys[i], label ) ) return ( KEY_ERR
);
    gzprintf( out, "%s ", label );
  }
  gzprintf( out, "\n" );
  max = ElleMaxFlynns();
  for ( i = 0; i < max && !err; i++ )
  {
    if ( ElleFlynnIsActive( i ) )
    {
      gzprintf( out, "%d", i );
      for ( j = 0; j < count; j++ )
      {
        ElleGetFlynnRealAttribute( i, & val, keys[j] );
        if ( gzprintf( out, " %.8e", val ) < 0 ) err = WRITE_ERR;
      }
      gzprintf( out, "\n" );
    }
  }
  return ( err );
}

int SaveZIPMineralData( gzFile out )
{
  char label[20], val[MAX_MINERAL_LEN + 1];
  int i, err = 0, mineral_id, dflt, num;
  int max;
                                                                                     
  if ( !id_match( FileKeys, MINERAL, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );
  dflt = ElleDefaultFlynnMineral();
  if ( !id_match( MineralKeys, dflt, val ) ) return ( KEY_ERR );
  if ( ( num = gzprintf( out, "%s %s\n", SET_ALL, val ) ) < 0 )
    return ( WRITE_ERR );
  max = ElleMaxFlynns();
  for ( i = 0; i < max; i++ )
  {
    if ( ElleFlynnIsActive( i ) && ElleFlynnHasAttribute( i, MINERAL ) )
    {
      ElleGetFlynnMineral( i, & mineral_id );
      if ( mineral_id != dflt )
      {
        if ( !id_match( MineralKeys, mineral_id, val ) ) return (
KEY_ERR );
        if ( ( num = gzprintf( out, "%d %s\n", i, val ) ) < 0 )
          return ( WRITE_ERR );
      }
    }
  }
  return ( err );
}

int SaveZIPRegionAttributeData( gzFile out )
{
  char label[20];
  int err = 0, i, j;
  int * active, maxa, * keys;
                                                                                     
  ElleFlynnDfltAttributeList( & active, & maxa );
  for ( i = 0; i < maxa && !err; i++ )
  {
    switch ( active[i] )
    {
      case EXPAND:
      case COLOUR:
      case GRAIN:
      case SPLIT:
        err = SaveZIPFlynnIntAttribData( out, active[i] );
      break;
      case MINERAL:
        err = SaveZIPMineralData( out );
      break;
      case CAXIS:
        err = SaveZIPCAxisData( out );
//        if ( ElleEnergyLUTSize() > 0 )
 //         err = SaveZIPGBEnergyLUT( out );
        while ( i < ( maxa - 1 )
             && ( active[i + 1] == CAXIS_Z || active[i + 1] == CAXIS_Y
|| active[i + 1] == CAXIS_X || active[i + 1] == CAXIS ) )
               i++;
      break;
      case E3_ALPHA:
        err = SaveZIPEuler3Data( out );
//        if ( ElleEnergyLUTSize() > 0 )
//          err = SaveZIPGBEnergyLUT( out );
        while ( i < ( maxa - 1 ) && ( active[i + 1] == E3_BETA ||
active[i + 1] == E3_GAMMA ) ) i++;
      break;
      case F_INCR_S:
      case F_BULK_S:
      case E_XX:
      case E_XY:
      case E_YX:
      case E_YY:
      case E_ZZ:
        if ( ( keys = ( int * ) malloc( NUM_FLYNN_STRAIN_VALS * sizeof(
int ) ) ) == 0 )
          OnError( "ElleZIPRegionAttributeData", MALLOC_ERR );
        for ( j = 0; j < NUM_FLYNN_STRAIN_VALS; j++ ) keys[j] = NO_INDX;
        j = 0;
        while ( i < maxa && id_match( FlynnStrainKeys, active[i], label
) )
        {
          keys[j] = active[i];
          j++; i++;
        }
        i--;
        /* for loop will increment again */
        err = SaveZIPFlynnStrainData( out, keys, j );
        if (keys) free( keys );
      break;
      case AGE:
      case CYCLE:
        if ( ( keys = ( int * ) malloc( NUM_FLYNN_AGE_VALS * sizeof( int
) ) ) == 0 )          OnError( "ElleZiPRegionAttributeData", MALLOC_ERR
);
        for ( j = 0; j < NUM_FLYNN_AGE_VALS; j++ ) keys[j] = NO_INDX;
        j = 0;
        while ( i < maxa && id_match( FlynnAgeKeys, active[i], label ) )
        {
          keys[j] = active[i];
          j++; i++;
        }
        i--;
        /* for loop will increment again */
        err = SaveZIPAgeData( out, keys, j );
        if (keys) free( keys );
      break;
      break;
      case VISCOSITY:
      case ENERGY:
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
        err = SaveZIPFlynnRealAttribData( out, active[i] );
      break;
      default:
        err = ATTRIBID_ERR;
      break;
    }
  }
  if ( active ) free( active );
  return ( err );
}

int SaveZIPRegionData( gzFile out )
{
  int err = 0;
                                                                                     
  if ( err = SaveZIPFlynnData( out ) ) return ( err );
  err = SaveZIPRegionAttributeData( out );
  return ( err );
}

int SaveZIPViscosityData( gzFile out )
{
  char label[20];
  int i, err = 0, num;
  int max;
  float val, dflt;
                                                                                     
  if ( !id_match( FileKeys, VISCOSITY, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );
  dflt = ElleDefaultFlynnViscosity();
  if ( ( num = gzprintf( out, "%s %f\n", SET_ALL, dflt ) ) < 0 )
    return ( WRITE_ERR );
  max = ElleMaxFlynns();
  for ( i = 0; i < max; i++ )
  {
    if ( ElleFlynnIsActive( i ) && ElleFlynnHasAttribute( i, VISCOSITY )
)
    {
      ElleGetFlynnViscosity( i, & val );
      if ( val != dflt )
      {
        if ( ( num = gzprintf( out, "%d %f\n", i, val ) ) < 0 )
          return ( WRITE_ERR );
      }
    }
  }
  return ( err );
}

/*******************
 * phase out below this line
 *****************/
int ElleReadExpandData(FILE *fp, char str[])
{
    int err=0, num;
    float val;
    int full_id;
    int s, u;

    ElleInitFlynnAttribute(EXPAND);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%f\n", &val)) != 1)
                return(READ_ERR);
                ElleSetDefaultFlynnIntAttribute((int)val,EXPAND);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if (Region_ver)
                if ((num = fscanf(fp,"%d %d",&s,&u))!=2)
                    return(READ_ERR);
            if ((num = fscanf(fp,"%f", &val))!=1) return(READ_ERR);
            /* do range checks on val */
            if (ElleFlynnIsActive(full_id))
                ElleSetFlynnIntAttribute(full_id,(int)val,EXPAND);
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleReadSplitData(FILE *fp, char str[])
{
    int err=0, num;
    float val;
    int full_id;

    ElleInitFlynnAttribute(SPLIT);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%f\n", &val)) != 1)
                return(READ_ERR);
                ElleSetDefaultFlynnSplit((int)val);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if ((num = fscanf(fp,"%f", &val))!=1) return(READ_ERR);
            /* do range checks on val */
            if (ElleFlynnIsActive(full_id))
                ElleSetFlynnSplit(full_id,(int)val);
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleReadGrainData(FILE *fp, char str[])
{
    int err=0, num;
    float val;
    int full_id;

    ElleInitFlynnAttribute(GRAIN);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%f\n", &val)) != 1)
                return(READ_ERR);
                ElleSetDefaultFlynnGrain((int)val);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if ((num = fscanf(fp,"%f", &val))!=1) return(READ_ERR);
            /* do range checks on val */
            if (ElleFlynnIsActive(full_id))
                ElleSetFlynnGrain(full_id,(int)val);
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleReadColourData(FILE *fp, char str[])
{
    int err=0, num;
    float val;
    int full_id;
    int s, u;

    ElleInitFlynnAttribute(COLOUR);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp)) return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%f\n", &val)) != 1)
                return(READ_ERR);
                ElleSetDefaultFlynnColour((int) val);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id= atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if (Region_ver)
                if ((num = fscanf(fp,"%d %d",&s,&u))!=2)
                    return(READ_ERR);
            if ((num = fscanf(fp,"%f", &val))!=1) return(READ_ERR);
            /* do range checks on val */
            if (ElleFlynnIsActive(full_id))
                ElleSetFlynnColour(full_id,(int)val);
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleReadViscosityData(FILE *fp, char str[])
{
    int err=0, num;
    float val;
    int full_id;
    int s, u;

    ElleInitFlynnAttribute(VISCOSITY);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%f\n", &val)) != 1)
                return(READ_ERR);
                ElleSetDefaultFlynnViscosity(val);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if (Region_ver)
                if ((num = fscanf(fp,"%d %d",&s,&u))!=2)
                    return(READ_ERR);
            if ((num = fscanf(fp,"%f", &val))!=1) return(READ_ERR);
            /* do range checks on val */
            if (ElleFlynnIsActive(full_id))
                ElleSetFlynnViscosity(full_id,val);
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleReadDislocDenData(FILE *fp, char str[])
{
    int err=0, num;
    float val;
    int full_id;
    int s, u;

    ElleInitFlynnAttribute(DISLOCDEN);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%f\n", &val)) != 1)
                return(READ_ERR);
                ElleSetDefaultFlynnDislocDen(val);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if (Region_ver)
                if ((num = fscanf(fp,"%d %d",&s,&u))!=2)
                    return(READ_ERR);
            if ((num = fscanf(fp,"%f", &val))!=1) return(READ_ERR);
            /* do range checks on val */
            if (ElleFlynnIsActive(full_id))
                ElleSetFlynnDislocDen(full_id,val);
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleReadEnergyData(FILE *fp, char str[])
{
    int err=0, num;
    float val;
    int full_id;
    int s, u;

    ElleInitFlynnAttribute(ENERGY);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp)) return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%f\n", &val)) != 1)
                return(READ_ERR);
            ElleSetDefaultFlynnEnergy(val);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id= atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if (Region_ver)
                if ((num = fscanf(fp,"%d %d",&s,&u))!=2)
                    return(READ_ERR);
            if ((num = fscanf(fp,"%f", &val))!=1) return(READ_ERR);
            /* do range checks on val */
            if (ElleFlynnIsActive(full_id))
                ElleSetFlynnEnergy(full_id,val);
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleWriteGrainData(FILE *fp)
{
    char label[20];
    int i, err=0, val, dflt, num;
    int max;

    if (!id_match(FileKeys,GRAIN,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    dflt = ElleDefaultFlynnGrain();
    if ((num = fprintf(fp,"%s %d\n",SET_ALL,dflt))<0)
        return(WRITE_ERR);
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i) && ElleFlynnHasAttribute(i,GRAIN)) {
            ElleGetFlynnGrain(i,&val);
            if (val!=dflt) {
                if ((num = fprintf(fp,"%d %d\n", i, val))<0)
                    return(WRITE_ERR);
            }
        }
    }
    return(err);
}

int ElleWriteExpandData(FILE *fp)
{
    char label[20];
    int i, err=0, val, dflt, num;
    int max;

    if (!id_match(FileKeys,EXPAND,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    dflt = ElleDefaultFlynnExpand();
    if ((num = fprintf(fp,"%s %d\n",SET_ALL,dflt))<0)
        return(WRITE_ERR);
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i) && ElleFlynnHasAttribute(i,EXPAND)) {
            ElleGetFlynnIntAttribute(i,&val,EXPAND);
            if (val!=dflt) {
                if ((num = fprintf(fp,"%d %d\n", i, val))<0)
                    return(WRITE_ERR);
            }
        }
    }
    return(err);
}

int ElleWriteColourData(FILE *fp)
{
    char label[20];
    int i, err=0, val, num, dflt;
    int max;

    if (!id_match(FileKeys,COLOUR,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    dflt = ElleDefaultFlynnColour();
    if ((num = fprintf(fp,"%s %d\n",SET_ALL,dflt))<0)
        return(WRITE_ERR);
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i) && ElleFlynnHasAttribute(i,COLOUR)) {
            ElleGetFlynnColour(i,&val);
            if (val!=dflt) {
                if ((num = fprintf(fp,"%d %d\n", i, val))<0)
                    return(WRITE_ERR);
            }
        }
    }
    return(err);
}

int ElleWriteEnergyData(FILE *fp)
{
    char label[20];
    int i, err=0, num;
    int max;
    float val, dflt;
    if (!id_match(FileKeys,ENERGY,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    dflt = ElleDefaultFlynnEnergy();
    if ((num = fprintf(fp,"%s %f\n",SET_ALL,dflt))<0)
        return(WRITE_ERR);
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i) && ElleFlynnHasAttribute(i,ENERGY)) {
            ElleGetFlynnEnergy(i,&val);
            if (val!=dflt) {
                if ((num = fprintf(fp,"%d %f\n", i, val))<0)
                    return(WRITE_ERR);
            }
        }
    }
    return(err);
}

int ElleWriteDislocDenData(FILE *fp)
{
    char label[20];
    int i, err=0, num;
    int max;
    float val, dflt;

    if (!id_match(FileKeys,DISLOCDEN,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    dflt = ElleDefaultFlynnDislocDen();
    if ((num = fprintf(fp,"%s %f\n",SET_ALL,dflt))<0)
        return(WRITE_ERR);
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i) && ElleFlynnHasAttribute(i,DISLOCDEN)) {
            ElleGetFlynnDislocDen(i,&val);
            if (val!=dflt) {
                if ((num = fprintf(fp,"%d %f\n", i, val))<0)
                    return(WRITE_ERR);
            }
        }
    }
    return(err);
}

int ElleWriteViscosityData(FILE *fp)
{
    char label[20];
    int i, err=0, num;
    int max;
    float val, dflt;

    if (!id_match(FileKeys,VISCOSITY,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    dflt = ElleDefaultFlynnViscosity();
    if ((num = fprintf(fp,"%s %f\n",SET_ALL,dflt))<0)
        return(WRITE_ERR);
    max = ElleMaxFlynns();
    for (i=0;i<max;i++) {
        if (ElleFlynnIsActive(i) && ElleFlynnHasAttribute(i,VISCOSITY)) {
            ElleGetFlynnViscosity(i,&val);
            if (val!=dflt) {
                if ((num = fprintf(fp,"%d %f\n", i, val))<0)
                    return(WRITE_ERR);
            }
        }
    }
    return(err);
}
