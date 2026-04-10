 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile: file.cc,v $
 * Revision:  $Revision: 1.17 $
 * Date:      $Date: 2014/05/23 12:02:18 $
 * Author:    $Author: levans $
 *
 ******************************************************/
#include <stdio.h>
#include "bflynns.h"
#include "version.h"
#include <cstring>
#include "nodes.h"
#include "string_utils.h"
#include "runopts.h"
#include "lut.h"
#include "regions.h"
#include "general.h"
#include "interface.h"
#include "unodes.h"
#include "segments.h"
#include "file.h"
#include "stats.h"
#include "error.h"
#include "misorient.h"

#include "display.h"

/*****************************************************

static const char rcsid[] =
       "$Id: file.cc,v 1.17 2014/05/23 12:02:18 levans Exp $";

******************************************************/
#define ELLE_EXT ".elle"
#define HEADER_MAX 100


int ElleReadData( char *filename)
{
    const int MAX_KEYS=20;
    char str[256], *ptr, *start;
    int num, key, keys[MAX_KEYS], finished=0, err=0;
    int i;
    FILE *fp;

    if ((fp=fopen(filename,"r"))==NULL) return(OPEN_ERR);
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
        case REGIONS :
        case FLYNNS  : if (err = ElleReadFlynnData(fp,str))
                           finished = 1;
                       break;
        case PARENTS : if (err = ElleReadParentData(fp,str))
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
                                  i<MAX_KEYS && !finished) {
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
                       /*ElleVoronoiUnodes();*/
                       break;
        case SEGMENTS:
            if (err = ElleReadSegmentData(fp,str))
                finished = 1;
            break;
        case LOCATION: if (err = ElleReadLocationData(fp,str)) 
                           finished = 1;
                       break;
        case AGE:
                       /*
                        * read the order of age vals
                        */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          start = str;
                          while ((ptr=strtok(start," \t\r\n")) && 
                                  i<NUM_FLYNN_AGE_VALS && !finished) {
                              validate(ptr,&keys[i],FlynnAgeKeys);
                              if (keys[i]<0) finished = 1;
                              else {
                                  start = 0;
                                  i++;
                              }
                           }
                           if (i<1) finished = 1;
                       }
                       if (finished) err = READ_ERR;
                       else if (err = ElleReadAgeData(fp,str,keys,i)) 
                           finished = 1;
                       break;
        case ENERGY  :
        case VISCOSITY:
        case S_EXPONENT:
        case DISLOCDEN:
        case F_ATTRIB_A:
        case F_ATTRIB_B:
        case F_ATTRIB_C:
        case F_ATTRIB_D:
        case F_ATTRIB_E:
        case F_ATTRIB_F: if (err = ElleReadFlynnRealAttribData(fp,str,key)) 
                           finished = 1;
                       break;
        case F_ATTRIB_I:
        case F_ATTRIB_J:
        case F_ATTRIB_K:
        case EXPAND  :
        case COLOUR  :
        case SPLIT:
        case GRAIN:    if (err = ElleReadFlynnIntAttribData(fp,str,key))
                           finished=1;
                       break;
        case VELOCITY: dump_comments( fp ); /* read VEL_X VEL_Y */
                       if (err = ElleReadVelocityData(fp,str)) 
                           finished = 1;
                       break;
        case CONC_A   :
        case ATTRIB_A:
        case ATTRIB_B:
        case ATTRIB_C:
        case ATTRIB_D:
        case ATTRIB_E:
        case ATTRIB_F:
                       if (err=ElleReadNodeRealAttribData(fp,str,key))
                           finished = 1;
                       break;
        case N_MCONC_A:
                       if (err=ElleReadNodeMarginAttribData(fp,str,key))
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
        case STRAIN: /*
                        * read the order of strain vals
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
        case CAXIS   : if (err = ElleReadCAxisData(fp,str)) 
                           finished = 1;
                       /*ElleCheckFlynnDefaults(CAXIS);*/
                       break;
        case EULER_3 : if (err = ElleReadEuler3Data(fp,str)) 
                           finished = 1;
                       break;
        case GBE_LUT   : if (err = ElleReadGBEnergyLUT(fp,str)) 
                           finished = 1;
                       break;
        case MINERAL  : if (err = ElleReadMineralData(fp,str)) 
                           finished = 1;
                       /*ElleCheckFlynnDefaults(MINERAL);*/
                       break;
        case FLYNN_STRAIN: /*
                        * read the order of strain vals
                        */
                      i=0;
                      if (fgets(str,80,fp)!=NULL && !feof(fp)) {
                          start = str;
                          ptr=strtok(start," \t\r\n");
                          while (ptr && 
                                  i<NUM_FLYNN_STRAIN_VALS && !finished) {
                              validate(ptr,&keys[i],FlynnStrainKeys);
                              if (keys[i]<0) finished = 1;
                              else {
                                  start = 0;
                                  ptr=strtok(start," \t\r\n");
                                  i++;
                              }
                           }
                           if (i<1) finished = 1;
                       }
                       if (finished) err = READ_ERR;
                       else if (err = ElleReadFlynnStrainData(fp,str,keys,i)) 
                           finished = 1;
                       break;
        case FLYNN_STRESS  : /*
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
                       else if (err = ElleReadFlynnStressData(fp,str,keys,i)) 
                           finished = 1;
                       break;
        case U_CONC_A  :
        case U_ATTRIB_A:
        case U_ATTRIB_B:
        case U_ATTRIB_C:
        case U_ATTRIB_D:
        case U_ATTRIB_E:
        case U_ATTRIB_F:
        case U_ENERGY:
        case U_S_EXPONENT:
        case U_PHASE:
        case U_FRACTURES:
        case U_TEMPERATURE:
        case U_DIF_STRESS:
        case U_MEAN_STRESS:
        case U_DENSITY:
        case U_YOUNGSMODULUS:
        case U_DISLOCDEN:
        case U_VISCOSITY:
                       if (key==U_CONC_A) key=CONC_A;
                       if (err=ElleReadUnodeRealAttribData(fp,str,&key,1))
                           finished = 1;
                       break;
        case U_FINITE_STRAIN: /*
                        * read the order of finite strain vals
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
        case U_EULER_3: /*
                        * read the 3 euler angle values
                        */
                       keys[0] = E3_ALPHA;
                       keys[1] = E3_BETA;
                       keys[2] = E3_GAMMA;
                       if (err =
                               ElleReadUnodeRealAttribData(fp,str,keys,3)) 
                           finished = 1;
                       break;
        case U_CAXIS: /*
                        * read the orientation and plunge
                        */
                       keys[0] = CAXIS;
                       if (err =
                               ElleReadUnodeRealAttribData(fp,str,keys,1)) 
                           finished = 1;
                       break;
        case S_PERMEABILITY:
        case S_FLUID_FLUX:
                       if (err = ElleReadSegmentAttribData(fp,str,key))
                           finished = 1;
                       break;
#if XY
#endif
        case COMMENT : dump_comments(fp);
                       num = fscanf(fp,"%[^A-Za-z#]",str);
                       num = fscanf(fp,"%s",str);
                       break;
        default:       err = KEY_ERR;
                       finished = 1;
                       break;
        }
    }
    fclose(fp);
#ifndef NO_DSP
    if (ElleDisplay()) ElleUpdateSettings();
#endif
    ElleSetCount(0);
    return(err);
}

int ElleWriteData( const char *filename )
{
    char header[HEADER_MAX],*end;
    std::string creation;
    int i,err=0;
    FILE *fp;

    if ((fp=fopen(filename,"w"))==NULL) return(OPEN_ERR);
    for (i=0;i<HEADER_MAX;i++) header[i]='\0';
                                                                                
    /* write version */
    creation = ElleGetCreationString();
    strncpy(header, creation.c_str(),HEADER_MAX-1);
    //strncpy(header, ElleGetCreationCString(),HEADER_MAX-1);
    if ((end=strrchr(header,'\n'))!=NULL) {
        end++;
        *end = '\0';
    } 
    if (fprintf(fp,"%s\n",header)<0) return(WRITE_ERR);
    if (!(err=ElleWriteOptionData(fp)))
        if (!(err=ElleWriteRegionData(fp)))
            err=ElleWriteNodeData(fp);
    fclose(fp);
    if (!err) err=ElleWriteUnodeData(filename);
    if (!err) err=ElleWriteSegmentData(filename);
    return(err);
}

int ElleCheckFiles()
{
    int count, interval=0,st_interval=0;

    count = ElleCount();
    interval = EllesaveInterval();
    if (count>0 && interval>0)
        if (count>0 && count%interval==0)
            ElleAutoWriteFile(count);
    st_interval = ES_statsInterval();
    if (st_interval>0) 
        if (count==0 || count%st_interval==0)
            ES_WriteStatistics(count);
}

int ElleAutoWriteFile(int cnt)
{
    char fname[FILENAME_MAX];
    int err=0;

    sprintf(fname,"%s%03d%s",ElleSaveFileRoot(),cnt,ELLE_EXT);
    err = ElleWriteData((const char *)fname);
    return(err);
}

void ElleAttributeNotInFile(char *fname,int id)
{
    char msg[FILENAME_MAX]="";
    int i;

    strncpy(msg,fname,FILENAME_MAX-1);
    strncat(msg," ",FILENAME_MAX-1);
    i = strlen(msg);
    id_match(FileKeys,id,&(msg[i]));
    OnError(msg,INVALIDF_ERR);
}

#if XY
int ElleReadDisplayOptions(char *filename)
{
    unsigned char finished=0, initial=1;
    char input_str[256];

    input_str[0] = '\0';
    if ((fp=fopen(filename,"r"))==NULL) return(OPEN_ERR);
    if ((num = fscanf(fp,"%s", input_str))!=1) return(READ_ERR);
    else {
        err=ReadOptions(fp,input_str,initial);
    }
    return( err );
}

#endif
