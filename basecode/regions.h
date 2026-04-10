 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef E_regions_h
#define E_regions_h
#include <zlib.h>
#include <string.h>

int ElleZIPAgeData(gzFile *fp, int *keys, int count);
int ElleZIPFlynnRealAttribData(gzFile *fp, int id);
int ElleZIPFlynnIntAttribData(gzFile *fp, int id);
int ElleZIPMineralData(gzFile *fp);
int ElleZIPCAxisData(gzFile *fp);
int ElleZIPEuler3Data(gzFile *fp);
int ElleZIPFlynnStrainData(gzFile *fp,int *keys,int count);
int ElleZIPRegionData(gzFile *fp);
int ElleZIPRegionAttributeData(gzFile *fp);
int SaveZIPCAxisData( gzFile out );
int SaveZIPFlynnData( gzFile out );
 int SaveZIPColourData( gzFile out );
 int SaveZIPDislocDenData( gzFile out );
 int SaveZIPEnergyData( gzFile out );
 int SaveZIPEuler3Data( gzFile out );
 int SaveZIPExpandData( gzFile out );
 int SaveZIPFlynnIntAttribData( gzFile out, int key );
 int SaveZIPFlynnRealAttribData( gzFile out, int id );
 int SaveZIPFlynnStrainData( gzFile out, int * keys, int count );
 int SaveZIPGrainData( gzFile out );
 int SaveZIPMineralData( gzFile out );
 int SaveZIPRegionAttributeData( gzFile out );
 int SaveZIPRegionData( gzFile out );
 int SaveZIPAgeData( gzFile out, int * keys, int count );
 int SaveZIPViscosityData( gzFile out );
 int LoadZIPCAxisData(gzFile in, char str[]);
 int LoadZIPAgeData(gzFile in, char str[], int *keys, int count);
 int LoadZIPFlynnRealAttribData(gzFile in, char str[], int id);
 int LoadZIPFlynnIntAttribData(gzFile in, char str[], int id);
 int LoadZIPEuler3Data(gzFile in, char str[]);
 int LoadZIPMineralData(gzFile in, char str[]);
 int LoadZIPFlynnStrainData(gzFile in,char str[],int *keys,int count);

#ifdef __cplusplus
extern "C" {
#endif
int ElleWriteRegionData(FILE *fp);
int ElleWriteRegionAttributeData(FILE *fp);
int ElleReadCAxisData(FILE *fp, char str[]);
int ElleReadAgeData(FILE *fp, char str[], int *keys, int count);
int ElleReadFlynnRealAttribData(FILE *fp, char str[], int id);
int ElleReadFlynnIntAttribData(FILE *fp, char str[], int id);
int ElleReadEuler3Data(FILE *fp, char str[]);
int ElleReadMineralData(FILE *fp, char str[]);
int ElleReadFlynnStrainData(FILE *fp,char str[],int *keys,int count);
int ElleReadFlynnStressData(FILE *fp,char str[],int *keys,int count);
int ElleWriteAgeData(FILE *fp, int *keys, int count);
int ElleWriteFlynnRealAttribData(FILE *fp, int id);
int ElleWriteFlynnIntAttribData(FILE *fp, int id);
int ElleWriteMineralData(FILE *fp);
int ElleWriteCAxisData(FILE *fp);
int ElleWriteEuler3Data(FILE *fp);
int ElleWriteFlynnStrainData(FILE *fp,int *keys,int count);
int ElleWriteFlynnStressData(FILE *fp,int *keys,int count);
void ElleSetRegionFirstNodes();
/*********************************
 * functions below this line are being phased out
 *********************************/
int ElleReadSplitData(FILE *fp, char str[]);
int ElleReadGrainData(FILE *fp, char str[]);
int ElleReadEnergyData(FILE *fp, char str[]);
int ElleReadViscosityData(FILE *fp, char str[]);
int ElleReadDislocDenData(FILE *fp, char str[]);
int ElleReadExpandData(FILE *fp, char str[]);
int ElleReadColourData(FILE *fp, char str[]);
int ElleWriteExpandData(FILE *fp);
int ElleWriteColourData(FILE *fp);
int ElleWriteEnergyData(FILE *fp);
int ElleWriteViscosityData(FILE *fp);
int ElleWriteDislocDenData(FILE *fp);
int ElleWriteSplitData(FILE *fp);
int ElleWriteGrainData(FILE *fp);
#ifdef __cplusplus
}
#endif
#endif
