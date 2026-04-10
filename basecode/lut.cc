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
#include <string>
#include "lut.h"
#include "file.h"
#include "general.h"
#include "error.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

EnergyLUT E_lut;

int ElleReadGBEnergyLUT(FILE *fp, char str[])
{
	char * start=0, *ptr=0;
    int err=0,num,max,i;
    int size=91;
    double val;

    if (fgets(str,80,fp)!=NULL && !feof(fp)) {
		start=str;
        if (ptr=strtok(start,"\t\r\n")) {
		size = atoi(str);
        if (size>LUT_MAX)
			OnError("ElleReadGBEnergyLUT",LIMIT_ERR);
    }
    }
    ElleInitEnergyLUT(size);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            val = atof(str);
            i=0;
            max = ElleEnergyLUTSize();
            ElleSetEnergyLUT(i,val);
            i++;
            while (i<max) {
                if ((num = fscanf(fp,"%lf", &val))!=1) return(READ_ERR);
                ElleSetEnergyLUT(i,val);
                i++;
            }
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

void ElleInitEnergyLUT(int size)
{
    int i;

    E_lut.size = size;
    E_lut.data = 0;
    if (size>0) {
        if ((E_lut.data=(double *)malloc(size*sizeof(double)))==0)
            OnError("ElleInitEnergyLUT",MALLOC_ERR);
        for (i=0;i<E_lut.size;i++) E_lut.data[i] = 1.0;
    }
}

int ElleEnergyLUTSize()
{
    return(E_lut.size);
}

void ElleSetEnergyLUT(int index,double val)
{
    if (index<0 || index>E_lut.size)
        OnError("ElleSetEnergyLUT",RANGE_ERR);
    if (E_lut.data!=0) E_lut.data[index] = val;
}

double ElleEnergyLUTValue(int index)
{
    if (E_lut.data!=0) {
        if (index<0 || index>E_lut.size)
            OnError("ElleEnergyLUTValue",RANGE_ERR);
        return(E_lut.data[index]);
    }
    return(1.0);
}

int ElleWriteGBEnergyLUT(FILE *fp)
{
    char label[20];
    int err=0, i, j, max;

    if (!id_match(FileKeys,GBE_LUT,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);

    max = ElleEnergyLUTSize();
    for (i=0,j=0;i<max;i++)  {
        if (fprintf(fp,"%lf ",ElleEnergyLUTValue(i))<0)
            return(WRITE_ERR);
        j++;
        if (j==5) {
            fprintf(fp,"\n");
            j=0;
        }
    }
    fprintf(fp,"\n");
    return(err);
}

void ElleRemoveEnergyLUT()
{
    E_lut.size = 0;
    if (E_lut.data) free(E_lut.data);
}

int LoadZIPGBEnergyLUT( gzFile in, char str[] )
{
  int err = 0, num, max, i;
  double val;

  ElleInitEnergyLUT( 91 );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      val = atof( str );
      i = 0;
      max = ElleEnergyLUTSize();
      ElleSetEnergyLUT( i, val );
      i++;
      while ( i < max )
      {
        val = atof( ( gzReadSingleString( in ) ).c_str() );
        ElleSetEnergyLUT( i, val );
        i++;
      }
      // fscanf(fp,"\n");
    }
  }
  return ( err );
}
int SaveZIPGBEnergyLUT( gzFile in )
{
  char label[20];
  int err = 0, i, j, max;

  if ( !id_match( FileKeys, GBE_LUT, label ) ) return ( KEY_ERR );
  gzprintf( in, "%s\n", label );

  max = ElleEnergyLUTSize();
  for ( i = 0, j = 0; i < max; i++ )
  {
    if ( gzprintf( in, "%lf ", ElleEnergyLUTValue( i ) ) < 0 )
      return ( WRITE_ERR );
    j++;
    if ( j == 5 )
    {
      gzprintf( in, "\n" );
      j = 0;
    }
  }
  gzprintf( in, "\n" );
  return ( err );
}

