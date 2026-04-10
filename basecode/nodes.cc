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
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include "nodes.h"
#include "unodes.h"
#include "bflynns.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "convert.h"
#include "general.h"
#include "interface.h"
#include "crossings.h"
#include "check.h"
#include "polygon.h"
#include "polyutils.h"
#include "log.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

using std::vector;

void ElleRotateSegment(int node1,int node2,double angle);
void ElleRotateTripleJ(int node1,Coords *xy_mp,int nb1, int nb2,
                       Coords *xynew);
void RecalculateTripleSwitchCONC(int node1,int node2,int node3,
                                    int node4,int node5,int node6);
void ElleTripleJMoveDir(int node,Coords *xy_mp,int nb1, int nb2,
                       Coords *dir);
void MoveTowardsPoint(Coords *start, Coords *dir, Coords *newxy, double len);
double ElleSwitchLength();
double ElleSmallSwitchLength();

int SaveZIPNodeLocation( gzFile out )
{
  /* * writes location data in a node file * location label then position data on a separate
  * line for each node with the format: %d %f %f * (node id number, x, y) */
  char label[20];
  int i, max;
  Coords pos;

  max = ElleMaxNodes();
  if ( !id_match( FileKeys, LOCATION, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );
  for ( i = 0; i < max; i++ )
  {
    if ( ElleNodeIsActive( i ) )
    {
      ElleNodePosition( i, & pos );
      gzprintf( out, "%d %.10lf %.10lf\n", i, pos.x, pos.y );
    }
  }
  return ( 0 );
}

int LoadZIPLocationData( gzFile in, char str[] )
{
  /* * routine to be called after the location keyword has * been read in an Elle file
  * Reads lines with the format %d %lf %lf (node id, x, y) */
  int err = 0, num, nn;
  Coords current;
  double tmpx, tmpy;

  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( nn = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR );
      ElleSetNodeActive( nn );
      /* * Alphas rounding to %.6f so read as double */
      current.x = atof( ( gzReadSingleString( in ) ).c_str() );
      current.y = atof( ( gzReadSingleString( in ) ).c_str() );
      /* ElleSetPosition(nn,&current); */
      /* ElleSetPrevPosition(nn,&current); */
      ElleCopyToPosition( nn, & current );
      ElleCopyToPrevPosition( nn, & current );
    }
  }
  return ( err );
}
int LoadZIPStressData( gzFile in, char str[], int * keys, int count )
{
  /* * Read stress data in a node file. * Stress labels then stress data on a separate
  * line for each node with the format: %d %f %f %f %f %f * (node id number, tau xx, tau yy, tau zz, tau xy, pressure
  * or the order read into keys) */
  int i;
  int nn, num, err = 0;
  int index[NUM_STRESS_VALS];
  double val[NUM_STRESS_VALS];

  for ( i = 0; i < count; i++ )
  {
    ElleInitNodeAttribute( keys[i] );
    /* change this to index[i] = ElleFindNodeAttribIndex(keys[i]) */
    /** no defaults set for stresses ******** */
    switch ( keys[i] )
    {
      case TAU_XX:
        if ( ( index[i] = ElleTauXXIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      case TAU_YY:
        if ( ( index[i] = ElleTauYYIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      case TAU_ZZ:
        if ( ( index[i] = ElleTauZZIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      case TAU_XY:
        if ( ( index[i] = ElleTauXYIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      case TAU_1:
        if ( ( index[i] = ElleTau1Index() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      case PRESSURE:
        if ( ( index[i] = EllePressureIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      default:
        return ( ID_ERR );
      break;
    }
  }
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );

    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    /* * SET_ALL not valid for stress vals */
    else
    {
      if ( ( nn = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR );
      ElleSetNodeActive( nn );
      for ( i = 0; i < count; i++ )
      {
        val[i] = atof( ( gzReadSingleString( in ) ).c_str() );
        ElleSetNodeAttribute( nn, val[i], keys[i] );
      }
    }
  }
  return ( 0 );
}
int LoadZIPVelocityData( gzFile in, char str[] )
{
  /* * routine to be called after the velocity keyword has * been read in an Elle file
  * Reads lines with the format %d %f %f(node id, velocity in * x-direction, velocity in y-direction) */
  int j, err = 0, num, nn;
  int max, indxx, indxy;
  double valx, valy;

  /* this should be written like stress - pass index of attrib ids */
  ElleInitNodeAttribute( VEL_X );
  ElleInitNodeAttribute( VEL_Y );
  if ( ( indxx = ElleVelocityXIndex() ) == NO_INDX )
    return ( MAXATTRIB_ERR );
  if ( ( indxy = ElleVelocityYIndex() ) == NO_INDX )
    return ( MAXATTRIB_ERR );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );

    else if ( !strcmp( str, SET_ALL ) )
    {
      valx = atof( ( gzReadSingleString( in ) ).c_str() );
      valy = atof( ( gzReadSingleString( in ) ).c_str() );
      max = ElleMaxNodes();
      for ( j = 0; j < max; j++ )
        if ( ElleNodeIsActive( j ) )
        {
          ElleSetNodeAttribute( j, valx, VEL_X );
          ElleSetNodeAttribute( j, valy, VEL_Y );
        }
      ElleSetNodeDefaultVelocityX( valx );
      ElleSetNodeDefaultVelocityY( valy );
    }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( nn = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR );
      ElleSetNodeActive( nn );
      valx = atof( ( gzReadSingleString( in ) ).c_str() );
      valy = atof( ( gzReadSingleString( in ) ).c_str() );
      ElleSetNodeAttribute( nn, valx, VEL_X );
      ElleSetNodeAttribute( nn, valy, VEL_Y );
    }
  }
  return ( err );
}
int SaveZIPNodeData( gzFile out )
{
  char label[20];
  int err = 0;
  int i, j, * active=0, maxa, * keys=0;

  err = SaveZIPNodeLocation( out );
  ElleNodeAttributeList( & active, & maxa );
  for ( i = 0; i < maxa && !err; i++ )
  {
    switch ( active[i] )
    {
      case VEL_X:
        err = SaveZIPNodeVelocity( out );
        i++;
      break;
      case TAU_XX:
      case TAU_YY:
      case TAU_ZZ:
      case TAU_XY:
      case TAU_1:
      case PRESSURE:
        if ( ( keys = ( int * ) malloc( maxa * sizeof( int ) ) ) == 0 )
          OnError( "ElleWriteNodeData", MALLOC_ERR );
        for ( j = 0; j < maxa; j++ ) keys[j] = NO_INDX;
        j = 0;
        while ( id_match( StressKeys, active[i], label ) )
        {
          keys[j] = active[i];
          j++; i++;
        }
        i--;
        /* for loop will increment again */
        err = SaveZIPStressData( out, keys, j );
        if (keys) free( keys );
      break;
      case N_MCONC_A:
      case N_MCONC_A_1:
        err = SaveZIPNodeMarginAttribData( out, N_MCONC_A );
      break;
      case CONC_A:
        err = SaveZIPNodeConcA( out );
      break;
      case INCR_S:
      case BULK_S:
        if ( ( keys = ( int * ) malloc( maxa * sizeof( int ) ) ) == 0 )
          OnError( "ElleWriteNodeData", MALLOC_ERR );
        for ( j = 0; j < maxa; j++ ) keys[j] = NO_INDX;
        j = 0;
        while ( id_match( StrainKeys, active[i], label ) )
        {
          keys[j] = active[i];
          j++; i++;
        }
        i--;
        /* for loop will increment again */
        err = SaveZIPStrainData( out, keys, j );
        if (keys) free( keys );
      break;
      case N_ATTRIB_A:
      case N_ATTRIB_B:
      case N_ATTRIB_C:
      case N_ATTRIB_D:
      case N_ATTRIB_E:
      case N_ATTRIB_F:
        err = SaveZIPNodeRealAttribute( out, active[i] );
      break;
      case N_MATTRIB_A:
      case N_MATTRIB_A_1:
        err = SaveZIPNodeMarginAttribData( out, N_MATTRIB_A );
      break;
        // these have already been written, so step over
      case N_MCONC_A_2:
      case N_MCONC_A_3:
      case N_MATTRIB_A_2:
      case N_MATTRIB_A_3:
      case N_MATTRIB_B_2:
      case N_MATTRIB_B_3:
      case N_MATTRIB_C_2:
      case N_MATTRIB_C_3:
      break;
    }
  }
  if ( active != 0 ) free( active );
  return ( err );
}
int LoadZIPNodeMarginAttribData( gzFile in, char str[], int attr_id )
{
  /* * routine to be called after a node margin attrib keyword has * been read in an Elle file
  * Reads lines with the format %d %lf %lf %lf(node id, val for *        2 or 3 flynns) */
  char * ptr, * start;
  int i, j, err = 0, num, nn;
  int max, loc_id;
  int id[3], rgn[3], nb_id[3];
  double val, val_3[3] =
  {
    0, 0, 0
  };

  switch ( attr_id )
  {
    case N_MCONC_A:
      loc_id = CONC_A;
      id[0] = N_MCONC_A_1;
      id[1] = N_MCONC_A_2;
      id[2] = N_MCONC_A_3;
    break;
    case N_MATTRIB_A:
      loc_id = ATTRIB_A;
      id[0] = N_MATTRIB_A_1;
      id[1] = N_MATTRIB_A_2;
      id[2] = N_MATTRIB_A_3;
    break;
    case N_MATTRIB_B:
      loc_id = ATTRIB_B;
      id[0] = N_MATTRIB_B_1;
      id[1] = N_MATTRIB_B_2;
      id[2] = N_MATTRIB_B_3;
    break;
    case N_MATTRIB_C:
      loc_id = ATTRIB_C;
      id[0] = N_MATTRIB_C_1;
      id[1] = N_MATTRIB_C_2;
      id[2] = N_MATTRIB_C_3;
    break;
    default:
      return ( ATTRIBID_ERR );
  }
  ElleInitNodeMarginAttribute( loc_id );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
    else if ( !strcmp( str, SET_ALL ) )
    {
      val_3[0] = atof( ( gzReadSingleString( in ) ).c_str() );
      val_3[1] = atof( ( gzReadSingleString( in ) ).c_str() );
      val_3[2] = atof( ( gzReadSingleString( in ) ).c_str() );
      max = ElleMaxNodes();
      for ( j = 0; j < max; j++ )
        if ( ElleNodeIsActive( j ) )
        {
          for ( i = 0; i < 3; i++ )
            ElleSetNodeAttribute( j, val_3[i], id[i] );
        }
    }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( nn = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR );
      if ( gzgets( in, str, 256 ) == NULL ) return ( READ_ERR );
      start = str;
      ptr = strtok( start, " \t\r\n" );
      while ( ptr )
      {
        rgn[0] = atoi( ptr );
        start = 0;
        ptr = strtok( start, " \t\r\n" );
        if ( ptr )
        {
          val_3[0] = atof( ptr );
          start = 0;
          ElleSetNodeMarginAttribute( nn, val_3[0], loc_id, rgn[0] );
        }
        ptr = strtok( start, " \t\r\n" );
      }
    }
  }
  return ( err );
}
int SaveZIPStressData( gzFile out, int * keys, int count )
{
  /* * Writes stress data in an elle file. * Stress data on a separate line for each node * with the format: %d %e %e %e %e %e
  * (node id number, count values in the order defined by keys) */
  char label[20];
  int i, j, max;
  int index[NUM_STRESS_VALS];

  if ( !id_match( FileKeys, STRESS, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s ", label );
  for ( i = 0; i < count; i++ )
  {
    if ( !id_match( StressKeys, keys[i], label ) ) return ( KEY_ERR );
    gzprintf( out, "%s ", label );
  }
  gzprintf( out, "\n" );
  max = ElleMaxNodes();
  for ( i = 0; i < count; i++ )
  {
    switch ( keys[i] )
    {
      case TAU_XX:
        if ( ( index[i] = ElleTauXXIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      case TAU_YY:
        if ( ( index[i] = ElleTauYYIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      case TAU_ZZ:
        if ( ( index[i] = ElleTauZZIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      case TAU_XY:
        if ( ( index[i] = ElleTauXYIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      case TAU_1:
        if ( ( index[i] = ElleTau1Index() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      case PRESSURE:
        if ( ( index[i] = EllePressureIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      default:
        return ( ID_ERR );
      break;
    }
  }
  for ( i = 0; i < max; i++ )
  {
    if ( ElleNodeIsActive( i ) )
    {
      gzprintf( out, "%d", i );
      for ( j = 0; j < count; j++ )
      {
        gzprintf( out, " %.8e", ElleNodeAttribute( i, keys[j] ) );
      }
      gzprintf( out, "\n" );
    }
  }
  return ( 0 );
}
int SaveZIPNodeMarginAttribData( gzFile out, int attr_id )
{
  /* * writes margin real attribute data in a node file * line for each node with the format: %d %lf %lf %lf
  * (node id number, val for 3 flynns) */
  char label[20];
  int j, i, max, loc_id;
  int id[3], rgn[3], prnt = 0;
  double val_3[3], dflt_3[3] =
  {
    0, 0, 0
  };

  max = ElleMaxNodes();

  switch ( attr_id )
  {
    case N_MCONC_A:
      loc_id = CONC_A;
      id[0] = N_MCONC_A_1;
      id[1] = N_MCONC_A_2;
      id[2] = N_MCONC_A_3;
    break;
    case N_MATTRIB_A:
      loc_id = ATTRIB_A;
      id[0] = N_MATTRIB_A_1;
      id[1] = N_MATTRIB_A_2;
      id[2] = N_MATTRIB_A_3;
    break;
    case N_MATTRIB_B:
      loc_id = ATTRIB_B;
      id[0] = N_MATTRIB_B_1;
      id[1] = N_MATTRIB_B_2;
      id[2] = N_MATTRIB_B_3;
    break;
    case N_MATTRIB_C:
      loc_id = ATTRIB_C;
      id[0] = N_MATTRIB_C_1;
      id[1] = N_MATTRIB_C_2;
      id[2] = N_MATTRIB_C_3;
    break;
    default:
      return ( ATTRIBID_ERR );
  }

  if ( !id_match( FileKeys, attr_id, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );
  /* this need fixing - get dflts from dflt array */
  gzprintf( out, "%s %.8e %.8e %.8e\n", SET_ALL, dflt_3[0], dflt_3[1], dflt_3[2] );
  for ( j = 0; j < max; j++ )
  {
    if ( ElleNodeIsActive( j ) )
    {
      ElleRegions( j, rgn );
      for ( i = 0, prnt = 0; i < 3; i++ )
      {
        val_3[i] = dflt_3[i];
        if ( rgn[i] != NO_NB )
          val_3[i] = ElleNodeMarginAttribute( j, loc_id, rgn[i] );
        if ( val_3[i] != dflt_3[i] ) prnt = 1;
      }
      if ( prnt )
      {
        gzprintf( out, "%d", j );
        for ( i = 0; i < 3; i++ )
        {
          if ( rgn[i] != NO_NB && val_3[i] != dflt_3[i] )
            gzprintf( out, " %d %.8e", rgn[i], val_3[i] );
        }
        gzprintf( out, "\n" );
      }
    }
  }
  return ( 0 );
}
int SaveZIPStrainData( gzFile out, int * keys, int count )
{
  /* * Writes strain data in an elle file. * Strain data on a separate line for each node * with the format: %d %f %f
  * (node id number, count values in the order defined by keys) */
  char label[20];
  int i, j, max;
  int index[NUM_STRAIN_VALS];

  if ( !id_match( FileKeys, STRAIN, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s ", label );
  for ( i = 0; i < count; i++ )
  {
    if ( !id_match( StrainKeys, keys[i], label ) ) return ( KEY_ERR );
    gzprintf( out, "%s ", label );
  }
  gzprintf( out, "\n" );
  max = ElleMaxNodes();
  for ( i = 0; i < count; i++ )
  {
    switch ( keys[i] )
    {
      case INCR_S:
        if ( ( index[i] = ElleStrainIncrIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      case BULK_S:
        if ( ( index[i] = ElleBulkStrainIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      default:
        return ( ID_ERR );
      break;
    }
  }
  for ( i = 0; i < max; i++ )
  {
    if ( ElleNodeIsActive( i ) )
    {
      gzprintf( out, "%d", i );
      for ( j = 0; j < count; j++ )
      {
        gzprintf( out, " %.8e", ElleNodeAttribute( i, keys[j] ) );
      }
      gzprintf( out, "\n" );
    }
  }
  return ( 0 );
}
int SaveZIPNodeRealAttribute( gzFile out, int id )
{
  /* * writes real attribute data in a node file * line for each node with the format: %d %f * (node id number, val) */
  char label[20];
  int i, max;
  double val, dflt;
  double eps = 1e-15;

  max = ElleMaxNodes();

  if ( !id_match( FileKeys, id, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );

  dflt = 0;
  /* this need fixing - store in dflt array */
  gzprintf( out, "%s %.8e\n", SET_ALL, dflt );
  for ( i = 0; i < max; i++ )
  {
    if ( ElleNodeIsActive( i ) )
    {
      val = ElleNodeAttribute( i, id );
      if ( val != dflt && ( id != CONC_A || val > dflt + eps || val < dflt - eps ) )
        gzprintf( out, "%d %.8e\n", i, val );
    }
  }
  return ( 0 );
}
int SaveZIPNodeConcA( gzFile out )
{
  /* * writes ConcA data in a node file * line for each node with the format: %d %f * (node id number, conc) */
  char label[20];
  int i, max;
  int indx;
  double val, dflt;
  double eps = 1e-15;

  max = ElleMaxNodes();
  if ( ( indx = ElleConcAIndex() ) == NO_INDX )
    return ( MAXATTRIB_ERR );

  if ( !id_match( FileKeys, CONC_A, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );

  dflt = ElleNodeDefaultConcA();
  gzprintf( out, "%s %.8e\n", SET_ALL, dflt );
  for ( i = 0; i < max; i++ )
  {
    if ( ElleNodeIsActive( i ) )
    {
      val = ElleNodeAttribute( i, CONC_A );
      if ( val > dflt + eps || val < dflt - eps )
        gzprintf( out, "%d %.8e\n", i, val );
    }
  }
  return ( 0 );
}
int SaveZIPNodeVelocity( gzFile out )
{
  /* * writes velocity data in a node file * velocity label then velocity data on a separate
  * line for each node with the format: %d %f %f * (node id number, x direction velocity, y direction velocity) */
  char label[20];
  int i, max;
  int indxx, indxy;
  double valx, valy, dfltx, dflty;

  max = ElleMaxNodes();
  if ( ( indxx = ElleVelocityXIndex() ) == NO_INDX )
    return ( MAXATTRIB_ERR );
  if ( ( indxy = ElleVelocityYIndex() ) == NO_INDX )
    return ( MAXATTRIB_ERR );

  if ( !id_match( FileKeys, VELOCITY, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s ", label );
  if ( !id_match( VelocityKeys, VEL_X, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s ", label );
  if ( !id_match( VelocityKeys, VEL_Y, label ) ) return ( KEY_ERR );
  gzprintf( out, "%s\n", label );

  dfltx = ElleNodeDefaultVelocityX();
  dflty = ElleNodeDefaultVelocityY();
  gzprintf( out, "%s %.8e %.8e\n", SET_ALL, dfltx, dflty );
  for ( i = 0; i < max; i++ )
  {
    if ( ElleNodeIsActive( i ) )
    {
      valx = ElleNodeAttribute( i, VEL_X );
      valy = ElleNodeAttribute( i, VEL_Y );
      if ( valx != dfltx || valy != dflty )
        gzprintf( out, "%d %.8e %.8e\n", i, valx, valy );
    }
  }
  return ( 0 );
}

int LoadZIPNodeRealAttribData( gzFile in, char str[], int id )
{
  /* * routine to be called after real attribute keyword has * been read in an Elle file
  * Reads lines with the format %d %f (node id, conc) */
  int j, err = 0, num, nn;
  int max, indx;
  double val;

  ElleInitNodeAttribute( id );
  if ( ( indx = ElleFindNodeAttribIndex( id ) ) == NO_INDX )
    return ( MAXATTRIB_ERR );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );

    else if ( !strcmp( str, SET_ALL ) )
    {
      val = atof( ( gzReadSingleString( in ) ).c_str() );
      max = ElleMaxNodes();
      for ( j = 0; j < max; j++ )
        if ( ElleNodeIsActive( j ) )
        {
          ElleSetNodeAttribute( j, val, id );
        }
      /* need to set up an array of defaults, rather than hardwired in code. Assumes the default for these is 0 */
    }
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( nn = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR );
      /* make this a warning message */
      if ( ElleNodeIsActive( nn ) )
      {
        val = atof( ( gzReadSingleString( in ) ).c_str() );
        ElleSetNodeAttribute( nn, val, id );
      }
      else {
        sprintf( logbuf, "Ignoring file attribute %d for inactive node %d\n", id, nn );
        Log( 2, logbuf );
      }
    }
  }
  return ( err );
}
int LoadZIPStrainData( gzFile in, char str[], int * keys, int count )
{
  /* * Read strain data in a node file. * Strain labels then strain data on a separate
  * line for each node with the format: %d %f %f * (node id number, psr1, bulk psr1 or the order read into keys) */
  int i;
  int nn, num, err = 0;
  int index[NUM_STRAIN_VALS];
  double val[NUM_STRAIN_VALS];

  for ( i = 0; i < count; i++ )
  {
    ElleInitNodeAttribute( keys[i] );
    /* change this to index[i] = ElleFindNodeAttribIndex(keys[i]) */
    /** no defaults set for stresses ******** */
    switch ( keys[i] )
    {
      case INCR_S:
        if ( ( index[i] = ElleStrainIncrIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      case BULK_S:
        if ( ( index[i] = ElleBulkStrainIndex() ) == NO_INDX )
          return ( MAXATTRIB_ERR );
      break;
      default:
        return ( ID_ERR );
      break;
    }
  }
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );


    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    /* * SET_ALL not valid for strain-rate vals */
    else
    {
      if ( ( nn = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR );
      ElleSetNodeActive( nn );
      for ( i = 0; i < count; i++ )
      {
        val[i] = atof( ( gzReadSingleString( in ) ).c_str() );
        ElleSetNodeAttribute( nn, val[i], keys[i] );
      }
    }
  }
  return ( 0 );
}

int ElleInsertDoubleJ(int node1,int node2,int *nn,float frac)
{
    /*
     * node1,node2 are the nodes defining the segment
     * nn is the new node
     * frac determines the position of nn on the segment (eg frac=0.5
     * inserts nn halfway between node1 and node2)
     */
    Coords xy1, xy2, newxy;

    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node2,&xy2,&xy1);
    newxy.x = xy1.x - (xy1.x-xy2.x)*frac;
    newxy.y = xy1.y - (xy1.y-xy2.y)*frac;
    return(ElleInsertDoubleJAtPoint(node1,node2,nn,&newxy));
}

int ElleInsertDoubleJAtPoint(int node1,int node2,int *nn,Coords *pos)
{
    /*
     * node1,node2 are the nodes defining the segment
     * nn is the new node
     * pos determines the position of nn on the segment 
     */
    char msg[30];
    int err=0;
    int index1, index2;
    ERegion rgn1, rgn2;

    sprintf(msg,"%s %d %d","InsertDoubleJ",node1,node2);
    *nn = ElleFindSpareNode();
    ElleSetPosition(*nn,pos);
    ElleSetPrevPosition(*nn,pos);
    ElleNeighbourRegion(node1,node2,&rgn1);
    ElleNeighbourRegion(node2,node1,&rgn2);
    if ((index1 = ElleFindNbIndex(node1,node2))==NO_NB)
        OnNodeError(node1,msg,NONB_ERR);
    /* check which boundary - may be 2-sided grain */
    if ((index2 = ElleFindBndNbIndex(node2,node1,rgn2))==NO_NB)
                                          OnNodeError(node2,msg,NONB_ERR);
    ElleSetNeighbour(*nn,0,node1,&rgn2);
    ElleSetNeighbour(*nn,1,node2,&rgn1);
    ElleSetNeighbour(node1,index2,*nn,0);
    ElleSetNeighbour(node2,index1,*nn,0);
    ElleIntrpAttributes(*nn);
//    ElleSetNodeChange(1);
    return(err);
}

int ElleInsertDoubleJOnBnd(int node1,int bnd1,int node2,int bnd2,
                           int *nn,float frac)
{
    /*
     * node1,node2 are the nodes defining the segment
     * bnd1 specifies the boundary node1->node2(in case of 2-sided flynn)
     * bnd2 specifies the boundary node2->node1(in case of 2-sided flynn)
     * nn is the new node
     * frac determines the position of nn on the segment (eg frac=0.5
     * inserts nn halfway between node1 and node2)
     */
    char msg[30];
    int err=0;
    int index1, index2;
    Coords xy1, xy2, newxy;

    sprintf(msg,"%s %d %d","InsertDoubleJOnBnd",node1,node2);
    *nn = ElleFindSpareNode();
    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node2,&xy2,&xy1);
    newxy.x = xy1.x - (xy1.x-xy2.x)*frac;
    newxy.y = xy1.y - (xy1.y-xy2.y)*frac;
    ElleSetPosition(*nn,&newxy);
    ElleSetPrevPosition(*nn,&newxy);
    if ((index1 = ElleFindBndNbIndex(node1,node2,bnd2))==NO_NB)
                                          OnNodeError(node1,msg,NONB_ERR);
    if ((index2 = ElleFindBndNbIndex(node2,node1,bnd1))==NO_NB)
                                          OnNodeError(node2,msg,NONB_ERR);
    ElleSetNeighbour(*nn,1,node2,&bnd1);
    ElleSetNeighbour(*nn,0,node1,&bnd2);
    ElleSetNeighbour(node1,index2,*nn,0);
    ElleSetNeighbour(node2,index1,*nn,0);
    ElleIntrpAttributes(*nn);
//    ElleSetNodeChange(1);
    return(err);
}

int ElleDeleteDoubleJ(int id)
{
    /*
     * id is the double node to be deleted
     * after deleting the node, checks for and removes any 2-sided
     * grain or single node
     */
    char msg[30];
    int i;
    int nb1, nb2, id_indx1, id_indx2, nb;
    int nbnodes[3];
    int  angcnt=0, count=0, delete_err=0;
    ERegion rgn1,rgn2;
    double minarea;
    Coords xy, xy1, xy2, tmpxy;
    Intersection isect[MAX_I_CNT];
    Angle_data ang[3];

    sprintf(msg,"%s %d","DeleteDoubleJ",id);
    ElleNeighbourNodes(id,nbnodes);
    i=0;
    while (i<3 && nbnodes[i]==NO_NB) i++;
    nb1 = nbnodes[i];
    ElleNeighbourRegion(id,nb1,&rgn1);
    if ((id_indx1 = ElleFindBndNbIndex(id,nb1,rgn1))==NO_NB)
        OnNodeError(id,msg,NONB_ERR);
    i++;
    while (i<3 && nbnodes[i]==NO_NB) i++;
    nb2 = nbnodes[i];
    if (nb1==nb2) ElleRemoveHangingNode(id,nb1);
    else {
        ElleNeighbourRegion(id,nb2,&rgn2);
        if((id_indx2 = ElleFindBndNbIndex(id,nb2,rgn2))==NO_NB)
            OnNodeError(id,msg,NONB_ERR);
        ElleNodePosition(id,&xy);
        ElleNodePlotXY(nb1,&xy1,&xy);
        ElleNodePlotXY(nb2,&xy2,&xy);
        // deleting the DJ straightens the boundary -
        // same as moving the DJ to the midpt of the neighbours
        tmpxy.x = xy1.x - (xy1.x-xy2.x)*0.5;
        tmpxy.y = xy1.y - (xy1.y-xy2.y)*0.5;
        /*
         * check whether the node movement causes an angle flip
         */
        ElleCheckAngles(id,&tmpxy,ang,&angcnt);
        if (ElleFindIntersection(id,NO_NB,rgn1,&xy,
                                     &tmpxy,&isect[0]))
            count++;
        if (ElleFindIntersection(id,nb1,rgn2,&xy,
                                 &tmpxy,&isect[0]))
            count++;
        if (ElleFindIntersection(id,NO_NB,rgn2,&xy,
                                     &tmpxy,&isect[0]))
            count++;
        if (ElleFindIntersection(id,nb2,rgn1,&xy,
                                 &tmpxy,&isect[0]))
            count++;
        /*
         * if flynn is tiny, deleting will increase its area and it may
         * cause topo problems so merge it instead
         * use area of 3-sided grain with sides of minNodeSep
         */
        /*minarea = ElleminNodeSep()*ElleminNodeSep()*SIN60*0.5;*/
        minarea = ElleminNodeSep()*ElleminNodeSep()*SIN60*0.25;
        if (fabs(ElleRegionArea(rgn1)) < (double)minarea) 
            ElleMergeFlynnsNoCheck(rgn2,rgn1);
        else if (fabs(ElleRegionArea(rgn2)) < (double)minarea)
            ElleMergeFlynnsNoCheck(rgn1,rgn2);
        else if (!angcnt && !count) {
            if (ElleNodeCONCactive())
                ElleRedistributeNodeCONC(id,nb1,nb2);
            ElleSetNeighbour(nb1,id_indx1,nb2,0);
            ElleSetNeighbour(nb2,id_indx2,nb1,0);
            ElleUpdateFirstNodes(id,nb1,rgn1);
            ElleUpdateFirstNodes(id,nb2,rgn2);
            ElleClearNodeEntry(id);
            while (ElleCheckForTwoSidedGrain(nb1,&nb)) {
                if (ElleNodeIsTriple(nb1) && ElleNodeIsTriple(nb))
                    ElleRemoveTripleJLink(nb1,nb);
                else if (ElleNodeIsTriple(nb1)) ElleRemoveHangingNode(nb,nb1);
                else ElleRemoveHangingNode(nb1,nb);
            }
        }
        else delete_err=1;
    }
                
    return(delete_err);
}

int ElleDeleteDoubleJNoCheck(int id)
{
    /*
     * id is the double node to be deleted
     * after deleting the node, checks for and removes any 2-sided
     * grain or single node
     */
    char msg[30];
    int i;
    int nb1, nb2, id_indx1, id_indx2, nb;
    int nbnodes[3];
    ERegion rgn1,rgn2;
    Coords xy1, xy2;

    sprintf(msg,"%s %d","DeleteDoubleJ",id);
    ElleNeighbourNodes(id,nbnodes);
    i=0;
    while (i<3 && nbnodes[i]==NO_NB) i++;
    nb1 = nbnodes[i];
    ElleNeighbourRegion(id,nb1,&rgn1);
    if ((id_indx1 = ElleFindBndNbIndex(id,nb1,rgn1))==NO_NB)
        OnNodeError(id,msg,NONB_ERR);
    i++;
    while (i<3 && nbnodes[i]==NO_NB) i++;
    nb2 = nbnodes[i];
    if (nb1==nb2) ElleRemoveHangingNode(id,nb1);
    else {
        ElleNeighbourRegion(id,nb2,&rgn2);
        if((id_indx2 = ElleFindBndNbIndex(id,nb2,rgn2))==NO_NB)
            OnNodeError(id,msg,NONB_ERR);
        if (ElleNodeCONCactive())
            ElleRedistributeNodeCONC(id,nb1,nb2);
        ElleNodePosition(nb1,&xy1);
        ElleNodePlotXY(nb2,&xy2,&xy1);
        ElleSetNeighbour(nb1,id_indx1,nb2,0);
        ElleSetNeighbour(nb2,id_indx2,nb1,0);
        ElleUpdateFirstNodes(id,nb1,rgn1);
        ElleUpdateFirstNodes(id,nb2,rgn2);
        ElleClearNodeEntry(id);
        while (ElleCheckForTwoSidedGrain(nb1,&nb)) {
            if (ElleNodeIsTriple(nb1) && ElleNodeIsTriple(nb))
                ElleRemoveTripleJLink(nb1,nb);
            else if (ElleNodeIsTriple(nb1)) ElleRemoveHangingNode(nb,nb1);
            else ElleRemoveHangingNode(nb1,nb);
        }
    }
    return(0);
}

void ElleDeleteSingleJ(int node )
{
    /*
     * node is the single node to be deleted - does nothing if
     * node is not single
     * deletes single nodes resulting from the deletion
     */
    int i, nb, nbnodes[3];
    ERegion rgn;

    while (ElleNodeIsSingle(node)) {
        ElleNeighbourNodes(node,nbnodes);
        i=0; while (nbnodes[i]==NO_NB && i<3) i++;
        if (i==3) OnNodeError(node,"ElleDeleteSingleJ",NONB_ERR);
        nb = nbnodes[i];
        if (ElleNodeCONCactive())
            ElleRedistributeNodeCONC(node,nb,NO_NB);
        ElleNeighbourRegion(node,nb,&rgn);
        ElleUpdateFirstNodes(node,nb,rgn);
        ElleNeighbourNodes(nb,nbnodes);
        i=0; while (nbnodes[i]!=node && i<3) i++;
        ElleClearNeighbour(nb,i);
        ElleClearNodeEntry(node);
        node = nb;
    }
}

void ElleRemoveHangingNode(int dblnode,int trplnode)
{
    /*
     *                 __
     *   newfirst--trpl  dbl
     *                 --
     *  particular case of 2-sided grain
     */
    char msg[30];
    int j=0, found = 0, newfirst=0, nbnodes[3];
    ERegion full_id, full_id_rm, tmp[3];

    sprintf(msg,"%s %d %d","RemoveHangingNode",dblnode,trplnode);

    ElleNeighbourNodes(trplnode,nbnodes);
    while (nbnodes[j]==dblnode) j++;
    newfirst = nbnodes[j];

/* fix this when invisible boundaries impemented */

    ElleNeighbourRegion(trplnode,newfirst,&full_id);
    ElleNeighbourNodes(dblnode,nbnodes);
    ElleRegions(dblnode,tmp);
    for (j=0;j<3&&!found;j++) {
        if (nbnodes[j]!=NO_NB) {
            full_id_rm = tmp[j];
            if (!ElleIdMatch(&full_id,&full_id_rm)) found = 1;
        }
    }
    j = ElleFindNbIndex(trplnode,newfirst);
    if (newfirst==NO_NB||j==NO_NB) OnNodeError(trplnode,msg,NONB_ERR);
    if (ElleNodeCONCactive()) {
        ElleRedistributeNodeCONC(dblnode,trplnode,trplnode);
        ElleRedistributeNodeCONC(trplnode,newfirst,NO_NB);
    }
    ElleClearNeighbour(newfirst,j);
    ElleUpdateFirstNodes(dblnode,newfirst,full_id);
    ElleUpdateFirstNodes(trplnode,newfirst,full_id);
    ElleRemoveShrinkingFlynn(full_id_rm);
    ElleClearNodeEntry(dblnode);
    ElleClearNodeEntry(trplnode);
}

void ElleRemoveTripleJLink(int node1,int node2)
{
    /*        |
     *        1
     *      /   \
     * [1] | [0] | [2]
     *      \   /
     *        2
     *        |
     *  called to remove 2-sided grain
     */
    char msg[30];
    ERegion full_id[4],tmp[3];
    int rmindx1=NO_NB,rmindx2=NO_NB,idindx=NO_NB;
    int i, found, nbnodes1[3], nbnodes2[3], nb1=NO_NB, nb2=NO_NB;
    int j, maxa=0, *active=0;
    double len1,len2,len3,conc1,conc2,conc1new,conc2new;

    sprintf(msg,"%s %d %d","RemoveTripleJLink",node1,node2);
    /*
     * find indices and full id of node not in 2-sided grain
     */
    ElleNeighbourNodes(node1,nbnodes1);
    for (i=0;i<3;i++) if (nbnodes1[i]!=node2)
        nb1 = nbnodes1[i];
    ElleNeighbourNodes(node2,nbnodes2);
    for (i=0;i<3;i++) if (nbnodes2[i]!=node1)
        nb2 = nbnodes2[i];
    if (nb2==NO_NB || nb1==NO_NB)
        OnNodeError(node1,msg,NONB_ERR);
    if (ElleNodeCONCactive()) {
        len1 = ElleNodeSeparation(node1,nb1);
        len2 = ElleNodeSeparation(node2,nb2);
        len3 = ElleNodeSeparation(node1,node2);
        ElleNodeAttributeList(&active,&maxa);
        for (j=0;j<maxa;j++) {
            if (ElleNodeCONCAttribute(active[j])) {
                conc1 = ElleNodeAttribute(node1,active[j]);
                conc1new = conc1*(len1+len3+len3)/(len1+len3);
                ElleSetNodeAttribute(node1,conc1new,active[j]);
                conc2 = ElleNodeAttribute(node2,active[j]);
                conc2new = conc2*(len2+len3+len3)/(len2+len3);
                ElleSetNodeAttribute(node2,conc2new,active[j]);
            }
        }
        if (active) free(active);
    }
    ElleNeighbourRegion(node1,nb1,&full_id[1]);
    ElleNeighbourRegion(node2,nb2,&full_id[2]);
    /*
     * find full id of unit to be removed
     */
    ElleRegions(node1,tmp);
    for (i=0,found=0;i<3&&!found;i++) {
        full_id[0] = tmp[i];
        if (!ElleIdMatch(&full_id[0],&full_id[1]) &&
                                !ElleIdMatch(&full_id[0],&full_id[2]))
           found = 1;
    }
    /* double check */
    ElleRegions(node2,tmp);
    for (i=0,found=0;i<3&&!found;i++) {
        full_id[3] = tmp[i];
        if (!ElleIdMatch(&full_id[3],&full_id[1]) &&
                                !ElleIdMatch(&full_id[3],&full_id[2]))
           found = 1;
    }
    if (!ElleIdMatch(&full_id[0],&full_id[3]))
        OnNodeError(node2,msg,NONB_ERR);
    idindx = 1;
    rmindx2 = ElleFindBndIndex(node2,full_id[idindx]);
    idindx = 0;
    rmindx1 = ElleFindBndIndex(node1,full_id[idindx]);
    if (rmindx1==NO_NB || rmindx2==NO_NB)
        OnNodeError(node1,msg,NONB_ERR);
    /*
     * remove link
     */
    ElleClearNeighbour(node1,rmindx1);
    ElleClearNeighbour(node2,rmindx2);

    rmindx2 = ElleFindBndIndex(node2,full_id[idindx]);
    /*
     *  change unit for remaining link -
     *  previously grain to be removed
     */
    ElleSetRegionEntry(node2,rmindx2,full_id[1]);

    ElleRemoveShrinkingFlynn(full_id[idindx]);

    /*
     * if region has wrapped around, full_id[1]==full_id[2]
     * check whether it is too large or enclosing another region
     */
    if (ElleIdMatch(&full_id[1],&full_id[2])) {
        OnError(msg,RGNWRP_ERR);
    }
    //ElleSetNodeChange(1);
}

/*
 * calculates a position in the centre of the triangle
 * with vertices node2, node3, midpt of node-node1 boundary
 */
void ElleFindCentre(int node,int node1,int node2,int node3,
                    Coords *xy)
{
    Coords xy1,xy2,xy3;
    Coords xy_mp;

    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node,&xy_mp,&xy1);
    xy_mp.x = (xy1.x + xy_mp.x)*0.5;
    xy_mp.y = (xy1.y + xy_mp.y)*0.5;
    ElleNodePlotXY(node2,&xy2,&xy1);
    ElleNodePlotXY(node3,&xy3,&xy1);
    xy->x = (xy_mp.x + xy2.x + xy3.x)/3.0;
    xy->y = (xy_mp.y + xy2.y + xy3.y)/3.0;
}

/*
 * moves node to a position in the centre of the triangle
 * with vertices node2, node3, midpt of node-node1 boundary
 */
void ElleMoveToCentre(int node,int node1,int node2,int node3)
{
    Coords xy1;
    Coords xyold, increment;

    ElleFindCentre(node,node1,node2,node3,&xy1);
    ElleNodePlotXY(node,&xyold,&xy1);
    increment.x = xy1.x - xyold.x;
    increment.y = xy1.y - xyold.y;
    ElleUpdatePosition(node,&increment);
    /*ElleSetPosition(node,&xy1);*/
}

/*
 * find a point on the line which bisects the angle
 * nb1-node-nb2
 * (point is rotation of  coords of nb half the angle)
 */
void ElleTripleJMoveDir(int node,Coords *xy_mp,int nb1, int nb2,
                       Coords *dir)
{
    int index,index1,index2;
    Coords xy,xy_nb1,xy_nb2;
    double xnew,ynew;
    double ang0,ang1,ang2,ang;
	std::vector< std::pair<double,int> > ang_order;
	std::vector< std::pair<double,int> >::iterator it;

    ElleNodePlotXY(node,&xy,xy_mp);
    ElleNodePlotXY(nb1,&xy_nb1,xy_mp);
    ElleNodePlotXY(nb2,&xy_nb2,xy_mp);
    ang0 = polar_angle(xy_mp->x-xy.x,xy_mp->y-xy.y);
    ang1 = polar_angle(xy_nb1.x-xy.x,xy_nb1.y-xy.y);
    ang2 = polar_angle(xy_nb2.x-xy.x,xy_nb2.y-xy.y);
	std::pair<double,int> p0(ang0,node);
    ang_order.push_back(p0);
	std::pair<double,int> p1(ang1,nb1);
    ang_order.push_back(p1);
	std::pair<double,int> p2(ang2,nb2);
    ang_order.push_back(p2);
	std::sort(ang_order.begin(),ang_order.end());
/*
for (it=ang_order.begin();it!=ang_order.end();it++)
cout << it->first << ' ' << it->second << endl;
cout << endl;
*/
    index = 0;
    while (ang_order[index].second!=node) index++;
    index1 = (index+1)%3;
    index2 = (index1+1)%3;
/*
    angle(xy.x,xy.y,
          xy_nb1.x,xy_nb1.y,
          xy_mp->x,xy_mp->y,
          &ang1);
    if (ang1<0) ang1 = 2.0*PI+ang1;
    angle(xy.x,xy.y,
          xy_nb2.x,xy_nb2.y,
          xy_mp->x,xy_mp->y,
          &ang2);
    if (ang2<0) ang2 = 2.0*PI+ang2;
    ang=(ang2-ang1)*0.5;
    if (ang<0) ang = 2.0*PI+ang;
*/
    /*if (ang1<ang2) ang = 2.0*PI+ang1-ang2;*/
    /*else ang = ang1-ang2;*/
    /*if (ang>PI) ang = ang-2.0*PI;*/
    /*ang *= 0.5;*/
    /*rotate_coords_ac(xy_nb2.x,xy_nb2.y,*/
                  /*xy.x,xy.y,*/
                  /*&xnew, &ynew, ang);*/
    ang = ang_order[index2].first - ang_order[index1].first;
    if (ang<0) ang += 2*PI;
    ang *= 0.5;
    if (ang_order[index1].second==nb2)
    rotate_coords_ac(xy_nb2.x,xy_nb2.y,
                  xy.x,xy.y,
                  &xnew, &ynew, ang);
    else
    rotate_coords_ac(xy_nb1.x,xy_nb1.y,
                  xy.x,xy.y,
                  &xnew, &ynew, ang);
    dir->x = xnew; 
    dir->y = ynew;
}

/*
 * rotate to lie on the line which bisects the angle 
 * nb1-mid_pt-nb2 (mid_pt is mid_point of node1, node2)
 * check this before using
 */
void ElleRotateTripleJ(int node1,Coords *xy_mp,int nb1, int nb2,
                       Coords *xynew)
{
    Coords xy1,xy_nb1,xy_nb2,dir;
    double xnew,ynew;
    double ang1, ang2, ang;
    double len;

    ElleNodePlotXY(node1,&xy1,xy_mp);
    ElleNodePlotXY(nb1,&xy_nb1,xy_mp);
    ElleNodePlotXY(nb2,&xy_nb2,xy_mp);
    angle(xy_mp->x,xy_mp->y,
          xy_nb1.x,xy_nb1.y,
          xy1.x,xy1.y,
          &ang1);
    if (ang1<0) ang1 = 2.0*PI+ang1;
    angle(xy_mp->x,xy_mp->y,
          xy_nb2.x,xy_nb2.y,
          xy1.x,xy1.y,
          &ang2);
    if (ang2<0) ang2 = 2.0*PI+ang2;
    if ((ang=(ang2-ang1))<0) ang = 2.0*PI+ang;
    rotate_coords_ac(xy_nb1.x,xy_nb1.y,
              xy_mp->x,xy_mp->y,
              &xnew, &ynew, ang*0.5);
    /*len = EllePointSeparation(&xy_mp,&xy1);*/
    len = ElleSwitchLength()*0.75;
    dir.x = xnew; 
    dir.y = ynew; 
    MoveTowardsPoint(xy_mp,&dir,xynew,len);
}

void ElleRotateSegment(int node1,int node2,double angle)
{
    Coords xy1,xy2;
    Coords xy_mp;
    double xnew,ynew;

    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node2,&xy2,&xy1);
    xy_mp.x = (xy1.x + xy2.x)*0.5;
    xy_mp.y = (xy1.y + xy2.y)*0.5;
    rotate_coords((double)xy1.x,(double)xy1.y,
              (double)xy_mp.x,(double)xy_mp.y,
              &xnew, &ynew, angle);
    xy1.x = xnew;  xy1.y = ynew;
    ElleSetPosition(node1,&xy1);
    rotate_coords((double)xy2.x,(double)xy2.y,
              (double)xy_mp.x,(double)xy_mp.y,
              &xnew, &ynew, angle);
    xy2.x = xnew;  xy2.y = ynew;
    ElleSetPosition(node2,&xy2);
}

unsigned char AngleProblem(int node1,int node2,int node3,int node4,
                           int node5,int node6)
{
    /* 
     * returns 1 in either of these cases:
     *          1---2                 3       5       
     *        //     \\              4\      /6
     *       3/       \5              \\    //
     *        4       6                1----2
     */
    unsigned char problem=0;
    Coords xy1,xy2,xy3,xy4,xy5,xy6;
    double ang1=0, ang2=0;

    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node2,&xy2,&xy1);
    ElleNodePlotXY(node3,&xy3,&xy1);
    angle(xy1.x,xy1.y,xy3.x,xy3.y,
          xy2.x,xy2.y,&ang1);
    if (ang1>PI) {
        ElleNodePlotXY(node5,&xy5,&xy1);
        angle(xy2.x,xy2.y,xy1.x,xy1.y,
              xy5.x,xy5.y,&ang2);
        if (ang2>PI) problem = 1;
    }
    else {
        ElleNodePlotXY(node6,&xy6,&xy1);
        angle(xy2.x,xy2.y,xy6.x,xy6.y,
              xy1.x,xy1.y,&ang1);
        if (ang1>PI) {
            ElleNodePlotXY(node4,&xy4,&xy1);
            angle(xy1.x,xy1.y,xy4.x,xy4.y,
                  xy2.x,xy2.y,&ang2);
            if (ang2>PI) problem = 1;
        }
    }
    return(problem);
}

#if XY
int ElleSwitchTripleNodes(int node1,int node2)
{
    int err=0;
    Coords xy1, xy2;

    ElleNodePosition(node1,&xy1);
    ElleNodePosition(node2,&xy2);
    ElleSetPosition(node1,&xy2);
    ElleSetPosition(node2,&xy1);
    if (ElleNodeIsActive(node1)) err = ElleNodeTopologyCheck(node1);
    return(err);
}

int ElleSwitchTripleNodesForced(int node1,int node2)
{
    int err=0;
    Coords xy1, xy2;

    ElleNodePosition(node1,&xy1);
    ElleNodePosition(node2,&xy2);
    ElleSetPosition(node1,&xy2);
    ElleSetPosition(node2,&xy1);
    if (ElleNodeIsActive(node1)) err = ElleNodeTopologyCheck(node1);
    return(err);
}
#endif

/************these versions are from 1.2.2**************/
int ElleSwitchTripleNodes(int node1,int node2)
{
    unsigned char checked = 0;
    char msg[60], buf[30];
    int indx1in2, indx2in1, indx4in1, indx1in4;
    int i, indx2in5, indx5in2;
    int indx6in2, indx3in1;
    int node6, node5, node4, node3;
    int nb;
    ERegion full_id[4];
    ERegion tmp, rgn[3];
    int newf1, newf2, err=0;
    double sep1, sep2, sep3;
    Coords xy, xya, xyb, xyc, xy_mp;
    Intersection isect[MAX_I_CNT];
    Angle_data ang;
    int same1,same2,min1,min2;

#if XY
    if(Elle2NodesSameMineral(node1,node2)==0) /*mwj 12/12/99*/
#endif
    if(EllePhaseBoundary(node1,node2)) /*le 14/2/03*/
    return(0);

    sprintf(msg,"%s %d %d","SwitchTripleNodes",node1,node2);
    /*
     *       3     [1]    5
     *        \          /
     *    [0]  1--------2  [2]
     *        /          \
     *       4     [3]    6
     */

    indx1in2 = ElleFindNbIndex(node1,node2);
    indx2in1 = ElleFindNbIndex(node2,node1);
    ElleNeighbourRegion(node2,node1,&full_id[3]);
    ElleNeighbourRegion(node1,node2,&full_id[1]);
    /*
     * check whether switch will cause region wrap
     */
    ElleRegions(node1,rgn);
    i=0;
    while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
    if (i==3) OnError(msg,ID_ERR);
    full_id[0] = rgn[i];
    ElleRegions(node2,rgn);
    i=0;
    while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
    if (i==3) OnError(msg,ID_ERR);
    full_id[2] = rgn[i];
    if (ElleIdMatch(&full_id[0],&full_id[2]) &&
                ElleFlynnNodeCount(full_id[1])>3 &&
                        ElleFlynnNodeCount(full_id[3])>3) {
//        if (err=ElleSplitWrappingFlynn(full_id[0],
 //                            node1, node2, &newf1, &newf2))
            return(err);
        EllePromoteFlynn(newf1);
        EllePromoteFlynn(newf2);
        ElleRemoveFlynn(full_id[0]);
        ElleRegions(node1,rgn);
        i=0;
        while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
        if (i==3) OnError(msg,ID_ERR);
        full_id[0] = rgn[i];
        ElleRegions(node2,rgn);
        i=0;
        while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
        if (i==3) OnError(msg,ID_ERR);
        full_id[2] = rgn[i];
    }
    node5 = ElleFindBndNb(node2,full_id[1]);
    indx5in2 = ElleFindBndIndex(node2,full_id[1]);
    indx2in5 = ElleFindNbIndex(node2,node5);
    ElleNeighbourRegion(node5,node2,&tmp);
    if (ElleIdMatch(&tmp,&full_id[3])) {
        node6 = node5;
        indx6in2 = indx5in2;
        for(i=0;i<3;i++) if (i!=indx1in2 && i!=indx6in2) indx5in2 = i;
        node5 = ElleIndexNb(node2,indx5in2);
        indx2in5 = ElleFindNbIndex(node2,node5);
    }
    else {
        for(i=0;i<3;i++) if (i!=indx1in2 && i!=indx5in2) indx6in2 = i;
       node6 = ElleIndexNb(node2,indx6in2);
    }
    ElleNeighbourRegion(node2,node6,&full_id[2]);
    indx4in1 = ElleFindBndIndex(node1,full_id[3]);
    node4 = ElleIndexNb(node1,indx4in1);
    indx1in4 = ElleFindNbIndex(node1,node4);
    ElleNeighbourRegion(node4,node1,&tmp);
    if (ElleIdMatch(&tmp,&full_id[1])) {
        node3 = node4;
        indx3in1 = indx4in1;
        for(i=0;i<3;i++) if (i!=indx2in1 && i!=indx3in1) indx4in1 = i;
        node4 = ElleIndexNb(node1,indx4in1);
        indx1in4 = ElleFindNbIndex(node1,node4);
    }
    else {
        for(i=0;i<3;i++) if (i!=indx2in1 && i!=indx4in1) indx3in1 = i;
       node3 = ElleIndexNb(node1,indx3in1);
    }
    ElleNeighbourRegion(node1,node3,&full_id[0]);
    /*
     * error check before continuing
     */
    if (indx1in2==NO_NB||indx2in1==NO_NB||indx5in2==NO_NB||
         indx2in5==NO_NB||indx4in1==NO_NB||indx1in4==NO_NB)
        OnError(msg,NONB_ERR);
    if (node3==NO_NB||node4==NO_NB||node5==NO_NB||node6==NO_NB)
        OnError(msg,NONB_ERR);
    /*
     * if starting topology means switch will cause crossing
     * do nothing
    if (AngleProblem(node1,node2,node3,node4,node5,node6))
        return(1);
     */
    /*
     * Insert DJ on 2-5 or 1-4 (moving boundaries) to try and prevent
     * crossings with 1-3, 2-6
     * should only be done in certain cases - but what?
    sep1 = ElleNodeSeparation(node1,node4);
    sep2 = ElleNodeSeparation(node2,node6);
    sep3 = ElleNodeSeparation(node2,node6);
    if (sep1 > sep2) {
        ElleInsertDoubleJ(node1,node4,&i,(float)(sep2/sep1));
        node4 = i;
        indx1in4 = ElleFindNbIndex(node1,node4);
    }
    sep1 = ElleNodeSeparation(node2,node5);
    sep2 = ElleNodeSeparation(node1,node3);
    if (sep1 > sep2) {
        sep3 = ElleNodeSeparation(node2,node3);

        ElleInsertDoubleJ(node5,node2,&i,(float)(sep2/sep1));
        node5 = i;
        indx2in5 = ElleFindNbIndex(node2,node5);
    }
     */

    ElleSetNeighbour(node1,indx4in1,node5,&full_id[1]);
    ElleSetNeighbour(node2,indx5in2,node4,&full_id[3]);
    ElleSetRegionEntry(node1,indx2in1,full_id[2]);
    ElleSetRegionEntry(node2,indx1in2,full_id[0]);
    ElleSetNeighbour(node5,indx2in5,node1,0);
    ElleSetNeighbour(node4,indx1in4,node2,0);
    if (ElleNodeCONCactive() && !ElleUnodeCONCactive())
        RecalculateTripleSwitchCONC(node1,node2,node3,
                                    node4,node5,node6);
/*printf("mass_b4_ts = %.8e\n",ElleTotalNodeMass(CONC_A));*/

    ElleUpdateFirstNodes(node2,node1,full_id[1]);
    ElleUpdateFirstNodes(node1,node2,full_id[3]);

    /*
     * this is causing boundary crossings
     * check for it but do what if found???
     * problem reduced if SwitchDistance reduced
     * trying crossings check

    ElleNodePosition(node1,&xy);
    ElleFindCentre(node1,node2,node3,node5,&xya);
    xyb.x = xya.x-xy.x;
    xyb.y = xya.y-xy.y;
    ElleNodePosition(node2,&xy);
    ElleFindCentre(node2,node1,node4,node6,&xya);
    xyc.x = xya.x-xy.x;
    xyc.y = xya.y-xy.y;
    ElleCrossingsCheck(node1,&xyb);
    ElleCrossingsCheck(node2,&xyc);
     */
#if XY
    ElleNodePosition(node1,&xy);
    ElleNodePlotXY(node2,&xya,&xy);
    xy_mp.x = (xy.x + xya.x)*0.5;
    xy_mp.y = (xy.y + xya.y)*0.5;
    ElleRotateTripleJ(node1,&xy_mp,node3,node5);
    ElleRotateTripleJ(node2,&xy_mp,node6,node4);
#endif

    while (ElleNodeIsActive(node1) &&
                     ElleCheckForTwoSidedGrain(node1,&nb)) {
        checked = 1;
        if (ElleNodeIsTriple(nb)) ElleRemoveTripleJLink(node1,nb);
        else ElleRemoveHangingNode(nb,node1);
    }
    while (ElleNodeIsActive(node2) &&
                     ElleCheckForTwoSidedGrain(node2,&nb)) {
        checked = 1;
        if (ElleNodeIsTriple(nb)) ElleRemoveTripleJLink(node2,nb);
        else ElleRemoveHangingNode(nb,node2);
    }
    /*
     * if node is moved do it after removing 2-sided grains
	 * to avoid problems with routines involved in updating
	 * attributes and reassigning unodes
     */
	if (ElleNodeIsActive(node1) && ElleNodeIsActive(node2)) {
    	ElleFindCentre(node1,node2,node3,node5,&xy);
    	ElleMoveToCentre(node2,node1,node4,node6);
    	ElleSetPosition(node1,&xy);
	}
#if XY

    /*
     * check whether it is too large or enclosing another region
     */
    if (!checked) {
        if (ElleIdMatch(&full_id[1],&full_id[3]) 
         || ElleIdMatch(&full_id[0],&full_id[2])) {
            sprintf(msg,"Region=%d ",full_id[1]);
            OnError(msg,RGNWRP_ERR);
        }
        /*
         * check whether it caused a crossing
         */
         InitIntersection(isect);
         ElleNodePosition(node2,&xy);
         ElleNodePlotXY(node4,&xya,&xy);
         ElleNodePlotXY(node6,&xyb,&xy);
         angle(xy.x,xy.y,xya.x,xya.y,xyb.x,xyb.y,&(ang.prev_angle));
         if (ang.prev_angle>0 && ang.prev_angle<PI_2) {
             ang.pivot = node2;
             ang.nb = node6;
             if (ElleFindIntersection(node4,node2,full_id[3],
                       &xya,&xy,&isect[0])==1) 
                 ElleMoveLink(&isect[0],&ang,&xy);
         }
         ElleNodePosition(node1,&xy);
         ElleNodePlotXY(node5,&xya,&xy);
         ElleNodePlotXY(node3,&xyb,&xy);
         angle(xy.x,xy.y,xya.x,xya.y,xyb.x,xyb.y,&(ang.prev_angle));
         if (ang.prev_angle>0 && ang.prev_angle<PI_2) {
             ang.pivot = node1;
             ang.nb = node3;
             if (ElleFindIntersection(node5,node1,full_id[1],
                       &xya,&xy,&isect[0])==1) 
                 ElleMoveLink(&isect[0],&ang,&xy);
         }
    }
    InitIntersection(isect);
    for (i=0;i<4;i++) {
        if (ElleFlynnIsActive(full_id[i])) {
            if (!ElleRegionIsSimple(full_id[i],isect)) {
                strcpy(buf,"");
                strcpy(msg,"");
                sprintf(msg,"%s %d %d","SwitchTripleNodes",node1,node2);
                sprintf(buf," %d not simple",full_id[i]);
                strcat(msg,buf);
                fprintf(stderr,"%s\n",msg);
                //OnError(msg,0);
            }
        }
    }
#endif
    if (ElleNodeIsActive(node1)) err = ElleNodeTopologyCheck(node1);
    ElleSetNodeChange(1); //changes made even if check returns err?
    return(err);
}

int ElleSwitchTripleNodesForced(int node1,int node2)
{
    unsigned char checked = 0;
    char msg[60], buf[30];
    int indx1in2, indx2in1, indx4in1, indx1in4;
    int i, indx2in5, indx5in2;
    int indx6in2, indx3in1;
    int node6, node5, node4, node3;
    int nb;
    ERegion full_id[4];
    ERegion tmp, rgn[3];
    int newf1, newf2, err=0;
    Coords xy, xy2, xy5;
    int same1,same2,min1,min2;
    double len1, len2;
    Intersection isect[MAX_I_CNT];

    sprintf(msg,"%s %d %d","SwitchTripleNodesForced",node1,node2);
    /*
     *       3     [1]    5
     *        \          /
     *    [0]  1--------2  [2]
     *        /          \
     *       4     [3]    6
     */
    /*
     * only performs a switch if bnd 1-2 is not biphase and
     * bnd 1-3 or bnd 2-6 is not biphase
     * (ie for mineral type, [0]=[1]=[3] or [1]=[3]=[2]
     */

    indx1in2 = ElleFindNbIndex(node1,node2);
    indx2in1 = ElleFindNbIndex(node2,node1);
    ElleNeighbourRegion(node2,node1,&full_id[3]);
    ElleNeighbourRegion(node1,node2,&full_id[1]);
    /*
     * check whether switch will cause region wrap
     */
    ElleRegions(node1,rgn);
    i=0;
    while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
    if (i==3) OnError(msg,ID_ERR);
    full_id[0] = rgn[i];
    ElleRegions(node2,rgn);
    i=0;
    while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
    if (i==3) OnError(msg,ID_ERR);
    full_id[2] = rgn[i];
    if (ElleIdMatch(&full_id[0],&full_id[2]) &&
                ElleFlynnNodeCount(full_id[1])>3 &&
                        ElleFlynnNodeCount(full_id[3])>3) {
//        if (err=ElleSplitWrappingFlynn(full_id[0],
 //                            node1, node2, &newf1, &newf2))
            return(err);
        EllePromoteFlynn(newf1);
        EllePromoteFlynn(newf2);
        ElleRemoveFlynn(full_id[0]);
        ElleRegions(node1,rgn);
        i=0;
        while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
        if (i==3) OnError(msg,ID_ERR);
        full_id[0] = rgn[i];
        ElleRegions(node2,rgn);
        i=0;
        while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
        if (i==3) OnError(msg,ID_ERR);
        full_id[2] = rgn[i];
    }
    node5 = ElleFindBndNb(node2,full_id[1]);
    indx5in2 = ElleFindBndIndex(node2,full_id[1]);
    indx2in5 = ElleFindNbIndex(node2,node5);
    ElleNeighbourRegion(node5,node2,&tmp);
    if (ElleIdMatch(&tmp,&full_id[3])) {
        node6 = node5;
        indx6in2 = indx5in2;
        for(i=0;i<3;i++) if (i!=indx1in2 && i!=indx6in2) indx5in2 = i;
        node5 = ElleIndexNb(node2,indx5in2);
        indx2in5 = ElleFindNbIndex(node2,node5);
    }
    else {
        for(i=0;i<3;i++) if (i!=indx1in2 && i!=indx5in2) indx6in2 = i;
       node6 = ElleIndexNb(node2,indx6in2);
    }
    ElleNeighbourRegion(node2,node6,&full_id[2]);
    indx4in1 = ElleFindBndIndex(node1,full_id[3]);
    node4 = ElleIndexNb(node1,indx4in1);
    indx1in4 = ElleFindNbIndex(node1,node4);
    ElleNeighbourRegion(node4,node1,&tmp);
    if (ElleIdMatch(&tmp,&full_id[1])) {
        node3 = node4;
        indx3in1 = indx4in1;
        for(i=0;i<3;i++) if (i!=indx2in1 && i!=indx3in1) indx4in1 = i;
        node4 = ElleIndexNb(node1,indx4in1);
        indx1in4 = ElleFindNbIndex(node1,node4);
    }
    else {
        for(i=0;i<3;i++) if (i!=indx2in1 && i!=indx4in1) indx3in1 = i;
       node3 = ElleIndexNb(node1,indx3in1);
    }
    ElleNeighbourRegion(node1,node3,&full_id[0]);
    /*
     * error check before continuing
     */
    if (EllePhaseBoundary(node1,node2) ||
         (EllePhaseBoundary(node1,node3) &&
          EllePhaseBoundary(node2,node6)) )
        return(0);
    if (indx1in2==NO_NB||indx2in1==NO_NB||indx5in2==NO_NB||
         indx2in5==NO_NB||indx4in1==NO_NB||indx1in4==NO_NB)
        OnError(msg,NONB_ERR);
    if (node3==NO_NB||node4==NO_NB||node5==NO_NB||node6==NO_NB)
        OnError(msg,NONB_ERR);

    err = ElleSetNeighbour(node1,indx4in1,node5,&full_id[1]);
    err = ElleSetNeighbour(node2,indx5in2,node4,&full_id[3]);
    ElleSetRegionEntry(node1,indx2in1,full_id[2]);
    ElleSetRegionEntry(node2,indx1in2,full_id[0]);
    err = ElleSetNeighbour(node5,indx2in5,node1,0);
    err = ElleSetNeighbour(node4,indx1in4,node2,0);
    if (ElleNodeCONCactive() && !ElleUnodeCONCactive())
        RecalculateTripleSwitchCONC(node1,node2,node3,
                                    node4,node5,node6);
/*printf("mass_b4_ts = %.8e\n",ElleTotalNodeMass(CONC_A));*/
    /*
     * this is causing boundary crossings
     * check for it but do what if found???
     * problem reduced (removed) if SwitchDistance reduced
     */
    /*ElleMoveToCentre(node1,node2,node3,node5);*/
    /*ElleMoveToCentre(node2,node1,node4,node6);*/

    if(Elle2NodesSameMineral(node1,node2)==0) {
        if(Elle2NodesSameMineral(node1,node3)==0) {
    /* repositioning for biphase boundary 2,5 - keep boundary static */
            len1 = ElleNodeSeparation(node1,node2);
            len2 = ElleNodeSeparation(node2,node5);
            ElleNodePosition(node2,&xy2);
            ElleNodePlotXY(node5,&xy5,&xy2);
            xy.x = xy2.x - (xy2.x-xy5.x)*0.5;
            xy.y = xy2.y - (xy2.y-xy5.y)*0.5;
            ElleSetPosition(node1,&xy);
        }
        else if(Elle2NodesSameMineral(node2,node6)==0) {
    /* repositioning for biphase boundary 1,3 - keep boundary static */
            len1 = ElleNodeSeparation(node1,node2);
            len2 = ElleNodeSeparation(node1,node3);
            ElleNodePosition(node1,&xy2);
            ElleNodePlotXY(node3,&xy5,&xy2);
            xy.x = xy2.x - (xy2.x-xy5.x)*0.5;
            xy.y = xy2.y - (xy2.y-xy5.y)*0.5;
            ElleSetPosition(node2,&xy);
        }
    /* how did these TJs get so close? */
        else 
          Log( 0, "Problem with 2-phase triple switch\n" );
    }
    else {
        ElleFindCentre(node1,node2,node3,node5,&xy);
        ElleMoveToCentre(node2,node1,node4,node6);
        ElleSetPosition(node1,&xy);

      /*ElleRotateSegment(node1,node2,M_PI*0.5);*/
    }

    ElleUpdateFirstNodes(node2,node1,full_id[1]);
    ElleUpdateFirstNodes(node1,node2,full_id[3]);

    while (ElleNodeIsActive(node1) &&
                     ElleCheckForTwoSidedGrain(node1,&nb)) {
        checked = 1;
        if (ElleNodeIsTriple(nb)) ElleRemoveTripleJLink(node1,nb);
        else ElleRemoveHangingNode(nb,node1);
    }
    while (ElleNodeIsActive(node2) &&
                     ElleCheckForTwoSidedGrain(node2,&nb)) {
        checked = 1;
        if (ElleNodeIsTriple(nb)) ElleRemoveTripleJLink(node2,nb);
        else ElleRemoveHangingNode(nb,node2);
    }
#if XY
    /*
     * check whether it is too large or enclosing another region
     */
    if (!checked) {
        if (ElleIdMatch(&full_id[1],&full_id[3])
         || ElleIdMatch(&full_id[0],&full_id[2])) {
            sprintf(msg,"Region=%d ",full_id[1]);
            OnNodeError(node1,msg,RGNWRP_ERR);
        }
    }
    InitIntersection(isect);
    for (i=0;i<4;i++) {
        if (ElleFlynnIsActive(full_id[i])) {
            if (!ElleRegionIsSimple(full_id[i],isect)) {
#ifdef DEBUG
if (ElleNodeIsActive(node1)) writeNodeLocalPolyFile(node1);
#endif
                strcpy(buf,"");
                strcpy(msg,"");
                sprintf(msg,"%s %d %d","SwitchTripleNodes",node1,node2);
                sprintf(buf," %d not simple",full_id[i]);
                strcat(msg,buf);
                Log(1,msg);
                //OnError(msg,0);
            }
        }
    }
#endif
    if (ElleNodeIsActive(node1)) err = ElleNodeTopologyCheck(node1);
    ElleSetNodeChange(1); //changes made even if check returns err?
    return(err);
}
#if YZ
#endif

#if XY
int ElleSwitchTripleNodes(int node1,int node2)
{
    unsigned char checked = 0;
    char msg[60], buf[30];
    int indx1in2=NO_NB, indx2in1=NO_NB, indx4in1=NO_NB, indx1in4=NO_NB;
    int i, indx2in5=NO_NB, indx5in2=NO_NB;
    int indx6in2=NO_NB, indx3in1=NO_NB;
    int indx2in6=NO_NB, indx1in3=NO_NB;
    int node6, node5, node4, node3;
    int nb, sepcnt;
    ERegion full_id[4];
    ERegion tmp, rgn[3];
    int newf1, newf2, err=0;
    double sep1, sep2, sep3;
    Coords xy, xya, xyb, xyc, xy_mp;
    Coords xy1, xy2, increment,increment2;
Coords xy3, xy4, xy5, xy6;
    Coords xyold,xyold2;
    Intersection isect[MAX_I_CNT];
    Angle_data ang;

    if(Elle2NodesSameMineral(node1,node2)==0) /*mwj 12/12/99*/
    return(0);
#if XY
#endif

    strcpy(msg,"");
    sprintf(msg,"%s %d %d","SwitchTripleNodes",node1,node2);
    /*
     *       3     [1]    5
     *        \          /
     *    [0]  1--------2  [2]
     *        /          \
     *       4     [3]    6
     */

#ifdef DEBUG
if (ElleNodeIsActive(node1)) writeNodeLocalPolyFile(node1);
#endif
    indx1in2 = ElleFindNbIndex(node1,node2);
    indx2in1 = ElleFindNbIndex(node2,node1);
    ElleNeighbourRegion(node2,node1,&full_id[3]);
    ElleNeighbourRegion(node1,node2,&full_id[1]);
    /*
     * check whether switch will cause region wrap
     */
    ElleRegions(node1,rgn);
    i=0;
    while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
    if (i==3) OnNodeError(node1,msg,ID_ERR);
    full_id[0] = rgn[i];
    ElleRegions(node2,rgn);
    i=0;
    while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
    if (i==3) OnNodeError(node2,msg,ID_ERR);
    full_id[2] = rgn[i];
    if (ElleIdMatch(&full_id[0],&full_id[2]) &&
                ElleFlynnNodeCount(full_id[1])>3 &&
                        ElleFlynnNodeCount(full_id[3])>3) {
//        if (err=ElleSplitWrappingFlynn(full_id[0],
 //                            node1, node2, &newf1, &newf2))
            return(err);
        EllePromoteFlynn(newf1);
        EllePromoteFlynn(newf2);
        ElleRemoveFlynn(full_id[0]);
        ElleRegions(node1,rgn);
        i=0;
        while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
        if (i==3) OnNodeError(node1,msg,ID_ERR);
        full_id[0] = rgn[i];
        ElleRegions(node2,rgn);
        i=0;
        while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
        if (i==3) OnNodeError(node2,msg,ID_ERR);
        full_id[2] = rgn[i];
    }
    node5 = ElleFindBndNb(node2,full_id[1]);
    indx5in2 = ElleFindBndIndex(node2,full_id[1]);
    indx2in5 = ElleFindNbIndex(node2,node5);
    ElleNeighbourRegion(node5,node2,&tmp);
    if (ElleIdMatch(&tmp,&full_id[3])) {
        node6 = node5;
        indx6in2 = indx5in2;
        for(i=0;i<3;i++) if (i!=indx1in2 && i!=indx6in2) indx5in2 = i;
        node5 = ElleIndexNb(node2,indx5in2);
        indx2in5 = ElleFindNbIndex(node2,node5);
    }
    else {
        for(i=0;i<3;i++) if (i!=indx1in2 && i!=indx5in2) indx6in2 = i;
       node6 = ElleIndexNb(node2,indx6in2);
    }
    ElleNeighbourRegion(node2,node6,&full_id[2]);
    indx4in1 = ElleFindBndIndex(node1,full_id[3]);
    node4 = ElleIndexNb(node1,indx4in1);
    indx1in4 = ElleFindNbIndex(node1,node4);
    ElleNeighbourRegion(node4,node1,&tmp);
    if (ElleIdMatch(&tmp,&full_id[1])) {
        node3 = node4;
        indx3in1 = indx4in1;
        for(i=0;i<3;i++) if (i!=indx2in1 && i!=indx3in1) indx4in1 = i;
        node4 = ElleIndexNb(node1,indx4in1);
        indx1in4 = ElleFindNbIndex(node1,node4);
    }
    else {
        for(i=0;i<3;i++) if (i!=indx2in1 && i!=indx4in1) indx3in1 = i;
       node3 = ElleIndexNb(node1,indx3in1);
    }
    ElleNeighbourRegion(node1,node3,&full_id[0]);
    /*
     * error check before continuing
     */
    if (indx1in2==NO_NB||indx2in1==NO_NB||indx5in2==NO_NB||
         indx2in5==NO_NB||indx4in1==NO_NB||indx1in4==NO_NB)
        OnNodeError(node1,msg,NONB_ERR);
    if (node3==NO_NB||node4==NO_NB||node5==NO_NB||node6==NO_NB)
        OnNodeError(node1,msg,NONB_ERR);
    /*
     * if starting topology means switch will cause crossing
     * do nothing
    if (AngleProblem(node1,node2,node3,node4,node5,node6))
        return(1);
     */
    /*
     * Insert DJ on 2-5 or 1-4 (moving boundaries) to try and prevent
     * crossings with 1-3, 2-6
     * should only be done in certain cases - but what?
    sep1 = ElleNodeSeparation(node1,node4);
    sep2 = ElleNodeSeparation(node2,node6);
    sep3 = ElleNodeSeparation(node2,node6);
    if (sep1 > sep2) {
        ElleInsertDoubleJ(node1,node4,&i,(float)(sep2/sep1));
        node4 = i;
        indx1in4 = ElleFindNbIndex(node1,node4);
    }
    sep1 = ElleNodeSeparation(node2,node5);
    sep2 = ElleNodeSeparation(node1,node3);
    if (sep1 > sep2) {
        sep3 = ElleNodeSeparation(node2,node3);

        ElleInsertDoubleJ(node5,node2,&i,(float)(sep2/sep1));
        node5 = i;
        indx2in5 = ElleFindNbIndex(node2,node5);
    }
     */

    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node2,&xy2,&xy1);
    ElleNodePlotXY(node3,&xy3,&xy1);
    ElleNodePlotXY(node4,&xy4,&xy1);
    ElleNodePlotXY(node5,&xy5,&xy1);
    ElleNodePlotXY(node6,&xy6,&xy1);
    if (!lines_intersect(xy4.x,xy4.y,xy2.x,xy2.y,
                         xy5.x,xy5.y,xy1.x,xy1.y,&xyc.x,&xyc.y) &&
        !lines_intersect(xy4.x,xy4.y,xy2.x,xy2.y,
                         xy3.x,xy3.y,xy1.x,xy1.y,&xyc.x,&xyc.y) &&
        !lines_intersect(xy5.x,xy5.y,xy1.x,xy1.y,
                         xy6.x,xy6.y,xy2.x,xy2.y,&xyc.x,&xyc.y) ) {
    ElleSetNeighbour(node1,indx4in1,node5,&full_id[1]);
    ElleSetNeighbour(node2,indx5in2,node4,&full_id[3]);
    ElleSetRegionEntry(node1,indx2in1,full_id[2]);
    ElleSetRegionEntry(node2,indx1in2,full_id[0]);
    ElleSetNeighbour(node5,indx2in5,node1,0);
    ElleSetNeighbour(node4,indx1in4,node2,0);
    if (ElleNodeCONCactive() && !ElleUnodeCONCactive())
        RecalculateTripleSwitchCONC(node1,node2,node3,
                                    node4,node5,node6);
/*printf("mass_b4_ts = %.8e\n",ElleTotalNodeMass(CONC_A));*/

    ElleUpdateFirstNodes(node2,node1,full_id[1]);
    ElleUpdateFirstNodes(node1,node2,full_id[3]);
    /* move 1, 2 along a line bisecting the angle between their nbs */
    ElleNodePosition(node1,&xyold);
    ElleNodePlotXY(node2,&xyold2,&xyold);
    xy_mp.x = (xyold.x + xyold2.x)*0.5;
    xy_mp.y = (xyold.y + xyold2.y)*0.5;
    ElleTripleJMoveDir(node1,&xy_mp,node5,node3,&xy1);
    MoveTowardsPoint(&xyold,&xy1,&xy,ElleSwitchLength()*0.25);
    increment.x = xy.x - xyold.x;
    increment.y = xy.y - xyold.y;
    ElleTripleJMoveDir(node2,&xy_mp,node4,node6,&xy2);
    MoveTowardsPoint(&xyold2,&xy2,&xy,ElleSwitchLength()*0.25);
    increment2.x = xy.x - xyold2.x;
    increment2.y = xy.y - xyold2.y;

#if XY
    /* move to centroid */
    ElleFindCentre(node1,node2,node3,node5,&xy);
    ElleMoveToCentre(node2,node1,node4,node6);
    ElleNodePlotXY(node1,&xyold,&xy);
    increment.x = xy.x - xyold.x;
    increment.y = xy.y - xyold.y;
    /*ElleUpdatePosition(node1,&increment);*/
    /* repositioning to boundary 2,5 - keep boundary static */
    ElleNodePosition(node2,&xya);
    ElleNodePlotXY(node5,&xyb,&xya);
    xy_mp.x = xya.x - (xya.x-xyb.x)*0.5;
    xy_mp.y = xya.y - (xya.y-xyb.y)*0.5;
    ElleNodePlotXY(node1,&xyold,&xy_mp);
    increment2.x = xy_mp.x - xyold.x;
    increment2.y = xy_mp.y - xyold.y;
    /*ElleUpdatePosition(node1,&increment);*/
#endif

#ifdef DEBUG
if (ElleNodeIsActive(node1)) writeNodeLocalPolyFile(node1);
#endif
    }
    else if (!lines_intersect(xy3.x,xy3.y,xy2.x,xy2.y,
                         xy4.x,xy4.y,xy1.x,xy1.y,&xyc.x,&xyc.y) &&
        !lines_intersect(xy3.x,xy3.y,xy2.x,xy2.y,
                         xy6.x,xy6.y,xy1.x,xy1.y,&xyc.x,&xyc.y) &&
        !lines_intersect(xy6.x,xy6.y,xy1.x,xy1.y,
                         xy5.x,xy5.y,xy2.x,xy2.y,&xyc.x,&xyc.y) ) {
    indx2in6 = ElleFindNbIndex(node2,node6);
    indx1in3 = ElleFindNbIndex(node1,node3);
    ElleSetNeighbour(node1,indx3in1,node6,&full_id[2]);
    ElleSetNeighbour(node2,indx6in2,node3,&full_id[0]);
    ElleSetRegionEntry(node1,indx2in1,full_id[0]);
    ElleSetRegionEntry(node2,indx1in2,full_id[2]);
    ElleSetNeighbour(node6,indx2in6,node1,0);
    ElleSetNeighbour(node3,indx1in3,node2,0);
    if (ElleNodeCONCactive() && !ElleUnodeCONCactive())
        RecalculateTripleSwitchCONC(node1,node2,node3,
                                    node4,node5,node6);
/*printf("mass_b4_ts = %.8e\n",ElleTotalNodeMass(CONC_A));*/

    ElleUpdateFirstNodes(node2,node1,full_id[3]);
    ElleUpdateFirstNodes(node1,node2,full_id[1]);
    /* move 1, 2 along a line bisecting the angle between their nbs */
    ElleNodePosition(node1,&xyold);
    ElleNodePlotXY(node2,&xyold2,&xyold);
    xy_mp.x = (xyold.x + xyold2.x)*0.5;
    xy_mp.y = (xyold.y + xyold2.y)*0.5;
    ElleTripleJMoveDir(node1,&xy_mp,node6,node4,&xy1);
    MoveTowardsPoint(&xyold,&xy1,&xy,ElleSwitchLength()*0.25);
    increment.x = xy.x - xyold.x;
    increment.y = xy.y - xyold.y;
    ElleTripleJMoveDir(node2,&xy_mp,node3,node5,&xy2);
    MoveTowardsPoint(&xyold2,&xy2,&xy,ElleSwitchLength()*0.25);
    increment2.x = xy.x - xyold2.x;
    increment2.y = xy.y - xyold2.y;
    }
    else {
        return(1);
} //end lines_intersect

    while (ElleNodeIsActive(node1) &&
                     ElleCheckForTwoSidedGrain(node1,&nb)) {
        checked = 1;
        if (ElleNodeIsTriple(nb)) ElleRemoveTripleJLink(node1,nb);
        else ElleRemoveHangingNode(nb,node1);
    }
    while (ElleNodeIsActive(node2) &&
                     ElleCheckForTwoSidedGrain(node2,&nb)) {
        checked = 1;
        if (ElleNodeIsTriple(nb)) ElleRemoveTripleJLink(node2,nb);
        else ElleRemoveHangingNode(nb,node2);
    }

    if (ElleNodeIsActive(node1)) ElleUpdatePosition(node1,&increment);

    if (ElleNodeIsActive(node2)) ElleUpdatePosition(node2,&increment2);


#ifdef DEBUG
if (ElleNodeIsActive(node1)) writeNodeLocalPolyFile(node1);
#endif

    if (ElleNodeIsDouble(node3)) ElleCheckDoubleJ(node3);
    if (ElleNodeIsDouble(node4)) ElleCheckDoubleJ(node4);
    if (ElleNodeIsDouble(node5)) ElleCheckDoubleJ(node5);
    if (ElleNodeIsDouble(node6)) ElleCheckDoubleJ(node6);
    if (!ElleNodeIsActive(node3)||!ElleNodeIsActive(node4)||
        !ElleNodeIsActive(node5)||!ElleNodeIsActive(node6))
        checked=1;
#if XY
    for (i=0;i<4;i++) 
        if (ElleFlynnIsActive(full_id[i]))
            if (ElleCheckSmallFlynn(full_id[i])!=0)
                checked=1;
#endif
    /*
     * check whether it is too large or enclosing another region
     */
    if (!checked) {
        if (ElleIdMatch(&full_id[1],&full_id[3]) 
         || ElleIdMatch(&full_id[0],&full_id[2])) {
            sprintf(msg,"Region=%d ",full_id[1]);
            OnError(msg,RGNWRP_ERR);
        }
#if XY
        /*
         * check whether it caused a crossing
         */
         InitIntersection(isect);
         ElleNodePosition(node2,&xy);
         ElleNodePlotXY(node4,&xya,&xy);
         ElleNodePlotXY(node6,&xyb,&xy);
         angle(xy.x,xy.y,xya.x,xya.y,xyb.x,xyb.y,&(ang.prev_angle));
         if (ang.prev_angle>0 && ang.prev_angle<PI_2) {
if (ElleNodeIsActive(node1)) writeNodeLocalPolyFile(node1);
             ang.pivot = node2;
             ang.nb = node6;
             if (ElleFindIntersection(node4,node2,full_id[3],
                       &xya,&xy,&isect[0])==1) 
                 ElleMoveLink(&isect[0],&ang,&xy);
         }
         ElleNodePosition(node1,&xy);
         ElleNodePlotXY(node5,&xya,&xy);
         ElleNodePlotXY(node3,&xyb,&xy);
         angle(xy.x,xy.y,xya.x,xya.y,xyb.x,xyb.y,&(ang.prev_angle));
         if (ang.prev_angle>0 && ang.prev_angle<PI_2) {
if (ElleNodeIsActive(node1)) writeNodeLocalPolyFile(node1);
             ang.pivot = node1;
             ang.nb = node3;
             if (ElleFindIntersection(node5,node1,full_id[1],
                       &xya,&xy,&isect[0])==1) 
                 ElleMoveLink(&isect[0],&ang,&xy);
         }
#endif
    }
    InitIntersection(isect);
    for (i=0;i<4;i++) {
        if (ElleFlynnIsActive(full_id[i])) {
            if (!ElleRegionIsSimple(full_id[i],isect)) {
#ifdef DEBUG
if (ElleNodeIsActive(node1)) writeNodeLocalPolyFile(node1);
#endif
//failing to get rid of dj causing crossing
/*
                if (i==0 && ElleNodeIsDouble(node3) )
                    ElleDeleteDoubleJ(node3);
                else if (i==1 && ElleNodeIsDouble(node5) )
                    ElleDeleteDoubleJ(node5);
                else if (i==2 && ElleNodeIsDouble(node6) )
                    ElleDeleteDoubleJ(node6);
                else if (i==3 && ElleNodeIsDouble(node4) )
                    ElleDeleteDoubleJ(node4);
*/
                if (ElleFlynnIsActive(full_id[i]) &&
                      !ElleRegionIsSimple(full_id[i],isect)) {
#ifdef DEBUG
if (ElleNodeIsActive(node1)) writeNodeLocalPolyFile(node1);
#endif

                strcpy(buf,"");
                strcpy(msg,"");
                sprintf(msg,"%s %d %d","SwitchTripleNodes",node1,node2);
                sprintf(buf," %d not simple",full_id[i]);
                strcat(msg,buf);
                fprintf(stderr,"%s\n",msg);
#ifdef DEBUG
#endif
                //OnError(msg,0);
            }
            }
        }
    }
    return(0);
}

int ElleSwitchTripleNodesForced(int node1,int node2)
{
    unsigned char checked = 0;
    char msg[60], buf[30];
    int indx1in2=NO_NB, indx2in1=NO_NB, indx4in1=NO_NB, indx1in4=NO_NB;
    int i, indx2in5=NO_NB, indx5in2=NO_NB;
    int indx6in2=NO_NB, indx3in1=NO_NB;
    int node6, node5, node4, node3;
    int nb;
    ERegion full_id[4];
    ERegion tmp, rgn[3];
    int newf1, newf2, err=0;
    Coords xy, xy2, xy5, xynb, xy_mp;
    Coords xyold, xy1, xynew, increment;
    int min1,min2;
    Intersection isect[MAX_I_CNT];
    Angle_data ang;


    sprintf(msg,"%s %d %d","SwitchTripleNodesForced",node1,node2);
    /*
     *       3     [1]    5
     *        \          /
     *    [0]  1--------2  [2]
     *        /          \
     *       4     [3]    6
     */
    /*
     * only performs a switch if bnd 1-2 is not biphase and
     * bnd 1-3 or bnd 2-6 is not biphase
     * (ie for mineral type, [0]=[1]=[3] or [1]=[3]=[2]
     */
#ifdef DEBUG
if (ElleNodeIsActive(node1)) writeNodeLocalPolyFile(node1);
#endif

    indx1in2 = ElleFindNbIndex(node1,node2);
    indx2in1 = ElleFindNbIndex(node2,node1);
    ElleNeighbourRegion(node2,node1,&full_id[3]);
    ElleNeighbourRegion(node1,node2,&full_id[1]);
    /*
     * check whether switch will cause region wrap
     */
    err = ElleRegions(node1,rgn);
    i=0;
    while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
    if (i==3) OnNodeError(node1,msg,ID_ERR);
    full_id[0] = rgn[i];
    err = ElleRegions(node2,rgn);
    i=0;
    while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
    if (i==3) OnNodeError(node2,msg,ID_ERR);
    full_id[2] = rgn[i];
    if (ElleIdMatch(&full_id[0],&full_id[2]) &&
                ElleFlynnNodeCount(full_id[1])>3 &&
                        ElleFlynnNodeCount(full_id[3])>3) {
//        if (err=ElleSplitWrappingFlynn(full_id[0],
 //                            node1, node2, &newf1, &newf2))
            return(err);
        EllePromoteFlynn(newf1);
        EllePromoteFlynn(newf2);
        ElleRemoveFlynn(full_id[0]);
        err = ElleRegions(node1,rgn);
        i=0;
        while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
        if (i==3) OnNodeError(node1,msg,ID_ERR);
        full_id[0] = rgn[i];
        err = ElleRegions(node2,rgn);
        i=0;
        while (i<3 && (rgn[i]==full_id[1] || rgn[i]==full_id[3])) i++;
        if (i==3) OnNodeError(node2,msg,ID_ERR);
        full_id[2] = rgn[i];
    }
    node5 = ElleFindBndNb(node2,full_id[1]);
    indx5in2 = ElleFindBndIndex(node2,full_id[1]);
    indx2in5 = ElleFindNbIndex(node2,node5);
    ElleNeighbourRegion(node5,node2,&tmp);
    if (ElleIdMatch(&tmp,&full_id[3])) {
        node6 = node5;
        indx6in2 = indx5in2;
        for(i=0;i<3;i++) if (i!=indx1in2 && i!=indx6in2) indx5in2 = i;
        node5 = ElleIndexNb(node2,indx5in2);
        indx2in5 = ElleFindNbIndex(node2,node5);
    }
    else {
        for(i=0;i<3;i++) if (i!=indx1in2 && i!=indx5in2) indx6in2 = i;
       node6 = ElleIndexNb(node2,indx6in2);
    }
    ElleNeighbourRegion(node2,node6,&full_id[2]);
    indx4in1 = ElleFindBndIndex(node1,full_id[3]);
    node4 = ElleIndexNb(node1,indx4in1);
    indx1in4 = ElleFindNbIndex(node1,node4);
    ElleNeighbourRegion(node4,node1,&tmp);
    if (ElleIdMatch(&tmp,&full_id[1])) {
        node3 = node4;
        indx3in1 = indx4in1;
        for(i=0;i<3;i++) if (i!=indx2in1 && i!=indx3in1) indx4in1 = i;
        node4 = ElleIndexNb(node1,indx4in1);
        indx1in4 = ElleFindNbIndex(node1,node4);
    }
    else {
        for(i=0;i<3;i++) if (i!=indx2in1 && i!=indx4in1) indx3in1 = i;
       node3 = ElleIndexNb(node1,indx3in1);
    }
    ElleNeighbourRegion(node1,node3,&full_id[0]);
    /*
     * error check before continuing
     */
    if (EllePhaseBoundary(node1,node2) ||
         (EllePhaseBoundary(node1,node3) &&
          EllePhaseBoundary(node2,node6)) ) {
        ElleGetFlynnIntAttribute(full_id[0], &min1, MINERAL);
        ElleGetFlynnIntAttribute(full_id[2], &min2, MINERAL);
        if (min1!=min2) return(0);
    }
    if (indx1in2==NO_NB||indx2in1==NO_NB||indx5in2==NO_NB||
         indx2in5==NO_NB||indx4in1==NO_NB||indx1in4==NO_NB)
        OnNodeError(node1,msg,NONB_ERR);
    if (node3==NO_NB||node4==NO_NB||node5==NO_NB||node6==NO_NB)
        OnNodeError(node1,msg,NONB_ERR);

    err = ElleSetNeighbour(node1,indx4in1,node5,&full_id[1]);
    err = ElleSetNeighbour(node2,indx5in2,node4,&full_id[3]);
    ElleSetRegionEntry(node1,indx2in1,full_id[2]);
    ElleSetRegionEntry(node2,indx1in2,full_id[0]);
    err = ElleSetNeighbour(node5,indx2in5,node1,0);
    err = ElleSetNeighbour(node4,indx1in4,node2,0);
    if (ElleNodeCONCactive() && !ElleUnodeCONCactive())
        RecalculateTripleSwitchCONC(node1,node2,node3,
                                    node4,node5,node6);
/*printf("mass_b4_ts = %.8e\n",ElleTotalNodeMass(CONC_A));*/

    ElleUpdateFirstNodes(node2,node1,full_id[1]);
    ElleUpdateFirstNodes(node1,node2,full_id[3]);
    /*
     * this is causing boundary crossings
     * check for it but do what if found???
     * problem reduced (removed) if SwitchDistance reduced
     */
    /*ElleMoveToCentre(node1,node2,node3,node5);*/
    /*ElleMoveToCentre(node2,node1,node4,node6);*/

#ifdef DEBUG
if (ElleNodeIsActive(node1)) writeNodeLocalPolyFile(node1);
#endif
    if(Elle2NodesSameMineral(node1,node2)||
        Elle2NodesSameMineral(node1,node3)==0) {
    /* repositioning for biphase boundary 2,5 - keep boundary static */
            ElleNodePosition(node2,&xy2);
            ElleNodePlotXY(node5,&xy5,&xy2);
            MoveTowardsPoint(&xy2,&xy5,&xy,ElleNodeSeparation(node1,node2));
            ElleNodePlotXY(node1,&xyold,&xy);
            increment.x = xy.x - xyold.x;
            increment.y = xy.y - xyold.y;
            ElleUpdatePosition(node1,&increment);
#ifdef DEBUG
if (ElleNodeIsActive(node1)) writeNodeLocalPolyFile(node1);
#endif
            if (ElleSwitchGap(node1,node2) &&
                    ElleNodeSeparation(node1,node5)>ElleSmallSwitchLength()) {
                xy_mp.x = (xy.x + xy2.x)*0.5;
                xy_mp.y = (xy.y + xy2.y)*0.5;
                /*MoveTowardsPoint(&xy_mp,&xy,&xynew,ElleSmallSwitchLength());*/
                MoveTowardsPoint(&xy,&xy5,&xynew,ElleSmallSwitchLength());
                increment.x = xynew.x - xy.x;
                increment.y = xynew.y - xy.y;
                ElleUpdatePosition(node1,&increment);
            }
            if (ElleSwitchGap(node1,node2) &&
                    ElleNodeSeparation(node2,node6)>ElleSmallSwitchLength()) {
                ElleNodePlotXY(node6,&xynb,&xy2);
                ElleNodePlotXY(node1,&xy,&xy2);
                xy_mp.x = (xy.x + xy2.x)*0.5;
                xy_mp.y = (xy.y + xy2.y)*0.5;
                MoveTowardsPoint(&xy2,&xynb,&xynew,ElleSmallSwitchLength());
                increment.x = xynew.x - xy2.x;
                increment.y = xynew.y - xy2.y;
                ElleUpdatePosition(node2,&increment);
            }
            if (ElleNodeIsDouble(node5)) {
                ElleCheckDoubleJ(node5);
                if (ElleNodeIsDouble(node5)) {
                   ElleNeighbourRegion(node1,node3,&tmp);
                   if ( !ElleRegionIsSimple(tmp,isect)) {
                       ElleDeleteDoubleJNoCheck(node5);
                   }
                   else {
                       ElleNeighbourRegion(node1,node5,&tmp);
                       if (!ElleRegionIsSimple(tmp,isect)) 
                           ElleDeleteDoubleJNoCheck(node5);
                   }
                }
            }
            if (ElleNodeIsDouble(node6)) {
                ElleCheckDoubleJ(node6);
                if (ElleNodeIsDouble(node6)) {
                    ElleNeighbourRegion(node2,node6,&tmp);
                    if (!ElleRegionIsSimple(tmp,isect)) {
                        ElleDeleteDoubleJNoCheck(node6);
                    }
                    else {
                        ElleNeighbourRegion(node2,node4,&tmp);
                        if (!ElleRegionIsSimple(tmp,isect)) 
                            ElleDeleteDoubleJNoCheck(node6);
                    }
                }
            }
        }
        else if(Elle2NodesSameMineral(node2,node6)==0) {
    /* repositioning for biphase boundary 1,4 - keep boundary static */
            ElleNodePosition(node1,&xy2);
            ElleNodePlotXY(node4,&xy5,&xy2);
            MoveTowardsPoint(&xy2,&xy5,&xy,ElleNodeSeparation(node1,node2));
            ElleNodePlotXY(node2,&xyold,&xy);
            increment.x = xy.x - xyold.x;
            increment.y = xy.y - xyold.y;
            ElleUpdatePosition(node2,&increment);
            if (ElleSwitchGap(node1,node2) &&
                    ElleNodeSeparation(node2,node4)>ElleSmallSwitchLength()) {
                xy_mp.x = (xy.x + xy2.x)*0.5;
                xy_mp.y = (xy.y + xy2.y)*0.5;
                MoveTowardsPoint(&xy_mp,&xy,&xy1,ElleSmallSwitchLength());
                increment.x = xy1.x - xy.x;
                increment.y = xy1.y - xy.y;
                ElleUpdatePosition(node2,&increment);
            }
            if (ElleNodeIsDouble(node4)) {
                ElleNeighbourRegion(node2,node6,&tmp);
                if (!ElleRegionIsSimple(tmp,isect)) {
                    ElleDeleteDoubleJNoCheck(node4);
                }
                else {
                    ElleNeighbourRegion(node2,node4,&tmp);
                    if (!ElleRegionIsSimple(tmp,isect)) 
                        ElleDeleteDoubleJNoCheck(node4);
                }
            }
            if (ElleNodeIsDouble(node3)) {
                ElleNeighbourRegion(node1,node3,&tmp);
                if (!ElleRegionIsSimple(tmp,isect))  {
                    ElleDeleteDoubleJNoCheck(node3);
                }
                else {
                    ElleNeighbourRegion(node1,node5,&tmp);
                    if (!ElleRegionIsSimple(tmp,isect)) 
                        ElleDeleteDoubleJNoCheck(node3);
                }
            }
        }
    /* how did these TJs get so close? */
        else fprintf(stderr,"Problem with 2-phase triple switch\n");
#ifdef DEBUG
if (ElleNodeIsActive(node1)) writeNodeLocalPolyFile(node1);
#endif

    while (ElleNodeIsActive(node1) &&
                     ElleCheckForTwoSidedGrain(node1,&nb)) {
        checked = 1;
        if (ElleNodeIsTriple(nb)) ElleRemoveTripleJLink(node1,nb);
        else ElleRemoveHangingNode(nb,node1);
    }
    while (ElleNodeIsActive(node2) &&
                     ElleCheckForTwoSidedGrain(node2,&nb)) {
        checked = 1;
        if (ElleNodeIsTriple(nb)) ElleRemoveTripleJLink(node2,nb);
        else ElleRemoveHangingNode(nb,node2);
    }
    for (i=0;i<4;i++) 
        if (ElleFlynnIsActive(full_id[i]))
            if (ElleCheckSmallFlynn(full_id[i])!=0)
                checked=1;
    /*
     * check whether it is too large or enclosing another region
     */
    if (!checked) {
        if (ElleIdMatch(&full_id[1],&full_id[3])
         || ElleIdMatch(&full_id[0],&full_id[2])) {
            sprintf(msg,"Region=%d ",full_id[1]);
            OnNodeError(node1,msg,RGNWRP_ERR);
        }
    }
    InitIntersection(isect);
    for (i=0;i<4;i++) {
        if (ElleFlynnIsActive(full_id[i])) {
            if (!ElleRegionIsSimple(full_id[i],isect)) {
#ifdef DEBUG
if (ElleNodeIsActive(node1)) writeNodeLocalPolyFile(node1);
#endif
                strcpy(buf,"");
                strcpy(msg,"");
                sprintf(msg,"%s %d %d","SwitchTripleNodes",node1,node2);
                sprintf(buf," %d not simple",full_id[i]);
                strcat(msg,buf);
                fprintf(stderr,"%s\n",msg);
                //OnError(msg,0);
            }
        }
    }
    return(0);
}
#endif

int ElleTwoSidedGrainId(int node1, int node2)
{
    int id=NO_NB, i, j;
    int nbs1[3], nbs2[3];
    int rgns1[3], rgns2[3], tmp;

    ElleNeighbourNodes(node1,nbs1);
    ElleNeighbourNodes(node2,nbs2);
    ElleRegions(node1,rgns1);
    ElleRegions(node2,rgns2);
    for (i=0;i<3 && id==NO_NB;i++) {
        if (nbs1[i]==node2) {
            tmp = rgns1[i];
            for (j=0;j<3 && id==NO_NB;j++)
                //if (nbs2[j]==node1 && rgns2[j]==tmp) id=tmp;
                if (nbs2[j]==node1 && rgns2[i]==tmp) id=tmp;
        }
    }
    return(id);
}


int ElleIdMatch(ERegion *a, ERegion *b)
{
    /*
     * returns 1 if the unit a matches b 
     *  else returns 0
     */
    if (*a == *b) return(1);
    return(0);
}

int ElleFindNbIndex(int nb, int node)
{
    /*
     * find the index for the first occurrence of nb
     */
    int i, idx = NO_NB, nbnodes[3];
    int found=0;

    ElleNeighbourNodes(node,nbnodes);
    for (i=0;i<3 && !found;i++)
        if (nbnodes[i]==nb) {
            idx = i;
            found = 1;
        }
    return( idx );
}

int ElleFindBndIndex(int node, ERegion full_id)
{
    /*
     * find the index for the first occurrence
     * of the matching boundary
     */
    int i, idx = NO_NB;
    int found=0;
    ERegion tmp[3];

/*
    if ((nb=ElleFindBndNb(node,full_id))!=NO_NB) {
        ElleNeighbourRegion(nb,node,&tmp);
        idx = ElleFindBndNbIndex(nb,node,tmp);
    }
*/
    ElleRegions(node,tmp);
    for (i=0;i<3 && !found;i++) {
        if (tmp[i]==full_id ||
                (tmp[i]!=NO_NB && isChildOf(tmp[i],full_id))) {
            idx = i;
            found = 1;
        }
    }
    return( idx );
}

int ElleFindBndNbIndex(int nb, int node, ERegion full_id)
{
    /*
     * find the index for the nb. Check full id of boundary
     * in case it is a 2-sided grain
     */
    int i, idx = NO_NB, nbnodes[3];
    int  found=0;
    ERegion rgn[3];

/*
    ElleNeighbourNodes(node,nbnodes);
    idx = ElleFindNbIndex(nb,node);
    if (idx!=NO_NB) {
        ElleNeighbourRegion(node,nb,&tmp);
        if (full_id==tmp || isChildOf(tmp,full_id)) {
            idx++; if (idx>2) idx=0;
            i=idx;
            for (idx=NO_NB;i<3;i++)
                if (nbnodes[i]==nb) idx = i;
        }
    }
*/
    ElleNeighbourNodes(node,nbnodes);
    ElleRegions(node,rgn);
    i=0;
    while (i<3 && !found) {
         if (nbnodes[i]==nb) {
             if (rgn[i]!=full_id &&
                 !isChildOf(rgn[i],full_id)) {
                 idx = i;
                 found = 1;
             }
         }
         i++;
    }
    return( idx );
}

int ElleNewFindBndNb(int node_num, int prev_node,ERegion id)
{
    return(ElleFindBndNode(node_num,prev_node,id));
}

int ElleFindBndNode(int node_num, int prev_node,int id)
{
    /*
     * more rigorous that ElleFindBndNb, in case of more than
     * one node with id region eg. grain wrap, nucleation
     */
    int i=0, j, nbnodes[3], nbnbnodes[3], idx = NO_NB, found=0;
    ERegion rgn[3], nbrgn;

    ElleNeighbourNodes(node_num,nbnodes);
    ElleRegions(node_num,rgn);
    i=0;
    while (i<3 && !found) {
         if (nbnodes[i]!=NO_NB && nbnodes[i]!=prev_node &&
                 (rgn[i]==id || isChildOf(rgn[i],id))){
             idx = nbnodes[i];
             ElleNeighbourNodes(idx,nbnbnodes);
             j=0;
             while (nbnbnodes[j]!=node_num && j<3) j++;
             ElleNeighbourRegion(idx,nbnbnodes[j],&nbrgn);
             if (nbrgn!=id &&
                 !isChildOf(nbrgn,id))
                 found = 1;
             else i++;
         }
         else i++;
    }
    return( idx );
}
#if XY
#endif

int ElleFindBndNb(int node_num, ERegion id)
{
   /*
    *  returns the neighbour node of node_num along the boundary of
    *  the region specified by id.
    */
    int i=0, j, nbnodes[3], nbnbnodes[3], idx = NO_NB, found=0;
    ERegion rgn[3], nbrgn;

    ElleNeighbourNodes(node_num,nbnodes);
    ElleRegions(node_num,rgn);
    i=0;
    while (i<3 && !found) {
        if (rgn[i]==id ||
                (rgn[i]!=NO_NB && isChildOf(rgn[i],id))) {
            idx = nbnodes[i];
            ElleNeighbourNodes(idx,nbnbnodes);
            j=0;
            while (nbnbnodes[j]!=node_num) j++;
            ElleNeighbourRegion(idx,nbnbnodes[j],&nbrgn);
            if (nbrgn!=id && !isChildOf(nbrgn,id)) found=1;
            else i++;
        }
        else i++;
    }
    return( idx );
}

int ElleReadLocationData(FILE *fp, char str[])
{
    /* 
     * routine to be called after the location keyword has
     * been read in an Elle file
     * Reads lines with the format %d %lf %lf (node id, x, y)
     */
    int  err=0, num, nn;
    Coords current;
    double tmpx,tmpy;

    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1) return(READ_ERR);
        else if (str[0] == '#') dump_comments( fp );
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((nn = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            ElleSetNodeActive(nn);
            /*
             * Alphas rounding to %.6f so read as double
             */
            if ((num = fscanf(fp,"%lf %lf\n",
                             &current.x, &current.y)) != 2)
                return(READ_ERR);
            /*ElleSetPosition(nn,&current);*/
            /*ElleSetPrevPosition(nn,&current);*/
            ElleCopyToPosition(nn,&current);
            ElleCopyToPrevPosition(nn,&current);
        }
    }
    return(err);
}

int ElleReadConcAData(FILE *fp, char str[])
{
    /*
     * routine to be called after the CONC_A keyword has
     * been read in an Elle file
     * Reads lines with the format %d %f (node id, conc)
     */
    int j, err=0, num, nn;
    int max, indx;
    double val;

    ElleInitNodeAttribute(CONC_A);
    if ((indx = ElleConcAIndex())==NO_INDX)
        return(MAXATTRIB_ERR);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp)) return(READ_ERR);
        else if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%lf\n", &val)) != 1)
                return(READ_ERR);
            max = ElleMaxNodes();
            for (j=0;j<max;j++)
                if (ElleNodeIsActive(j)) {
                    ElleSetNodeAttribute(j,val,CONC_A);
                }
            ElleSetNodeDefaultConcA(val);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((nn = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            ElleSetNodeActive(nn);
            if ((num = fscanf(fp,"%lf\n", &val)) != 1)
                return(READ_ERR);
            ElleSetNodeAttribute(nn,val,CONC_A);
        }
    }
    return(err);
}

int ElleReadNodeMarginAttribData(FILE *fp, char str[],int attr_id)
{
    /*
     * routine to be called after a node margin attrib keyword has
     * been read in an Elle file
     * Reads lines with the format %d %lf %lf %lf(node id, val for 
     *        2 or 3 flynns)
     */
    char *ptr,*start;
    int i, j, err=0, num, nn;
    int max, loc_id;
    int id[3], rgn[3], nb_id[3];
    double val, val_3[3]={0,0,0};

    switch (attr_id) {
        case N_MCONC_A: 
                        loc_id = CONC_A;
                        id[0] = N_MCONC_A_1;
                        id[1] = N_MCONC_A_2;
                        id[2] = N_MCONC_A_3;
                        break;
        case N_MATTRIB_A: 
                        loc_id = ATTRIB_A;
                        break;
        case N_MATTRIB_B: 
                        loc_id = ATTRIB_B;
                        break;
        case N_MATTRIB_C: 
                        loc_id = ATTRIB_C;
                        break;
        default:        return(ATTRIBID_ERR);
    }
    ElleInitNodeMarginAttribute(loc_id);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp)) return(READ_ERR);
        else if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%lf %lf %lf\n",
                             &val_3[0],&val_3[1],&val_3[2])) != 3)
                return(READ_ERR);
            max = ElleMaxNodes();
            for (j=0;j<max;j++)
                if (ElleNodeIsActive(j)) {
                    for (i=0;i<3;i++)
                        ElleSetNodeAttribute(j,val_3[i],id[i]);
                }
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((nn = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            if (fgets(str,256,fp)==NULL) return(READ_ERR);
            start=str;
            ptr = strtok(start," \t\r\n");
            while (ptr) {
                rgn[0] = atoi(ptr);
                start = 0;
                ptr = strtok(start," \t\r\n");
                if (ptr) {
                    val_3[0] = atof(ptr);
                    start = 0;
                    ElleSetNodeMarginAttribute(nn,val_3[0],
                                                loc_id,rgn[0]);
                }
                ptr = strtok(start," \t\r\n");
            }
        }
    }
    return(err);
}

unsigned char ElleNodeMarginAttributeActive(int id)
{
    unsigned char found = 0;

    switch(id) {
    case CONC_A: found = ElleNodeAttributeActive(N_MCONC_A_1);
                 break;
    case N_MATTRIB_A:
    case ATTRIB_A: found = ElleNodeAttributeActive(N_MATTRIB_A_1);
                 break;
    case N_MATTRIB_B:
    case ATTRIB_B: found = ElleNodeAttributeActive(N_MATTRIB_B_1);
                 break;
    case N_MATTRIB_C:
    case ATTRIB_C: found = ElleNodeAttributeActive(N_MATTRIB_C_1);
                 break;
    default:     break;
    }
    return(found);
}

unsigned char ElleNodeAttributeActive(int id)
{
    unsigned char found = 0;

    if (ElleFindNodeAttribIndex(id)!=NO_INDX)
        found = 1;
    return(found);
}

int ElleFindNodeAttributeRange(int id, double *amin, double *amax)
{
    int i, max, err=0;
    int indx, set=0;
    double tmpmin, tmpmax, val;

    if ((indx = ElleFindNodeAttribIndex(id))==NO_INDX)
        err = ATTRIBID_ERR;
    else {
        max = ElleMaxNodes();
        for (i=0;i<max;i++) {
            if (ElleNodeIsActive(i)) {
                val = ElleNodeAttribute(i,id);
                if (!set) {
                    tmpmax = val;
                    tmpmin = val;
                    set = 1;
                }
                else {
                    if (val > tmpmax) tmpmax = val;
                    if (val < tmpmin) tmpmin = val;
                }
            }
        }
        *amax = tmpmax;
        *amin = tmpmin;
    }
}

int ElleReadNodeRealAttribData(FILE *fp, char str[], int id)
{
    /*
     * routine to be called after real attribute keyword has
     * been read in an Elle file
     * Reads lines with the format %d %f (node id, conc)
     */
    int j, err=0, num, nn;
    int max, indx;
    double val;

    ElleInitNodeAttribute(id);
    if ((indx = ElleFindNodeAttribIndex(id))==NO_INDX)
        return(MAXATTRIB_ERR);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp)) return(READ_ERR);
        else if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%lf\n", &val)) != 1)
                return(READ_ERR);
            max = ElleMaxNodes();
            for (j=0;j<max;j++)
                if (ElleNodeIsActive(j)) {
                    ElleSetNodeAttribute(j,val,id);
                }
            /* need to set up an array of defaults, rather than
               hardwired in code. Assumes the default for these is 0 */
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((nn = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            /* make this a warning message */
            if (ElleNodeIsActive(nn)) {
                if ((num = fscanf(fp,"%lf\n", &val)) != 1)
                    return(READ_ERR);
                ElleSetNodeAttribute(nn,val,id);
            }
            else
                fprintf(stderr,"Ignoring file attribute %d for inactive node %d\n",id,nn);
        }
    }
    return(err);
}

int ElleReadVelocityData(FILE *fp, char str[])
{
    /* 
     * routine to be called after the velocity keyword has
     * been read in an Elle file
     * Reads lines with the format %d %f %f(node id, velocity in
     * x-direction, velocity in y-direction)
     */
    int j, err=0, num, nn;
    int max, indxx, indxy;
    double valx,valy;

/* this should be written like stress - pass index of attrib ids */
    ElleInitNodeAttribute(VEL_X);
    ElleInitNodeAttribute(VEL_Y);
    if ((indxx = ElleVelocityXIndex())==NO_INDX)
        return(MAXATTRIB_ERR);
    if ((indxy = ElleVelocityYIndex())==NO_INDX)
        return(MAXATTRIB_ERR);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp)) return(READ_ERR);
        else if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%lf %lf\n", &valx, &valy)) != 2)
                return(READ_ERR);
            max = ElleMaxNodes();
            for (j=0;j<max;j++)
                if (ElleNodeIsActive(j)) {
                    ElleSetNodeAttribute(j,valx,VEL_X);
                    ElleSetNodeAttribute(j,valy,VEL_Y);
                }
            ElleSetNodeDefaultVelocityX(valx);
            ElleSetNodeDefaultVelocityY(valy);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((nn = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            ElleSetNodeActive(nn);
            if ((num = fscanf(fp,"%lf %lf\n", &valx, &valy)) != 2)
                return(READ_ERR);
            ElleSetNodeAttribute(nn,valx,VEL_X);
            ElleSetNodeAttribute(nn,valy,VEL_Y);
        }
    }
    return(err);
}

int ElleReadNodeFile(char *filename)
{
    /* No longer used */
    FILE *fp;
    char str[80];
    int  err=0, num, nn;
    Coords current;

    if ((fp=fopen(filename,"r"))==NULL) return(OPEN_ERR);
    while (!feof(fp)) {
        if ((num = fscanf(fp,"%s", str))!=1) return(READ_ERR);
        if (str[0] != '#') {
            nn = atoi(str);
            ElleSetNodeActive(nn);
            if ((num = fscanf(fp,"%lf %lf\n",
                             &current.x, &current.y)) != 2)
                return(READ_ERR);
            ElleSetPosition(nn,&current);
            ElleSetPrevPosition(nn,&current);
        }
        else dump_comments( fp );
    }
    fclose(fp);                            
    return(err);
}

int ElleWriteNodeData(FILE *fp)
{
    char label[20];
    int err=0;
    int i, j, *active=0, maxa, *keys=0;

    err=ElleWriteNodeLocation(fp);
    ElleNodeAttributeList(&active,&maxa);
    for (i=0;i<maxa && !err;i++)  {
        switch(active[i]) {
        case VEL_X: err = ElleWriteNodeVelocity(fp);
                    i++;
                    break;
        case TAU_XX:
        case TAU_YY:
        case TAU_ZZ:
        case TAU_XY:
        case TAU_1 :
        case PRESSURE:
                    if ((keys=(int *)malloc(maxa * sizeof(int)))==0)
                        OnError("ElleWriteNodeData",MALLOC_ERR);
                    for (j=0;j<maxa;j++) keys[j] = NO_INDX;
                    j=0;
                    while (id_match(StressKeys,active[i],label)) {
                        keys[j] = active[i];
                        j++; i++;
                    }
                    i--; /* for loop will increment again */
                    err = ElleWriteStressData(fp,keys,j);
                    if (keys) free(keys);
                    break;
        case N_MCONC_A:
        case N_MCONC_A_1: err = ElleWriteNodeMarginAttribData(fp,N_MCONC_A);
                    break;
        case N_MATTRIB_A:
        case N_MATTRIB_A_1:
                    err = ElleWriteNodeMarginAttribData(fp,N_MATTRIB_A);
                    break;
        case N_MATTRIB_B:
        case N_MATTRIB_B_1:
                    err = ElleWriteNodeMarginAttribData(fp,N_MATTRIB_B);
                    break;
        case N_MATTRIB_C:
        case N_MATTRIB_C_1:
                    err = ElleWriteNodeMarginAttribData(fp,N_MATTRIB_C);
                    break;
        case CONC_A: err = ElleWriteNodeConcA(fp);
                    break;
        case INCR_S :
        case BULK_S :
                    if ((keys=(int *)malloc(maxa * sizeof(int)))==0)
                        OnError("ElleWriteNodeData",MALLOC_ERR);
                    for (j=0;j<maxa;j++) keys[j] = NO_INDX;
                    j=0;
                    while (id_match(StrainKeys,active[i],label)) {
                        keys[j] = active[i];
                        j++; i++;
                    }
                    i--; /* for loop will increment again */
                    err = ElleWriteStrainData(fp,keys,j);
                    if (keys) free(keys);
                    break;
        case N_ATTRIB_A:
        case N_ATTRIB_B:
        case N_ATTRIB_C:
        case N_ATTRIB_D:
        case N_ATTRIB_E:
        case N_ATTRIB_F: err = ElleWriteNodeRealAttribute(fp,active[i]);
                    break;
                    // these have already been written, so step over
        case N_MCONC_A_2:
        case N_MCONC_A_3:
        case N_MATTRIB_A_2:
        case N_MATTRIB_A_3:
        case N_MATTRIB_B_2:
        case N_MATTRIB_B_3:
        case N_MATTRIB_C_2:
        case N_MATTRIB_C_3:
                    break;
        }
    }
    if (active!=0) free(active);
    return(err);
}

int ElleWriteNodeLocation(FILE *fp)
{
    /*
     * writes location data in a node file
     * location label then position data on a separate
     * line for each node with the format: %d %f %f
     * (node id number, x, y)
     */
    char label[20];
    int i,max;
    Coords pos;

    max = ElleMaxNodes();
    if (!id_match(FileKeys,LOCATION,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            ElleNodePosition(i,&pos); 
            fprintf(fp,"%d %.10lf %.10lf\n",i,pos.x, pos.y);
        }
    }
    return(0);
}


int ElleWriteNodeConcA(FILE *fp)
{
    /*
     * writes ConcA data in a node file
     * line for each node with the format: %d %f
     * (node id number, conc)
     */
    char label[20];
    int i,max;
    int indx;
    double val, dflt;
    double eps=1e-15;

    max = ElleMaxNodes();
    if ((indx = ElleConcAIndex())==NO_INDX)
        return(MAXATTRIB_ERR);

    if (!id_match(FileKeys,CONC_A,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);

    dflt = ElleNodeDefaultConcA();
    fprintf(fp,"%s %.8e\n",SET_ALL,dflt);
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            val = ElleNodeAttribute(i,CONC_A);
            if (val> dflt+eps || val< dflt-eps)
                fprintf(fp,"%d %.8e\n",i,val);
        }
    }
    return(0);
}

int ElleWriteNodeRealAttribute(FILE *fp,int id)
{
    /*
     * writes real attribute data in a node file
     * line for each node with the format: %d %f
     * (node id number, val)
     */
    char label[20];
    int i,max;
    double val, dflt;
    double eps=1e-15;

    max = ElleMaxNodes();

    if (!id_match(FileKeys,id,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);

    dflt = 0; /* this need fixing - store in dflt array */
    fprintf(fp,"%s %.8e\n",SET_ALL,dflt);
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            val = ElleNodeAttribute(i,id);
            if (val!=dflt && (id!=CONC_A || val> dflt+eps || val< dflt-eps))
                fprintf(fp,"%d %.8e\n",i,val);
        }
    }
    return(0);
}

int ElleWriteNodeMarginAttribData(FILE *fp,int attr_id)
{
    /*
     * writes margin real attribute data in a node file
     * line for each node with the format: %d %lf %lf %lf
     * (node id number, val for 3 flynns)
     */
    char label[20];
    int j, i,max,loc_id;
    int id[3], rgn[3], prnt=0;
    double val_3[3], dflt_3[3]={0,0,0};

    max = ElleMaxNodes();

    switch(attr_id) {
        case N_MCONC_A: loc_id = CONC_A;
                        id[0] = N_MCONC_A_1;
                        id[1] = N_MCONC_A_2;
                        id[2] = N_MCONC_A_3;
                        break;
        case N_MATTRIB_A: loc_id = ATTRIB_A;
                        id[0] = N_MATTRIB_A_1;
                        id[1] = N_MATTRIB_A_2;
                        id[2] = N_MATTRIB_A_3;
                        break;
        case N_MATTRIB_B: loc_id = ATTRIB_B;
                        id[0] = N_MATTRIB_B_1;
                        id[1] = N_MATTRIB_B_2;
                        id[2] = N_MATTRIB_B_3;
                        break;
        case N_MATTRIB_C: loc_id = ATTRIB_C;
                        id[0] = N_MATTRIB_C_1;
                        id[1] = N_MATTRIB_C_2;
                        id[2] = N_MATTRIB_C_3;
                        break;
        default:        return(ATTRIBID_ERR);
    }

    if (!id_match(FileKeys,attr_id,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    /* this need fixing - get dflts from dflt array */
    fprintf(fp,"%s %.8e %.8e %.8e\n",SET_ALL,
                               dflt_3[0],dflt_3[1],dflt_3[2]);
    for (j=0;j<max;j++) {
        if (ElleNodeIsActive(j)) {
            ElleRegions(j,rgn);
            for (i=0,prnt=0;i<3;i++) {
                val_3[i] = dflt_3[i];
                if (rgn[i]!=NO_NB) 
                  val_3[i]=ElleNodeMarginAttribute(j,loc_id,rgn[i]);
                if (val_3[i]!=dflt_3[i]) prnt=1;
            }
            if (prnt) {
                fprintf(fp,"%d",j);
                for (i=0;i<3;i++) {
                    if (rgn[i]!=NO_NB && val_3[i]!=dflt_3[i])
                        fprintf(fp," %d %.8e", rgn[i],val_3[i]);
                }
                fprintf(fp,"\n");
            }
        }
    }
    return(0);
}

int ElleWriteNodeVelocity(FILE *fp)
{
    /*
     * writes velocity data in a node file
     * velocity label then velocity data on a separate
     * line for each node with the format: %d %f %f
     * (node id number, x direction velocity, y direction velocity)
     */
    char label[20];
    int i,max;
    int indxx, indxy;
    double valx, valy, dfltx, dflty;

    max = ElleMaxNodes();
    if ((indxx = ElleVelocityXIndex())==NO_INDX)
        return(MAXATTRIB_ERR);
    if ((indxy = ElleVelocityYIndex())==NO_INDX)
        return(MAXATTRIB_ERR);

    if (!id_match(FileKeys,VELOCITY,label)) return(KEY_ERR);
    fprintf(fp,"%s ",label);
    if (!id_match(VelocityKeys,VEL_X,label)) return(KEY_ERR);
    fprintf(fp,"%s ",label);
    if (!id_match(VelocityKeys,VEL_Y,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);

    dfltx = ElleNodeDefaultVelocityX();
    dflty = ElleNodeDefaultVelocityY();
    fprintf(fp,"%s %.8e %.8e\n",SET_ALL,dfltx,dflty);
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            valx = ElleNodeAttribute(i,VEL_X);
            valy = ElleNodeAttribute(i,VEL_Y);
            if (valx!=dfltx || valy!=dflty)
                fprintf(fp,"%d %.8e %.8e\n",i,valx,valy);
        }
    }
    return(0);
}

int ElleReadStressData(FILE *fp,char str[],int *keys,int count)
{
    /*
     * Read stress data in a node file.
     * Stress labels then stress data on a separate
     * line for each node with the format: %d %f %f %f %f %f
     * (node id number, tau xx, tau yy, tau zz, tau xy, pressure
     * or the order read into keys)
     */
    int i;
    int nn, num, err=0;
    int index[NUM_STRESS_VALS];
    double val[NUM_STRESS_VALS];

    for (i=0;i<count;i++) {
        ElleInitNodeAttribute(keys[i]);
/* change this to index[i] = ElleFindNodeAttribIndex(keys[i]) */
/****no defaults set for stresses *********/
        switch(keys[i]) {
        case TAU_XX:
                     if ((index[i]=ElleTauXXIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        case TAU_YY: if ((index[i]=ElleTauYYIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        case TAU_ZZ: if ((index[i]=ElleTauZZIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        case TAU_XY: if ((index[i]=ElleTauXYIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        case TAU_1 : if ((index[i]=ElleTau1Index())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        case PRESSURE: if ((index[i]=EllePressureIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        default:     return(ID_ERR);
                     break;
        }
    }
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
            ElleSetNodeActive(nn);
            for (i=0;i<count;i++) {
                if ((num = fscanf(fp," %lf", &val[i])) != 1)
                    return(READ_ERR);
                ElleSetNodeAttribute(nn,val[i],keys[i]);
            }
        }
        fscanf(fp,"\n");
    }
    return(0);
}

int ElleReadStrainData(FILE *fp,char str[],int *keys,int count)
{
    /*
     * Read strain data in a node file.
     * Strain labels then strain data on a separate
     * line for each node with the format: %d %f %f
     * (node id number, psr1, bulk psr1 or the order read into keys)
     */
    int i;
    int nn, num, err=0;
    int index[NUM_STRAIN_VALS];
    double val[NUM_STRAIN_VALS];

    for (i=0;i<count;i++) {
        ElleInitNodeAttribute(keys[i]);
/* change this to index[i] = ElleFindNodeAttribIndex(keys[i]) */
/****no defaults set for stresses *********/
        switch(keys[i]) {
        case INCR_S :
                     if ((index[i]=ElleStrainIncrIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        case BULK_S : if ((index[i]=ElleBulkStrainIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        default:     return(ID_ERR);
                     break;
        }
    }
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        else if (str[0] == '#') dump_comments( fp );
        else if (str[0]<'0' || str[0]>'9') return(0);
        /*
         * SET_ALL not valid for strain-rate vals
         */
        else {
            if ((nn = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            ElleSetNodeActive(nn);
            for (i=0;i<count;i++) {
                if ((num = fscanf(fp," %lf", &val[i])) != 1)
                    return(READ_ERR);
                ElleSetNodeAttribute(nn,val[i],keys[i]);
            }
        }
        fscanf(fp,"\n");
    }
    return(0);
}

int ElleWriteStressData(FILE *fp, int *keys, int count)
{
    /*
     * Writes stress data in an elle file.
     * Stress data on a separate line for each node 
     * with the format: %d %e %e %e %e %e
     * (node id number, count values in the order defined by keys)
     */
    char label[20];
    int i,j,max;
    int index[NUM_STRESS_VALS];

    if (!id_match(FileKeys,STRESS,label)) return(KEY_ERR);
    fprintf(fp,"%s ",label);
    for (i=0;i<count;i++) {
        if (!id_match(StressKeys,keys[i],label)) return(KEY_ERR);
        fprintf(fp,"%s ",label);
    }
    fprintf(fp,"\n");
    max = ElleMaxNodes();
    for (i=0;i<count;i++) {
        switch(keys[i]) {
        case TAU_XX: if ((index[i]=ElleTauXXIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        case TAU_YY: if ((index[i]=ElleTauYYIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        case TAU_ZZ: if ((index[i]=ElleTauZZIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        case TAU_XY: if ((index[i]=ElleTauXYIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        case TAU_1 : if ((index[i]=ElleTau1Index())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        case PRESSURE: if ((index[i]=EllePressureIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        default:     return(ID_ERR);
                     break;
        }
    }
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            fprintf(fp,"%d",i);
            for (j=0;j<count;j++)  {
                fprintf(fp," %.8e",ElleNodeAttribute(i,keys[j]));
            }
            fprintf(fp,"\n");
        }
    }
    return(0);
}

int ElleWriteStrainData(FILE *fp, int *keys, int count)
{
    /*
     * Writes strain data in an elle file.
     * Strain data on a separate line for each node 
     * with the format: %d %f %f
     * (node id number, count values in the order defined by keys)
     */
    char label[20];
    int i,j,max;
    int index[NUM_STRAIN_VALS];

    if (!id_match(FileKeys,STRAIN,label)) return(KEY_ERR);
    fprintf(fp,"%s ",label);
    for (i=0;i<count;i++) {
        if (!id_match(StrainKeys,keys[i],label)) return(KEY_ERR);
        fprintf(fp,"%s ",label);
    }
    fprintf(fp,"\n");
    max = ElleMaxNodes();
    for (i=0;i<count;i++) {
        switch(keys[i]) {
        case INCR_S : if ((index[i]=ElleStrainIncrIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        case BULK_S: if ((index[i]=ElleBulkStrainIndex())==NO_INDX)
                         return(MAXATTRIB_ERR);
                     break;
        default:     return(ID_ERR);
                     break;
        }
    }
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            fprintf(fp,"%d",i);
            for (j=0;j<count;j++)  {
                fprintf(fp," %.8e",ElleNodeAttribute(i,keys[j]));
            }
            fprintf(fp,"\n");
        }
    }
    return(0);
}

double ElleVelocityX(int node)
{
    if (!ElleNodeIsActive(node))
       OnError("ElleVelocityX",NODENUM_ERR); 
    return(ElleNodeAttribute(node,VEL_X));
}

double ElleVelocityY(int node)
{
    if (!ElleNodeIsActive(node))
       OnError("ElleVelocityY",NODENUM_ERR); 
    return(ElleNodeAttribute(node,VEL_Y));
}

int ElleJoinTwoDoubleJ(int first, int last,ERegion old,ERegion rnew)
{
    int i, nbnodes[3], err=0;

    ElleNeighbourNodes(last,nbnodes);
    i=0;
    while (i<3 && nbnodes[i]!=NO_NB) i++;
    ElleSetNeighbour(last,i,first,&rnew);
    ElleNeighbourNodes(first,nbnodes);
    i=0;
    while (i<3 && nbnodes[i]!=NO_NB) i++;
    ElleSetNeighbour(first,i,last,&old);
    return(err);
}

void ElleSwitchTriples(int modfactor)
{
   /*
    * Parameters: modfactor - controls how many switches are done
    *             This function is only used on an initial mesh
    *             consisting of connected triples.
    * Action: Forced triple switch between node and its first triple
    *         nb
    */
    int j, nbnodes[3],max;

    max = ElleMaxNodes();
    for (j=0;j<max;j++)
        if (j%modfactor == 0 && ElleNodeIsActive(j)) {
            ElleNeighbourNodes(j,nbnodes);
            ElleSwitchTripleNodesForced(j,nbnodes[0]);
        }
}

double ElleSwitchLength()
{
    return(ElleminNodeSep());
}

int ElleSwitchGap(int node1,int node2)
{
   /*
    * Parameters: node1, node2 - integers specifying a node and a
    *             connected neighbour
    * Action: None
    * Return value: 1 if the distance between the nodes is 
    *               < min node separation else 0
    */
    double SwitchDist;
    Coords xy1,xy2;

    SwitchDist = ElleSwitchLength();
    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node2,&xy2,&xy1);
    if (sqrt((xy2.x - xy1.x)*(xy2.x - xy1.x) + 
                       (xy2.y - xy1.y)*(xy2.y - xy1.y)) < SwitchDist)
    return( 1 );
    else return( 0 );
}

double ElleSmallSwitchLength()
{
    return(ElleminNodeSep()*0.1);
}

int ElleSmallSwitchGap(int node1,int node2)
{
   /*
    **********currently this is the same as ElleSwitchGap ***********
    * Parameters: node1, node2 - integers specifying a node and a
    *             connected neighbour
    * Action: None
    * Return value: 1 if the distance between the nodes is < SwitchDist
    *               else 0
    */
    double SwitchDist;
    Coords xy1,xy2;

    SwitchDist = ElleSmallSwitchLength();
    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node2,&xy2,&xy1);
    if (sqrt((xy2.x - xy1.x)*(xy2.x - xy1.x) +
                       (xy2.y - xy1.y)*(xy2.y - xy1.y)) < SwitchDist)
    return( 1 );
    else return( 0 );
}

int ElleNodesCoincident(int node1,int node2)
{
   /*
    * Parameters: node1, node2 - integers specifying a node and a
    *             connected neighbour
    * Action: None
    * Return value: 1 if the distance between the nodes is < eps
    *               else 0
    */
    double eps = 1e-8;
    Coords xy1,xy2;

    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node2,&xy2,&xy1);
    if (sqrt((xy2.x - xy1.x)*(xy2.x - xy1.x) + 
                       (xy2.y - xy1.y)*(xy2.y - xy1.y)) < eps)
        return( 1 );
    else return( 0 );
}

int ElleGapTooSmall(int node1,int node2)
{
   /*
    * Parameters: node1, node2 - integers specifying a node and a
    *             connected neighbour
    * Action: None
    * Return value: 1 if the distance between the nodes is < MinNodeSep
    *               else 0
    */
    double minNodeSep;
    Coords xy1,xy2;

    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node2,&xy2,&xy1);
    minNodeSep = ElleminNodeSep();
    if (sqrt((xy2.x - xy1.x)*(xy2.x - xy1.x) + 
                       (xy2.y - xy1.y)*(xy2.y - xy1.y)) < minNodeSep)
    return( 1 );
    else return( 0 );
}

int ElleGapTooLarge(int node1,int node2)
{
   /*
    * Parameters: node1, node2 - integers specifying a node and a
    *             connected neighbour
    * Action: None
    * Return value: 1 if the distance between the nodes is > MaxNodeSep
    *               else 0
    */
    double maxNodeSep;
    Coords xy1,xy2;

    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node2,&xy2,&xy1);
    maxNodeSep = EllemaxNodeSep();
    if (sqrt((xy2.x - xy1.x)*(xy2.x - xy1.x) + 
                       (xy2.y - xy1.y)*(xy2.y - xy1.y) ) > maxNodeSep)
    return( 1 );
    else return( 0 );
}

int ElleNodeOnRgnBnd(int node, ERegion id)
{
    int nbnodes[3],found=0,nb;

    ElleNeighbourNodes(node,nbnodes);
    if ((nb=ElleFindBndNb(node,id))!=NO_NB) found=1;
/*
    for (i=0,found=0;i<3&&!found;i++) {
        if (nbnodes[i]==NO_NB) tmp=NO_NB;
        else ElleNeighbourRegion(node,nbnodes[i],&tmp);
        if (ElleIdMatch(&tmp,id)) found = 1;
    }
*/
    return(found);
}

void ElleRelPosition(Coords *orig, int nb, Coords *xyrel, double *sep)
{
    Coords xynb;
    double tmp;

    ElleNodePlotXY(nb,&xynb,orig);
    xyrel->x = xynb.x - orig->x;
    xyrel->y = xynb.y - orig->y;
    tmp = xyrel->x * xyrel->x +
                  xyrel->y * xyrel->y;
    *sep = sqrt(tmp);
}

double ElleNodeSeparation(int node1, int node2)
{
    Coords xy1,xy2;
    double tmp;

    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node2,&xy2,&xy1);
    tmp = (double)(xy1.x - xy2.x) * (xy1.x-xy2.x) +
                  (xy1.y - xy2.y) * (xy1.y - xy2.y);
    return(sqrt(tmp));
}

double ElleNodePrevSeparation(int node1, int node2)
{
    Coords prevxy1,xy2;
    double tmp;

    ElleNodePrevPosition(node1,&prevxy1);
    ElleNodePlotXY(node2,&xy2,&prevxy1);
    tmp = (prevxy1.x - xy2.x) * (prevxy1.x-xy2.x) +
                  (prevxy1.y - xy2.y) * (prevxy1.y - xy2.y);
    return(sqrt(tmp));
}

void ElleNumberOfNodes(int *doubles, int *triples)
{
    int i, max, numdble=0, numtrple=0;

    max = ElleMaxNodes();
    for (i=0;i<max;i++) 
        if (ElleNodeIsActive(i)) {
            if (ElleNodeIsDouble(i)) numdble++;
            else if (ElleNodeIsTriple(i)) numtrple++;
        }
    *doubles = numdble;
    *triples = numtrple;
}

int ElleResetNodeConc(int start, int reset_cnt, double val, int attrib_id)
{
    int k=0, rgns[3], cnt, prev, curr, nb, err=0;

    if (ElleNodeIsActive(start) && ElleNodeIsDouble(start)) {
        ElleRegions(start,rgns);
        ElleSetNodeAttribute(start,val,attrib_id);
        while (k<3) {
            if (rgns[k]!=NO_NB) {
              curr = prev = start;
              cnt=0;
              while (cnt<reset_cnt &&
                    (nb = ElleFindBndNode(curr,prev,rgns[k]))!=start
                      && ElleNodeIsDouble(nb)) {
                  ElleSetNodeAttribute(nb, val, attrib_id);
                  prev = curr;
                  curr = nb;
                  cnt++;
              }
            }
            k++;
        }
    }
    else err=1;
    return(err);
}

void RecalculateTripleSwitchCONC(int node1,int node2,int node3,
                                    int node4,int node5,int node6)
{
    int i,k;
    int maxa=0, *active=0, nbs[3];
    double conc, newconc;
    double m1=0.0, m2=0.0, m4=0.0, m5=0.0;
    double sep14, sep15, sep24, sep25, len=0.0, newlen=0.0;

    ElleNodeAttributeList(&active,&maxa);
    len = ElleNodeSeparation(node1,node2);
    len += ElleNodeSeparation(node1,node3);
    sep15 = ElleNodeSeparation(node1,node5);
    newlen = len + sep15;
    sep14 = ElleNodeSeparation(node1,node4);
    len += sep14;
    for (k=0;k<maxa;k++) {
        if (ElleNodeCONCAttribute(active[k])) {
            conc = ElleNodeAttribute(node1,active[k]);
            newconc = conc*len/newlen;
            ElleSetNodeAttribute(node1,newconc,active[k]);
        }
    }
    len = ElleNodeSeparation(node2,node1);
    len += ElleNodeSeparation(node2,node6);
    sep24 = ElleNodeSeparation(node2,node4);
    newlen = len + sep24;
    sep25 = ElleNodeSeparation(node2,node5);
    len += sep25;
    for (k=0;k<maxa;k++) {
        if (ElleNodeCONCAttribute(active[k])) {
            conc = ElleNodeAttribute(node2,active[k]);
            newconc = conc*len/newlen;
            ElleSetNodeAttribute(node2,newconc,active[k]);
        }
    }
    ElleNeighbourNodes(node4,nbs);
    newlen = 0.0;
    for (i=0;i<3;i++)
        if (nbs[i]!=NO_NB) newlen += ElleNodeSeparation(node4,nbs[i]);
    len = newlen - sep24 + sep14;
    for (k=0;k<maxa;k++) {
        if (ElleNodeCONCAttribute(active[k])) {
            conc = ElleNodeAttribute(node4,active[k]);
            newconc = conc*len/newlen;
            ElleSetNodeAttribute(node4,newconc,active[k]);
        }
    }
    ElleNeighbourNodes(node5,nbs);
    newlen = 0.0;
    for (i=0;i<3;i++)
        if (nbs[i]!=NO_NB) newlen += ElleNodeSeparation(node5,nbs[i]);
    len = newlen - sep15 + sep25;
    for (k=0;k<maxa;k++) {
        if (ElleNodeCONCAttribute(active[k])) {
            conc = ElleNodeAttribute(node5,active[k]);
            newconc = conc*len/newlen;
            ElleSetNodeAttribute(node5,newconc,active[k]);
        }
    }
    if (active) free(active);
    
}

double ElleNodeMassToConc(int node, double mass, int attr_id)
{
    int i, nbs[3];
    double conc=0.0, len=0.0;

    if (ElleNodeIsActive(node)) {
        ElleNeighbourNodes(node,nbs);
        for (i=0;i<3;i++) {
            if (nbs[i]!=NO_NB)
                len += ElleNodeSeparation(node,nbs[i]);
        }
        conc = 2*mass/len;
        ElleSetNodeAttribute(node,conc,attr_id);
    }
    return(conc);
}

double ElleNodeConcToMass(int node, int attr_id)
{
    int i, nbs[3];
    double conc, len=0.0;

    if (ElleNodeIsActive(node)) {
        conc=ElleNodeAttribute(node,attr_id);
        ElleNeighbourNodes(node,nbs);
        for (i=0;i<3;i++) {
            if (nbs[i]!=NO_NB)
                len += ElleNodeSeparation(node,nbs[i]);
        }
    }
    return(conc*len*0.5);
}

// assumes conc is in Elle units
// and boundary segments are constant width (ElleBndWidth() is real units)
double ElleTotalNodeMass(int attr_id)
{
    int j, i, max, nbs[3];
    double conc, total=0.0, len=0.0, wdth=0.0;

    wdth=ElleBndWidth()/ElleUnitLength();
    if (ElleNodeAttributeActive(attr_id)) {
        max = ElleMaxNodes();
        for (j=0;j<max;j++) {
            if (ElleNodeIsActive(j)) {
                len=0.0;
                conc=ElleNodeAttribute(j,attr_id);
                ElleNeighbourNodes(j,nbs);
                for (i=0;i<3;i++) {
                    if (nbs[i]!=NO_NB)
                        len += ElleNodeSeparation(j,nbs[i]);
                }
                total += conc*len*wdth;
            }
        }
    }
    return(total*0.5);
}

int ElleDeleteDoubles()
{
    int i, err=0, max;

    max = ElleMaxNodes();
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i) && ElleNodeIsDouble(i)) {
           err = ElleDeleteDoubleJ(i);
        }
    }
    return(err);
}

int ElleAddDoubles()
{
    char msg[30];
    int i, j, err=0, nn, next, addnode, max, nbnodes[3];

    max = ElleMaxNodes();
    do {
        addnode=0;
        for (i=0;i<max;i++) {
            if (ElleNodeIsActive(i)) {
                ElleNeighbourNodes(i,nbnodes);
                for (j=0;j<3;j++) {
                    next = nbnodes[j];
                    if (next!=NO_NB && ElleGapTooLarge(i,next)) {
                        if (err = ElleInsertDoubleJ(i,next,&nn,0.5)) {
                            sprintf(msg,"%s %d %d","AddDoubles",i,next);
                            OnNodeError(i,msg,err);
                        }
                        addnode++;
                    }
                }
            }
        }
    } while (addnode>0);
    return(err);
}

void ResetRegionEntries(int start, int *prev, int end,
                        int old, int newrgn)
{
    int nb1, i, last, next;

    last = end;
    next = start;
    do {
        if ((nb1=ElleFindBndNode(next,*prev,old))==NO_NB)
            OnNodeError(next,"ResetRegionEntries",NONB_ERR);
        if ((i = ElleFindNbIndex(nb1,next))==NO_NB)
            OnNodeError(next,"ResetRegionEntries nb index",NONB_ERR);
        ElleSetRegionEntry(next,i,newrgn);
        *prev = next;
        next = nb1;
    } while (next!=last);
}

void ElleFindLocalBBox(Coords *centre, Coords *bbox, int factor)
{
    Coords xy;

    double len = EllemaxNodeSep() * factor;
    xy.x = centre->x - len;
    xy.y = centre->y - len;
    bbox[BASELEFT] = xy;
    xy.x = centre->x + len;
    xy.y = centre->y - len;
    bbox[BASERIGHT] = xy;
    xy.x = centre->x + len;
    xy.y = centre->y + len;
    bbox[TOPRIGHT] = xy;
    xy.x = centre->x - len;
    xy.y = centre->y + len;
    bbox[TOPLEFT] = xy;
}

void MoveTowardsPoint(Coords *start, Coords *dir, Coords *newxy, double len)
{
    double ang;

    ang = atan2( (dir->y-start->y), (dir->x-start->x) );
    newxy->x = start->x + len*cos(ang);
    newxy->y = start->y + len*sin(ang);
}

int ElleWriteNodeFile(char *filename)
{
    /* No longer used */
    int i,max;
    Coords pos;
    FILE *fp;

    max = ElleMaxNodes();
    if ((fp=fopen(filename,"w"))==NULL) return(OPEN_ERR);
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            ElleNodePosition(i,&pos); 
            fprintf(fp,"%d %f %f\n",i,pos.x, pos.y);
        }
    }
    fclose(fp);                            
    return(0);
}

void EllePrintNodes(FILE *fp)
{
    /* No longer used */
    int i,max;
    Coords current;

    max = ElleMaxNodes();
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            ElleNodePosition(i,&current);
            fprintf(fp,"%d %f %f\n",i,current.x,current.y);
        }
    }
}


void GetRay(int node1,int node2,int node3,double *ray,Coords *movedist);
#if XY
int MoveDoubleJ(int node1, Coords *incr);
int MoveTripleJ(int node1, Coords *incr);
    /* move to centre of 1-2 boundary and allow surface energy move */
    ElleNodePosition(node1,&xya);
    ElleNodePlotXY(node2,&xyb,&xya);
    xy_mp.x = xya.x - (xya.x-xyb.x)*0.5;
    xy_mp.y = xya.y - (xya.y-xyb.y)*0.5;
    ElleNodePrevPosition(node1,&xyold);
    ElleNodePrevPosition(node2,&xyold2);
    ElleSetPosition(node1,&xy_mp);
    ElleSetPosition(node2,&xy_mp);
    MoveTripleJ(node1,&increment);
    MoveTripleJ(node2,&increment2);
    ElleSetPrevPosition(node1,&xyold);
    ElleSetPrevPosition(node2,&xyold2);
    ElleUpdatePosition(node1,&increment);
    ElleUpdatePosition(node2,&increment2);
int MoveDoubleJ(int node1,Coords *incr)
{
    int i, nghbr[2], nbnodes[3], err;
    double maxV,gb_energy,ray,deltaT,vlen;
    double switchDist, speedUp;
    Coords xy1, movedist;

    switchDist = ElleSwitchdistance();
    speedUp = ElleSpeedup() * switchDist * switchDist * 0.02;
    maxV = ElleSwitchdistance()/5.0;
    /*
     * allows speedUp to be 1 in input file
     */
    gb_energy = speedUp;
    deltaT = 0.0;
    /*
     * find the node numbers of the neighbours
     */
    if (err=ElleNeighbourNodes(node1,nbnodes))
        OnError("MoveDoubleJ",err);
    i=0;
    while (i<3 && nbnodes[i]==NO_NB) i++;
    nghbr[0] = nbnodes[i]; i++;
    while (i<3 && nbnodes[i]==NO_NB) i++;
    nghbr[1] = nbnodes[i];

    GetRay(node1,nghbr[0],nghbr[1],&ray,&movedist);
    if (ray > 0.0) {
    /*if (ray > ElleSwitchdistance()/100.0) {*/
        vlen = gb_energy/ray;
        if (vlen > maxV) {
            vlen = maxV;
            deltaT = 1.0;
        }
        if (vlen>0.0) {
            movedist.x *= vlen;
            movedist.y *= vlen;
        }
        else {
            movedist.x = 0.0;
            movedist.y = 0.0;
        }
        /*TotalTime += deltaT;*/
        /*ElleUpdatePosition(node1,&movedist);*/
        incr->x = movedist.x;
        incr->y = movedist.y;
    }
    else {
        vlen = 0.0;
    }
}

int MoveTripleJ(int node1, Coords *incr)
{
    int i, nghbr[3], finished=0, err=0;
    double maxV,gb_energy[3],ray[3],deltaT,vlen[3],vlenTriple;
    double switchDist, speedUp;
    Coords xy1, movedist[3], movedistTriple;

    switchDist = ElleSwitchdistance();
    /*
     * allows speedUp to be 1 in input file
     */
    speedUp = ElleSpeedup() * switchDist * switchDist * 0.02;
    maxV = switchDist/5.0;
    for (i=0;i<3;i++) gb_energy[i] = speedUp;
    deltaT = 0.0;
    /*
     * find the node numbers of the neighbours
     */
    if (err=ElleNeighbourNodes(node1,nghbr))
        OnError("MoveTripleJ",err);

    GetRay(node1,nghbr[0],nghbr[1],&ray[0],&movedist[0]);
    GetRay(node1,nghbr[1],nghbr[2],&ray[1],&movedist[1]);
    GetRay(node1,nghbr[2],nghbr[0],&ray[2],&movedist[2]);
    for(i=0;i<3;i++) {
        if (ray[i] > 0.0) {
        /*if (ray[i] > switchDist/100.0) {*/
            vlen[i] = gb_energy[i]/ray[i];
            /*if (vlen[i] > maxV) vlen[i] = maxV;*/
        }
        else {
            vlen[i] = 0.0;
            finished = 1;
        }
    }
    if (!finished) {
        for(i=0;i<3;i++) {
            if (vlen[i] < maxV) {
                movedist[i].x *= vlen[i];
                movedist[i].y *= vlen[i];
            }
            else {
                movedist[i].x *= maxV;
                movedist[i].y *= maxV;
            }
        }
        movedistTriple.x = movedist[0].x+movedist[1].x+movedist[2].x;
        movedistTriple.y = movedist[0].y+movedist[1].y+movedist[2].y;
        vlenTriple = sqrt(movedistTriple.x*movedistTriple.x + 
                          movedistTriple.y*movedistTriple.y);
        if (vlenTriple > maxV) {
            vlenTriple = maxV/vlenTriple;
            movedistTriple.x *= vlenTriple;
            movedistTriple.y *= vlenTriple;
            deltaT = 1.0;
        }
        if (vlenTriple <= 0.0) movedistTriple.x = movedistTriple.y = 0.0;
          
        /*TotalTime += deltaT;*/
    }
    else {
        ElleNodePosition(node1,&xy1);
        ElleNodePlotXY(nghbr[0],&movedist[0],&xy1);
        ElleNodePlotXY(nghbr[1],&movedist[1],&xy1);
        ElleNodePlotXY(nghbr[2],&movedist[2],&xy1);
        for(i=0;i<3;i++) {
            movedist[i].x = movedist[i].x - xy1.x;
            movedist[i].y = movedist[i].y - xy1.y;
        }
        movedistTriple.x = (movedist[0].x+movedist[1].x+movedist[2].x)/2.0;
        movedistTriple.y = (movedist[0].y+movedist[1].y+movedist[2].y)/2.0;
#if XY
        vlenTriple = sqrt(movedistTriple.x*movedistTriple.x + 
                          movedistTriple.y*movedistTriple.y);
        if (vlenTriple > maxV) {
            vlenTriple = maxV/vlenTriple;
            movedistTriple.x *= vlenTriple;
            movedistTriple.y *= vlenTriple;
            deltaT = 1.0;
        }
#endif
    }
    ElleUpdatePosition(node1,&movedistTriple);
    incr->x = movedistTriple.x;
    incr->y = movedistTriple.y;
    
}
#endif

void GetRay(int node1,int node2,int node3,double *ray,Coords *movedist)
{
    double dx2,dy2,dx3,dy3,tmpx,tmpy;
    double k, x0,y0;
    double switchDist;
    double eps = 1e-8;
    double r;
    Coords xy1, xy2, xy3;

    switchDist = ElleSwitchdistance();
    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node2,&xy2,&xy1);
    ElleNodePlotXY(node3,&xy3,&xy1);
    dx2 = xy2.x - xy1.x;
    dy2 = xy2.y - xy1.y;
    dx3 = xy3.x - xy1.x;
    dy3 = xy3.y - xy1.y;
    if (dx2==0.0) {
        tmpx = dx2;
        tmpy = dy2;
        dx2 = dx3;
        dy2 = dy3;
        dx3 = tmpx;
        dy3 = tmpy;
    }
    *ray = 0.0;
    movedist->x = movedist->y = 0.0;

    if (dx2>eps || dx2<-eps) {
        k = 2.0 * dx3 * dy2/dx2 - 2.0*dy3;
        if (k!=0.0) {
            y0 = ((dx3/dx2)*(dx2*dx2 + dy2*dy2)-dx3*dx3-dy3*dy3)/k;
            x0 = (dx2*dx2+dy2*dy2 - 2.0*y0*dy2)/(2.0*dx2);
            r = sqrt((double)(x0*x0+y0*y0));
            if (r!=0.0) {
                *ray = r;
                movedist->x = x0/ *ray;
                movedist->y = y0/ *ray;
                if (*ray < switchDist/3.0) *ray = switchDist/3.0;
            }
        }
    }
}                

int ElleOrderNbsOnAngle(int node, int *nb)
{
    int i, j, node_nbs[3];
    double eps = 1e-6;
    vector<Coords> nb_coords;
    Coords nodexy, xy;

    ElleNeighbourNodes(node,node_nbs);
    ElleNodePosition(node, &nodexy);
    i=0;
    while (i<3 && node_nbs[i]==NO_NB) i++;
    while (i<3 && node_nbs[i]!=NO_NB) {
        ElleNodePlotXY(node_nbs[i], &xy, &nodexy);
        nb_coords.push_back(xy);
        i++;
    }
    sortCoordsOnAngle(nb_coords,&nodexy);
    for (i=0;i<3;i++) nb[i] = NO_NB;
    for (i=0;i<3;i++) {
        if (node_nbs[i]!=NO_NB) {
          ElleNodePlotXY(node_nbs[i], &xy, &nodexy);
          for (j=0;j<3;j++) {
            if (fabs(nb_coords[j].x-xy.x) < eps &&
              fabs(nb_coords[j].y-xy.y)<eps)
              nb[j] = node_nbs[i];
          }
        }
    }
}
