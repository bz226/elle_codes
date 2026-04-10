 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nodes.h"
#include "bflynns.h"
#include "runopts.h"
#include "error.h"
#include "file.h"
#include "convert.h"
#include "general.h"
#include "interface.h"
#include "log.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/
int Region_ver=0;
#define XY  0

void ElleCleanArrays(void)
{

    ElleCleanFlynnArray();
    ElleCleanNodeArray();
    ElleCleanUnodeArray();
}

void ElleRemoveArrays(void)
{
    /*ElleRemoveGrainArray();*/
    ElleRemoveNodeArray();
}

int ElleWriteFlynnData(FILE *fp)
{
    char label[20];
    int i,j,max,num,err=0, parents=0;
    int *node_ids,num_nodes;

    max = ElleMaxFlynns();

    /*
     * write the lowest level flynns (defined by nodes)
     */
    if (!id_match(FileKeys,FLYNNS,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    for (i=0;i<max && !err;i++) {
        if (ElleFlynnIsActive(i) && !isParent(i)) {
            ElleFlynnNodes(i, &node_ids, &num_nodes);
            if ((num=fprintf(fp,"%d %d",i,num_nodes))<0)
                   err==WRITE_ERR;
            else {
                j=0;
                do {
                    if ((num=fprintf(fp," %d",node_ids[j]))<0)
                        err=WRITE_ERR;
                    j++;
                } while (j<num_nodes && !err);
                fprintf(fp,"\n");
            }
            if (node_ids) free(node_ids);
        }
    }
    for (i=0;i<max && !parents;i++) 
        if (ElleFlynnIsActive(i) && isParent(i)) parents=1;
    /*
     * write the higher level flynns (defined by other flynns)
     */
    if (!err && parents) err = ElleWriteParentData(fp);
    return(err);
}

int ElleReadFlynnData(FILE *fp, char str[])
{
    int i, err=0, numnodes, nn, num;
    int last, first;
    int id,curr;
    int key, s, u;
    ERegion full_id;


    /*
     * read old elle files as well
     */
    validate(str,&key,FileKeys);
    Region_ver = (key==REGIONS);
    if (Region_ver) {
        sprintf( logbuf, "Reading old file format\n" );
        Log( 2, logbuf );
    }
    curr = NO_NB;
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1) return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((full_id = id = atoi(str))==MAX_INDX)
                return(MAXINDX_ERR);
            if (curr != id) {
                ElleSetFlynnActive(id);/* fixes id>MaxGrains */
                curr = id;
            }
            if (Region_ver)
                if ((num = fscanf(fp,"%d %d",&s,&u))!=2)
                    return(READ_ERR);
            if ((num = fscanf(fp,"%d", &numnodes))!=1) return(READ_ERR);
            if (numnodes<=2) return(NODENUM_ERR);
            if ((num = fscanf(fp,"%d", &nn))!=1) return(READ_ERR);
            first = nn;
            ElleSetNodeActive(nn);
            ElleSetFlynnFirstNode(id,nn);
            for (i=1;i<numnodes;i++) {
               last = nn;
               if ((num = fscanf(fp,"%d", &nn))!=1) return(READ_ERR);
               ElleSetNodeActive(nn);
               if (err=ElleSetNeighbour(last,NO_NB,nn,&full_id))
                   return(err);
            }
            if (err=ElleSetNeighbour(nn,NO_NB,first,&full_id))
                return(err);
            num=fscanf(fp,"\n");
        }
    }
    return(err);
}

int SaveZIPFlynnData( gzFile in )
{
  char label[20];
  int i, j, max, num, err = 0, parents = 0;
  int * node_ids, num_nodes;
                                                                                     
  max = ElleMaxFlynns();
                                                                                     
  /* * write the lowest level flynns (defined by nodes) */
  if ( !id_match( FileKeys, FLYNNS, label ) ) return ( KEY_ERR );
  gzprintf( in, "%s\n", label );
  for ( i = 0; i < max && !err; i++ )
  {
    if ( ElleFlynnIsActive( i ) && !isParent( i ) )
    {
      ElleFlynnNodes( i, & node_ids, & num_nodes );
      if ( ( num = gzprintf( in, "%d %d", i, num_nodes ) ) < 0 )
        err == WRITE_ERR;
      else
      {
        j = 0;
        do
        {
          if ( ( num = gzprintf( in, " %d", node_ids[j] ) ) < 0 )
            err = WRITE_ERR;
          j++;
        }
        while ( j < num_nodes && !err );
        gzprintf( in, "\n" );
      }
      if ( node_ids ) free( node_ids );
    }
  }
  for ( i = 0; i < max && !parents; i++ )
    if ( ElleFlynnIsActive( i ) && isParent( i ) ) parents = 1;
  /* * write the higher level flynns (defined by other flynns) */
  if ( !err && parents ) err = SaveZIPParentData( in );
  return ( err );
}
                                                                                     
int LoadZIPFlynnData( gzFile in, char str[] )
{
  int i, err = 0, numnodes, nn, num;
  int last, first;
  int id, curr;
  int key, s, u;
  ERegion full_id;
                                                                                     
                                                                                     
  /* * read old elle files as well */
  validate( str, & key, FileKeys );
  Region_ver = ( key == REGIONS );
  if (Region_ver) {
      sprintf( logbuf, "Reading old file format\n" );
      Log( 2, logbuf );
  }
  curr = NO_NB;
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( full_id = id = atoi( str ) ) == MAX_INDX )
        return ( MAXINDX_ERR );
      if ( curr != id )
      {
        ElleSetFlynnActive( id );
        /* fixes id>MaxGrains */
        curr = id;
      }
      if ( Region_ver )
      {
        s = atoi( ( gzReadSingleString( in ) ).c_str() );
        u = atoi( ( gzReadSingleString( in ) ).c_str() );
      }
      numnodes = atoi( ( gzReadSingleString( in ) ).c_str() );
                                                                                     
      if ( numnodes <= 2 ) return ( NODENUM_ERR );
      nn = atoi( ( gzReadSingleString( in ) ).c_str() );
      first = nn;
      ElleSetNodeActive( nn );
      ElleSetFlynnFirstNode( id, nn );
      for ( i = 1; i < numnodes; i++ )
      {
        last = nn;
        nn = atoi( ( gzReadSingleString( in ) ).c_str() );
        ElleSetNodeActive( nn );
        if ( err = ElleSetNeighbour( last, NO_NB, nn, & full_id ) )
          return ( err );
      }
      if ( err = ElleSetNeighbour( nn, NO_NB, first, & full_id ) )
        return ( err );
      //       fscanf(fp,"\n");
    }
  }
  return ( err );
}
                                                                                     
int LoadZIPParentData( gzFile in, char str[] )
{
  int i, err = 0, numkids, nn, num;
  int curr;
                                                                                     
  curr = NO_NB;
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
                                                                                     
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      if ( ( curr = atoi( str ) ) == MAX_INDX )
        return ( MAXINDX_ERR );
      /* if ((num = fscanf(fp,"%d", &first))!=1) return(READ_ERR); */
      numkids = atoi( ( gzReadSingleString( in ) ).c_str() );
      ElleSetFlynnActive( curr );
      /* fixes id>MaxGrains */
      /* ElleSetFlynnFirstNode(curr,first); */
      for ( i = 0; i < numkids; i++ )
      {
        nn = atoi( ( gzReadSingleString( in ) ).c_str() );
        ElleAddFlynnChild( curr, nn );
      }
      //       fscanf(fp,"\n");
    }
  }
  return ( err );
}
                                                                                     
int SaveZIPParentData( gzFile in )
{
  char label[20];
  int i, max, err = 0;
                                                                                     
  max = ElleMaxFlynns();
                                                                                     
  if ( !id_match( FileKeys, PARENTS, label ) ) return ( KEY_ERR );
  gzprintf( in, "%s\n", label );
  for ( i = 0; i < max && !err; i++ )
  {
    if ( ElleFlynnIsActive( i ) && isParent( i ) )
    {
      err = SaveZIPParentFlynn( i, in );
      gzprintf( in, "\n" );
    }
  }
  return ( err );
                                                                                     
}

int ElleReadParentData(FILE *fp, char str[])
{
    int i, err=0, numkids, nn, num;
    int curr;

    curr = NO_NB;
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1) return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((curr = atoi(str))==MAX_INDX)
                return(MAXINDX_ERR);
            /*if ((num = fscanf(fp,"%d", &first))!=1) return(READ_ERR);*/
            if ((num = fscanf(fp,"%d", &numkids))!=1) return(READ_ERR);
            ElleSetFlynnActive(curr);/* fixes id>MaxGrains */
            /*ElleSetFlynnFirstNode(curr,first);*/
            for (i=0;i<numkids;i++) {
               if ((num = fscanf(fp,"%d", &nn))!=1) return(READ_ERR);
               ElleAddFlynnChild(curr,nn);
            }
            num=fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleWriteParentData(FILE *fp)
{
    char label[20];
    int i,max,err=0;

    max = ElleMaxFlynns();

    if (!id_match(FileKeys,PARENTS,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    for (i=0;i<max && !err;i++) {
        if (ElleFlynnIsActive(i) && isParent(i)) {
            err = ElleWriteParentFlynn(i, fp);
            fprintf(fp,"\n");
        }
    }
    return(err);
}

int ElleCheckForTwoSidedGrain(int node, int *nb)
{
    int found=0, nbnodes[3];

    *nb = NO_NB;
    ElleNeighbourNodes(node,nbnodes);
    if (nbnodes[0]==nbnodes[1] || nbnodes[0]==nbnodes[2])
            *nb = nbnodes[0];
    else if (nbnodes[1]==nbnodes[2])
            *nb = nbnodes[1];
    if (*nb != NO_NB) found = 1; /* 2-sided grain */
    return(found);
}


int ElleNewGrain(int first, int last, ERegion old, ERegion *rnew)
{
    int i,j,n,prev;
    int err=0;
    ERegion rgn;

    *rnew = ElleFindSpareFlynn();
    ElleSetFlynnFirstNode(*rnew,first);
    rgn = *rnew;
    prev = last;
    n = first;
    do {
        if ((j=ElleNewFindBndNb(n,prev,old))==NO_NB)
            OnNodeError(n,"New grain",NONB_ERR);
        if ((i = ElleFindNbIndex(j,n))==NO_NB)
            OnNodeError(n,"New flynn nb index",NONB_ERR);
        ElleSetRegionEntry(n,i,rgn);
        prev = n;
        n = j;
    } while (n!= last);

    if (ElleNodeIsDouble(first) &&
        ElleNodeIsDouble(last) && ElleFindNbIndex(first,last)==NO_NB) {
        err = ElleJoinTwoDoubleJ(first,last,old,*rnew);
    }
    else {
        if ((i = ElleFindNbIndex(first,last))==NO_NB)
            OnNodeError(first,"New unit nb index",NONB_ERR);
        ElleSetRegionEntry(last,i,*rnew);
    }
    ElleSetFlynnFirstNode(old, first);
    ElleCopyFlynnAttributes(old,*rnew);

    return(err);
}

void ElleNucleateGrainFromNode( int node, ERegion *new_rgn )
{
    int i,nbnodes[3],nn, nb[3];
    double ang;
    Coords xy, xy0, xy1, midpt;
    ERegion rgn;

    ElleNeighbourNodes(node,nbnodes);
    /*
     * ensure all neighbours are double nodes
     */
    for (i=0;i<3;i++) {
        if (nbnodes[i]!=NO_NB && ElleNodeIsTriple(nbnodes[i])) {
            ElleInsertDoubleJ(node,nbnodes[i],&nn,0.5);
            nbnodes[i] = nn;
        }
    }
    ElleNeighbourRegion(nbnodes[0],node,&rgn);
    if ((nb[0] = ElleFindBndNb(node, rgn))==NO_NB)
        OnNodeError(node,"ElleNucleateTripleJ - Bnd",0);
    if (ElleNodeIsTriple(node)) {
        ElleNeighbourRegion(nbnodes[0],node,&rgn);
        if ((nb[0] = ElleFindBndNb(node, rgn))==NO_NB)
            OnNodeError(node,"ElleNucleateTripleJ - Bnd",0);
        /*
         * form new grain
         */
        ElleNewGrain(nbnodes[0],nb[0],rgn,new_rgn);
        /*
         * move TJ to position of 3rd neighbour and delete that nb
         */
        if (nbnodes[1]==nb[0]) nb[0] = nbnodes[2];
        else nb[0] = nbnodes[1];
        ElleNodePosition(nb[0],&xy);
        ElleSetPosition(node,&xy);
        ElleDeleteDoubleJ(nb[0]);
    }
    else if (ElleNodeIsDouble(node)) {
        i=1;
        if (nbnodes[0]!=NO_NB && nbnodes[0]!=nb[0]) nb[1]=nbnodes[0];
        else if (nbnodes[1]!=NO_NB) nb[1] = nbnodes[1];
        else nb[1] = nbnodes[2];
        ElleNeighbourRegion(nb[0],node,&rgn);
        ElleNewGrain(nb[0],nb[1],rgn,new_rgn);
        ElleInsertDoubleJ(nb[0],nb[1],&nn,0.5);
        ElleNodePosition(node,&xy);
        ElleNodePosition(nb[0],&xy0);
        ElleNodePlotXY(nb[1],&xy1,&xy0);
        ang = M_PI*0.5*0.333;
        midpt.x = xy0.x - (xy0.x - xy1.x)*0.5;
        midpt.y = xy0.y - (xy0.y - xy1.y)*0.5;
        xy1.x = (midpt.x-xy0.x) * cos(ang) +
                          (midpt.y-xy0.y) * sin(ang);
        xy1.y = -(midpt.x-xy0.x) * sin(ang) +
                          (midpt.y-xy0.y) * cos(ang);
        xy1.x += xy0.x; xy1.y += xy0.y;
        ElleSetPosition(node,&xy1);
        xy1.x = (midpt.x-xy0.x) * cos(-ang) +
                          (midpt.y-xy0.y) * sin(-ang);
        xy1.y = -(midpt.x-xy0.x) * sin(-ang) +
                          (midpt.y-xy0.y) * cos(-ang);
        xy1.x += xy0.x; xy1.y += xy0.y;
        ElleSetPosition(nn,&xy1);
    }
}

