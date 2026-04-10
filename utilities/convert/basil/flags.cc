#include <stdio.h>
#include <stdlib.h>
#include "flags.h"
#include "attrib.h"
#include "nodes.h"
#include "errnum.h"
#include "error.h"

typedef struct {
    int maxflags;
    int *elems;
} FlagArray;

FlagArray BoundaryFlags, CheckFlags;


int CnvtNodeCheck(int node)
{
    return(
        (CheckFlags.maxflags>node) ? CheckFlags.elems[node] : 0);
}

int CnvtNodeOnBoundary(int node)
{
    return(
        (BoundaryFlags.maxflags>node) ? BoundaryFlags.elems[node] : 0);
}

void CnvtClearCheckFlag(int index)
{
    int max, err;

    max = CnvtMaxCheckFlags();
    if (index >= max)
        OnError( "CnvtClearCheckFlag",NODENUM_ERR );
    CheckFlags.elems[index] = 0;
}

void CnvtClearBoundaryFlag(int index)
{
    int max, err;

    max = CnvtMaxBoundaryFlags();
    if (index >= max)
        OnError( "CnvtClearBoundaryFlag",NODENUM_ERR );
    BoundaryFlags.elems[index] = 0;
}

void CnvtSetCheckFlag(int index)
{
    int max, err;

    max = CnvtMaxCheckFlags();
    if (index >= max)  {
        max = (int)(1.5*index);
        if (err = CnvtReallocateCheckFlagArray(max))
            OnError( "Reallocating flag array",err );
    }
    CheckFlags.elems[index] = 1;
}

void CnvtSetBoundaryFlag(int index, int val)
{
    int max, err;

    max = CnvtMaxBoundaryFlags();
    if (index >= max)  {
        max = (int)(1.5*index);
        if (err = CnvtReallocateBoundaryFlagArray(max))
            OnError( "Reallocating flag array",err );
    }
    BoundaryFlags.elems[index] = val;
}

int CnvtMaxCheckFlags()
{
    return(CheckFlags.maxflags);
}

int CnvtMaxBoundaryFlags()
{
    return(BoundaryFlags.maxflags);
}

void CnvtInitBoundaryFlagArray()
{
    int max,err=0;

    BoundaryFlags.maxflags = 0;
    BoundaryFlags.elems = 0;
    max = ElleMaxNodes();
    if (err = CnvtReallocateBoundaryFlagArray(max))
        OnError( "Reallocating Flag array",err );
}

void CnvtInitCheckFlagArray()
{
    int max,err=0;

    CheckFlags.maxflags = 0;
    CheckFlags.elems = 0;
    max = ElleMaxNodes();
    if (err = CnvtReallocateCheckFlagArray(max))
        OnError( "Reallocating Flag array",err );
}

int CnvtReallocateCheckFlagArray(int size)
{
    int oldsize, i;
    int err=0;
    /*
     * reallocate memory for the flag array
     */
    oldsize = CheckFlags.maxflags;
    if ((CheckFlags.elems = (int *)realloc(CheckFlags.elems,
                                        size*sizeof(int))) ==0)
         return( MALLOC_ERR );
    CheckFlags.maxflags = size;
    for (i=oldsize;i<size;i++) CheckFlags.elems[i]=0;
    return( 0 );
}

int CnvtReallocateBoundaryFlagArray(int size)
{
    int oldsize, i;
    int err=0;
    /*
     * reallocate memory for the flag array
     */
    oldsize = BoundaryFlags.maxflags;
    if ((BoundaryFlags.elems = (int *)realloc(BoundaryFlags.elems,
                                        size*sizeof(int))) ==0)
         return( MALLOC_ERR );
    BoundaryFlags.maxflags = size;
    for (i=oldsize;i<size;i++) BoundaryFlags.elems[i]=0;
    return( 0 );
}

void CnvtClearCheckFlagArray()
{
    if (CheckFlags.elems) free(CheckFlags.elems);
}

void CnvtClearBoundaryFlagArray()
{
    if (BoundaryFlags.elems) free(BoundaryFlags.elems);
}
