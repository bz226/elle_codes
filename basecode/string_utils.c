/*----------------------------------------------------------------
 *    Elle:   string_utils.c  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/
#include <stdio.h>
#include <string.h>
#include "file.h"
#include "string_utils.h" /* sort this out, valid_terms def */

int validate( char *str,int *key,valid_terms *keywords )
{
    int err=0;

    if (str[0]=='#') *key = COMMENT;
    else *key = name_match(str,keywords);
    return( err );
}

/*
 * searches the given set of valid_terms and returns
 * the associated id if a name match is found
 * else returns -1
 */
int name_match( char *str,valid_terms *set )
{
    unsigned char found;
    int i=0;

    found=0;
    while (set[i].name && !found) {
        if (!strcmp(str,set[i].name)) found=1;
        else i++;
    }
    return( found ? set[i].id : -1 );
}

/*
 * If id matches a member of set,
 * copy the corresponding name to the string name and return 1
 * else name has length zero and return 0
 */
int id_match(valid_terms set[],int id,char *name)
{
    int found=0;
    int i=0;

    strcpy(name,"");
    while (set[i].name && !found) {
        if (id==set[i].id) found=1;
        else i++;
    }
    if (found) strcpy(name,set[i].name);
    return(found);
}
