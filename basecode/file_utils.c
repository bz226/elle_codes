 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <sys/types.h>
#include <sys/stat.h>
#include "file_utils.h"
#include "string_utils.h"
#include "errnum.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

int ElleSkipLine(FILE *fp, char *str)
{
    int c;

    c = getc(fp);
     /*skip the rest of the line */
    while (c!='\n' && c!=EOF) {
        c = getc(fp);
    }
    return(0);
}

int ElleCopyLine(FILE *fp, FILE *fpout, char *str)
{
    int c;

        /* write the first word */
    fprintf(fpout,"%s",str);
    c = getc(fp);
     /*write the rest of the line */
    while (c!='\n' && c!=EOF) {
        putc(c,fpout);
        c = getc(fp);
    }
    if (c!=EOF) {
        if (putc(c,fpout)==EOF) return(WRITE_ERR); /* write the \n */
        else return(0);
    }
    return(0);
}

int ElleSkipSection(FILE *fp, char *str,valid_terms keyterms[])
{
    int key, finished = 0;
    int err=0,num;

    /*
     * skip over input until eof or next keyword
     */
    while (!feof(fp) && !finished ) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp)) return(READ_ERR);
        if (!feof(fp)) {
            validate(str,&key,keyterms);
            if (key<0) {
                if (err = ElleSkipLine(fp,str)) return(err);
            }
            else finished = 1;
        }
    }
    return(0);
}

int ElleCopySection(FILE *fp, FILE *fpout, char *str,
                valid_terms keyterms[])
{
    int err, num, key, finished=0;

    /* write the keyword */
    if (err = ElleCopyLine(fp,fpout,str)) return(err);
        /*
         * copy til next keyword
         */
    while (!feof(fp) && !finished ) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        if (!feof(fp)) {
            validate(str,&key,keyterms);
            if (key<0) {
                if (err = ElleCopyLine(fp,fpout,str)) return(err);
            }
            else finished = 1;
        }
    }
    return(0);
}

int FileExists(char *filename)
{
    struct stat status;

    if (stat(filename,&status) == 0) return(1);
    return(0);
}
