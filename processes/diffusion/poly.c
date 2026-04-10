/*****************************************************************************/
/*                                                                           */
/*  (diffusion.c)                                                            */
/*     routine to diffuse some attribute across a delauney mesh given        */
/*        a dimensionless diffusion constant and time step                   */
/*                                                                           */
/*****************************************************************************/

/* If SINGLE is defined when triangle.o is compiled, it should also be       */
/*   defined here.  If not, it should not be defined here.                   */

/* #define SINGLE */

#ifdef SINGLE
#define REAL float
#else /* not SINGLE */
#define REAL double
#endif /* not SINGLE */

#include <stdio.h>
#include <string.h>
#include "poly.h"

#ifndef _STDLIB_H_
extern void *malloc();
extern void free();
#endif /* _STDLIB_H_ */

#ifdef SUN
int attribute_set(infile,key,max,rgnattribs,dflt)
char *infile;
const char *key;
int max;
double *val, *dflt;
#else
int attribute_set(char *infile,const char *key,int max,
                  double *rgnattribs, double *dflt)
#endif
{
    FILE *fp=0;
    char buf[81], str[32];
    int found = 0, finished=0, i;
    double val;

    if((fp=fopen(infile,"r"))==NULL) {
        fprintf(stderr,"cannot open file %s\n",infile);
        return(1);
    }
    /*
     * find keyword
     */
    while (!found && !feof(fp)) {
        fgets(buf,80,fp);
        if (isalpha(buf[0])) {
            sscanf(buf,"%s",str);
            if (!strcmp(str,key)) found = 1;
        }
    }
    if (found) {
    /*
     * set default value
     */
        fgets(buf,80,fp);
        if (isalpha(buf[0])) {
            sscanf(buf,"%s",str);
            if (!strcmp(str,DEFAULT)) {
                sscanf(&buf[strlen(DEFAULT)],"%lf",dflt);
                for (i=0;i<max;i++) rgnattribs[i] = *dflt;
                fgets(buf,80,fp);
            }
            else finished = 1;
        }
        while (!(finished=(!isdigit(buf[0]))) && !feof(fp)) {
            sscanf(buf,"%d %lf",&i,&val);
            if (i<max) rgnattribs[i] = val;
            else fprintf(stderr,"Invalid region id %d\n",i);
            fgets(buf,80,fp);
        }
    }
    if (fp) fclose(fp);
    return(found);
}
