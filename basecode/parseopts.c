/*----------------------------------------------------------------
 *    Elle:   parseopts.c  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runopts.h"
#include "string_utils.h"
#include "errnum.h"
#include "parseopts.h"
#include "log.h"

#ifdef __cplusplus
extern "C" {
#endif
extern int ValidColmapOption(char *);
#ifdef __cplusplus
}
#endif

static  char *  CommandOptions= "i:o:s:f:h:v:d:e:c:u:n";

int Getopt(int argc, char **argv, char *optstring);

/*!
 * Filenames cannot start with '-'
 * -s and -f options cannot be followed by -ve numbers
 */
int ParseOptions(int argc,char **argv)
{
    register int c;
    int i, itmp, err=0;
	double dtmp;
    UserData udata;
    extern int optindx;

    optindx= 1;

    ElleSetAppName(argv[0]);

    while (!err && (c = Getopt(argc, argv, CommandOptions)) != EOF)
        switch (c) {
        /*
         * this should be phased out - set in showelle.in file
         * rather than via command line
         */
#ifndef NODISP
/*
        case 'c':   
            if (optindx<=argc) {
                if ((i=ValidColmapOption(argv[optindx-1]))==-1)
                    fprintf (stderr,
                         "\n%s: ignoring unknown colourmap option %s\n",
                                           argv[0], argv[optindx-1]);
                else SetColourmapOption(i);
            }
            break;
*/
#endif
        case 'd':   
            if (optindx<=argc) {
				if (argv[optindx-1][0]!='-')
                	ElleSetDspOptsFile(argv[optindx-1]);
				else optindx--;
			}
            break;
        case 'e':   
            if (optindx<=argc ) {
				if (argv[optindx-1][0]!='-')
                	ElleSetExtraFile(argv[optindx-1]);
				else optindx--;
			}
            break;
        case 'f':   
            if (optindx<=argc) {
				if (argv[optindx-1][0]!='-') {
					if (sscanf(argv[optindx-1],"%d",&itmp)==1)
	                	ElleSetSaveFrequency(itmp);
					else err=SYNTAX_ERR;
				}
				else optindx--;
			}
            break;
        case 'h':
            /*ElleDisplayHelpMsg();*/
            err=HELP_ERR;
            break;
        case 'i':   
            if (optindx<=argc ) {
				if (argv[optindx-1][0]!='-')
                	ElleSetFile(argv[optindx-1]);
				else optindx--;
			}
            break;
        case 'n':   
            ElleSetDisplay(0);
            break;
        case 'o':   
            if (optindx<=argc ) {
				if (argv[optindx-1][0]!='-')
                	ElleSetOutFile(argv[optindx-1]);
				else optindx--;
			}
            break;
        case 's':   
            if (optindx<=argc) {
				if (argv[optindx-1][0]!='-') {
					if (sscanf(argv[optindx-1],"%d",&itmp)==1)
	                	ElleSetStages(itmp);
					else err=SYNTAX_ERR;
				}
				else optindx--;
			}
            break;
        case 'u':
            ElleUserData(udata);
            i=0;
            while (optindx<=argc && i<MAX_U_DATA &&
                   (sscanf(argv[optindx-1],"%lf",&udata[i])==1)) {
                optindx++;
                i++;
            }
            if (i>MAX_U_DATA) return(RANGE_ERR);
            ElleSetUserData(udata);
            optindx--;
            break;
        case 'v':   
            if (optindx<=argc) {
                if (sscanf(argv[optindx-1],"%lf",&dtmp)==1)
                	ElleSetBndVelocity(dtmp);
				else optindx--;
			}
            break;
        default:
            /*ElleDisplayHelpMsg();*/
            err=SYNTAX_ERR;
            break;
        }
    return(err);
}

/*
 * Getopt - get option letter from argv
 * by Henry Spencer
 * posted to Usenet net.sources list
 */
char    *optargv;        /* Global argument pointer. */
int     optindx = 0;     /* Global argv index. */

static char     *scan = NULL;   /* Private scan pointer. */

int Getopt(int argc, char **argv, char *optstring)
{
    register char   c;
    register char  *place;

    optargv = NULL;

    if (scan == NULL || *scan == '\0') {

        if (optindx >= argc || argv[optindx][0] != '-' ||
                                   argv[optindx][1] == '\0')
            return (EOF);
        if (strcmp (argv[optindx], "--") == 0) {
            optindx++;
            return (EOF);
        }
        scan = argv[optindx] + 1;
        optindx++;
    }
    c = *scan++;
    place = strchr(optstring, c);

    if (place == NULL || c == ':') {
        sprintf (logbuf, "\n%s: unknown option -%c\n", argv[0], c);
        Log(2,logbuf);
        return ('h');
    }
    place++;
    if (*place == ':') {
        if (*scan != '\0') {
            optargv = scan;
            scan = NULL;
        }
        else {
            optargv = argv[optindx];
            optindx++;
        }
    }
    return (c);
}

