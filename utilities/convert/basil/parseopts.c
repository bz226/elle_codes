/*--------------------------------------------------------------------
 *    Basil / Sybil:   parseopts.c  1.0  13 May 1997
 *
 *    Copyright (c) 1997 by G.A. Houseman, T.D. Barr, & L.A. Evans
 *    See README file for copying and redistribution conditions.
 *--------------------------------------------------------------------*/

/*
 * parses command line options and sets the values
 * of the passed variables
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "errnum.h"

static  char *  CommandOptions= "i:o:r:h";

void help_message();
int ParseOptions();
int Getopt();

int ParseOptions(int argc,char **argv,char **in,char **out,int *record)
{
    register int c;
    extern int optindex;
    extern char *optstr;
    char *val;

    optindex= 1;

    while ((c = Getopt(argc, argv, CommandOptions)) != EOF)
        switch (c) {
        case 'h':
            help_message(); 
            return(HELP_ERR);
        case 'i':   
            *in = argv[optindex-1];
            /*strcpy(infile,argv[optindex-1]);*/
            break;
        case 'o':   
            *out = argv[optindex-1];
            /*strcpy(outfile,argv[optindex-1]);*/
            break;
        case 'r':   
            *record = atoi(argv[optindex-1]);
            /*strcpy(outfile,argv[optindex-1]);*/
            break;
        default:
            help_message(); 
            return(SYNTAX_ERR);
            break;
        }
    return(0);
}

/*
 * Getopt - get option letter from argv
 * by Henry Spencer
 * posted to Usenet net.sources list
 */
char    *optstr;        /* Global argument pointer. */
int     optindex = 0;     /* Global argv index. */

static char     *scan = NULL;   /* Private scan pointer. */

int Getopt(argc, argv, optstring)
int  argc;
char **argv;
char *optstring;
{
    register char   c;
    register char  *place;

    optstr = NULL;

    if (scan == NULL || *scan == '\0') {

        if (optindex >= argc || argv[optindex][0] != '-' || argv[optindex][1] == '\0')
            return (EOF);
        if (strcmp (argv[optindex], "--") == 0) {
            optindex++;
            return (EOF);
        }
        scan = argv[optindex] + 1;
        optindex++;
    }
    c = *scan++;
    place = strchr(optstring, c);

    if (place == NULL || c == ':') {
        fprintf (stderr, "\n%s: unknown option -%c\n", argv[0], c);
        return ('h');
    }
    place++;
    if (*place == ':') {
        if (*scan != '\0') {
            optstr = scan;
            scan = NULL;
        }
        else {
            optstr = argv[optindex];
            optindex++;
        }
    }
    return (c);
}

void help_message()
{
        fprintf(stderr, "options:\n" );
        fprintf(stderr, "   -i   name of input file (basil solution file)\n" );
        fprintf(stderr, "   -o   name for Elle output file\n" );
        fprintf(stderr, "   -r   record number for input file [1]\n" );
        fprintf(stderr, "   -h   Print this message\n" );
        fprintf(stderr,
         "Example:  basil2elle -i FD.sols/basout -r 2 -o res.elle\n" );
}
