 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_file_utils_h
#define _E_file_utils_h

#include <stdio.h>
#include "string_utils.h"

#ifdef __WIN32__
const char E_DIR_SEPARATOR='\\';
#else
const char E_DIR_SEPARATOR='/';
#endif

#ifdef __cplusplus
extern "C" {
#endif
int ElleSkipLine(FILE *fp,char *str);
int ElleSkipSection(FILE *fp,char *str,valid_terms keyterms[]);
int ElleCopyLine(FILE *fp, FILE *fpout, char *str);
int ElleCopySection(FILE *fp, FILE *fpout, char *str,
                valid_terms keyterms[]);
int FileExists(char *filename);
#ifdef __cplusplus
}
#endif
#endif
