/*----------------------------------------------------------------
 *    Elle:   filedat.h  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/
#ifndef _S_filedat_h
#define _S_filedat_h
typedef struct {
	char fname[FILENAME_MAX];
	FILE *fp;
	int rec_req;
	int rec_curr;
	int ref_req;
	int ref_curr;
	int rec_max;
} file_data;

#endif
