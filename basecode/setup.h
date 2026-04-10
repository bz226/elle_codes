/*----------------------------------------------------------------
 *    Elle:   setup.h  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/
#ifndef _E_setup_h
#define _E_setup_h

#ifdef __cplusplus
extern "C" {
#endif

int StartApp();
int Run_App(FILE *);
int SetupApp(int, char **);
void Init_Data();
#ifdef __cplusplus
}
#endif
#endif
