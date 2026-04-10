/*----------------------------------------------------------------
 *    Elle:   defines.h  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/
#ifndef _defines_h
#define _defines_h
#ifdef XSYB
#define fontdat XFontStruct *
#define colourdat XColor
#define BLACK_PEN  0
#else
#define fontdat int
#define colourdat int
#define BLACK_PEN  1
#define XPointer int*
#endif
#endif
