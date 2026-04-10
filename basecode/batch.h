 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_display_h
#define _E_display_h
/* matches def in menus.c */
#define GRAINS    12
#define SUBGRAINS 13
#define UNITS     14
#define TRIANGLES 15

#ifdef __cplusplus
extern "C" {
#endif
void ElleUpdateDisplay();
void ElleUpdateSettings();

#ifdef __cplusplus
}
#endif
#endif
