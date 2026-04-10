 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_error_h
#define _E_error_h

#ifndef _E_errnum_h
#include "errnum.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif
void OnError(const char *message,int err_num);
void CleanUp(void);
#ifdef __cplusplus
}
#endif
#endif
