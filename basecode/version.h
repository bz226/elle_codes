 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_version_h
#define _E_version_h
#include <string>

std::string ElleGetLibVersionString();
std::string ElleGetLocalTimeString();
std::string ElleGetCreationString();
const char *ElleGetCreationCString(void);

#endif
