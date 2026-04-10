 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include "attrib.h"
#include "versiondef.h"
#include "version.h"
#include "timefn.h"
#include "runopts.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/
std::string ElleGetLibVersionString()
{
	std::string vers = Version_num;
    vers += Patch_level;
    return(vers);
}

std::string ElleGetLocalTimeString()
{
	std::string x;
    x += GetLocalTime();
    return(x);
}

std::string ElleGetCreationString()
{
	std::string x = "# Created by ";
    x += ElleAppName();
    x += ": elle version ";
    x += ElleGetLibVersionString();
    x += "  ";
    x += GetLocalTime();
    return(x);
}

const char *ElleGetCreationCString(void)
{
	std::string x = "# Created by ";
    x += ElleAppName();
    x += ": elle version ";
    x += ElleGetLibVersionString();
    x += "  ";
    x += GetLocalTime();
    return(x.c_str());
}
