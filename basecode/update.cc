 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include "update.h"
#include "display.h"
#include "runopts.h"
#include "file.h"
/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

int ElleUpdate()
{
    ElleIncrementCount();
    if (ElleDisplay()) ElleUpdateDisplay();
	/*ElleCheckFiles();*/
}
