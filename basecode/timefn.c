 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <time.h>
#include "timefn.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

char *GetLocalTime(void)
{
    struct tm *tmval;
    time_t current;

    current = time(NULL);
    tmval = localtime(&current);
    return(asctime(tmval));
}

/*
 *   returns yy/mm/dd in day[]
 *   returns seconds after midnight if flag=0
 *   else seconds since last call with flag=1
 */

int datime_(int *day,int *secs,int *flag)
{
    time_t now;
    struct tm *timedat;
    static long jdays,elapsedsecs;

    time(&now);
    timedat = localtime(&now);
    day[0] = timedat->tm_year;
    day[1] = timedat->tm_mon+1;
    day[2] = timedat->tm_mday;

    *secs = timedat->tm_hour*3600 + timedat->tm_min*60 + timedat->tm_sec;
   if (*flag) {
        *secs -= (elapsedsecs-(timedat->tm_yday - jdays)*86400);
    }
    else {
        elapsedsecs = *secs;
        jdays = timedat->tm_yday;
    }
    return(0);
}

long currenttime()
{
    time_t now;

    time(&now);
    return((long)now);
}
