 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "string_utils.h"
#include "init.h"
#include "bflynns.h"
#include "runopts.h"
#include "setup.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/
#ifdef __cplusplus
extern "C" {
#endif
extern void ElleInitTriAttributes();
extern void ES_Init_Stats();
extern void ElleInitNodeArray(void);
extern void Init_Data();
void ElleInitEnergyLUT(int size);
void ElleRemoveEnergyLUT();
#ifdef __cplusplus
}
#endif

int ElleInit()
{
    ElleInitRandom((int)time(0));
    srand((unsigned int) time(0));
    /*
     * the following line is for testing against gsl/rng/test.c
     */
    /*ElleInitRandom(1);*/
    ElleInitRunOptions(); 
    if (ElleDisplay()) Init_Data();
    ElleInitNodeArray();
    ElleInitTriAttributes();
    ElleInitEnergyLUT(0);
    ES_Init_Stats();
    return(0);
}

int ElleExit()
{
    CleanUp();
}

/*
void ElleReinitArrays()
{
    int max;

    max = ElleMaxNodes();
    ElleClearNodeArray(0,max-1);
    max = ElleMaxGrains();
    ElleClearGrainArray(0,max-1);
}
*/

int ElleReinit()
{
    ElleCleanArrays();
    ElleClearTriAttributes();
    ElleRemoveEnergyLUT();
    ElleSetCount(0);
    return(0);
}
