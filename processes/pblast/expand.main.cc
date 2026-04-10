
/*
 *  main.cc
 */

#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "parseopts.h"
#include "init.h"
#include "runopts.h"
#include "setup.h"
#include "file.h"
#include "expand.h"

main(int argc, char **argv)
{
    int err=0;
    UserData udata;
    extern int InitExpand(void);

    ElleInit();

    ElleSetInitFunction(InitExpand);

    /*ElleSetDisplay(0);*/
    /*ElleSetSaveFrequency(10);*/
    ElleUserData(udata);
    /* initialise speedup,resetnode,num_nbs,reset val,attribute*/
    udata[Velocity] = 1.0;
    udata[NodeId] = NO_NB;
    udata[AttribId] = CONC_A;
    udata[AttribValue] = udata[ResetNum] = 0;
    udata[AreaFlag] = 0;
    udata[VectorCalc] = BisectAngle;
    ElleSetUserData(udata);
    ElleSetOptNames("Vel Factor","Bnode Id","Attrib Value","Reset","Attribute","Area Flag","Direction alg","unused","unused");

    if (err=ParseOptions(argc,argv))
        OnError("",err);

    ElleSetSaveFileRoot("expand");

    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * run init and run functions
     */
    StartApp();

    return(0);
} 
