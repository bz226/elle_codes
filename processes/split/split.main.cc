
/*
 *  main.c
 */

#include <stdio.h>
#include <stdlib.h>

#include "attrib.h"
#include "errnum.h"
#include "error.h"
#include "init.h"
#include "string_utils.h"
#include "runopts.h"
#include "file.h"
#include "stats.h"
#include "setup.h"
#include "parseopts.h" 
#include "init.h" 
#include "split.h" 

float TotalTime;

main(int argc, char **argv)
{
    int err=0;
    ElleRunFunc init;
    extern int InitThisProcess(void);
    UserData userdata;

    ElleInit();

    ElleUserData(userdata);
    userdata[SMode]=0; // Change default calculation mode
    ElleSetUserData(userdata);

    ElleSetOptNames("SplitMode","unused","unused","unused","unused","unused","unused","unused","unused");
    if (err=ParseOptions(argc,argv)) 
        OnError("",err); 
 

    ElleSetInitFunction(InitThisProcess);
    
    
    ElleSetSaveFileRoot("split");

    if (ElleDisplay()) SetupApp(argc,argv);

    StartApp();
    return(0); 
} 
