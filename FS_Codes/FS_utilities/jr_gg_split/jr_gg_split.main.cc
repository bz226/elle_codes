
/*
 *  main.cc
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "error.h"
#include "parseopts.h"
#include "init.h"
#include "runopts.h"
#include "stats.h"
#include "setup.h"
#include "interface.h"



int main(int argc, char **argv)
{
    int err=0;
    extern int InitGG_Split(void);

    ElleInit();

    ElleSetInitFunction(InitGG_Split);

    if (err=ParseOptions(argc,argv))
        OnError("",err);
    //ES_SetstatsInterval(50);

  /*  for (i=0;i<argc;i++)
    	printf("%s", *(argv+i));
    printf("\n");
*/
    ElleSetSaveFileRoot("gg_split");

    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * run init and run functions
     */
    StartApp();

    return(0);
}
