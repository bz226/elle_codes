#include <stdio.h>
#include <math.h>
#include <string.h>
#include "display.h"
#include "error.h"
#include "file.h"
#include "init.h"
#include "runopts.h"

#include "interface.h"
int ShowFile();

int InitShow()
{
    int err=0;
    char *infile;

    ElleReinit();
    ElleSetRunFunction(ShowFile);

    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
    }
}

int ShowFile()
{
    if (ElleDisplay()) EllePlotRegions(ElleCount());
}
