#include <stdio.h>
#include <math.h>
#include <string.h>
#include "attrib.h"
#include "globals.h"
#include "display.h"
#include "error.h"
#include "file.h"
#include "init.h"
#include "string_utils.h"
#include "runopts.h"
#include "interface.h"

extern "C" void SetFlynnAttribOption(int,float,float);
extern "C" void GetFlynnAttribOption(int *);

int ShowFile();

int InitShow()
{
    int err=0;
    int i,max;
    int attr;
    int set=0;
    float val, amax, amin;
    float b, c;
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
    int i, j, k;
    int interval=0,err=0,max;
    int attr, set=0;
    float val,amin,amax;
    float b,c;

#if XY
    GetFlynnAttribOption(&attr);

    if (ElleFlynnAttributeActive(attr)) {
        amax = amin = 0;
        max = ElleMaxFlynns();
        for (i=0;i<max;i++) {
            if (ElleFlynnIsActive(i) && ElleFlynnHasAttribute(i,attr)) {
                switch(attr) {
                case E_XX:
                case E_XY:
                case E_YY:
                case E_ZZ: ElleGetFlynnStrain(i,&val,attr);
                           break;
                case EULER_3: ElleGetFlynnEuler3(i,&val,&b,&c);
                               break;
                }
                if (!set) {
                    amax = val;
                    amin = val;
                    set = 1;
                }
                else {
                    if (val > amax) amax = val;
                    if (val < amin) amin = val;
                }
            }
        }
        SetFlynnAttribOption(attr,amin,amax);
    }
#endif
    if (ElleDisplay()) EllePlotRegions(ElleCount());
}
