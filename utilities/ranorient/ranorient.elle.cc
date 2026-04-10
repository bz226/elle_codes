#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "init.h"
#include "mat.h"
#include "update.h"

int InitThisProcess(), ProcessFunction();

/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitThisProcess()
{
    char *infile;
    int err=0;
    
    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
                /*
         * check for any necessary attributes
         *  STRAIN, SPLIT, GRAIN
         */
        if (!ElleFlynnAttributeActive(MINERAL))
            OnError(infile,INVALIDF_ERR);
        if (!ElleFlynnAttributeActive(EULER_3))
            ElleInitFlynnAttribute(EULER_3);
    }

    /*
     * this process just runs once - no display
     */
    ProcessFunction();
}

int ProcessFunction()
{
    int i, j, k, l,debug_flag=1;
    int interval=0,err=0,max,mintype;
    double curra, currb, currc,dflt1,dflt2,dflt3;
    double rmap[3][3];
    double eps = 1e-5;
    FILE *debugout;

    printf("ProcessFunction()!!!!!!!!!!!!!!!!!!\n");

    interval = EllesaveInterval();
   
    max = ElleMaxFlynns();
    for (j=0;j<max;j++) {
        if (ElleFlynnIsActive(j) && !isParent(j))
        {
            
            ElleGetFlynnMineral(j, &mintype);

            if(mintype==QUARTZ)
            {
                ElleDefaultFlynnEuler3(&dflt1,&dflt2,&dflt3);
                ElleGetFlynnEuler3(j,&curra,&currb,&currc);
            
                /*if (curra==dflt1 && currb==dflt2 && currc==dflt3 ) */
                if (fabs(curra-dflt1) < eps &&
                    fabs(currb-dflt2) < eps &&
                    fabs(currc-dflt3) < eps ) 
                {
                    orientZXZ(rmap);
                    uneulerZXZ(rmap,&curra,&currb,&currc);

                    ElleSetFlynnEuler3(j, curra, currb, currc);
                }

            }
        }
    }
    
    ElleUpdate();
} 
