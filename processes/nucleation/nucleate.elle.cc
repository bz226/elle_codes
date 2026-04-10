#include <cstdio>
#include <cmath>
#include <iostream>
#include "attrib.h"
#include "nodes.h"
#include "update.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "init.h"
#include "interface.h"
#include "timefn.h"
#include <stdlib.h>
int DoSomethingToNode(int node);
int ProcessFunction(), InitThisProcess();

using std::cout;
using std::endl;
/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitThisProcess()
{
    char *infile;
    int err=0;
	long tmp;
	 tmp = currenttime();

	 #ifdef DEBUG
	 	tmp = 1028731772;
	 #endif
	 cout << "seed= " << tmp << endl;
	 srand(tmp);
    /*
     * clear the data structures
     */
    ElleReinit();

    ElleSetRunFunction(ProcessFunction);

    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError("",err);
        /*
         * check for necessary node attributes which
         * are not in the input file
         * if necessary call
        ElleAttributeNotInFile(infile,validAttr);
         */
    }
}

int ProcessFunction()
{
    int i, j, k;
    int err=0,max;

    ElleCheckFiles();

    for (i=0;i<EllemaxStages();i++) // loop for number of stages
	{
        max = ElleMaxNodes();	// index of maximum node used in model
        /*
         * alternate up and down the node array
         */
        if (i%2) {
            for (j=0;j<max;j++) {
                if (ElleNodeIsActive(j)) // process node if it is active
				{
                    err = DoSomethingToNode(j);
                    if(err)
					break;
                }
            }
        }
        else {
            for (j=max-1;j>=0;j--) {
                if (ElleNodeIsActive(j)) {
                    err = DoSomethingToNode(j);
                    if(err)
						break;
                }
            }
        }
        ElleUpdate();
    }
}

int DoSomethingToNode(int node)
{
    double test,prox;
    int nbnodes[3];
    int newid;
    int count;

	prox=ElleNodeAttribute(node, N_ATTRIB_A);

	if(prox > .1) // very close to a garnet
		test =1.0;
	else if(prox > 1e-12 && prox < .1) // anti-clustering rule
		test=rand()/(double)RAND_MAX*(1.0/-log(prox));
	else // no existing garnets
		test=rand()/(double)RAND_MAX*.1;

	if(ElleNodeIsTriple(node) && test < 0.001)
	{

		ElleNucleateFlynnFromNode(node, &newid);
		ElleSetFlynnIntAttribute(newid,1,EXPAND);
		ElleSetFlynnRealAttribute(newid,50.0,VISCOSITY);
		return(1);
	}
	return(0);
}
