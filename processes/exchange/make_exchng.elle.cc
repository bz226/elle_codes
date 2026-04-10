#include <stdio.h>
#include <math.h>
#include "attrib.h"
#include "nodes.h"
#include "update.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "init.h"
#include "triattrib.h"
#include "unodes.h"

int InitSetUnodes(), SetUnodes();
void ResetSetUnodeAttributeByMineral(int flynnid,double val);

/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitSetUnodes()
{
    char *infile;
    int err=0;
    
    /*
     * clear the data structures
     */
    ElleReinit();

    ElleSetRunFunction(SetUnodes);

    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
        /*
         * check for any necessary attributes which may
         * not have been in the elle file
         */
    }
}

int SetUnodes()
{
    int i, j, k;
    int max_stages, max_flynns, max_unodes;
    int min_type;
    UserData udata;
    double qtz_val=0.0,mica_val=5.4,gt_val=8.0,val=0.0;
    ElleUserData(udata);
    int mintype;
    
    if(udata[0]>0)
    	qtz_val=udata[0];
    if(udata[1]>0)
    	mica_val=udata[1];
    if(udata[2]>0)
    	gt_val=udata[2];
    	
    ElleCheckFiles();
    
    max_stages = EllemaxStages(); // number of stages 
    max_flynns = ElleMaxFlynns(); // maximum flynn number used
    max_unodes = ElleMaxUnodes(); // maximum unode number used
    for (i=0;i<max_stages;i++)   // cycle through stages
	{
        for (j=0;j<max_flynns;j++) // cycle through flynns
		{
            if (ElleFlynnIsActive(j))  // process if flynn is active 
			{
                ElleClearTriAttributes();  // reset triangulation
                TriangulateUnodes(j,MeshData.tri); // triangulate one flynns unodes
                
                ElleGetFlynnIntAttribute(j, &mintype, MINERAL); //mwj*** 20/06/2003
                
                if(mintype==QUARTZ)
                	val=qtz_val;
                else if(mintype==MICA)
                	val=mica_val;
                else if(mintype==GARNET)
                	val=gt_val;
                else 
                	val=0.0;
                	
				ResetSetUnodeAttributeByMineral(j,val);  // process this triangulation
            }
        }
        ElleUpdate();
    }
} 

void ResetSetUnodeAttributeByMineral(int flynnid, double val)
{
    int i;
    int count;


    std::vector<int> unodelist;  // create a vector list of unodes
    ElleGetFlynnUnodeList(flynnid,unodelist); // get the list of unodes for a flynn
    count = unodelist.size();
    
    for (i=0; i<count; i++) 			
		ElleSetUnodeAttribute(unodelist[i],CONC_A, val); // set new unode attribute value

}
