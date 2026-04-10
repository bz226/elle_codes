/*************************************************************************
 * Please do not delete/modify next information! 
 * Author: Albert Griera
 * Institution: Departament Geologia, Universitat Autònoma de Barcelona
 * Email: albert.griera@uab.cat
 * Utility: New flynn nucleation using subgrain information stored in unodes layer
 * version:2.0
 * Date: 10.2011
 * This code is non-transferable without previous agreement between user and authors
 ***************************************************************************/
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "flynnarray.h"
#include "attrib.h"
#include "nodes.h"
#include "display.h"
#include "check.h"
#include "errnum.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "polygon.h"
#include "stats.h"
#include "init.h"
#include "mineraldb.h"
#include "unodes.h"
#include "update.h"
#include "dislocden.h"
#include "parseopts.h"
#include "setup.h"
#include "unode2flynn.h"


#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

int AdjustFlynnAttribute(int flynn_id,int attrib_id,double factor);
int InitThisProcess(), ProcessFunction();

int main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
 
    /*
     * initialise
     */
    ElleInit();
	  
    ElleUserData(userdata);
    userdata[N_rows]=256; // unode rows 
    userdata[MinUnodes]=10; // unode's cluster size
    userdata[2]=0; // Set to 1 to only use the flynn indicated in userdata[3]
    userdata[3]=0; // the flynn ID (see description above)
    ElleSetUserData(userdata);
    ElleSetOptNames("N_rows","ClusterSize","UseOnlyOneFlynn","Flynn","unused","unused","unused","unused","unused"); // to be modified
    
    if (err=ParseOptions(argc,argv))
        OnError("",err);

    /*
     * set the function
     */
    ElleSetInitFunction(InitThisProcess);

    /*
     * set the interval for writing to the stats file
     */

    /*
     * set the base for naming statistics and elle files
     */
	
    string rootname="u2f_sgb";
    ElleSetSaveFileRoot((char *)(rootname.c_str()));
    if (strlen(ElleOutFile())==0) {
        rootname += ".elle";
        ElleSetOutFile((char *)(rootname.c_str()));
    }

    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * run your initialisation function and start the application
     */
    StartApp();
    
     return(0);
} 
#define FLUID_PRESENCE 1
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
     * clear the data structures
     */


    ElleReinit();

    ElleSetRunFunction(ProcessFunction);
    
    UserData userdata;
    ElleUserData(userdata);

    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
        /*
         * check for any necessary attributes
         */
        if (!ElleUnodeAttributeActive(U_DISLOCDEN))
            ElleInitUnodeAttribute(U_DISLOCDEN);
		if (!ElleUnodeAttributeActive(U_ATTRIB_C))
            ElleInitUnodeAttribute(U_ATTRIB_C);
		if (!ElleUnodeAttributeActive(U_DENSITY)) 
			OnError("the subgrain preprocess must to be run previously to u2f_sgb",0);
		if (!ElleFlynnAttributeActive(F_ATTRIB_B))
			ElleInitFlynnAttribute(F_ATTRIB_B);		
			
    }
}

int ProcessFunction()
{
    bool FSDEBUG=false;
    bool bDisplay = false;
    UserData userdata;
    ElleUserData(userdata);
    int iDoOnlyOneFLynn = (int)userdata[2];
    
    
    int i, j, l, mm, max, mintype,type;

    int err=0,  new_flynn=NO_VAL;
    list< pair<double,int> >ordered_list;
    list< pair<double,int> >::iterator ito;
    
	vector<int> tmplist, tmpunodes;
	int ugrid_size, min_unodes, m, k, count;	
	int max_flynns,idummy, unode_id; 	
	double dummy, area_crit, test;
	Coords pos;	

	vector<int> allnodelist, nodelist, ran;

    ugrid_size=(int)userdata[N_rows]; // set the grid size 
    min_unodes=(int)userdata[MinUnodes]; //minimum size of unode's cluster to define a new grain 
    bool still_loop = true;

        
   // if ( ElleminNodeSep() >= (1.0/ugrid_size)) area_crit=min_unodes/(ElleminNodeSep()*ElleminNodeSep());    
   //	else area_crit=min_unodes/(ugrid_size*ugrid_size); // set the flynn's critical minimum area to split a grain  	

	area_crit=5e-4;
	
	ElleCheckFiles();
	max_flynns=ElleMaxFlynns();
	
    for ( l = 0; l < EllemaxStages(); l++ )
    {
		
		for (m=0;m<max_flynns; m++)  
			if (ElleFlynnIsActive(m)) 
				ran.push_back(m); // copy flynns id to  <vector> ran
							
        random_shuffle( ran.begin(), ran.end() );

	
        for ( mm = 0; mm < ran.size(); mm++ )
        {
					                                                                        
		m=ran.at(mm);			
		if (ElleFlynnIsActive(m)) {

            if (iDoOnlyOneFLynn)
            {
                if (m==(int)userdata[3])
                {
                    bDisplay=true;
                    area_crit=5e-4;
                    printf("USING ONLY FLYNN %u\n",m);
                }
                else 
                {
                    bDisplay=false;
                    area_crit=5e9;
                    
                }   
            }
            
			if (bDisplay) printf("flynn %i",m);
					  
			if (ElleRegionArea(m) >= area_crit){ // Area must to be higher than critical value 
			  
				tmpunodes.clear();
				tmplist.clear();
				ElleGetFlynnUnodeList(m,tmpunodes);
	
				// Subgrain ID is stored in U_DENSITY (provisional)
				// Subgrain list inside a flynn and sort according by size or number of unodes	
		
				for (k=0;k<tmpunodes.size();k++) { 
					ElleGetUnodeAttribute(tmpunodes[k], &dummy,U_DENSITY);
					tmplist.push_back( int(dummy) ); // copy subgrains id to the <vector> tmpunodes
				}
	
			
				sort(tmplist.begin(), tmplist.end());		  
				if (bDisplay) printf(" size unodes %lu %lu \n",tmpunodes.size(), tmplist.size());
						  
				idummy=*tmplist.begin();
				count=0;  

				// generate an ordered list, using a list
				list< pair<int,int> >ordered_list;
				list< pair<int,int> >::iterator ito;

				for(k=0;k<tmplist.size();k++) 
                {
					if (tmplist[k] != idummy)
                    {
						pair<int,int> tmp(idummy, count);
						 ito=ordered_list.begin();
						 while (ito!=ordered_list.end() &&
                            tmp.second < (*ito).second) ito++;
							ordered_list.insert(ito,tmp);
						count=0;
						
						idummy=tmplist[k];
		
					} else count++;
				}
									
				pair<int,int> tmp(idummy, count); //add last one, and do ordered list
				ito=ordered_list.begin();
				while (ito!=ordered_list.end() &&
					tmp.second < (*ito).second) ito++;
					ordered_list.insert(ito,tmp);						

				// print list 			
				ito=ordered_list.begin();
				count=0;
				while (ito!=ordered_list.end()) {
				if (bDisplay) cout << (*ito).first << " " <<
				(*ito).second << endl;	
				count++;	
				ito++;
				}
		

		// take ordered list of subgrains, search for an unode near to the boundaries to start the new flynn
		err = 0;		
		ito=ordered_list.begin();
		if ( ito != ordered_list.end()) ito++; // first subgrain is the parent grain

		// while (ito!=ordered_list.end()) { // only is allowed one subgrain splitting for each flynn
		
			if((*ito).second >= min_unodes) { // if subgrain is tiny, don't do nothing
			
				tmplist.clear();
				for (k=0;k<tmpunodes.size();k++) {  //update the tmpunodes to the actual flynn after definition of a new flynn¿?
					ElleGetUnodeAttribute(tmpunodes[k], &dummy, U_DENSITY);	
					if ( int(dummy) == (*ito).first ) tmplist.push_back(tmpunodes[k]); // take unodes of the subgrain
					}		

				ElleFlynnNodes(m,allnodelist);
				
				k=0;
				bool found=false;
				double mindist; 
				while( k<tmplist.size() && found == false) {
 	
					ElleGetUnodePosition(tmplist[k], &pos);
					// nodelist.clear();
	                mindist= ElleUnodeROI(); 
					for (i=0;i<allnodelist.size();i++) {
						
						int j;
						Coords xy, uxy;
						ElleNodePosition(allnodelist[i],&xy);
						ElleCoordsPlotXY(&xy, &pos);
						test= pointSeparation(&xy, &pos); 
						
						if ( pointSeparation(&xy, &pos) < mindist){
							if (bDisplay) cout << test << " " << mindist << endl;
							// nodelist.push_back(allnodelist[i]);
							found=true; 
							unode_id=tmplist[k];
							mindist=test;  
						}
						}
					

					k++;
					}
					
        //if (bDisplay) 
            //printf (" initial %u %lu \n", unode_id, tmplist.size());	
					
			    if (tmplist.size() > 2 && found == true ) { // added  

 				 new_flynn = NO_VAL;
                 printf("unode_id %u tmplist.size() %lu nodelist.size() %lu\n",unode_id,tmplist.size(),nodelist.size());
            
                // FOR DEBUGGING: 
                if (iDoOnlyOneFLynn)
                {
                    fstream fMyFile;
                    Coords cUnodeXY;
                    
                    fMyFile.open ( "Subgrain.txt", fstream::out | fstream::trunc);
                    
                    ElleGetUnodePosition(unode_id,&cUnodeXY);
                    
                    fMyFile << unode_id << " " << cUnodeXY.x << " " << cUnodeXY.y << endl;
                    
                    for (int i=0;i<tmplist.size();i++)
                    {
                    
                        ElleGetUnodePosition(tmplist[i],&cUnodeXY);
                        
                        fMyFile << tmplist[i] << " " << cUnodeXY.x << " " << cUnodeXY.y << endl;
                        
                    }
                    
                    
                    fMyFile.close();
                }
                int old_flynn=NO_VAL;
                new_flynn=NO_VAL;
				 err=Unode2Flynn2 (unode_id,tmplist, &new_flynn, &old_flynn);
				// err=Unode2Flynn2 (unode_id,tmplist,nodelist, &new_flynn);				
				
				// ElleSetFlynnRealAttribute(new_flynn, 1.0, F_ATTRIB_B);
                if (ElleFlynnAttributeActive(AGE))
                    UpdateFlynnAges(new_flynn,old_flynn);
                    
                /* FS: Add the possibility to track split events
                 * Write logfile of splits happen with following columns:
                 * 1 split_type phase_oldgrain phase_newgrain id_oldgrain id_newgrain
                 * split_type=1: rotation recrystallisation (always in this code)
                 * split_type=2: grain dissection (never in this code)
                 */
                 bool bTrackSplits=true;
                 if (bTrackSplits)
                 {
                     // find phases of old and new flynn
                     int iPhaseOld = 0, iPhaseNew = 0;
                     double dPhaseTmp = 0.0;
                     if (ElleFlynnAttributeActive(VISCOSITY))
                     {
                         ElleGetFlynnRealAttribute(old_flynn,&dPhaseTmp,VISCOSITY);
                         iPhaseOld = (int)dPhaseTmp;
                         ElleGetFlynnRealAttribute(new_flynn,&dPhaseTmp,VISCOSITY);
                         iPhaseNew = (int)dPhaseTmp;
                     }
                     
                     // write file
                    fstream fSplitLog;
                    fSplitLog.open ( "Track_SplitEvents.txt", fstream::out | fstream::app);

                    fSplitLog << "1 1 " << iPhaseOld << " " << iPhaseNew << " "
                              << old_flynn << " " << new_flynn << endl;
                    fSplitLog.close(); 
                 }
			}

		}	

    }
	}
	

}
		
       // ElleWriteData(ElleOutFile());
      			ElleUpdate();
  }
  
  printf("Finished ...\n");
  ElleWriteData("u2f.elle");
    return(0);
}
/*
int AdjustFlynnAttribute(int flynn_id,int attrib_id,double factor)
{
    int err=0;
    double val;
    
    if (ElleFlynnIsActive(flynn_id) &&
            ElleFlynnAttributeActive(attrib_id)) {
        ElleGetFlynnRealAttribute(flynn_id, &val, attrib_id);
        val *= factor;
        ElleSetFlynnRealAttribute(flynn_id, val, attrib_id);
    }
    return(err);
}
*/

/* BY FLORIAN:
 * 
 * Update the flynn ages by setting them to (-1), ATTENTION: After this, we 
 * still need to finally increase all ages of flynns by 1, but this needs to be 
 * done when ALL recrystallization is over.
 * 
 * We do this after every call of "Unode2Flynn2", if a flynn was split
 * 
 * New flynn will always be the smaller one:
 * Do not assign new age to old flynn, if it is more than 10x larger than new 
 * one 
 */
void UpdateFlynnAges(int iNew,int iOld)
{
    
    if (ElleFlynnIsActive(iNew) && ElleFlynnIsActive(iOld))
    {
        double dAreaNew=ElleRegionArea(iNew);
        double dAreaOld=ElleRegionArea(iOld);
        
        if ((dAreaOld/dAreaNew)<10.0)
        {
            ElleSetFlynnRealAttribute(iNew,(-1.0),AGE);
            ElleSetFlynnRealAttribute(iOld,(-1.0),AGE);
        } 
        else
        {
            ElleSetFlynnRealAttribute(iNew,(-1.0),AGE);
        }
    }
    
}
