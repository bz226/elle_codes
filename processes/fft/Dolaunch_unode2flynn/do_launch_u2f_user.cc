#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "attrib.h"
#include "nodes.h"
#include "update.h"
#include "error.h"
#include "parseopts.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "init.h"
#include "setup.h"
#include "triattrib.h"
#include "unodes.h"
#include "polygon.h"

using std::ios;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::string;
using std::vector;
using std::cin;

int Input_tex(), Init_input();
int temp_file();

main(int argc, char **argv)
{
    int err=0;
    UserData udata;

    /*
     * initialise
     */
    ElleInit();

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(Init_input);

	ElleUserData(udata);
	ElleSetUserData(udata);
	
    if (err=ParseOptions(argc,argv))
        OnError("",err);
    ElleSetSaveFrequency(1);

    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("u2f");

    /*
     * run your initialisation function and start the application
     */
    StartApp();

    CleanUp();

    return(0);
} 


/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int Init_input()
{
    char *infile;
    int err=0;
    
    /*
     * clear the data structures
     */
    ElleReinit();

    ElleSetRunFunction(temp_file);

    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);

    }
}


/* Do a script file to launch "unode2flynn" (nucleation process)
 * user data:
 * udata[0] number of unodes neighbours to use
 * udata[1] critical value of dislocation density
 *
 * provisional  until fix errors in unode2flynn process 
 */

// add a function to generate the provisional launch file 
int temp_file()
{	 
 int err=0;	
 int i, j=0, dd, count,n_neighbours;
 double val, max_unodes; 
 double d_crit;
 UserData udata; 
 FILE *tmpstats;  

vector <int> ran_unodes; 
	 
	 ElleUserData(udata);
	 n_neighbours=(int)udata[0];
	 d_crit= udata[1];
	 
    tmpstats = fopen("launch_u2f.shelle","w");  
	max_unodes = ElleMaxUnodes();

	 // List of critical unodes
	 
    for (i=0;i<max_unodes;i++)
    {

		ElleGetUnodeAttribute(i,U_DISLOCDEN,&val);
		
		if (val > d_crit) {
		ran_unodes.push_back(i);			
			j=j+1;	
		}

	}

	// Random the list of unodes 
    std::random_shuffle(ran_unodes.begin(), ran_unodes.end());

	// Write the script file
	count=ran_unodes.size();

	for (i=0;i<count; i++){
		dd= ran_unodes[i];
		fprintf(tmpstats,"/g/elle_2009b/elle/elle/binwx/unode2flynn_user -i u2f.elle -u %d %d %e -n\n",dd,n_neighbours,d_crit);

	}
	
    fprintf(tmpstats,"\n");
    fclose(tmpstats);
	
	printf("%i\n", j);
	

}
