
/*
 *  main.cc
 */

#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "parseopts.h"
#include "init.h"
#include "runopts.h"
#include "file.h"
#include "setup.h"
#include "experiment.h"

Experiment *MyExperiment;


main(int argc, char **argv)
{
    int err=0;
    UserData userdata;

    //*-----------------------------------------
    //* initialise
    //*-----------------------------------------

    ElleInit();
	
    //extern int Init_Experiment(void);
	
	MyExperiment = new Experiment;

    //*--------------------------------------------------
    //* set the function to the one in your process file
    //*--------------------------------------------------

    //ElleSetInitFunction(InitSetMike);
	
	//ElleSetInitFunction(MyExperiment->Init_Experiment);
	
	ElleSetInitFunction(Init_Experiment);

    ElleUserData(userdata);
    userdata[0]=0; // Change default calculation mode
      // -u 0 read experiment type and parameters from a zip file
      //      e.g. if input elle file is res50.elle then read res50.zip
      // -u 1 fracturing
      // -u 2 fracture boudinage
      // -u 3 expanding inclusions
      // -u 4 shrinkage cracks
      // -u 5 viscoelastic deformation
      // -u 6 grooves
      // -u 7 Stylolites
      // -u 8 combine graingrowth and fractures
      // -u 9 solid solid phase change
      // -u 10 heat flow
      // -u 11 pure grain growth
      // -u 12 Lattice gas diffusion
      // -u 13 Lattice gas fluid flow
    ElleSetUserData(userdata);

    ElleSetOptNames("Experiment","unused","unused","unused","unused","unused","unused","unused","unused");

    if (err=ParseOptions(argc,argv))
        OnError("",err);

	
    //*-------------------------------------------------------
    //* set the base for naming statistics and elle files
    //*------------------------------------------------------

    ElleSetSaveFileRoot("my_experiment");

	
    //*-------------------------------------
    //* set up the X window
    //*-------------------------------------

    if (ElleDisplay()) SetupApp(argc,argv);

    //*--------------------------------------------------------------
    //* run your initialisation function and start the application
    //*--------------------------------------------------------------
	
	

    StartApp();
	
	
    CleanUp();

    return(0);
} 
extern int Init_Experiment(void)
{
	
		//-----------------------------------------------------
	  // Set the run function
	  //-----------------------------------------------------

		 ElleSetRunFunction(Run_Experiment);
		MyExperiment->Init();
}
extern int Run_Experiment()
	{
		MyExperiment->Run();
	}
