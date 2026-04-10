#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <math.h>
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
#include "log.h"

using std::ios;
using std::vector;
using std::ifstream;
using std::cout;
using std::endl;

// funcions definides
int Input_tex(), Init_input();
int SetUnodeAttributesFromFile2(char *fname);

main(int argc, char **argv)
{
    int err=0;
    UserData userdata;

    /*
     * initialise
     */
    ElleInit();
    ElleUserData(userdata);
     
    userdata[0]=12; // Default: 12 = No import for U_ATTRIB_A
    userdata[1]=12; // Default: 12 = No import for U_ATTRIB_B
    userdata[2]=12; // Default: 12 = No import for U_ATTRIB_D
    userdata[3]=12; // Default: 12 = No import for U_ATTRIB_E
    userdata[4]=12; // Default: 12 = No import for U_ATTRIB_F
    
    ElleSetUserData(userdata);   
    ElleSetOptNames("U_ATTRIB_A","U_ATTRIB_B","U_ATTRIB_D","U_ATTRIB_E","U_ATTRIB_F","unused","unused","unused","unused");
	//std::cout<<"Starting ..."<<std::endl;
    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(Init_input);

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
    ElleSetSaveFileRoot("fft_out");

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
int Init_input()   //InitMoveBnodes()
{
    char *infile;
    int err=0;
    
    /*
     * clear the data structures
     */
    ElleReinit();

    ElleSetRunFunction(Input_tex);

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
        /*!
         * The following is now happening in SetUnodeAttributesFromFile2
         */
        //if (!ElleUnodeAttributeActive(U_ATTRIB_A))
            //ElleInitUnodeAttribute(U_ATTRIB_A);
        //if (!ElleUnodeAttributeActive(U_ATTRIB_B))
            //ElleInitUnodeAttribute(U_ATTRIB_B);
		//if (!ElleUnodeAttributeActive(U_ATTRIB_C))
            //ElleInitUnodeAttribute(U_ATTRIB_C);
		//if (!ElleUnodeAttributeActive(U_ATTRIB_D))
            //ElleInitUnodeAttribute(U_ATTRIB_D);
		//if (!ElleUnodeAttributeActive(U_ATTRIB_E))
            //ElleInitUnodeAttribute(U_ATTRIB_E);
		//if (!ElleUnodeAttributeActive(U_ATTRIB_F))
            //ElleInitUnodeAttribute(U_ATTRIB_F);
            
    // Never happening, because done in fft2elle:
		////if (!ElleUnodeAttributeActive(U_DISLOCDEN))
	 	////	ElleInitUnodeAttribute(U_DISLOCDEN);
    }
}

int Input_tex()
{
    int err=0, i;
    err = SetUnodeAttributesFromFile2("tex.out"); //file is tex.out
    if (err) OnError("tex.out",err);

    /*
     * write the updated Elle file
     */
    if (err=ElleWriteData("fft_out.elle"))
        OnError("",err);
}


int SetUnodeAttributesFromFile2(char *fname)
{
    int err=0,opt_1,opt_2,opt_3,opt_4,opt_5;
    int id=0, i, j,max_unodes,jj ;
    Coords xy;
	double val[12],dd;

	UserData userdata;
    ElleUserData(userdata);    
    opt_1=(int)userdata[0];
    opt_2=(int)userdata[1];
    opt_3=(int)userdata[2];
    opt_4=(int)userdata[3];
    opt_5=(int)userdata[4];
    
        if (opt_1!= 12 && !ElleUnodeAttributeActive(U_ATTRIB_A))
            ElleInitUnodeAttribute(U_ATTRIB_A);
        if (opt_2!= 12 && !ElleUnodeAttributeActive(U_ATTRIB_B))
            ElleInitUnodeAttribute(U_ATTRIB_B);
		if (!ElleUnodeAttributeActive(U_ATTRIB_C)) // Always do this
            ElleInitUnodeAttribute(U_ATTRIB_C);
		if (opt_3!= 12 && !ElleUnodeAttributeActive(U_ATTRIB_D))
            ElleInitUnodeAttribute(U_ATTRIB_D);
		if (opt_4!= 12 && !ElleUnodeAttributeActive(U_ATTRIB_E))
            ElleInitUnodeAttribute(U_ATTRIB_E);
		if (opt_5!= 12 && !ElleUnodeAttributeActive(U_ATTRIB_F))
            ElleInitUnodeAttribute(U_ATTRIB_F);

	//sprintf(logbuf,"*** Export data from FFT to ELLE ***\n[4] normalized strain rate\n[5] normalized stress\n[6] activity basal mode\n[7] activity prismatic mode\n");
	//Log( 1,logbuf );
/*	
	printf("\t*** Export data from FFT to ELLE ***\n");	
	printf("\t*** using as default tex.out file ***\n");

	printf("Import data to store in U_ATTRIB_A and U_ATTRIB_B\n");
	printf("NON import dislocation density to U_DISLOCDEN !!\n");	
	printf("[4] normalized strain rate\n");
	printf("[5] normalized stress\n");
	printf("[6] activity basal mode\n");
	printf("[7] activity prismatic mode\n");
	printf("[8] Geometrical necessary dislocation density ()\n");
	printf("[9] Statistical dislocation density (¿?)\n");	
	printf("[10] identification of Fourier Point \n");	
	printf("[11] FFT grain nunmber\n");
    printf("[12] Import nothing for this attribute\n");
	printf("?? (exp. 4 5 +<return>)\n");	
*/	
	
	/*printf("Option 1: %u\n",opt_1);
	printf("Option 2: %u\n",opt_2);
	printf("Option 3: %u\n",opt_3);
	printf("Option 4: %u\n",opt_4);
	printf("Option 5: %u\n",opt_5);*/
	
	id=0;
    ifstream datafile(fname);
    if (!datafile) return(OPEN_ERR);
    while (datafile) {
	
datafile>>val[0]>>val[1]>>val[2]>>val[3]>>val[4]>>val[5]>>val[6]>>val[7]>>val[8]>>val[9]>>val[10]>>val[11];	

		
            if(opt_1!=12) ElleSetUnodeAttribute(id,val[opt_1],U_ATTRIB_A);
            if(opt_2!=12) ElleSetUnodeAttribute(id,val[opt_2],U_ATTRIB_B);
            if(opt_3!=12) ElleSetUnodeAttribute(id,val[opt_3],U_ATTRIB_D);
            if(opt_4!=12) ElleSetUnodeAttribute(id,val[opt_4],U_ATTRIB_E);
            if(opt_5!=12) ElleSetUnodeAttribute(id,val[opt_5],U_ATTRIB_F);

				// ElleSetUnodeAttribute(id,val[8],U_DISLOCDEN);	
		id++;	
    }
    datafile.close();
	
	max_unodes = ElleMaxUnodes();
    for (i=0;i<max_unodes;i++)   // cycle through unodes 
    {
	    jj=ElleUnodeFlynn(i);
	    ElleSetUnodeAttribute(i,U_ATTRIB_C, double(jj)); 
    }
	
    return(err);
}
