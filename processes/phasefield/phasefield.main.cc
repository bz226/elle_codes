
/*
 *  main.cc
 */


#include "phasefield.h"

main(int argc, char **argv)
{
    int err=0;
    extern int InitSetUnodes(void);
    UserData userdata;
 
    /*
     * initialise
     */
    ElleInit();
    
    Reference();
    
    ElleUserData(userdata);
    
    userdata[uRESET] 	= 0;    // reset unode values
    userdata[uK] 		= 1.8;    // Latent heat
    userdata[uEPS] 		= 0.01;    // interfacial width
    userdata[uDELTA] 	= 0.02;    // modulation of the interfacial width
    userdata[uANGLE0] 	= 1.57;    // orientation of the anisotropy axis
    userdata[uANISO] 	= 6.0;    // anisotropy 2*PI/ANISO
    userdata[uALPHA] 	= 0.9;    // m(T) = ALPHA/PI * atan(GAMMA*(TEQ-T))
    userdata[uGAMMA] 	= 10;    // 

    ElleSetUserData(userdata);
    ElleSetOptNames("Reset unodes","Latent heat","Eps","Delta","Angle0","Aniso","Alpha","Gamma","unused");
    
    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitSetUnodes);
    ElleSetOptNames("Reset Unode Values","Latent heat","Interfacial width","Width modulation","Orientation","Anisotropy","Alpha","Gamma","unused");

    if (err=ParseOptions(argc,argv))
        OnError("",err);

/*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("phasefield");
    ElleSetAppName("phasefield");
    /*
     * set up the X window
     */
    if (ElleDisplay()) 
        SetupApp(argc,argv);

    StartApp();
//    CleanUp();

    return(0);
} 

void Reference()
{
	
	printf("----------------------------------------------------------------------------------------\n");
	printf("Phase Field model of crystallisation from a melt\n");
	printf("Code based on code found in paper by Thierry BIBEN, 2005\n");
	printf("European Journal of Physics, 26, S47-S55\n");
	printf("----------------------------------------------------------------------------------------\n");
	printf("User data options\n");
	printf("elle_phasefield -u a b c d e f g h\n");
	printf("a = flag to reset elle file unodes to 300x300 square grid (default = 0, reset = 1) \n");
	printf("b = Latent Heat (default = 1.8)\n");
	printf("c = Interfacial width (default = 0.01)\n");
	printf("d = Modulation of the interfacial width (default = 0.02)\n");
	printf("e = Orientation of the anisotropy axis (in radians, default = 1.57)\n");
	printf("f = Anisotropy (default which is hexagonal = 6)\n");
	printf("g = Alpha (default = 0.9)\n");
	printf("h = Gamma (default = 10.0)\n");
	printf("----------------------------------------------------------------------------------------\n");
	
}
