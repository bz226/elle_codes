#include "FS_euler3random.h"
using namespace std;

int main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    /*
     * Initialise and set option names
     */
    ElleInit();
    ElleSetOptNames("InUnodes","NoiseInFlynns","Anisotropy","AnisoAlpha","AnisoBeta","AnisoGamma","AnisoNoise","AnisoOnlyForPhase","unused");
    /* 
     * Give default values for userdata:
     */
    ElleUserData(userdata);
    /*
     * User input options:
     */
    userdata[InUnodes]=0;   // If set to 1: Store angles in unodes instead of flynns, default 0
    userdata[NoiseInFlynns]=0; // Setting it to anything higher than 0 means adding a noise to otherwise constant LPO within one flynn, the angles are then stored in unodes and removed from flynns
                               // ATTETION: Only working if InUnodes == 0
    userdata[Anisotropy]=0; // If set to 1: Code will create anisotropic distribution of euler angles with maximum values defined in the FOUR following input parameters, default: 0
    userdata[AnisoAlpha]=0; // Only if Anisotropy==1: Value for euler alpha 
    userdata[AnisoBeta]=0;  // Only if Anisotropy==1: Value for euler beta 
    userdata[AnisoGamma]=0; // Only if Anisotropy==1: Value for euler gamma 
    userdata[AnisoNoise]=0; // Only if Anisotropy==1: Values for alpha,beta,gamma will have a noise of +- this value
    userdata[AnisoOnlyFor1Phase]=0; // If euler angles are already in flynns: If setting to anything else than 0: Only change values in flynns with VISCOSITY = userdata[AnisoOnlyForPhase]
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    /*
     * Eventually set the interval for writing file, stages etc. if e.g.
	 * this utility should only run once etc.
     */
    ElleSetSaveFrequency(1);
    ElleSetStages(1);
    /*
     * Set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);	
    /*
     * Set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);
    /*
     * Run your initialisation function and start the application
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
int InitThisProcess()
{
    char *infile;
    int err=0;
    /*
     * Clear the data structures
     */
    ElleSetRunFunction(ProcessFunction);
    /*
     * Read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) 
    {
        if (err=ElleReadData(infile)) OnError(infile,err);
    }
    /*
     * Now that the Elle file is loaded, the user code could potentially check 
     * if e.g. unodes are in the file in case they are necessary or check for
     * attributes in flynns, bnodes or unodes that are needed in this code
     */
    /*! EXAMPLE:
    if(!ElleUnodesActive()) 
    {
        printf("Error: No unodes in file\n\n");  
        return 0;
    } 
    */
     
}

/* 
 * Anything can now be in the ProcessFunction itself:
 */ 
int ProcessFunction()
{
    int err=0;
    UserData userdata;              // Initialize the "userdata" array
    ElleUserData(userdata);         // Load the input data
    int iInUnodes = (int)userdata[InUnodes]; 
    double dNoiseInFlynns = userdata[NoiseInFlynns];
    double dAnisoOnlyFor1Phase = userdata[AnisoOnlyFor1Phase];
    double dFlynnPhase = 0.0; // stores phase of flynn if dAnisoOnlyFor1Phase!=0
    int iAniso = (int)userdata[Anisotropy];
    //double dAlpha = 0.0, dBeta = 0.0, dGamma = 0.0;
    double dAlphaMin, dAlphaMax, dBetaMin, dBetaMax, dGammaMin, dGammaMax;
    
    double rmap[3][3];
    double dEulers[3];
    bool bInRange = false; // To check if angles are in the desired interval
    
    if (!iAniso)
    {
        dAlphaMin = -180; dAlphaMax = 180;
        dBetaMin = 0; dBetaMax = 90;
        dGammaMin = -180; dGammaMax = 180;
    }
    else // values will be centered around a single maximum with a certain noise
    {
        dAlphaMin = userdata[AnisoAlpha]-userdata[AnisoNoise];
        dAlphaMax = userdata[AnisoAlpha]+userdata[AnisoNoise];
        //dBetaMin = userdata[AnisoBeta]-(userdata[AnisoNoise] *2); 
        //dBetaMax = userdata[AnisoBeta];
        dBetaMin = userdata[AnisoBeta]-userdata[AnisoNoise]; 
        dBetaMax = userdata[AnisoBeta]+userdata[AnisoNoise];
        dGammaMin = userdata[AnisoGamma]-userdata[AnisoNoise];
        dGammaMax = userdata[AnisoGamma]+userdata[AnisoNoise];
    }
    
    /*
     * Initialise randomisation
     */
    srand ( time(NULL) );
    
    /*
     * Save Euler angles to Unodes or Flynns:
     */
     
    /*
     * Check if user only wants to check the angles, which is achieved by using
     * userdata[0]=99 to use flynns or -99 to use unodes
     */
    if (iInUnodes==99) // flynn version
    {
        cout << "Only rearranging flynn angles in correct interval" << endl;
        CheckAllAngles(0);
        printf("Finished ...\n");

        err=ElleWriteData("FS_euler3random.elle");
        if(err) OnError("",err);
        return 0;
    }
    if (iInUnodes==-99) // unode version
    {
        cout << "Only rearranging unode angles in correct interval" << endl;
        CheckAllAngles(1);
        printf("Finished ...\n");

        err=ElleWriteData("FS_euler3random.elle");
        if(err) OnError("",err);
        return 0;
    }
    if (iInUnodes==0)
    {
        if (!ElleFlynnAttributeActive(EULER_3))
            ElleInitFlynnAttribute(EULER_3);
        
        printf("Saving random Euler angles to FLYNN \"EULER_3\"\n");
        printf("Data ranges:\n");
        printf("Euler alpha: [%i %i]\n",(int)dAlphaMin,(int)dAlphaMax);
        printf("Euler beta: [%i %i]\n",(int)dBetaMin,(int)dBetaMax);
        printf("Euler gamma: [%i %i]\n",(int)dGammaMin,(int)dGammaMax);
        
        if (iAniso) printf("\nFinding eulers that are in the correct range:\nThis might take a few moments ... \n\n");
        
        for (int i=0;i<ElleMaxFlynns();i++)
        {
            if (ElleFlynnIsActive(i))
            {
                /* Check if the user only wishes to change eulers in flynns 
                 * with specific phase: 
                 */
                if (dAnisoOnlyFor1Phase!=0)
                {
                    if (!ElleFlynnAttributeActive(VISCOSITY))
                    {
                        printf("ERROR: When dAnisoOnlyFor1Phase is set to anything else than 0, flynn attribute VISCOSITY has to be active!\n");
                    }
                    ElleGetFlynnRealAttribute(i,&dFlynnPhase,VISCOSITY);
                    if (dFlynnPhase!=dAnisoOnlyFor1Phase)
                    {
                        // stop loop here and go to the next flynn:
                        continue; 
                    }
                }
                
                ///* To check if the value is in the desired range according to the 
                 //* user input: Otherwise create new eulers */
                //bInRange = false;
                //while (!bInRange)
                //{   
                    //dEulers[0] = dEulers[1] = dEulers[2] = 0.0;
                    //orientZXZ(rmap);
                    //uneulerZXZ(rmap,&dEulers[0],&dEulers[1],&dEulers[2]);
            
                    ///* Bring eulers in general in the ranges from 
                     //* -180-180 (alpha, gamma) or 0-90 (beta)*/
                    //CheckEulers(dEulers);
                
                    ///* Do the checking*/
                    //if (dEulers[0] >= dAlphaMin && dEulers[0] <= dAlphaMax &&   
                        //dEulers[1] >= dBetaMin && dEulers[1] <= dBetaMax &&     
                        //dEulers[2] >= dGammaMin && dEulers[2] <= dGammaMax)
                    //{
                        //bInRange = true; 
                    //}          
                //}
                dEulers[0] = GetRandomNumberElleFunc(dAlphaMin,dAlphaMax);
                dEulers[1] = GetRandomNumberElleFunc(dBetaMin,dBetaMax);
                dEulers[2] = GetRandomNumberElleFunc(dGammaMin,dGammaMax);
                
                CheckEulers(dEulers);
                
                ElleSetFlynnEuler3(i,dEulers[0],dEulers[1],dEulers[2]);
            }
        }
        
        if (dNoiseInFlynns!=0)
            AddNoise2Flynns(dNoiseInFlynns);
    }
    else
    {
        if (!ElleUnodesActive())
        {        
            printf("Error: Cannot save Euler angles to unodes:\nNo unodes in file\n");  
            return 0;        
        }
        
        if (!ElleUnodeAttributeActive(EULER_3))
            ElleInitUnodeAttribute(EULER_3);
        
        printf("Saving random Euler angles to UNODE \"U_EULER_3\"\n");
        printf("Data ranges:\n");
        printf("Euler alpha: [%i %i]\n",(int)dAlphaMin,(int)dAlphaMax);
        printf("Euler beta: [%i %i]\n",(int)dBetaMin,(int)dBetaMax);
        printf("Euler gamma: [%i %i]\n",(int)dGammaMin,(int)dGammaMax);
        
        if (iAniso) printf("\nFinding eulers that are in the correct range:\nThis might take a few moments ... \n\n");
        
        for (int i=0; i<ElleMaxUnodes(); i++)
        {
            /* To check if the value is in the desired range according to the 
             * user input: Otherwise create new eulers */
            bInRange = false;
            while (!bInRange)
            {   
                dEulers[0] = dEulers[1] = dEulers[2] = 0.0;
                orientZXZ(rmap);
                uneulerZXZ(rmap,&dEulers[0],&dEulers[1],&dEulers[2]);
        
                /* Bring eulers in general in the ranges from 
                 * -180-180 (alpha, gamma) or 0-90 (beta)*/
                CheckEulers(dEulers);
            
                /* Do the checking*/
                if (dEulers[0] >= dAlphaMin && dEulers[0] <= dAlphaMax &&   
                    dEulers[1] >= dBetaMin && dEulers[1] <= dBetaMax &&     
                    dEulers[2] >= dGammaMin && dEulers[2] <= dGammaMax)
                {
                    bInRange = true; 
                }          
            }

            /* Alternatively create a random number, however this will not be
             * completely random:
             * You can change this bit according to yours needs */
            //dEulers[0] = GetRandomNumber(-180,180,(double)rand());
            //dEulers[1] = GetRandomNumber(-5,5,(double)rand());
            //dEulers[2] = GetRandomNumber(-180,180,(double)rand()); 
            
            ElleSetUnodeAttribute(i,dEulers[0],dEulers[1],dEulers[2],EULER_3);
            //printf("Done with %f %% of %u unodes\n",100.0*((double)i/(double)ElleMaxUnodes()),ElleMaxUnodes());
        }
    }
    
    printf("Finished ...\n");

    err=ElleWriteData("FS_euler3random.elle");
    if(err) OnError("",err);
    
    return 0;
}

/*
 * Output an array of random number between two values A and B of type double
 */
double GetRandomNumber(double dA, double dB, double dRandomisation)
{       
    double dRandomNumber = 0.0;
        
    dRandomNumber = ( dA+(dRandomisation*(dB-dA)) )/RAND_MAX;
    
    if (dA<0)
    {
        dRandomNumber -= dA*(-1.0);       
    }    
    else
    {
        dRandomNumber += dA;
    }
    
    return (dRandomNumber);
}

/*
 * Output an array of random number between two values A and B of type double
 */
double GetRandomNumberElleFunc(double dA, double dB)
{       
    double dRandomNumber = 0.0;
        
    dRandomNumber = dA+(dB-dA)*ElleRandomD();
    
    return (dRandomNumber);
}

/*
 * Add some noise to euler angles within flynns (so initially constant LPO IN 
 * ONE FLYNN has now a noise
 * ---> For that purpose angles have to be transferred from flynns to unodes
 * ---> INPUT: Noise will be +- dNoisePlusMinus
 */
void AddNoise2Flynns(double dNoisePlusMinus)
{
    if (!ElleUnodesActive())
    {
        printf("ERROR (AddNoise2Flynns): Unodes are not active\n");
        return;
    }
    
    if (!ElleUnodeAttributeActive(EULER_3))
        ElleInitUnodeAttribute(EULER_3);
    
    int iMaxFlynns = ElleMaxFlynns();
    vector<int> vUnodesInFlynn;
    int iUnodeID = 0;
    
    double dNoise = 0.0;
    double dAlphaFlynn,dBetaFlynn,dGammaFlynn;
    double dAlpha,dBeta,dGamma;

    for (int i=0;i<iMaxFlynns;i++)
    {
        if (ElleFlynnIsActive(i))
        {
            ElleGetFlynnUnodeList(i,vUnodesInFlynn);
            ElleGetFlynnEuler3(i,&dAlphaFlynn,&dBetaFlynn,&dGammaFlynn);
            for (int j=0;j<vUnodesInFlynn.size();j++)
            {              
                dAlpha = dAlphaFlynn;
                dBeta = dBetaFlynn;
                dGamma = dGammaFlynn;
                iUnodeID = vUnodesInFlynn.at(j);
                /*
                 * Noise has to be determined three times, for alpha, beta and 
                 * gamma, afterwards the angle has be pushed back in the desired 
                 * interval of -180° to 180° or 0-90° respectively
                 */
                dNoise = GetRandomNumber(-dNoisePlusMinus,dNoisePlusMinus,(double)rand());  
                dAlpha += dNoise;
                if (dAlpha < -180) dAlpha += 180;
                if (dAlpha > 180) dAlpha -= 180;
                
                dNoise = GetRandomNumber(-dNoisePlusMinus,dNoisePlusMinus,(double)rand());  
                dBeta += dNoise;
                if (dBeta < 0) dBeta += 90;
                if (dBeta > 90) dBeta -= 90;
                
                dNoise = GetRandomNumber(-dNoisePlusMinus,dNoisePlusMinus,(double)rand());  
                dGamma += dNoise;
                if (dGamma < -180) dGamma += 180;
                if (dGamma > 180) dGamma -= 180;
                                
                ElleSetUnodeAttribute(iUnodeID,dAlpha,dBeta,dGamma,EULER_3); 
            }
            vUnodesInFlynn.clear();
        }
    }
}

/*
 * This function checks if the angles are in the correct intervals for FFT,
 * which are:
 * alpha = [-180,180];
 * beta  = [0, 90 ];
 * gamma = [-180,180];
 * Depending on the input iUseUnodes, it checks the unode or flynn angles, if
 * iUseUnodes == 0, it uses flynns
 *
 * If the angles are not in the correct interval, they are pushed in the correct
 * interval
 */
void CheckAllAngles(int iUseUnodes)
{
    double dAlpha,dBeta,dGamma;
    
    if (iUseUnodes==0)
    {
        // use flynns
        for (int i=0;i<ElleMaxFlynns();i++)
        {
            if (ElleFlynnIsActive)
            {
                ElleGetFlynnEuler3(i,&dAlpha,&dBeta,&dGamma);                
                if (dBeta > 90) // beta should still not be larger than 180
                {
                    dBeta = 90-(dBeta-90);
                    dAlpha += 180;
                }
                
                if (dAlpha < -180) dAlpha += 180;
                if (dAlpha>180) dAlpha -= 360;
                
                if (dGamma < -180) dGamma += 180;
                if (dGamma>180) dGamma -= 360;
            
                ElleSetFlynnEuler3(i,dAlpha,dBeta,dGamma);
            }
        }
    }
    else 
    {
        // use unodes
        for (int i=0;i<ElleMaxUnodes();i++)
        {
            ElleGetUnodeAttribute(i,&dAlpha,&dBeta,&dGamma,EULER_3); 
                           
            if (dBeta > 90)  // beta should still not be larger than 180
            {
                dBeta = 90-(dBeta-90);
                dAlpha += 180;
            }
            
            if (dAlpha < -180) dAlpha += 180;
            if (dAlpha>180) dAlpha -= 360;
            
            if (dGamma < -180) dGamma += 180;
            if (dGamma>180) dGamma -= 360;
        
            ElleSetUnodeAttribute(i,dAlpha,dBeta,dGamma,EULER_3);
        }     
    }
}

/*
 * Does more or less the same than CheckAllAngles, but only for one Euler 
 * orientation specified in the inputs
 */
void CheckEulers(double dEulers[3])
{
    double dAlpha,dBeta,dGamma;
    
    dAlpha = dEulers[0];
    dBeta  = dEulers[1];
    dGamma = dEulers[2];
              
    if (dBeta > 90)   // beta should still not be larger than 180
    {
        dBeta = 90-(dBeta-90);
        dAlpha += 180;
    }
    
    if (dAlpha < -180) dAlpha += 180;
    if (dAlpha>180) dAlpha -= 360;
    
    if (dGamma < -180) dGamma += 180;
    if (dGamma>180) dGamma -= 360;
    
    dEulers[0] = dAlpha;
    dEulers[1] = dBeta;
    dEulers[2] = dGamma;            
}

/* ALTERED FROM TIDY TO WORK WITH UNODES */
int RanorientQuartzUnodes()
{
    int max,unode;
    double curra, currb, currc,dflt1,dflt2,dflt3;
    double dEulers[3];
    double rmap[3][3];
    double eps = 1e-5;
    int mintype;
	
	max = ElleMaxUnodes();
    
	for (unode=0;unode<max;unode++) 		// loop though all unodes
	{
        dEulers[0] = dEulers[1] = dEulers[2] = 0.0;
        orientZXZ(rmap);
        uneulerZXZ(rmap,&dEulers[0],&dEulers[1],&dEulers[2]);
   
        CheckEulers(dEulers);
        
        ElleSetUnodeAttribute(unode,dEulers[0],dEulers[1],dEulers[2],EULER_3);
    }
    
    CheckAllAngles(1);

}

/* FROM TIDY: CALL BY ONLY SETTING FIRST INPUT TO 999*/
int RanorientQuartz()
{
    int max,flynn;
    double curra, currb, currc,dflt1,dflt2,dflt3;
    double rmap[3][3];
    double eps = 1e-5;
    int mintype;
	
	max = ElleMaxFlynns();		// index of maximum flynn used in model
    
	for (flynn=0;flynn<max;flynn++) 		// loop though all flynns
	{
        if (ElleFlynnIsActive(flynn)) // process flynn if it is active
		{
           // std::cout << "blah" << std::endl;
			ElleGetFlynnIntAttribute(flynn, &mintype, MINERAL);
            //std::cout << "blah2" << std::endl;

            if(mintype==QUARTZ || mintype==ICE) // "|| mintype==ICE" added 12th August 2014 FSteinbach
            {
                ElleDefaultFlynnEuler3(&dflt1,&dflt2,&dflt3);
                ElleGetFlynnEuler3(flynn,&curra,&currb,&currc);
                /*printf("curr %.8e %.8e %.8e\ndflt %.8e %.8e %.8e\n\n",
                    curra,currb,currc,
                    dflt1,dflt2,dflt3);*/
            
                /*if (curra==dflt1 && currb==dflt2 && currc==dflt3 ) */
                if (fabs(curra-dflt1) < eps &&
                    fabs(currb-dflt2) < eps &&
                    fabs(currc-dflt3) < eps ) 
                {
                    orientZXZ(rmap);
                    uneulerZXZ(rmap,&curra,&currb,&currc);

                    ElleSetFlynnEuler3(flynn, curra,currb,currc);
                }

            }
       }
    }
    CheckAllAngles(0);

}
