#include "FS_flynn2unode_attribute.h"
using namespace std;

main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    /*
     * initialise
     */
    ElleInit();
    ElleSetOptNames("Viscosity","Euler_3","Dislocden","F_Attrib_A","unused","unused","unused","unused","unused");
    /* 
     * Give default values for userdata:
     */
    ElleUserData(userdata);
    userdata[Viscosity] =0; // Default: 0
    userdata[Euler_3]   =0; // Default: 0
    userdata[Dislocden] =0; // Default: 0
    userdata[F_Attrib_A]=0; // Default: 0
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    ElleSetSaveFrequency(1);
    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitF2U);	
	/*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("flynn2unode_attribute.elle");
    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);
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
int InitF2U()
{
    char *infile;
    int err=0;
    /*
     * clear the data structures
     */
    ElleSetRunFunction(Start_F2U);
    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
        /*
         * This process assumes unodes are in the elle file
         */
    }
} 
int Start_F2U()
{
    // 0. Optional: Check if all unodes are assigned to the correct flynn
    // and re-assign if not
    CheckUnodesFlynn();
    
    // 1. Check if unodes are in the file:
    if(!ElleUnodesActive())
    {
        printf("\nError: There are no unodes in this elle-file\n\n");
        return 0;
    }
    // 2. Get user input:    
    UserData userdata;
    ElleUserData(userdata);
    int iTransferVisc       =(int)userdata[0];
    int iTransferEuler      =(int)userdata[1];
    int iTransferDD         =(int)userdata[2]; 
    int iTransferAttrib_A   =(int)userdata[3];
    
    // 3. Check which attributes are to be transferred:
    if(iTransferVisc) TransferViscosity();  
    if(iTransferEuler) TransferEuler3();
    if(iTransferDD) TransferDislocden();
    if(iTransferAttrib_A) TransferAttribA();  
   
    // Write the data to new elle-file:    
    int err=ElleWriteData(ElleSaveFileRoot());
    if(err) OnError("",err);
    return 0;
}

/*
 * This Function transfers the VISCOSITY Flynn Attribute to the relevant unodes:
 */
void TransferViscosity()
{
    int iFlynnID=-1;
    double dValue = 0.0;
    
    // 1. Check if VISCOSITY is a flynn attribute
    if (!ElleFlynnAttributeActive(VISCOSITY)) 
    {
        printf("\nError: \"VISCOSITY\" is not a flynn attribute!\n\n");
        return;
    }
    printf("Transferring flynn data \"VISCOSITY\" to unode attrubite"
            " \"U_VISCOSITY\"\n");    
    // 2. Check if unode attribute U_VISCOSITY exists, if not: initialize it:
    if (!ElleUnodeAttributeActive(U_VISCOSITY)) 
        ElleInitUnodeAttribute(U_VISCOSITY);
    
    // 3. Store flynns' VISCOSITY in relevant unodes' U_VISCOSITY
    for (int iUnodeID=0;iUnodeID<ElleMaxUnodes();iUnodeID++)  // cycle through unodes 
    {
	    iFlynnID=ElleUnodeFlynn(iUnodeID);
        ElleGetFlynnRealAttribute(iFlynnID,&dValue,VISCOSITY);
	    ElleSetUnodeAttribute(iUnodeID,U_VISCOSITY,dValue); 
    }
    
}

/*
 * This Function transfers the EULER_3 Flynn Attribute to the relevant unodes:
 */
void TransferEuler3()
{
    int iFlynnID=-1;
    double dEulerAngles[3] = {0.0,0.0,0.0};
    
    // 1. Check if VISCOSITY is a flynn attribute
    if (!ElleFlynnAttributeActive(EULER_3)) 
    {
        printf("\nError: \"EULER_3\" is not a flynn attribute!\n\n");
        return;
    }
    printf("Transferring flynn data \"EULER_3\" to unode attrubite"
            " \"U_EULER_3\"\n");
    // 2. Check if unode attribute U_EULER_3 exists, if not: initialize it:
    if (!ElleUnodeAttributeActive(EULER_3)) 
        ElleInitUnodeAttribute(EULER_3);

    // 3. Store flynns' EULER_3 in relevant unodes' U_EULER_3
    for (int iUnodeID=0;iUnodeID<ElleMaxUnodes();iUnodeID++)  // cycle through unodes 
    {
	    iFlynnID=ElleUnodeFlynn(iUnodeID);
        ElleGetFlynnEuler3(iFlynnID,
            &dEulerAngles[0],
            &dEulerAngles[1],
            &dEulerAngles[2]);
	    ElleSetUnodeAttribute(iUnodeID,
            dEulerAngles[0],
            dEulerAngles[1],
            dEulerAngles[2],EULER_3); 
    }
    
}

/*
 * This Function transfers the DISLOCDEN Flynn Attribute to the relevant unodes:
 */
void TransferDislocden()
{
    int iFlynnID=-1;
    double dValue = 0.0;
    
    // 1. Check if DISLOCDEN is a flynn attribute
    if (!ElleFlynnAttributeActive(DISLOCDEN)) 
    {
        printf("\nError: \"DISLOCDEN\" is not a flynn attribute!\n\n");
        return;
    }
    printf("Transferring flynn data \"DISLOCDEN\" to unode attrubite"
            " \"U_DISLOCDEN\"\n");    
    // 2. Check if unode attribute U_DISLOCDEN exists, if not: initialize it:
    if (!ElleUnodeAttributeActive(U_DISLOCDEN)) 
        ElleInitUnodeAttribute(U_DISLOCDEN);
    
    // 3. Store flynns' DISLOCDEN in relevant unodes' U_DISLOCDEN
    for (int iUnodeID=0;iUnodeID<ElleMaxUnodes();iUnodeID++)  // cycle through unodes 
    {
	    iFlynnID=ElleUnodeFlynn(iUnodeID);
        ElleGetFlynnRealAttribute(iFlynnID,&dValue,DISLOCDEN);
	    ElleSetUnodeAttribute(iUnodeID,U_DISLOCDEN,dValue); 
    }
    
}

/*
 * This Function transfers the F_ATTRIB_A Flynn Attribute to the relevant unodes:
 */
void TransferAttribA()
{
    int iFlynnID=-1;
    double dValue = 0.0;
    
    // 1. Check if F_ATTRIB_A is a flynn attribute
    if (!ElleFlynnAttributeActive(F_ATTRIB_A)) 
    {
        printf("\nError: \"F_ATTRIB_A\" is not a flynn attribute!\n\n");
        return;
    }
    printf("Transferring flynn data \"F_ATTRIB_A\" to unode attrubite"
            " \"U_ATTRIB_A\"\n");    
    // 2. Check if unode attribute U_ATTRIB_A exists, if not: initialize it:
    if (!ElleUnodeAttributeActive(U_ATTRIB_A)) 
        ElleInitUnodeAttribute(U_ATTRIB_A);
    
    // 3. Store flynns' F_ATTRIB_A in relevant unodes' U_ATTRIB_A
    for (int iUnodeID=0;iUnodeID<ElleMaxUnodes();iUnodeID++)  // cycle through unodes 
    {
	    iFlynnID=ElleUnodeFlynn(iUnodeID);
        ElleGetFlynnRealAttribute(iFlynnID,&dValue,F_ATTRIB_A);
	    ElleSetUnodeAttribute(iUnodeID,U_ATTRIB_A,dValue); 
    }
    
}

void CheckUnodesFlynn()
{
/*
 * FS: In the beginning of the process, this function checks if each unode is
 * assigned to the correct flynn -> i.e. if flynn ID is correctly stored in
 * U_ATTRIB_C (or I should say in: iUnodeFlynnNumber), also the flynn ID should 
 * be correctly stored in a flynn attribute, this is also checked.
 */
    printf("FS_flynn2unode_attribute: Checking if all unodes are ");
    printf("assigned to correct flynn\n");
    int iFlynnID;
    Coords cUnodeXY;
    
    for (int unode=0;unode<ElleMaxUnodes();unode++)
    {
        ElleGetUnodePosition(unode,&cUnodeXY);
        iFlynnID = ElleUnodeFlynn(unode);
        
        /* Check if unode is really in this flynn: 
         * If yes, update unode and flynn attribute storing the correct
         * flynn id, if not find the flynn, the unode is in at the moment
         * and update everything afterwards*/
        if (EllePtInRegion(iFlynnID,&cUnodeXY))
        {
            /* unode is assigned to the correct flynn */
            ElleSetUnodeAttribute(unode,U_ATTRIB_C,(double)iFlynnID);
            ElleSetFlynnRealAttribute(iFlynnID,(double)iFlynnID,F_ATTRIB_C);            
        }
        else
        {
            /* unode is NOT assigned to the correct flynn, now we will have to 
             * loop through all flynn to find the correct one */
            bool bFound = false;
            int flynn = 0;
            while (flynn<ElleMaxFlynns() && !bFound)
            {
                if (ElleFlynnIsActive(flynn))
                {
                    if (EllePtInRegion(flynn,&cUnodeXY))
                    {
                        /* THIS is the correct flynn, update values now: */
                        ElleAddUnodeToFlynn(flynn,unode);
                        ElleSetUnodeAttribute(unode,U_ATTRIB_C,(double)flynn);
                        ElleSetFlynnRealAttribute(flynn,(double)flynn,F_ATTRIB_C);  
                        bFound = true;                      
                    }                   
                }
                flynn++;
            } // end of while loop going through all flynns
        } // end of if if (EllePtInRegion(iFlynnID,&cUnodeXY))
        
    }  // end of looping through all unodes
}
