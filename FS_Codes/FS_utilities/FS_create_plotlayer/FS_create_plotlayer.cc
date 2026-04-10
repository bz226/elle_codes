#include "FS_create_plotlayer.h"
using namespace std;

main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    /*
     * initialise
     */
    ElleInit();
    ElleSetOptNames("UnodeDeletion","MergeFlynns","ScaleFactor","Phase2Value","Attribute","unused","unused","unused","unused");
    /* 
     * Give default values for userdata:
     */
    ElleUserData(userdata);
    userdata[UnodeDeletion]=-1; // Default: -1. Set to X if all unodes @ U_VISCOSITY=X should be deleted, otherwise, set to 0 and the following parameters are regarded:
    userdata[MergeFlynns]=0; // Default: 0, set to another value if you wish to us it, which is the value of the Phase from which flynns should be merged for each cluster
    userdata[ScaleBox]=0; // Default: 0, do not scale the box and bnodes/unodes at all. To use scaling type value > 1 to upsacle and < 1 to downscale the box
    userdata[Phase2Value]=-1; // Default: -1 (i.e. do nothing) 0, means 2nd phase is set to max. value of attribute, any other input will be the max. value (type double)
    userdata[Attribute]=0; // Default: 0, used for U_DISLOCDEN
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    /*
     * Set the interval for writing file and stages: Topology checks only need 
     * to be performed once 
     */
    ElleSetSaveFrequency(1);
    ElleSetStages(1);
    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);	
	/*
     * set the base for naming statistics and elle files
     */
    char cFileroot[] = "with_plotlayer.elle";
    ElleSetSaveFileRoot(cFileroot);
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
}

/* 
 * Anything can now be in the ProcessFunction itself:
 */ 
int ProcessFunction()
{
    UserData userdata;            
    ElleUserData(userdata);  
    int iUUnodeDeletion = (int)userdata[UnodeDeletion];
    int iUMergeFlynns = (int)userdata[MergeFlynns];  
    double dScaleFactor = userdata[ScaleBox]; 
    int iUDoPlotlayer = (int)userdata[Phase2Value];
    int err = 0;
    
    if(!ElleUnodeAttributeActive(U_VISCOSITY) && iUUnodeDeletion > 0)
    {
        printf("\nATTENTION: Unode attribute \"U_VISCOSITY\" is not active:\n");
        printf("Please add it and use FS_flynn2unode_attribute and\nadd flynn viscosity to unode viscosty before!!\n\n");
        return 0;
    }    
    
    if (dScaleFactor!=0)
    {
        ScaleModelBox(dScaleFactor); 
        if (iUUnodeDeletion==0 & iUMergeFlynns==0 & iUDoPlotlayer==-1)
        {
            // nothing else to be done, write Elle file
            if(err=ElleWriteData(ElleSaveFileRoot())) OnError("",err); 
        }            
    }
    
    // Check for Merge Flynn function
    if (iUMergeFlynns!=0)
    {
        cout << "Merging Flynns of phase " << iUMergeFlynns << endl;  
        if (err=MergePhaseFlynns(iUMergeFlynns))
            OnError("Merging of flynns failed",err);   
        
        if(err=ElleWriteData(ElleSaveFileRoot())) 
            OnError("",err); 
    }
    
    // Unode deletion: HAS TO BE THE LAST STEP SINCE UNODES ARE ADDED "MANUALLY WITHOUT THE ElleWriteData" FUNCTION
    if (iUUnodeDeletion == 0 && iUDoPlotlayer>=0)
    {
        cout << "Using plotlayer function" << endl;
        if (err=CreatePlotlayer())
            OnError("",err);
    }
    if (iUUnodeDeletion > 0)
    {
        cout << "Using unode deletion for phase " << iUUnodeDeletion << endl;
        cout << " ---> Attribute U_FINITE_STRAIN will be lost because of this " << endl;   
        if (err=DeleteUnodes())
                OnError("",err);
    }
    
    if (iUUnodeDeletion<0 && iUMergeFlynns==0)
    {
        // Still necessary to save the file:
        if(err=ElleWriteData(ElleSaveFileRoot())) 
            OnError("",err); 
    }
        
    return 0;
} 

void ScaleModelBox(double dScaleFactor)
{
    // Upate unit cell:
    printf("Scaling the box by a factor of %f\n",dScaleFactor);
    CellData unitcell;
    ElleCellBBox(&unitcell);
    Coords  corners[4];
    
    corners[BASELEFT].x = unitcell.cellBBox[BASELEFT].x*dScaleFactor;
    corners[BASELEFT].y = unitcell.cellBBox[BASELEFT].y*dScaleFactor;
    corners[BASERIGHT].x = unitcell.cellBBox[BASERIGHT].x*dScaleFactor;
    corners[BASERIGHT].y = unitcell.cellBBox[BASERIGHT].y*dScaleFactor;
    corners[TOPRIGHT].x = unitcell.cellBBox[TOPRIGHT].x*dScaleFactor;
    corners[TOPRIGHT].y = unitcell.cellBBox[TOPRIGHT].y*dScaleFactor;
    corners[TOPLEFT].x = unitcell.cellBBox[TOPLEFT].x*dScaleFactor;
    corners[TOPLEFT].y = unitcell.cellBBox[TOPRIGHT].y*dScaleFactor;
    
    ElleSetCellBBox(&corners[BASELEFT], &corners[BASERIGHT],&corners[TOPRIGHT], &corners[TOPLEFT]);
    
    // Update bnode positions
    Coords cBnodeXY;
    
    for (int i=0;i<ElleMaxNodes();i++)
    {
        if (ElleNodeIsActive(i))
        {
            ElleNodePosition(i,&cBnodeXY);
            cBnodeXY.x *= dScaleFactor;    
            cBnodeXY.y *= dScaleFactor; 
            ElleSetPosition(i,&cBnodeXY);        
        }
    }
    
    // Update unode positions
    Coords cUnodeXY;
    
    for (int i=0;i<ElleMaxUnodes();i++)
    {
        ElleGetUnodePosition(i,&cUnodeXY);
        cUnodeXY.x *= dScaleFactor;
        cUnodeXY.y *= dScaleFactor; 
        ElleSetUnodePosition(i,&cUnodeXY);        
    }
}

int CreatePlotlayer()
{
    UserData userdata;            
    ElleUserData(userdata);    
    double d2ndPhaseValue = userdata[Phase2Value];  
    int iUserAttribute = (int)userdata[Attribute];
    int iAttribute = 7,err=0;
    bool bUseEulerCode = false;
    
    /*
     * Some checks:
     */
    if(!ElleUnodesActive()) 
    {
        printf("Attention: No unodes in file\n\n");  
    }     
    if (!ElleUnodeAttributeActive(iPlotLayerAttribute)) 
        ElleInitUnodeAttribute(iPlotLayerAttribute); 
    
    /*
     * End of checks
     */
    
    switch(iUserAttribute)
	{
		case 0: 
            printf("Processing data from U_DISLOCDEN.\n");
            iAttribute = U_DISLOCDEN;
			break; 
		case 1: 
            printf("Processing data from U_EULER_3.\n");
            printf("ATTENTION: Code is only transfering euler alpha.\n");
            iAttribute = EULER_3;
			break;
		case 2: 
            printf("Processing data from U_ATTRIB_A.\n");
            iAttribute = U_ATTRIB_A;
			break;
		case 3: 
            printf("Processing data from U_ATTRIB_B.\n");
            iAttribute = U_ATTRIB_B;
			break;
		case 4: 
            printf("Processing data from U_ATTRIB_C.\n");
            iAttribute = U_ATTRIB_C;
			break;
		case 5: 
            printf("Processing data from U_ATTRIB_D.\n");
            iAttribute = U_ATTRIB_D;
			break;
		case 6:
            printf("Processing data from U_ATTRIB_E.\n");
            iAttribute = U_ATTRIB_E;
			break;
        default : // in case of invalid user input
            printf("ERROR: %d is an invalid input for the attribute.\n",iUserAttribute);
            return 0;
	}
    
    
    // SOME CHECKS:
    // Check if unode attribute is in Elle-file
    if(!ElleUnodeAttributeActive(iAttribute))
    {
        printf("ERROR: Unode attribute not in file.\n");
        return 0;
    }
    
    // Check if Attribute was U_EULER_3, if yes, another code has to be used later:
    if(iAttribute==EULER_3)
        bUseEulerCode = true;
        
    // Perform the code to transfer the values etc:
    
    if(bUseEulerCode)
    {
        if(TransferAttributeEuler(d2ndPhaseValue,iAttribute))
            printf("TransferAttributeEuler - Some errors reading the unode attributes!\n");
    }
    else
    {
        if(TransferAttributes(d2ndPhaseValue,iAttribute))
            printf("TransferAttributes - Some errors reading the unode attributes!\n");
    }
    
    /*
     * Finally the data can potentially saved in a new Elle file:
     */  
    //if(err=ElleWriteData(ElleSaveFileRoot())) 
        //OnError("",err);
    
    return err;
}

int DeleteUnodes()
{
    /*
     * FS: My idea how to do that:
     *
     * 1) Loop through all unodes
     *      -> Get vector or array containing all unode attributes in file apart 
     *      from viscosity==desired value (those can potentially be id,x,y,
     *      dislocden,e-aplha,e-beta,e-gamma,u_attrib_a - f) by pushing back the
     *      value in the initially empty vector for the attribute
     * 
     *      -> count how often viscosity == desired value and store value for 
     *      new ID list
     * 
     * 2) Delete all unodes -> Their values are stored in the vectors
     * 3) Write output elle file without unodes
     * 4) Write vector content in outputelle file manually
     */
    int err = 0;
    UserData userdata;
    ElleUserData(userdata);
    int iDeletePhase = userdata[UnodeDeletion];
    int iCounter = 0;
    double dTempVisc = 0.0;
    double dTempAttribute1 = 0.0;
    double dTempAttribute2 = 0.0;
    double dTempAttribute3 = 0.0;
    Coords cXYUnode;
    
    vector<int> vUnodeID;
    vector<double> vX,vY,vDD,vAlpha,vBeta,vGamma,vA,vB,vC,vD,vE,vF;
    
    for (int i=0;i<ElleMaxUnodes();i++)
    {
        ElleGetUnodeAttribute(i,&dTempVisc,U_VISCOSITY);
        
        /*
         * If this attribute is not belonging to the phase from which unodes
         * shall be deleted:
         * Add attribute value at the end of the relevant vector and store 
         * correct new unode ID by using the counting value iCounter
         */
        if ((int)dTempVisc != iDeletePhase)
        {
            vUnodeID.push_back(iCounter);
            
            ElleGetUnodePosition(i,&cXYUnode);
            vX.push_back(cXYUnode.x);
            vY.push_back(cXYUnode.y);
            
            if (ElleUnodeAttributeActive(U_DISLOCDEN))
            {
                ElleGetUnodeAttribute(i,U_DISLOCDEN,&dTempAttribute1);
                vDD.push_back(dTempAttribute1);
            }
            if (ElleUnodeAttributeActive(EULER_3))
            {
                ElleGetUnodeAttribute(i,&dTempAttribute1,&dTempAttribute2,&dTempAttribute3,EULER_3);
                vAlpha.push_back(dTempAttribute1);
                vBeta.push_back(dTempAttribute2);
                vGamma.push_back(dTempAttribute3);
            }
            if (ElleUnodeAttributeActive(U_ATTRIB_A))
            {
                ElleGetUnodeAttribute(i,U_ATTRIB_A,&dTempAttribute1);
                vA.push_back(dTempAttribute1);
            }
            if (ElleUnodeAttributeActive(U_ATTRIB_B))
            {
                ElleGetUnodeAttribute(i,U_ATTRIB_B,&dTempAttribute1);
                vB.push_back(dTempAttribute1);
            }
            if (ElleUnodeAttributeActive(U_ATTRIB_C))
            {
                ElleGetUnodeAttribute(i,U_ATTRIB_C,&dTempAttribute1);
                vC.push_back(dTempAttribute1);
            }
            if (ElleUnodeAttributeActive(U_ATTRIB_D))
            {
                ElleGetUnodeAttribute(i,U_ATTRIB_D,&dTempAttribute1);
                vD.push_back(dTempAttribute1);
            }
            if (ElleUnodeAttributeActive(U_ATTRIB_E))
            {
                ElleGetUnodeAttribute(i,U_ATTRIB_E,&dTempAttribute1);
                vE.push_back(dTempAttribute1);
            }
            if (ElleUnodeAttributeActive(U_ATTRIB_F))
            {
                ElleGetUnodeAttribute(i,U_ATTRIB_F,&dTempAttribute1);
                vF.push_back(dTempAttribute1);
            }
            
            iCounter++;
        }
    }
    
    /*
     * Delete all unodes: Their values are stored in the vectors now!
     */    
    UnodesClean();
    /*
     * Save new Elle file - unodes are added manually later:
     */  
    if(err=ElleWriteData(ElleSaveFileRoot())) 
        OnError("",err);
    
    /*
     * Now add Unodes where they should not be deleted manually:
     */
    fstream fElleFile;
    fElleFile.open ( ElleSaveFileRoot(), fstream::out | fstream::app);
        
    fElleFile << "UNODES" << endl;    
    for (int j=0;j<iCounter;j++)
        fElleFile << j << " " << vX.at(j) << " " << vY.at(j) << endl;
        
    if (vDD.size()>0)
    {
        fElleFile << "U_DISLOCDEN" << endl;    
        for (int j=0;j<iCounter;j++)
            fElleFile << j << " " << vDD.at(j) << endl;
    }    
    if (vAlpha.size()>0)
    {
        fElleFile << "U_EULER_3" << endl;    
        for (int j=0;j<iCounter;j++)
            fElleFile << j << " " << vAlpha.at(j) << " " << vBeta.at(j) << " " << vGamma.at(j) << endl;
    }    
    if (vA.size()>0)
    {
        fElleFile << "U_ATTRIB_A" << endl;    
        for (int j=0;j<iCounter;j++)
            fElleFile << j << " " << vA.at(j) << endl;
    }    
    if (vB.size()>0)
    {
        fElleFile << "U_ATTRIB_B" << endl;    
        for (int j=0;j<iCounter;j++)
            fElleFile << j << " " << vB.at(j) << endl;
    }    
    if (vC.size()>0)
    {
        fElleFile << "U_ATTRIB_C" << endl;    
        for (int j=0;j<iCounter;j++)
            fElleFile << j << " " << vC.at(j) << endl;
    }    
    if (vD.size()>0)
    {
        fElleFile << "U_ATTRIB_D" << endl;    
        for (int j=0;j<iCounter;j++)
            fElleFile << j << " " << vD.at(j) << endl;
    }    
    if (vE.size()>0)
    {
        fElleFile << "U_ATTRIB_E" << endl;    
        for (int j=0;j<iCounter;j++)
            fElleFile << j << " " << vE.at(j) << endl;
    }    
    if (vF.size()>0)
    {
        fElleFile << "U_ATTRIB_F" << endl; 
        fElleFile << "Default 0" << endl; 
        for (int j=0;j<iCounter;j++)
            fElleFile << j << " " << vF.at(j) << endl;
    }
           
    fElleFile.close();
    
    return err;
}

void TransferFlynnViscosity2Unodes()
{
    
}

int TransferAttributes(double d2ndPhaseValue,int iAttribute)
{
    double d2ndPhaseValueTemp = (-1e6),dPhase=0.0;
    int iMaxUnodes = ElleMaxUnodes();
    int i=0,err=0,iFlynnID=0;
    
    if (d2ndPhaseValue==0) // USE MAXIMUM VALUE IN THE RELEVANT ATTRIBUTE
    {
        // Get maximum value of this attribute:
        for(i=0;i<iMaxUnodes;i++)
        {
            if(ElleGetUnodeAttribute(i,&d2ndPhaseValueTemp,iAttribute))
            {
                printf("ERROR: No attribute found for unode %d\n",i);
                err=1;
            }
            if(d2ndPhaseValueTemp>d2ndPhaseValue)
                d2ndPhaseValue = d2ndPhaseValueTemp;
        }
        
        // Set all unodes in U_ATTRIB_F to this value if flynn VISCOSITY==2
        for(i=0;i<iMaxUnodes;i++)
        {
            iFlynnID=ElleUnodeFlynn(i);
            ElleGetFlynnRealAttribute(iFlynnID,&dPhase,VISCOSITY);
            if(dPhase==2)
            {
                ElleSetUnodeAttribute(i,iPlotLayerAttribute,d2ndPhaseValue);
            }
            else
            {
                if(ElleGetUnodeAttribute(i,&d2ndPhaseValueTemp,iAttribute))
                {
                    printf("ERROR: No attribute found for unode %d\n",i);
                    err=1;
                }
                
                ElleSetUnodeAttribute(i,iPlotLayerAttribute,d2ndPhaseValueTemp);                        
            }
        }            
    }
    else // d2ndPhaseValue!=0: USE USER INPUT FOR THE VALUE FOR 2ND PHASE UNODES
    {
        // Set all unodes in U_ATTRIB_F to this value if flynn VISCOSITY==2
        for(i=0;i<iMaxUnodes;i++)
        {
            iFlynnID=ElleUnodeFlynn(i);
            ElleGetFlynnRealAttribute(iFlynnID,&dPhase,VISCOSITY);
            if(dPhase==2)
            {
                ElleSetUnodeAttribute(i,iPlotLayerAttribute,d2ndPhaseValue);
            }
            else
            {
                if(ElleGetUnodeAttribute(i,&d2ndPhaseValueTemp,iAttribute))
                {
                    printf("ERROR: No attribute found for unode %d\n",i);
                    err=1;
                }
                
                ElleSetUnodeAttribute(i,iPlotLayerAttribute,d2ndPhaseValueTemp);                        
            }
        } 
    }
    return err;
}

int TransferAttributeEuler(double d2ndPhaseValue,int iAttribute)
{
    double d2ndPhaseValueTemp = (-1e6),dPhase=0.0;
    int iMaxUnodes = ElleMaxUnodes();
    int i=0,err=0,iFlynnID=0;
    double dBeta=0.0,dGamma=0.0; // at the moment those are just dummy attributes
    
    if (d2ndPhaseValue==0) // USE MAXIMUM VALUE IN THE RELEVANT ATTRIBUTE
    {
        // Get maximum value of this attribute:
        for(i=0;i<iMaxUnodes;i++)
        {
            if(ElleGetUnodeAttribute(i,&d2ndPhaseValueTemp,&dBeta,&dGamma,iAttribute))
            {
                printf("ERROR: No attribute found for unode %d\n",i);
                err=1;
            }
            if(d2ndPhaseValueTemp>d2ndPhaseValue)
                d2ndPhaseValue = d2ndPhaseValueTemp;
        }
        
        // Set all unodes in U_ATTRIB_F to this value if flynn VISCOSITY==2
        for(i=0;i<iMaxUnodes;i++)
        {
            iFlynnID=ElleUnodeFlynn(i);
            ElleGetFlynnRealAttribute(iFlynnID,&dPhase,VISCOSITY);
            if(dPhase==2)
            {
                ElleSetUnodeAttribute(i,d2ndPhaseValue,iPlotLayerAttribute);
            }
            else
            {
                if(ElleGetUnodeAttribute(i,&d2ndPhaseValueTemp,&dBeta,&dGamma,iAttribute))
                {
                    printf("ERROR: No attribute found for unode %d\n",i);
                    err=1;
                }
                ElleSetUnodeAttribute(i,d2ndPhaseValueTemp,iPlotLayerAttribute);                        
            }
        }            
    }
    else // d2ndPhaseValue!=0: USE USER INPUT FOR THE VALUE FOR 2ND PHASE UNODES
    {
        // Set all unodes in U_ATTRIB_F to this value if flynn VISCOSITY==2
        for(i=0;i<iMaxUnodes;i++)
        {
            iFlynnID=ElleUnodeFlynn(i);
            ElleGetFlynnRealAttribute(iFlynnID,&dPhase,VISCOSITY);
            if(dPhase==2)
            {
                ElleSetUnodeAttribute(i,d2ndPhaseValue,iPlotLayerAttribute);
            }
            else
            {
                if(ElleGetUnodeAttribute(i,&d2ndPhaseValueTemp,&dBeta,&dGamma,iAttribute))
                {
                    printf("ERROR: No attribute found for unode %d\n",i);
                    err=1;
                }
                ElleSetUnodeAttribute(i,d2ndPhaseValueTemp,iPlotLayerAttribute);                        
            }
        } 
    }
    return err;
}

int MergePhaseFlynns(int iPhase)
{
    /*
     * IDEA: Merge Flynns that have the same phase and hence belong to the same 
     * cluster
     */   
    int err = 0, iCounter = 1;
    bool bFlynnHasSamePhaseNbs;
    float fFlynnViscFlynn1 = 0.0, fFlynnViscFlynn2 = 0.0;
	list<int> lFlynnsNbs;
    
    for (int i=0;i<ElleMaxFlynns();i++)
    {
        iCounter = 1;
        if (ElleFlynnIsActive(i))
        {
            ElleGetFlynnViscosity(i,&fFlynnViscFlynn1);
            if ((int)fFlynnViscFlynn1 == iPhase)
            {
                /*
                 * Iteratively find all same phase neighbours of a flynn and 
                 * merge them together until no neighbours are found any more:
                 *
                 * Flynn with ID "i" is growing because of this. Hence, two 
                 * "while" loops are necessary: First one just checks of there
                 * are any neighbours left, 2nd one loops through those 
                 * neighbours
                 */
                 
                while (iCounter!=0) // i.e.: while there are flynns being merged
                {
                    ElleFlynnNbRegions( i, lFlynnsNbs );
                    iCounter = 0;
                    while (lFlynnsNbs.size() > 0)
                    {
                        ElleGetFlynnViscosity(lFlynnsNbs.front(),&fFlynnViscFlynn2);
                        if (fFlynnViscFlynn2 == iPhase)
                        {
                            // Phases are the same --> Merge the flynns:
                            if (CheckPossibleWrappingFlynn(i,lFlynnsNbs.front()))
                            {
                                printf("Warning - Not merging flynn %u and %u: Merge will cause wrapping flynn\n",i,lFlynnsNbs.front());
                            }
                            else
                            {
                                ElleMergeFlynns(i,lFlynnsNbs.front());
                                iCounter++;
                            }
                        }
                        /* 
                         * Remove the first element in neighbour flynns list 
                         * since this first flynn has already been checked and 
                         * eventually merged by now:
                         */
                        lFlynnsNbs.pop_front();                            
                    }
                    lFlynnsNbs.clear();
                }                               
            }
        }
    }
    
    return err;
}

/*
 * This function checks if a flynn will become a wrapping flynn when merging two
 * other flynns. If yes it returns 1, if not 0.
 */
int CheckPossibleWrappingFlynn(int iKeepFlynn,int iRemoveFlynn)
{    
    int iCheck = 0; //Will be 1 if flynn wraps after merge
    int iNode, *ids, start=NO_NB, finish=NO_NB, iNumberNodes = 0;
    int iCount = 0;
    vector<Flynn> vFlynnVector;
    
    vFlynnVector[iKeepFlynn].flynnNodes(iKeepFlynn, &ids, &iNumberNodes);
    for (int i=0;i<iNumberNodes;i++) 
    {
        iNode = ids[i];
        if (ElleNodeIsTriple(iNode) &&
            ElleFindBndIndex(iNode,iRemoveFlynn)!=NO_NB) 
        {
            if (start==NO_NB) start=iNode;
            else if (finish==NO_NB) finish=iNode;
            iCount++;
        }
    }
    free(ids);
    
    // iCount should be 2 if valid for split
    // iCount of 4 may indicate split wrapping grain which should
    //   not be merged
    if (iCount!=2) 
    {
        iCheck=1;
        //if (iCount==4) 
        //{
            //cout << "merge will cause wrapping flynn" << endl;
        //}
        //else if (iCount) 
            //cout << "invalid TJ count " << iCount << endl;
    }
    
    return iCheck;    
}
