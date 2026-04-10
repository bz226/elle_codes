#include "FS_startGBS.h"
using namespace std;

int main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    /*
     * Initialise and set option names
     */
    ElleInit();
    ElleSetOptNames("GB-phase","ROI-fact","re-space-bnodes","StoreOldProps","unused","unused","unused","unused","unused");
    /* 
     * Give default values for userdata:
     */
    ElleUserData(userdata);
    for (int i=0;i<9;i++) 
        userdata[i]=0; // By default, set all user inputs to 0
        
    userdata[0] = 2; // By default, the ID of the grain boundary phase is = 2
    userdata[1] = 1; // By default, the factor to increase ROI is 1, no increase
                     // and roi radius will be mean unode spacing
                     // --> ROI around bnodes to find near unodes will be 
                     // approx. 1st neighbours
    userdata[2] = 0; // By default, bnodes are not re-spaced to improve 
                     // description of the boundaries in unodes, can be switched
                     // on by typing a dummy switch distance
    userdata[3] = 0; // By default, the old properties of boundary-unodes are 
                     // not stored in a separate textfile. Set to 1 to do this
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    
    ElleSetSaveFrequency(1);
    ElleSetStages(1);
    
    ElleSetInitFunction(InitThisProcess);	
    
    char cFileroot[] = "FS_startGBS.elle";  
    ElleSetSaveFileRoot(cFileroot);
    if (ElleDisplay()) SetupApp(argc,argv);
    
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

    if(!ElleUnodesActive()) 
    {
        printf("Error: No unodes in file\n\n");  
        return 0;
    }      
}

int ProcessFunction()
{
    int err=0;
    UserData userdata;              
    ElleUserData(userdata); 
    int iGBphaseID = (int)userdata[0];
    double dRoiFact = userdata[1];
    double dReSpaceSd = userdata[2];
    int iTmpStoreProps = userdata[3];
    
    /* Prepare filename to store old properties*/
    char cFname[]="TempGBunodesProps.txt";
    if (iTmpStoreProps!=0)
    {
        /* Initialise this temporary file*/
        fstream fPropsFile;
        fPropsFile.open ( cFname, fstream::out | fstream::trunc);
        
        /* Set first line that indicates what columns mean: "isactive?" will
         * either be 0 or 1 depending on if the following attribute is
         * active (1) or not (0)*/
        fPropsFile << "unodeID isactive? U_VISCOSITY isactive? U_DISLOCDEN ";
        fPropsFile << "isactive? E_ALPHA E_BETA E_GAMMA isactive? U_ATTRIB_A ";
        fPropsFile << "isactive? U_ATTRIB_B isactive? U_ATTRIB_C ";
        fPropsFile << "isactive? U_ATTRIB_D isactive? U_ATTRIB_E ";
        fPropsFile << "isactive? U_ATTRIB_F" << endl;
        fPropsFile.close();
    }
    
    double dRoi = 0.0;
    double dRealSwitchDist = ElleSwitchdistance();
    vector<int> vUnodeIDs;       
    if(!CheckFile()) return 0;    
    
    printf("~~~ Start GBS ~~~\n");
    
    /* Set roi radius to "dRoiFact-times" the mean unode spacing.
     * dRoiFact is user input and 1 by default*/
    double dA,dWidth,dHeight,dSSOffset;
    GetBoxInfo(&dA,&dWidth,&dHeight,&dSSOffset); // some of these properties remain unused
    double dMeanUnodeDist = 0.0;
    
    dMeanUnodeDist = dWidth/sqrt(ElleMaxUnodes());    
    dMeanUnodeDist += dHeight/sqrt(ElleMaxUnodes());
    dMeanUnodeDist = dMeanUnodeDist/2.0;
    
    dRoi = dMeanUnodeDist*dRoiFact;
    printf("Set grain boundary phase id (%u) to unodes closer than %e to bnodes\n",iGBphaseID,dRoi);
    
    /* Set bnodes distance to a very low dummy distance to improve the 
     * description of the boundary in unodes: Only if user sets the 3rd input
     * > 0 and > than the actual switch distance */
    if (dReSpaceSd>0)
        if (dReSpaceSd<dRealSwitchDist)
        {
            printf("Temporarily setting a lower switch distance of %e\n",dReSpaceSd);
            ReSpaceBnodesTidy(dReSpaceSd);        
        }
    
    /* Find nearest unode in ROI to each bnode, set to grain boundary phase*/
    printf("Finding near unodes and setting them to grain boundary phase\n");
    for (int bnode=0;bnode<ElleMaxNodes();bnode++)
    {
        vUnodeIDs.clear();
        if (ElleNodeIsActive(bnode))
        { 
            int iCheck = FindNearUnodesToBnode(bnode,dRoi,vUnodeIDs);
            if(iCheck!=0)
            {
                for (int i=0;i<vUnodeIDs.size();i++)
                {
                    // if user wants to: store old props of these unodes
                    if(iTmpStoreProps!=0) TmpStoreProps(vUnodeIDs[i],cFname);                    
                    // set to gb unode
                    ElleSetUnodeAttribute(vUnodeIDs[i],(double)iGBphaseID,iUnodePhaseAttrib);
                }
            }
            else
            {
                if(iCheck<0)
                {
                    printf("Skipping inactive bnode %u\n",bnode);
                    // That means bnode is inactive, should never happen, but
                    // just in case: Go to next bnode
                }
                else
                {
                    printf("WARNING: No unodes found in ROI near bnode %u\n",bnode);
                    /*No near unode found in ROI*/
                    // WHAT TO DO NOW??
                }
            }     
        }
    }
    
    /* Reset bnode spacing */
    if (dReSpaceSd>0)
        if (dReSpaceSd<dRealSwitchDist)
        {
            printf("Resetting to real switch distance (%e)\n",dRealSwitchDist);
            ReSpaceBnodesTidy(dRealSwitchDist); 
        }
    
    if (ElleWriteData(ElleSaveFileRoot())) OnError("",err);
    
    return 0;
}

int FindNearUnodesToBnode(int iBnode,double dROI,vector<int> &vUnodeIds)
{
    /* 
     * Find all unodes within ROI around the bnode, regardless of flynns.
     * Write unode IDs in vector<int> and return number of unodes
     */
    Coords cBnodePos;
    Coords cUnodePos;
    int iCounter=0;
    vUnodeIds.clear();
    
    if (!ElleNodeIsActive(iBnode)) return (-1);
        
    ElleNodePosition(iBnode,&cBnodePos);
    
    for (int unode=0;unode<ElleMaxUnodes();unode++)
    {
        ElleGetUnodePosition(unode,&cUnodePos);
        ElleCoordsPlotXY(&cUnodePos,&cBnodePos);
        
        if (pointSeparation(&cUnodePos,&cBnodePos)<=dROI)
        {
            /* Found a unode within ROI */
            vUnodeIds.push_back(unode);
            iCounter++;            
        }       
    }
    
    return (iCounter);    
}

int ReSpaceBnodesTidy(double spacing)
{
    /* The same function that tidy uses to change bnode spacing */
	int i,j,l;
	int max=0,newmax=0;
    int numtrp=0, numdbl=0, newnumdbl=0;
	
	if(spacing > 0.0)
		ElleSetSwitchdistance(spacing);
	
	/*for(i=5;i>0;i--)*/
    ElleNumberOfNodes(&newnumdbl,&numtrp);
    while (newnumdbl!=numdbl)
	{
		numdbl = newnumdbl;
        max = ElleMaxNodes();
        	for (j=0,l=0;j<max;j++) 
			{
        	    if (ElleNodeIsActive(j))
				/*{*/
	    			/*l++;*/
					/*if(l%i == 0)*/
                		if (ElleNodeIsDouble(j)) 
							ElleCheckDoubleJ(j);
						/*else */
							/*ElleCheckTripleJ(j);*/

        	    /*}*/
        	}
        ElleNumberOfNodes(&newnumdbl,&numtrp);
	}
    ElleAddDoubles();

}

bool CheckFile()
{
    UserData userdata;              
    ElleUserData(userdata); 
    int iGBphase = (int)userdata[0];
    /*
     * Returning true if file is okay, false if there is something wrong
     */
    if(!ElleUnodesActive()) 
    {
        printf("Error: No unodes in file\n\n");  
        return (false);
    }     
    // If phase attribute is inactive, initialse and set all unodes to 1 by
    // default
    if(!ElleUnodeAttributeActive(iUnodePhaseAttrib)) 
    {
        ElleInitUnodeAttribute(iUnodePhaseAttrib);
        if (iGBphase!=1)
            ElleSetDefaultUnodeAttribute(1.0,iUnodePhaseAttrib);        
        else
            ElleSetDefaultUnodeAttribute(2.0,iUnodePhaseAttrib);                
    }       
    
    return (true);
}

void TmpStoreProps(int iUnode,const char *cFname)
{
    /* Store old properties of this unode in a separate textfile. Later, in 
     * FS_endGBS, these properties can be added to the unode again 
     * For each unode, one line with */
    double dAttrib=0.0;
    double dE3[3];
    int iAttrib = 0;
    
    fstream fPropsFile;
    fPropsFile.open ( cFname, fstream::out | fstream::app);
    fPropsFile << iUnode << " ";
    
    iAttrib=U_VISCOSITY;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iUnode,&dAttrib,iAttrib);
        fPropsFile << setprecision(13) << "1 " << dAttrib << " ";
    }
    else
        fPropsFile << "0 0 ";
    
    iAttrib=U_DISLOCDEN;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iUnode,&dAttrib,iAttrib);
        fPropsFile << setprecision(13) << "1 " << dAttrib << " ";
    }
    else
        fPropsFile << "0 0 ";
    
    iAttrib=EULER_3;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iUnode,&dE3[0],&dE3[1],&dE3[2],iAttrib);
        fPropsFile << setprecision(13) << "1 " << dE3[0] << " ";
        fPropsFile << setprecision(13) << dE3[1] << " " << dE3[2] << " ";
    }
    else
    {
        // Still, we need some value and use flynn props:
        int iFlynn = 0;
        iFlynn= ElleUnodeFlynn(iUnode);
        if (ElleFlynnIsActive(iFlynn))
        {
            ElleGetFlynnEuler3(iFlynn,&dE3[0],&dE3[1],&dE3[2]);
            fPropsFile << setprecision(13) << "1 " << dE3[0] << " ";
            fPropsFile << setprecision(13) << dE3[1] << " " << dE3[2] << " ";
        }
        else
        {
            printf("ERROR (TmpStoreProps): Flynn %u of unode %u",iFlynn,iUnode);
            printf(" is inactive\n");
            return;
        }
        
    }
        
    iAttrib=U_ATTRIB_A;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iUnode,&dAttrib,iAttrib);
        fPropsFile << setprecision(13) << "1 " << dAttrib << " ";
    }
    else
        fPropsFile << "0 0 "; 
    
    iAttrib=U_ATTRIB_B;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iUnode,&dAttrib,iAttrib);
        fPropsFile << setprecision(13) << "1 " << dAttrib << " ";
    }
    else
        fPropsFile << "0 0 "; 
    
    iAttrib=U_ATTRIB_C;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iUnode,&dAttrib,iAttrib);
        fPropsFile << setprecision(13) << "1 " << dAttrib << " ";
    }
    else
        fPropsFile << "0 0 "; 
    
    iAttrib=U_ATTRIB_D;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iUnode,&dAttrib,iAttrib);
        fPropsFile << setprecision(13) << "1 " << dAttrib << " ";
    }
    else
        fPropsFile << "0 0 "; 
    
    iAttrib=U_ATTRIB_E;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iUnode,&dAttrib,iAttrib);
        fPropsFile << setprecision(13) << "1 " << dAttrib << " ";
    }
    else
        fPropsFile << "0 0 ";
    
    iAttrib=U_ATTRIB_F;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iUnode,&dAttrib,iAttrib);
        fPropsFile << setprecision(13) << "1 " << dAttrib << " ";
    }
    else
        fPropsFile << "0 0 ";  
 
    fPropsFile << endl;
    fPropsFile.close();
}

void GetBoxInfo(double *dArea,double *dWidth,double *dHeight,double *dSSOffset)
{
    CellData unitcell;
    ElleCellBBox(&unitcell);
    
    Coords box_xy[4];
    
    box_xy[0].x = unitcell.cellBBox[BASELEFT].x;
    box_xy[0].y = unitcell.cellBBox[BASELEFT].y;
    box_xy[1].x = unitcell.cellBBox[BASERIGHT].x;
    box_xy[1].y = unitcell.cellBBox[BASERIGHT].y;
    box_xy[2].x = unitcell.cellBBox[TOPRIGHT].x;
    box_xy[2].y = unitcell.cellBBox[TOPRIGHT].y;
    box_xy[3].x = unitcell.cellBBox[TOPLEFT].x;
    box_xy[3].y = unitcell.cellBBox[TOPLEFT].y;
    
    // Calculate with gaussian euqation for polygon areas with n corners:
    // 2*Area = Σ(i=1 to n)  (y(i)+y(i+1))*(x(i)-x(i+1))
    // if i+1>n use i=1 again (or here in C++ i=0)
    int i2 = 0;
    for (int i=0;i<4;i++)    
    {
        i2 = fmod(i,4)+1;
        *dArea += ( (box_xy[i].y+box_xy[i2].y)*(box_xy[i].x-box_xy[i2].x) )/2;   
    }
    
    *dWidth  = box_xy[1].x-box_xy[0].x;
    *dHeight = box_xy[3].y-box_xy[0].y;
    *dSSOffset = ElleSSOffset();
}
