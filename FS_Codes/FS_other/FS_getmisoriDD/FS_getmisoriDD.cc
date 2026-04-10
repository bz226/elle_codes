#include "FS_getmisoriDD.h"

int iMisoriAttrib = U_ATTRIB_F;
int iPhaseAttrib = VISCOSITY;
int symm_op;
double  symm[24][3][3];
    
int main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    /*
     * Initialise and set option names
     */
    ElleInit();
    ElleSetOptNames("HAGB","ExcludePhase","NoDDUpdate","unused","unused","unused","unused","unused","unused");
    /* 
     * Give default values for userdata (all zero):
     */
    ElleUserData(userdata);
    for (int i=0;i<9;i++)
        userdata[i] = 0;
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    /*
     * We actually only need to run this for one step:
     */
    ElleSetSaveFrequency(1);
    ElleSetStages(1);
    /*
     * Set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);	
	/*
     * Set the default output name for elle files
     */
    char cFileroot[] = "FS_getmisoriDD.elle";
    ElleSetSaveFileRoot(cFileroot);
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
}

int ProcessFunction()
{
    int err=0;
    UserData userdata; 
    ElleUserData(userdata);      
    double dHAGB = userdata[UHAGB];
    double dExcludePhase = userdata[UExcludePhase];
    double dPhaseTmp = 0.0; 
    double dMeanMis = 0.0;    
    double dSumMeanMis = 0.0; 
    
    int iSumUnodesPhase = 0;
        
    InitialChecks();
    
    /*
     * Initialize symmetry.symm file
     */
    symmetry(symm);
    
    /* 
     * Loop through all flynns, check if they are not "excluded" and get their 
     * unodelist
     */
    vector<int> vUnodeList;
    for (int flynn=0;flynn<ElleMaxFlynns();flynn++)
    {
        if (ElleFlynnIsActive(flynn))
        {
            //printf("Processing flynn %u\n",flynn);
            ElleGetFlynnRealAttribute(flynn,&dPhaseTmp,VISCOSITY);
            if (dPhaseTmp!=dExcludePhase)
            {
                /* Get misorientations of unodes of this flynn */
                ElleGetFlynnUnodeList(flynn,vUnodeList);
                GetUnodeMisorientationsInFlynn(flynn,vUnodeList,&dMeanMis);
                
                /* To later calculate the mean misorientation */
                dSumMeanMis += dMeanMis;
                iSumUnodesPhase += vUnodeList.size();
            }
            vUnodeList.clear();
            dMeanMis = 0.0;
        }
    }
    
    /* 
     * Calculate mean misorientation: not sure what to do with it 
     */
    dMeanMis = dSumMeanMis/(double)iSumUnodesPhase;
    
    //printf("Mean misorientation is %f°\n",dMeanMis);
    
    err=ElleWriteData(ElleSaveFileRoot());
    if(err) OnError("",err);
    
    return 0;
}

/*
 * This function stores all kernel average misorientations in the unode 
 * attribute iMisoriAttrib for all unodes in vector<int> vUnodeList
 */
void GetUnodeMisorientationsInFlynn(int iFlynn,vector<int> &vUnodeList,double *dMeanMis)
{
    UserData userdata; 
    ElleUserData(userdata);      
    double dHAGB = userdata[UHAGB];
    
    int iNumbUnodes = vUnodeList.size();
    double dEuler[3];
    double **dNbOri = 0;
    double dMisoris[8];
    double dTotalMisori = 0.0;
    double dTotalDist = 0.0;
    double dKernelAvMisori = 0.0;
    double dMeanDist = 0.0;
    double dDislocden = 0.0;
    double dBurgesVector = 0.0;
    
    int iUnode;
    int iMineralType = 0;
    int iNumNbs=0;
    vector<int> vBndFlag,vNbUnodes;
    
    Coords cUnodeXY;
    
    /* Triangulate unodes: 
     * Necesarry to later find the correct unode neighbours */
    ElleClearTriAttributes();
    TriangulateUnodes(iFlynn,MeshData.tri);
    
    /* To update dislcoation densities from misorientations: Set what MINERAL 
     * this flynn should be and get burgers vector*/
    ElleGetFlynnIntAttribute(iFlynn,&iMineralType,MINERAL);
    dBurgesVector = GetMineralAttribute(iMineralType,BURGERS_VECTOR);
    
    for (int i;i<iNumbUnodes;i++)
    {
        iUnode = vUnodeList.at(i);
        dTotalMisori = 0.0;
        dTotalDist = 0.0;
        
        /* Get unode euler angles and position */
        ElleGetUnodeAttribute(iUnode,&dEuler[0],&dEuler[1],&dEuler[2],EULER_3);
        ElleGetUnodePosition(iUnode,&cUnodeXY); 
        
        /* Convert into radians */                
        dEuler[0]*=DTOR; 
        dEuler[1]*=DTOR; 
        dEuler[2]*=DTOR; 
        
        /* Get information of neighbour unodes */
        ElleGetTriPtNeighbours(iUnode,vNbUnodes,vBndFlag,0);
        iNumNbs = vNbUnodes.size();
        
        if (iNumNbs>0)
        {
            vector<Coords> vcUnodeNbXY(iNumNbs);
            /* Get info for each of nb nodes and put into matrix */    		
            for (int k=0; k<iNumNbs;k++)
            {	
                /* Like in recovery code: Setup a matrix to hold the euler 
                 * angle info plus num of nbs */   			
                if (dNbOri==0) dNbOri=dmatrix(0,iNumNbs,0,5);

                /*
                * FS (!): dNbOri stores orientation of the "k-th" 
                * neighbour. Number of neighbours is thereofre = k
                * i.e.: k = nbnodes.size();
                * Neighbour-id [0]: Euler-alpha [1]: Euler-beta [2]: Euler-gamma  [3] Normalised Distance and [4] Actual distance to center unode (iUnodeID)
                *              dNbOri[k][0]   dNbOri[][1] dNbOri[][2] dNbOri[][3]  dNbOri[][4]
                *      0           53          50          176         0.9        4e-4 
                *      1           ..          ..          ..          ..          ..
                *      2           ..          ..          ..          ..          ..
                *      ..          ..          ..          ..          ..          ..
                *      k-1         ..          ..          ..          ..          ..
                *      k           ..          ..          ..          ..          ..
                */
                ElleGetUnodeAttribute(vNbUnodes[k],
                    &dNbOri[k][0],&dNbOri[k][1],&dNbOri[k][2],EULER_3);
                
                ElleGetUnodePosition(vNbUnodes[k],&vcUnodeNbXY[k]);
                
                /* Transfer all angle to radians */
                dNbOri[k][0]*=DTOR; 
                dNbOri[k][1]*=DTOR; 
                dNbOri[k][2]*=DTOR;                 
            }
            
            /* Get separation to neighbour unodes to normalise later: */
            FS_norm_sep(cUnodeXY,vcUnodeNbXY,dNbOri,iNumNbs);
            
            /* Loop again through all neighbours and determine misorientations*/                		
            for (int k=0; k<iNumNbs;k++)
            {                
                /* Get misorientations for this unode: CME_hex outputs angle
                 * in degree */
                dMisoris[k]=CME_hex(dNbOri[k][0],dNbOri[k][1],dNbOri[k][2],
                                    dEuler[0],dEuler[1],dEuler[2], symm);
                //if(iUnode==4396) printf("nb %u misori: %e\n",vNbUnodes[k],dMisoris[k]);
                if (dMisoris[k] >= dHAGB) dMisoris[k] = dHAGB;
                dTotalMisori += dMisoris[k]*dNbOri[k][3]; // weighted to distance
                dTotalDist += dNbOri[k][4]*dNbOri[k][3]; // weighted to distance
            }
            vcUnodeNbXY.clear();
            
        } // end of if (iNumNbs>0)
        
        /* We went through all neighbours (i.e. the kernel), find kernel average
         * misorientation now: */
        if (iNumNbs > 0)
        {
            dKernelAvMisori = dTotalMisori / (double)iNumNbs;
            dMeanDist = dTotalDist / (double)iNumNbs;
        }     
        else
        {
            dKernelAvMisori = 0.0;
            // With KAM being zero, it doesn't matter which distance we choose
            // as long as it is !=0
            dMeanDist = 1.0;        
        }
        //ElleSetUnodeAttribute(iUnode,(double)iNumNbs,U_ATTRIB_B);
       
        /* Write new orientation to unode attribute:*/
        ElleSetUnodeAttribute(iUnode,iMisoriAttrib,dKernelAvMisori);
        *dMeanMis += dKernelAvMisori;
        
        /* Calculate and set dislocation densities from misorientations*/   
        int iDoNotUpdateDDs=(int)userdata[2];
        if (iNumNbs>0 & iDoNotUpdateDDs==0)
        {     
            dDislocden = FS_Misori2DD(dKernelAvMisori,dMeanDist,dBurgesVector);     
            ElleSetUnodeAttribute(iUnode,U_DISLOCDEN,dDislocden);
        }
        
        /*Clear vectors and free matrix for nb orientations*/
        vNbUnodes.clear();
        vBndFlag.clear();
        if (dNbOri!=0) {free_dmatrix(dNbOri,0,iNumNbs,0,5);dNbOri=0;} 
                   
    } // End of looping through all unodes in the flynn
    
}

double FS_Misori2DD(double dKAM, double dMeanDist, double dBurgersVec)
{
    /* FS:  
     * Using the kernel average misorientation (KAM, in radians!!) to determine the 
     * corresponding dislocation density using:
     * 
     * --> IDEA: Ashby (1970), Borthwick et al. (2013): 
     * dd=KAM/(b*x)
     * 
     * where b is the length of the BUrgers vector and x is the mean distance of
     * all unodes in the kernel to unode of interest. 
     * Actually this is then depending on resolution.
     * 
     * INPUT: 
     * The kernel average misorientation (mean of weighted sum)
     * Mean of weighted sum of all distances from neighbours to this unode
     * The length of the Burgers vector of the material of interest
     */
    
    double dDislocden = 0.0;
    dMeanDist *= ElleUnitLength();    
    dKAM*=DTOR;

    dDislocden = dKAM/(dBurgersVec*dMeanDist);
    
    return (dDislocden);    
}

void symmetry(double symm[24][3][3])
{

    int i,n; 
	FILE *f=fopen("symmetry.symm","r");	
	double dum1,dum2,dum3;
    int ifscanf_return = 0; // FS (!) added this to remove warnings
	
	// first line of with the information of number of symmetry operation 
	ifscanf_return = fscanf(f,"%d",&symm_op);

	// store symmetry operators
     for(n=0;n<symm_op;n++) {
		 
	 	for (i=0;i<3;i++) {
			ifscanf_return = fscanf(f,"%lf %lf %lf",&dum1,&dum2,&dum3);	
			symm[n][i][0]=dum1;
			symm[n][i][1]=dum2;		
			symm[n][i][2]=dum3;
					
	 	}	
	}	

     fclose(f);		
}

double CME_hex(double curra1,double currb1,double currc1,double curra2,double currb2,double currc2, double symm[8][3][3])
{   
    double tmpA, tmpB, angle;
    double val, val2;
    double eps=1e-6;

    double rmap1[3][3];
    double rmap2[3][3];
    double rmapA[3][3],rmapAA[3][3];
    double a11, a22,a33;

	int n,i,j;
	double misang,minang, misang2;	
	double rmap1A[3][3],rmap2A[3][3],rmap1AA[3][3],rmap2AA[3][3];
	double aux_symm[3][3];
	
    //curra1 *= DTOR; currb1 *= DTOR; currc1 *= DTOR;
    //curra2 *= DTOR; currb2 *= DTOR; currc2 *= DTOR;

	orientmat2(rmap1, (double)curra1,(double)currb1,(double)currc1);
	orientmat2(rmap2, (double)curra2,(double)currb2,(double)currc2);
    // euler(rmap1,(double)curra1,(double)currb1,(double)currc1);// gives rotation matrix
    // euler(rmap2,(double)curra2,(double)currb2,(double)currc2);// gives rotation matrix

	// Ini symmetry operators... 
	minang=1000;
	
    for(n=0;n<symm_op;n++) {

		
		// auxiliar symmetry matrix [3][3] 
		for (i=0;i<3;i++) {
			for (j=0;j<3;j++) {
			
			aux_symm[i][j] = symm[n][i][j]; 
  		 		}				
			}		
	
// 1st	
    //calculation of tmpA where the inverse of rmap2 is taken for calculation
    matinverse(rmap2,rmap2A); 		
    matmult(rmap1,rmap2A,rmap1A);

	// symmetry operators			
	matmult(aux_symm,rmap1A,rmapA);	

// 2nd	
    //calculation of tmpAA where the inverse of rmap1 is taken for calculation
    matinverse(rmap1,rmap1A); 		
    matmult(rmap2,rmap1A,rmap2A);

	// symmetry operators			
	matmult(aux_symm,rmap2A,rmapAA);	

	// Take trace.. and misorientation angle	 
    a11=rmapA[0][0];
    a22=rmapA[1][1];
    a33=rmapA[2][2];
   
    val = (a11+a22+a33-1)/2;

    a11=rmapAA[0][0];
    a22=rmapAA[1][1];
    a33=rmapAA[2][2];

    val2 = (a11+a22+a33-1)/2;	
		
    if (val>1.0) val = 1.0;
    else if (val<-1.0) val = -1.0;
    if (val2>1.0) val2 = 1.0;
    else if (val2<-1.0) val2 = -1.0;
		
    misang=(acos(val));
	misang *= RTOD;
    misang2=(acos(val2));
	misang2 *= RTOD;

	if (misang2 < misang) {
		misang=misang2;	
		}
	
	if (misang < minang) {
		minang=misang;	
		}
	
 }
 
 // FS: I CHANGED THE FOLLOWING IF LOOP: IT SAID minang > 120 BEFORE!!!!)
 if (minang > 180 ) minang=180-minang; 

return(minang);
   
}

int InitialChecks()
{
    UserData userdata; 
    ElleUserData(userdata);      
    double dHAGB = userdata[UHAGB];
    double dExcludePhase = userdata[UExcludePhase];
    double dPhaseTmp = 0.0;

    /* FS (!): Check if symmetry.symm file is existing: */
    if (!fileExists("symmetry.symm"))
    {
        printf("\nERROR: File \"symmetry.symm\" is missing\n\n");
        return (0);
    }
    
    /* Check if user has set a valid HAGB angle: */
    if (dHAGB==0) 
    {
        printf("\nERROR: Set HAGB angle > 0°\n\n");
        return (0);
    }
    
    /* Check if flynn attribute MINERAL is active*/
    if (!ElleFlynnAttributeActive(MINERAL))
    {
        printf("\nERROR: Flynn attribute \"MINERAL\" should be active\n\n");
        return (0);        
    }
    
    /* Check if U_EULER_3 is active: */
    if (!ElleUnodeAttributeActive(EULER_3))
    {
        printf("\nERROR: Unode attribute \"U_EULER_3\" should be active\n\n");
        return (0);        
    }

    /* Check if the attribute in which misorientations are stored is 
     * active -> if not: Initiate it, if yes, RE-initiate it deleting all old
     * values: They will be updated by this code*/    
    if (!ElleUnodeAttributeActive(iMisoriAttrib))
    {
        ElleInitUnodeAttribute(iMisoriAttrib);
        ElleSetDefaultUnodeAttribute(0.0,iMisoriAttrib);
    }
    else
    {
        ElleRemoveDefaultUnodeAttribute(iMisoriAttrib);
        ElleInitUnodeAttribute(iMisoriAttrib);
        ElleSetDefaultUnodeAttribute(0.0,iMisoriAttrib);
    }
    
    /* Check if U_DISLOCDEN is active: If not initiate it and set default to 
     * some value*/
    if (!ElleUnodeAttributeActive(U_DISLOCDEN))
    {
        ElleInitUnodeAttribute(U_DISLOCDEN);
        double dDefDD = 0.0;
        ElleSetDefaultUnodeAttribute(dDefDD,U_DISLOCDEN);
    }
        
    /* Check for flynn phase attribute and make it active if it is not active
     * already. Set all phases to != Excluded phase*/
    if (!ElleFlynnAttributeActive(iPhaseAttrib))
    {
        ElleInitFlynnAttribute(iPhaseAttrib);
        if (dExcludePhase!=1)
            ElleSetDefaultFlynnRealAttribute(1.0,iPhaseAttrib);
        else
            ElleSetDefaultFlynnRealAttribute(2.0,iPhaseAttrib);
    }  
    
    return 0;    
}

void orientmat2(double a[3][3], double phi, double rho, double g)
{
    //double a[3][3];
    int i,j;

    a[0][0]=cos(phi)*cos(g)-sin(phi)*sin(g)*cos(rho);
    a[0][1]=sin(phi)*cos(g)+cos(phi)*sin(g)*cos(rho);
    a[0][2]=sin(g)*sin(rho);
    a[1][0]=-cos(phi)*sin(g)-sin(phi)*cos(g)*cos(rho);
    // a[1][1]=sin(phi)*sin(g)+cos(phi)*cos(g)*cos(rho);
    a[1][1]=-sin(phi)*sin(g)+cos(phi)*cos(g)*cos(rho);
    a[1][2]=cos(g)*sin(rho);
    a[2][0]=sin(phi)*sin(rho);
    a[2][1]=-cos(phi)*sin(rho);
    a[2][2]=cos(rho);
}

bool fileExists(const char *filename)
{
  ifstream ifile(filename);
  return ifile;
}

/* 
 * FS: Nearly the same function than in old recovery code, but nbori has 1 more
 * column here, that stores the actual unode separation, not the normalised
 * one
 */
int FS_norm_sep(Coords jxy, vector<Coords> &nbxy, double **nbori, int numnbs)
{
	
	/* this function checks the separation of the nbnodes to the central unode -
	 * makes sure that only nearest nbs are selected and factors in a diagonals
	 */
	 
	Coords  temp;
	int		i;
	int		error=0;
	double	mindist=0;
	
	for (i=0;i<numnbs;i++)
	{
	
		temp=nbxy[i];
	
		ElleCoordsPlotXY(&temp,&jxy);
	
		/* gets the xy positions for the unodes of interest */
	
		nbori[i][4]=pointSeparation(&temp,&jxy);
		
		/* an array of the separation between each points */
				
		if (i==0)
			mindist=nbori[i][4];
		
		else if (nbori[i][4]<mindist)
			mindist = nbori[i][4];
	}	
	
	if (mindist==0) 
    {
        error = 1;
        OnError("norm_sep=0",0);
    }
	
	/* error if the separation is equal to zero*/
	
	for (i=0;i<numnbs;i++)
	{
		nbori[i][3]=mindist/nbori[i][4];			
	
		/* factors in amount for diagonals - values should be either 1 for near
		 * nbs or 0.71 for diagonals in an undistored grid (otherwise closest unode will have nbori[i][3]=1, all others are below)
		 */
         
		//if (numnbs==8) printf("%lf ", nbori[i][3]);	
	}	
	//if (numnbs==8)
	 //printf("numnbs %u\n",numnbs);
	
	return(error);

}
