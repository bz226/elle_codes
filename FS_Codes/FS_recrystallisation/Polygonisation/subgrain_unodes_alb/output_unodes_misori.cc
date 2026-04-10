/*************************************************************************
 * Please do not delete/modify next information! 
 * Author: Albert Griera
 * Institution: Departament Geologia, Universitat Autònoma de Barcelona
 * Email: albert.griera@uab.cat
 * Utility: Define subgrains using unodes data
 * version:2.0
 * Date: 05.2014
 * This code is non-transferable without previous  agreement between user and authors 
 ***************************************************************************/

// Important Issues
/* This process makes a list of subgrains per flynn using unodes data and the relative disorientation. 
 * user[0] : Unodes grid size 
 * user[1] : Threshold between low and high angle grain boundaries
 * The code uses the symmetry.symm file
 * Provisionally subgrain data are stored in U_DENSITY. In the U_ATTRIB_B and U_ATTRIB_C store dummy variables that are unnecessary after the process  
 */

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include "error.h"
#include "file.h"
#include "init.h"
#include "runopts.h"
#include "interface.h"
#include "unodes.h"
#include "convert.h"
#include "mat.h"
#include "update.h"
#include "plot_ugrid.h"


int InitThisProcess(), ProcessFunction();
double misorientation(int unode_id1,int unode_id2, double symm[24][3][3],int symm_op);
void orientmat2(double a[3][3], double phi, double rho, double g);
int symmetry_load(double symm[24][3][3]);

int ugrid_size, totalgrain=0, sindex;
double lagb,hagb;
double symm[24][3][3];
int symm_op;

using std::vector;
using std::list;
using std::pair;
// using std::iterator;

/* 
 * FS Changes:
 * Introduced an "excluded phase" for which no subgrains will be created (i.e.
 * since it is air etc.)
 * 		--> Change that maybe to setting another critical angle to different 
 * 			phases, for air this angle could than be higher than any possible
 * 			value of misorientation
 */
int iPhaseAttrib = VISCOSITY;
int iExcludedPhase = 2;

/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
 
int InitThisProcess()
{
    char *infile;
    int err=0;
    UserData userdata;

    ElleUserData(userdata);
    ugrid_size=(int)userdata[UGridSize]; // Set grid size
    hagb= userdata[1]; // transition between low to high angle grain boundary
    iExcludedPhase = (int)userdata[2]; // FS added this
    sindex= (int)userdata[3];
    printf("hagb: %f°\n",hagb);	
    printf("excluded phase: %u\n",iExcludedPhase);	// FS added this
    printf("sindex: %u\n",sindex);	
	
    /*
     * clear the data structures
     */
    ElleSetRunFunction(ProcessFunction);

    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
			
		if (!ElleUnodeAttributeActive(U_ATTRIB_A))
            ElleInitUnodeAttribute(U_ATTRIB_A);
		if (!ElleUnodeAttributeActive(U_ATTRIB_B))
            ElleInitUnodeAttribute(U_ATTRIB_B);
		if (!ElleUnodeAttributeActive(U_ATTRIB_C))
            ElleInitUnodeAttribute(U_ATTRIB_C);
		if (!ElleUnodeAttributeActive(U_DENSITY)) // store the subgrain number, start with a default value of zero
            ElleInitUnodeAttribute(U_DENSITY);

    }
    
    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("unode_grain");	
   // if (err=ElleWriteData("unode_grain.elle"))
     //   OnError("",err);

	
}

int ProcessFunction()
{    
    /*
     * FS: Check if symmetry file exists
     */
    if(!fileExists("symmetry.symm"))
    {
        printf("ERROR: File \"symmetry.symm\" not found in directory\n");
        return(0);
    }
    
    int i,j,m;
	int unode_id[ugrid_size][ugrid_size];
	int unode_1, unode_2,i_1,j_1,j_2, i_2;
	double misangle, mean_misorient;
	double inc_shear, dummy=0, dummy2;
    double val[3], val1[3], minang;
    
    /* FS: To see if a unode belongs to a flynn of the excluded phase:*/
    double dPhase = 0.0;
    bool bDoSubgrain = true;
    double dSaveHAGB = 0.0;

	int grain, grain_2, k,l,idummy;
	int k_1, l_1; // sindex=3
	int iter, max_unodes, flynn_1;
	vector<int> tmpunodes;
	int count;
	
    CellData unitcell;
    UserData userdata;
	max_unodes = ElleMaxUnodes();
	
	// crystal symmetry
    symm_op=symmetry_load(symm);
	
	// store unode_id in an int(matrix)
	for (j=0;j<ugrid_size;j++) 
	{	
		for (i=0;i<ugrid_size;i++) 
		{	
			unode_id[i][j]=j*ugrid_size+i;
			ElleSetUnodeAttribute(unode_id[i][j],dummy,U_DENSITY); // init to zero value 
		}
	} 

	// loop over all unodes
	for (j=0;j<ugrid_size;j++) 
	{	
		for (i=0;i<ugrid_size;i++) 
		{			
			unode_1=unode_id[i][j];
			flynn_1=ElleUnodeFlynn(unode_1);

			tmpunodes.clear();

			/* FS: Here check if the flynn belongs to the excluded phase, 
			* if yes: Further down the misorientation angle is always set to a 
			* value < HAGB --> Means no subgrains in that flynn are created */
			bDoSubgrain = true;
			if (ElleFlynnAttributeActive(iPhaseAttrib))
			{
				dPhase = 0.0;
				ElleGetFlynnRealAttribute(flynn_1,&dPhase,iPhaseAttrib);
				if ((int)dPhase == iExcludedPhase) bDoSubgrain = false;
			}

			// 1st Check if unodes have been previously assigned to a subgrain, if not assign to a new subgrain	
			ElleGetUnodeAttribute(unode_1,&dummy,U_DENSITY);
			grain=int(dummy);		

			// 2nd check neighbours unodes 

			for (l=-sindex;l<=sindex;l++) 
			{	
				for (k=-sindex;k<=sindex;k++) 
				{

					// check  neigbhbours	unodes		
					k_1=i+k;
					l_1=j+l;

					// conditionals to wrapping unode_id information	
					if (k_1 < 0) k_1=ugrid_size+k_1;
					if (k_1 > ugrid_size-1) k_1 = k_1-ugrid_size;  
					if (l_1 < 0) l_1=ugrid_size+l_1;
					if (l_1 > ugrid_size-1) l_1 = l_1-ugrid_size; 

					idummy=unode_id[k_1][l_1]; 

					if (ElleUnodeFlynn(idummy) == flynn_1 )
					{	
						tmpunodes.push_back( idummy ); // copy unodes id to <vector> tmpunodes, only if they are at the same flynn than reference unode
					}
				}
			}	

			// then remove the unode itself 
			for (k=0; k<tmpunodes.size();k++) 
			{
				if (tmpunodes[k] == unode_1) iter=k;
			}

			tmpunodes.erase(tmpunodes.begin() + iter);		

			// 3rd Check if unodes have been previously assigned to a subgrain, if not 
			// 1st search nearest unodes with subgrain assignation and check misorientation 
			// else define a new subgrain number and assign to the pivote unode

			if( grain == 0) 
			{  // at this moment first trial with only unassigned unodes

				for(k=0;k<tmpunodes.size();k++) 
				{
					misangle=0;  
					minang=180;
					ElleGetUnodeAttribute(tmpunodes[k],&dummy,U_DENSITY);
					grain_2=int(dummy);

					if (grain_2 != 0) 
					{ 
						// select the unodes with minimum misorientation 
						misangle=misorientation(unode_1,tmpunodes[k],symm,symm_op);	
						if (misangle<=minang)
						{ 
							unode_2=grain_2;
							minang=misangle;
						}
					}	
				}	
				
				/*
				 * FS changes: Previously it has been checked if the unode of 
				 * interest belongs to a flynn of the excluded phase. If yes,
				 * no subgrains should be created in that flynn (U_DENSITY 
				 * within that flynn should always be the same), which will
				 * then be indicated by the bool variable bDoSubgrain being 
				 * false:
                 * Set minang always to be smaller than hagb by increasing hagb
				 */
				if (!bDoSubgrain)
				{
					minang = hagb-1;
				}
				
				if (minang <= hagb)
					ElleSetUnodeAttribute(unode_1, double(unode_2),U_DENSITY); 
				else 
				{
					totalgrain++;			
					ElleSetUnodeAttribute(unode_1,double(totalgrain),U_DENSITY);	
					grain=totalgrain;
				}			
			} 

			// 4th search list of neigbhbours unodes and check/assign as same grain of unode_1 if misori is below a critical value		
			for(k=0;k<tmpunodes.size();k++) 
			{			
				misangle=0;

				ElleGetUnodeAttribute(tmpunodes[k],&dummy,U_DENSITY);
				grain_2=int(dummy);

				if( grain_2 == 0) 
				{
					misangle=misorientation(unode_1,tmpunodes[k],symm,symm_op);				
					if (misangle <= hagb) 
						ElleSetUnodeAttribute(tmpunodes[k], double(grain),U_DENSITY); 
				}
			}
			// finish loop over neigbhbours unodes
		}	
	}

	// 5th new search and joint predefined grains   
	// loop over all unodes
	sindex=1;

	for (m=0;m<=1;m++)	// FS: This is just to do it two times!?
	{
		for (j=0;j<ugrid_size;j++) 
		{
			// if (fmod(j,10)==0) printf("unode %d \n", j);

			for (i=0;i<ugrid_size;i++) 
			{
				unode_1=unode_id[i][j];
				flynn_1=ElleUnodeFlynn(unode_1);
				tmpunodes.clear();

				// 1st Check if unodes have been previously assigned to a grain, if not assign to a new grain	
				ElleGetUnodeAttribute(unode_1,&dummy,U_DENSITY);
				grain=int(dummy);		

				// 2nd check neighbours unodes, only first neigbhbours 

				for (l=-sindex;l<=sindex;l++) 
				{	
					for (k=-sindex;k<=sindex;k++) 
					{
						// check  neigbhbours	unodes		
						k_1=i+k;
						l_1=j+l;

						// conditionals to wrapping unode_id information	
						if (k_1 < 0) k_1=ugrid_size+k_1;
						if (k_1 > ugrid_size-1) k_1 = k_1-ugrid_size;  
						if (l_1 < 0) l_1=ugrid_size+l_1;
						if (l_1 > ugrid_size-1) l_1 = l_1-ugrid_size; 

						idummy=unode_id[k_1][l_1]; 
						if (ElleUnodeFlynn(idummy) == flynn_1 )
						{	
							tmpunodes.push_back( idummy ); // copy unodes id to <vector> tmpunodes
						}
					}
				}	
				/*
				// then remove the unode itself 
				for (k=0; k<tmpunodes.size();k++) {
				if (tmpunodes[k] == unode_1) iter=k;
				}

				tmpunodes.erase(tmpunodes.begin() + iter);	// this part is duplicated !! define in two functions!
				*/
				//  check if grain number are different and below critical misorientation; 
				for(k=0;k<tmpunodes.size();k++) 
				{			
					misangle=0;
					count=0;
					ElleGetUnodeAttribute(tmpunodes[k],&dummy,U_DENSITY);
					grain_2=int(dummy);
					misangle=misorientation(unode_1,tmpunodes[k],symm,symm_op);

					if ( grain_2 != grain && misangle <= hagb) 
					{
						for (l=0;l<max_unodes;l++) 
						{ 
							// search all unodes with similar condition and reassign grain number. The new grain is the pivote unode
							ElleGetUnodeAttribute(l,&dummy ,U_DENSITY);
							if ( int(dummy) == grain_2 ) 
							{
								ElleSetUnodeAttribute(l,double(grain), U_DENSITY);
								count++;
							}
						}
						//printf("old grain %d new grain %d number %d \n",grain_2, grain, count);
					}
				}
			}
		}	
	} // end "m" loop 

	// sort and unique
    tmpunodes.clear();
	for (l=0;l<max_unodes;l++) { 
		ElleGetUnodeAttribute(l,&dummy ,U_DENSITY);
        tmpunodes.push_back( dummy ); // copy unodes id to <vector> tmpunodes
	}
	std::sort(tmpunodes.begin(), tmpunodes.end());

	idummy=*tmpunodes.begin(); // tmpunodes[0]	
	count=0;
	k=0;
	while (tmpunodes[k] >= idummy) {
	
		if (tmpunodes[k] != idummy){
			// printf("unode %d %d %d \n", tmpunodes[k], idummy, count); 
			for (l=0;l<max_unodes;l++) {
				ElleGetUnodeAttribute(l,&dummy ,U_DENSITY);	
				// if ( int(dummy) == idummy ) ElleSetUnodeAttribute(l,double(count), U_ATTRIB_C);
						
			}
		count++;
		idummy=tmpunodes[k];
		}
	k++;	

	}

	// std::unique(tmpunodes.begin(), tmpunodes.end());
	idummy=*tmpunodes.begin(); // tmpunodes[0]
	count=0;  

	list< pair<int,int> >ordered_list;
	list< pair<int,int> >::iterator ito;

	for(k=0;k<tmpunodes.size();k++) {
	
		if (tmpunodes[k] != idummy){
			pair<int,int> tmp(idummy, count);
			ito=ordered_list.begin();
			while (ito!=ordered_list.end() &&
				tmp.second < (*ito).second) ito++;
                ordered_list.insert(ito,tmp);
			count=0;
			idummy=tmpunodes[k];
		
		} else count++;
		
	}

	ito=ordered_list.begin();
	count=0;
	while (ito!=ordered_list.end()) {
//		std::cout << (*ito).first << " " <<
//					(*ito).second << std::endl;
					
			std::cout << (*ito).second << std::endl;
					
		for (l=0;l<max_unodes;l++) {
			ElleGetUnodeAttribute(l,&dummy ,U_DENSITY);	
			// if ( int(dummy) == (*ito).first ) ElleSetUnodeAttribute(l,double(count), U_ATTRIB_B);
			if ( int(dummy) == (*ito).first ) ElleSetUnodeAttribute(l,double((*ito).second), U_ATTRIB_B);
			}
		count++;	
		ito++;
	}	


	// finish loop over all unodes
	ElleUpdate();

} 

double misorientation(int unode_id1,int unode_id2, double symm[24][3][3],int symm_op)

{   
    double tmpA, tmpB, angle;
    double val[4];
    double eps=1e-6;

    double rmap1[3][3];
    double rmap2[3][3];
    double rmapA[3][3],rmapAA[3][3];
    double a11, a22,a33;

	int n,i,j;
	double misang,minang, misang2;	
	double rmap1A[3][3],rmap2A[3][3],rmap1AA[3][3],rmap2AA[3][3];
	double aux_symm[3][3];
    double curra1,currb1,currc1;
    double curra2,currb2,currc2; 
	
	ElleGetUnodeAttribute(unode_id1,&curra1,&currb1,&currc1,EULER_3);
	ElleGetUnodeAttribute(unode_id2,&curra2,&currb2,&currc2,EULER_3);
	
	curra1 *= DTOR; currb1 *= DTOR; currc1 *= DTOR;
    curra2 *= DTOR; currb2 *= DTOR; currc2 *= DTOR;
	
	orientmat2(rmap1, (double)curra1,(double)currb1,(double)currc1);// gives rotation matrix
	orientmat2(rmap2, (double)curra2,(double)currb2,(double)currc2);

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
   
		val[0] = (a11+a22+a33-1)/2.0;

		a11=rmapAA[0][0];
		a22=rmapAA[1][1];
		a33=rmapAA[2][2];

		val[1] = (a11+a22+a33-1)/2.0;	

		for(i=0;i<=1;i++){
		
			if (val[i]>1.0) val[i] = 1.0;
			else if (val[i]<-1.0) val[i] = -1.0;
		
			misang=(acos(val[i]));
			misang *= RTOD;
	
			if (misang < minang) {
				minang=misang;	
			}
		}
// if (minang<120) break; // first misorientation below zero, take it 
	}

 	return(minang);
	   
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

int symmetry_load(double symm[24][3][3])
{

	int i,j,n,symm_op; 
	FILE *f=fopen("symmetry.symm","r");	
	double dum1,dum2,dum3;

    // reset all values  
	for(n=0;n<24;n++) {
		for(i=0;i<3;i++) {
	 		for (j=0;j<3;j++) {
				symm[n][i][j]=0.0;
			}
		}
	}
			
	// first line of with the information of number of symmetry operation 
	fscanf(f,"%d",&symm_op);
	
	// store symmetry operators
     for(n=0;n<symm_op;n++) {
		 
	 	for (i=0;i<3;i++) {
			fscanf(f,"%lf %lf %lf",&dum1,&dum2,&dum3);	
			symm[n][i][0]=dum1;
			symm[n][i][1]=dum2;		
			symm[n][i][2]=dum3;
					
	 	}	
	}	

	fclose(f);
	
return(symm_op);	
}

bool fileExists(const char *filename)
{
  ifstream ifile(filename);
  return ifile.good();
}
