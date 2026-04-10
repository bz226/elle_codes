#include <stdio.h>
#include <math.h>
#include <vector>
#include "mat.h"
#include "attrib.h"
#include "nodes.h"
#include "update.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "init.h"
#include "triattrib.h"
#include "unodes.h"
#include "misorient.h"
#include "convert.h"

int InitSetUnodes(), SetUnodes();
void SetUnodeAttributeFromNbs(int flynnid,int attr_id);
void CalcMisorientEuler(double curra1,double currb1,double currc1,double curra2,double currb2,double currc2, double *orient);

using std::vector;

#define SPINS 6  // number of possible orientations for any given site
/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitSetUnodes()
{
    char *infile;
    int err=0;

    /*
     * clear the data structures
     */
    ElleReinit();

    ElleSetRunFunction(SetUnodes);

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
    }
}

int SetUnodes()
{
    int i, j, k;
    int max_stages, max_flynns, max_unodes;
	time_t now;

	time(&now);

	//srand48((long) now); // initialise random number generator from current time

    ElleCheckFiles();

    max_stages = EllemaxStages();
    max_flynns = ElleMaxFlynns();
    max_unodes = ElleMaxUnodes();
    for (i=0;i<max_stages;i++) {
        for (j=0;j<max_flynns;j++) {
            if (ElleFlynnIsActive(j)) {
                ElleClearTriAttributes();
                TriangulateUnodes(j,MeshData.tri);
                SetUnodeAttributeFromNbs(j,EULER_3);
				//break; // uncomment for debugging purposes to get first flynn working
            }
        }
        ElleUpdate();
    }
}

void SetUnodeAttributeFromNbs(int flynnid,int attr_id)
{
    int i,ii,j,randnb,loops;
    int id, num_nbs, count,slope,iso;
    double val,prob,m;
    int histo[SPINS],maxneigh,maxval,nodeval;
	double total1,total2;
	double transprobs[6];
	double energy;
    double e[3], k[3];
	double orient,misorientenergy,max_angle;
	int flag;
	double eps=1e-6;
	UserData udata;

        ElleUserData(udata);

        max_angle=(double)udata[0];
        slope=(int)udata[1];
        // default slope is set to 1 which means same energy for all misorientations
        if (slope!=1 || slope !=2 || slope !=3) slope==1;

	for(j=0;j<6;j++) // set up transition probabilities based on number of similar neighbours
	{
		transprobs[j]=0.5*exp((double) -j);
	}

    vector<int> unodelist; // define vector list of unodes
    ElleGetFlynnUnodeList(flynnid,unodelist); // get the list of unodes for a flynn
    count = unodelist.size();
	//printf("count: %d\n",count);

	for(loops=0;loops<1;loops++)
	{
	    	for (i=0; i<count; i++) {

			ii=(int)(ElleRandomD()*count);	// randomly select a unode from within flynn

			vector<int> nbnodes,bndflag;
        	ElleGetTriPtNeighbours(unodelist[ii],nbnodes,bndflag,0); //get the  list of neighbours for a unode
        	num_nbs = nbnodes.size();
			if(num_nbs==0)
				continue;
			//printf("num_nbs: %d\n",num_nbs);

			ElleGetUnodeAttribute(unodelist[ii],&k[0],&k[1],&k[2],attr_id); //  get the unodes attribute

			//printf("kvals: %lf %lf %lf\n",k[0],k[1],k[2]);

			for(j=0,flag=0;j<num_nbs;j++) // check to see if all neighbours are same
			{
				ElleGetUnodeAttribute(nbnodes[j],&e[0],&e[1],&e[2],attr_id); // get neighbouring unode attribute value
				if(fabs(e[0]-k[0])>eps || fabs(e[1]-k[1])>eps || fabs(e[2]-k[2])>eps)
					flag=1;

			}

			if(flag==0) // if all neighbours are same, don't bother
				continue;

        	for (j=0,total1=0; j<num_nbs; j++)  			// loop through all neighbouring unodes
			{
            	ElleGetUnodeAttribute(nbnodes[j],&e[0],&e[1],&e[2],attr_id); // get neighbouring unode attribute value

				orient = Misorient(k,e);
				misorientenergy=0.0;
				//printf("\n1 orient: %lf",orient);

			        if((fabs(orient)>max_angle || fabs(orient)==max_angle) && slope==1)
			        	{
										misorientenergy=1.0;
					// printf("%lf>%lf or iso = %i\n", orient,max_angle,iso);
                                  }
				else if (fabs(orient)<max_angle && slope==2)
				{	misorientenergy=fabs(orient)/max_angle;
				//printf(" energy_slope: %lf\n",misorientenergy);
				}
				else if (slope==3)
				  {       misorientenergy=fabs(orient/max_angle)*(1-(log(orient/max_angle)));	      //printf(" energy_shockly: %lf\n",misorientenergy);


				}

                                total1+=misorientenergy;
        	}


			randnb=(int)(ElleRandomD()*num_nbs);
			//printf("%d %d\n",k,nbnodes[k]);

			 ElleGetUnodeAttribute(nbnodes[randnb],&k[0],&k[1],&k[2],attr_id); // get neighbouring unode attribute value

			for (j=0,total2=0; j<num_nbs; j++)  			// loop through all neighbouring unodes
			{
            	ElleGetUnodeAttribute(nbnodes[j],&e[0],&e[1],&e[2],attr_id); // get neighbouring unode attribute value
				orient = Misorient(k,e);
				//printf("\n2 orient: %lf",orient);
			        if(fabs(orient)>max_angle || fabs(orient)==max_angle|| slope==1) {
					misorientenergy=1.0;
                                        //printf("%lf>%lf or iso = %i\n", orient,max_angle,iso);
                                  }
				else if (fabs(orient)<max_angle && slope==2)
				{	misorientenergy=fabs(orient)/max_angle;
				//	printf(" energy_slope: %lf\n",misorientenergy);
				}
                                else if (slope==3)
				  {       misorientenergy=fabs(orient/max_angle)*(1-(log(orient/max_angle)));	     // printf(" energy_shockly: %lf\n",misorientenergy);

				}

	total2+=misorientenergy;
        	}

			energy=(total1-total2);							// potential energy of change of node state
			//printf("\nt1 t2 e: %le %le %le\n",total1,total2,energy);

			if(energy >= 0)
				prob=1.0;									// probability of change of node state
			else
				prob=0.5*exp((double) energy);				// probability of change of node state

			m=ElleRandomD();								// get random number

			if(m<prob)
				ElleSetUnodeAttribute(unodelist[ii], k[0],k[1],k[2] ,attr_id); // set new unode attribute value

    	}
	}
}

#if XY
void CalcMisorientEuler(double curra1,double currb1,double currc1,double curra2,double currb2,double currc2, double *orient)
{
    double tmpA, tmpB, angle;
    double val;
    double eps=1e-6;

    double rmap1[3][3];
    double rmap2[3][3];
    double rmap3[3][3];
    double rmapA[3][3];
    double rmapB[3][3];
    double a11, a22,a33;

    curra1 *= DTOR; currb1 *= DTOR; currc1 *= DTOR;
    curra2 *= DTOR; currb2 *= DTOR; currc2 *= DTOR;

    euler(rmap1,(double)curra1,(double)currb1,(double)currc1);// gives rotation matrix
    euler(rmap2,(double)curra2,(double)currb2,(double)currc2);// gives rotation matrix

    /*calculation of tmpA where the inverse of rmap1 is taken for calculation*/
    matinverse(rmap1,rmap3);
    matmult(rmap2,rmap3,rmapA);

    a11=rmapA[0][0];
    a22=rmapA[1][1];
    a33=rmapA[2][2];

    val = (a11+a22+a33-1)/2;
   // if (val > 1.0+eps || val < -1.0-eps )
       // fprintf(stdout,"CalcMisorient - adjusting val for acos from %lf for bnd %d, %d\n",val,rgn1,rgn2);
    if (val>1.0) val = 1.0;
    else if (val<-1.0) val = -1.0;
    tmpA=acos(val);

    /*calculation of tmpB where the inverse of rmap1 is taken for calculation*/
    matinverse(rmap2,rmap3);
    matmult(rmap1,rmap3,rmapB);

    a11=rmapB[0][0];
    a22=rmapB[1][1];
    a33=rmapB[2][2];

    val = (a11+a22+a33-1)/2;
  //  if (val > 1.0+eps || val < -1.0-eps )
      //  fprintf(stdout,"CalcMisorient - adjusting val for acos from %lf for bnd %d, %d\n",val,rgn1,rgn2);
    if (val>1.0) val = 1.0;
    else if (val<-1.0) val = -1.0;
    tmpB=acos(val);

    if (tmpA<tmpB) angle=tmpA;
    else angle=tmpB;
    angle *= RTOD;
    *orient=angle;

}
#endif

