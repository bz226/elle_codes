#include <stdio.h>
#include <math.h>
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
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include "mat.h"
#include "phasefield.h"

using std::cout;
using std::endl;

struct vec2D { double x, y;};
struct vec2D	grad_theta[NX][NY], grad_eps2;

#if XY
double	**dmatrix(long nrl, long nrh, long ncl, long nch); /* added from numerical recipies to avoid >32k error */
double *dvector(long nl, long nh);
void nrerror(char error_text[]);
#endif

double get_angle( double x, double y )
{	double angle;
	if (x==0.0) 	angle = (y>0.0 ? 0.5*PI : -0.5*PI);
	else if (x>0.0)	angle = (y>0.0 ? atan(y/x) : 2.0*PI + atan(y/x));
	else 			angle = PI + atan(y/x);
	return angle; };


int InitSetUnodes(), SetUnodes();
double			**theta, **T, **eps2, **lap_T, **lap_theta, m,val;
double			**ax,**ay, epsilon, epsilon_prime, anglex, tet, dydx, dxdy, scal;

double K 	=	1.8 	;	/* Latent heat */
double TAU 	=	0.0003 	;	/* PF relaxation time */
double EPS 	=	0.01 	;	/* interfacial width */
double DELTA =	0.02 	;	/* modulation of the interfacial width */
double ANGLE0=	1.57 	;	/* orientation of the anisotropy axis */
double ANISO =	6.0 	;	/* anisotropy 2*PI/ANISO */

double ALPHA =	0.9 	;	/* m(T) = ALPHA/PI * atan(GAMMA*(TEQ-T)) */
double GAMMA =	10.0	;
double TEQ 	 =	1.0 	;	/* melting temperature */

double H 	 =	0.03 	;	/* spatial resolution */
double DT 	 =	2.e-4 	;	/* temporal resolution */



/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitSetUnodes()
{
    char *infile;
    int err=0,i,j,k;
	  UserData CurrData;
    int resetunodegrid=0;
    


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
    
    theta=dmatrix(0,NX-1,0,NY-1);
	T=dmatrix(0,NX-1,0,NY-1);
	eps2=dmatrix(0,NX-1,0,NY-1);
	lap_T=dmatrix(0,NX-1,0,NY-1);
	lap_theta=dmatrix(0,NX-1,0,NY-1);
	ax=dmatrix(0,NX-1,0,NY-1);
	ay=dmatrix(0,NX-1,0,NY-1);
	
        ElleUserData(CurrData);
				resetunodegrid=(int)CurrData[uRESET] ;	/* reset unode grid */
	
	

/* initial configuration: T=0 and theta=1 in a circle of radius 3H (0 elsewhere). */

	if(resetunodegrid==1)
	{
		printf("Initialising the run for %d x %d mesh \n", NX,NX);
		UnodesClean(); 
		ElleInitUnodes(NX, SQ_GRID);
		ElleInitUnodeAttribute(CONC_A);
		ElleInitUnodeAttribute(U_ATTRIB_A);
		for(i=0;i<NX;i++) for(j=0;j<NY;j++) 
		{
			T[i][j] = 0.0;
			theta[i][j] = ((i-NX/2)*(i-NX/2)+(j-NY/2)*(j-NY/2)<10)?1.0:0.0;
		}
	}
	else
	{
		for(i=0,k=0;i<NX;i++) for(j=0;j<NY;j++,k++) 
		{
			ElleGetUnodeAttribute(k, &val, CONC_A );
			theta[i][j]=val;
			ElleGetUnodeAttribute(k, &val, U_ATTRIB_A );
			T[i][j] = val;
		}

	}
    }
    
}

int SetUnodes()
{
    int max_stages, max_unodes;
	  int i, j, k, t, ip, im, jp, jm;
		UserData CurrData;

	    if (ElleCount()==0)
	    {
        ElleUserData(CurrData);
				K=CurrData[uK] ;	/* Latent heat */
				EPS=CurrData[uEPS] ;	/* interfacial width */
				DELTA=CurrData[uDELTA] ;	/* modulation of the interfacial width */
				ANGLE0=CurrData[uANGLE0] ;	/* orientation of the anisotropy axis */
				ANISO=CurrData[uANISO] ;	/* anisotropy 2*PI/ANISO  */
				ALPHA=CurrData[uALPHA] ;	/* m(T) = ALPHA/PI * atan(GAMMA*(TEQ-T))  */
				GAMMA=CurrData[uGAMMA] ;	       
				
        cout << "Latent heat= " << CurrData[uK] << endl;
        cout << "interfacial width= " << CurrData[uEPS] << endl;
        cout << "modulation of the interfacial width= " << CurrData[uDELTA] << endl;
        cout << "orientation of the anisotropy axis= " << CurrData[uANGLE0] << endl;
        cout << "anisotropy 2*PI/ANISO= " << CurrData[uANISO] << endl;
        cout << "ALPHA= " << CurrData[uALPHA] << endl;
        cout << "GAMMA= " << CurrData[uGAMMA] << endl;
         }

	/* temporal evolution */
	//for (t=0;t<NTIME;t++) 
	ElleCheckFiles();
    
    max_stages = EllemaxStages(); // number of stages 
    max_unodes = ElleMaxUnodes(); // maximum unode number used
    for (t=0;t<max_stages;t++)   // cycle through stages
	{
		{
		//printf ("doing time step %d of %d\n",t+1,NTIME);
		/* computation of various auxiliary quantities */
		for (i=0;i<NX;i++){ 
			for (j=0;j<NY;j++){
			ip = (i+1)%NX; 		/* i+1 with Periodic Boundaries */
			im = (NX+i-1)%NX; 	/* i-1 with Periodic Boundaries */
			jp = (j+1)%NY; 		/* j+1 with Periodic Boundaries */
			jm = (NY+j-1)%NY; 	/* j-1 with Periodic Boundaries */
			
	/* gradient and laplacians (order H^2, 9 points isotropic version for laplacians) */
			grad_theta[i][j].x = (theta[ip][j] - theta[im][j])/H;
			grad_theta[i][j].y = (theta[i][jp] - theta[i][jm])/H;
			
			lap_theta[i][j] = (2.0*(theta[ip][j]+theta[im][j]+theta[i][jp]+theta[i][jm]) +theta[ip][jp]+theta[im][jm]+theta[im][jp]+theta[ip][jm] - 12.0*theta[i][j])/(3.0*H*H);
			lap_T[i][j] = (2.0*(T[ip][j]+T[im][j]+T[i][jp]+T[i][jm])+T[ip][jp]+T[im][jm]+T[im][jp]+T[ip][jm] - 12.0*T[i][j])/(3.0*H*H);
	
	/* angular dependence of the interfacial width "espilon" */
			anglex = get_angle(grad_theta[i][j].x,grad_theta[i][j].y);
			epsilon = EPS*(1.0 + DELTA*cos(ANISO*(anglex-ANGLE0)));
			epsilon_prime = -EPS*ANISO*DELTA*sin(ANISO*(anglex-ANGLE0));
	
	/* auxiliary quantities appearing in the phase field equation */
			ay[i][j] = - epsilon*epsilon_prime * grad_theta[i][j].y;
			ax[i][j] = epsilon*epsilon_prime * grad_theta[i][j].x;
			eps2[i][j] = epsilon*epsilon;
			}
		}
	
	/* simple Euler step */
	for (i=0;i<NX;i++){ 
		for (j=0;j<NY;j++){
		/* first, some few spatial derivatives */
			ip = (i+1)%NX;
			im = (NX+i-1)%NX;
			jp = (j+1)%NY;
			jm = (NY+j-1)%NY;
		
			dxdy = (ay[ip][j] - ay[im][j])/H;
			dydx = (ax[i][jp] - ax[i][jm])/H;
			grad_eps2.x = (eps2[ip][j] - eps2[im][j])/H;
			grad_eps2.y = (eps2[i][jp] - eps2[i][jm])/H;
		
			tet = theta[i][j];
			m = ALPHA/PI * atan(GAMMA*(TEQ-T[i][j]));
			scal= grad_eps2.x*grad_theta[i][j].x+ grad_eps2.y*grad_theta[i][j].y;
	
		/* Euler step */
			theta[i][j] += (dxdy+dydx + eps2[i][j]*lap_theta[i][j] + scal + tet*(1.0-tet)*(tet-0.5+m))*DT/TAU;
			T[i][j] += lap_T[i][j]*DT + K*(theta[i][j] - tet);
		}}
	}
		
		
		for(i=0,k=0;i<NX;i++) for(j=0;j<NY;j++,k++) 
		{
			ElleSetUnodeAttribute(k,CONC_A, theta[i][j] );
			ElleSetUnodeAttribute(k,U_ATTRIB_A, T[i][j] );
		}

		ElleUpdate();

    }
} 



#if XY
double	**dmatrix(long nrl, long nrh, long ncl, long nch)
{
		long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	double **m;
	
	/*allocate pointers to rows*/
	m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
	if (!m) nrerror("allocation failure 1 in matrix()\n");
	m += NR_END;
	m -= nrl;
	
	/*allocate rows and set pointers to them */
	m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
	m[nrl] += NR_END;
	m[nrl] -= ncl;
	
	for (i=nrl+1;i<=nrh;i++)	m[i]=m[i-1]+ncol;
	return m;
}

double *dvector(long nl, long nh)
{
	double	*v;
	
	v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
	if (!v) nrerror("allocation failure in dvector()\n");
	return v-nl+NR_END;
}

void nrerrorx(char error_text[])
{
	printf("Numerical Recipes run-time error ...\n");
	printf("%s\n", error_text);
	printf("Now exiting system.....\n");
	exit(1);
}
#endif
