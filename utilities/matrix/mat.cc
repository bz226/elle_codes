#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "convert.h"
#include "erand.h"
#include "error.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_cblas.h>
#include "mat.h"

Erand angran;

void orientmat(double a[3][3], double phi, double rho, double g)
{
    //double a[3][3];
    int i,j;

	/* for rh ZYZ cba matrix is
     * row 0
	 * -sinphi1*sinphi2+cosphi1*cosphi2*cosPhi
	 * -cosphi1*sinphi2-sinphi1*cosphi2*cosPhi
     * -cosphi2*sinPhi
     * row 1
	 * -sinphi1*cosphi2+cosphi1*sinphi2*cosPhi
	 * cosphi1*cosphi2-sinphi1*sinphi2*cosPhi
     * sinphi2*sinPhi
     * row 2
     * cosphi1*sinPhi
	 * sinphi1*sinPhi
     * cosPhi
     * so why doesn't the code match? Leave as original?
     * Using ZXZ so it should not be called
	 */
    a[0][0]=cos(phi)*cos(g)-sin(phi)*sin(g)*cos(rho);
    a[0][1]=sin(phi)*cos(g)+cos(phi)*sin(g)*cos(rho);
    a[0][2]=sin(g)*sin(rho);
    a[1][0]=-cos(phi)*sin(g)-sin(phi)*cos(g)*cos(rho);
    a[1][1]=sin(phi)*sin(g)+cos(phi)*cos(g)*cos(rho);
    a[1][2]=cos(g)*sin(rho);
    a[2][0]=sin(phi)*sin(rho);
    a[2][1]=-cos(phi)*sin(rho);
    a[2][2]=cos(rho);

}

void orientmatZXZ(double a[3][3], double phi1, double Phi, double phi2)
{
    //double a[3][3];
    int i,j;

    a[0][0]=cos(phi1)*cos(phi2)-sin(phi1)*sin(phi2)*cos(Phi);
    a[0][1]=sin(phi1)*cos(phi2)+cos(phi1)*sin(phi2)*cos(Phi);
    a[0][2]=sin(phi2)*sin(Phi);
    a[1][0]=-cos(phi1)*sin(phi2)-sin(phi1)*cos(phi2)*cos(Phi);
    a[1][1]=-sin(phi1)*sin(phi2)+cos(phi1)*cos(phi2)*cos(Phi);
    a[1][2]=cos(phi2)*sin(Phi);
    a[2][0]=sin(phi1)*sin(Phi);
    a[2][1]=-cos(phi1)*sin(Phi);
    a[2][2]=cos(Phi);

}

#include<stdio.h>
void euler(double rmap[3][3], double phi, double rho, double g)
{
    double a[3][3],b[3][3],c[3][3];
	double eps=1.0e-7;
    int i,j;

	if (fabs(rho)<eps) rho = eps;
    a[0][0]=cos(phi);
    a[0][1]=-sin(phi);
    a[0][2]=0.0;
    a[1][0]=sin(phi);
    a[1][1]=cos(phi);
    a[1][2]=0.0;
    a[2][0]=0.0;
    a[2][1]=0.0;
    a[2][2]=1.0;

    b[0][0]=cos(rho);
    b[0][1]=0.0;
    b[0][2]=-sin(rho);
    b[1][0]=0.0;
    b[1][1]=1.0;
    b[1][2]=0.0;
    b[2][0]=sin(rho);
    b[2][1]=0.0;
    b[2][2]=cos(rho);

    c[0][0]=cos(g);
    c[0][1]=-sin(g);
    c[0][2]=0.0;
    c[1][0]=sin(g);
    c[1][1]=cos(g);
    c[1][2]=0.0;
    c[2][0]=0.0;
    c[2][1]=0.0;
    c[2][2]=1.0;

    matmult(b,a,rmap);
    matmult(c,rmap,rmap);

}

/*!
 * uneuler returns -90<*a<90, -180<*b<180, -90<*c<90
 */
void uneuler(double rmap[3][3],double *a, double *b, double *g)
{
	double eps=1.0e-10;
	double sb=0.0;
	double signb=1.0;
	/* for above rh ZYZ cba matrix is
	 * cosg*cosb*cosa - sing*sina	-sina*cosb*cosg - sing*cosa		-sinb*cosg
	 * sing*cosa*cosb + cosg*sina	-sina*sing*cosb + cosg*cosa		-sinb*sing
	 * sinb*cosa					-sina*sinb						cosb
	 */

	if (rmap[2][0]<0)
		signb=-1.0;

	/*
	 * acos returns 0 -> PI
	 * asin, atan return -PI/2 -> PI/2
	 * atan2 returns -PI -> PI
	 */
    *b= acos(rmap[2][2])*signb;

    if (fabs(*b) > eps) {
		sb = sin(*b);
        *a= atan( (rmap[2][1]/(-sb))/(rmap[2][0]/sb));
    	*g= atan((rmap[1][2]/(-sb))/(rmap[0][2]/(-sb)));
    }
    else {
    // *b is 0 or PI and cos(*b)=1
		// cannot calc *g
		// can only calculate *a if we assume *g==0
      *g = 0.0;
      *a = asin(rmap[1][0]);
    }
	*b *= RTOD;
	*a *= RTOD;
	*g *= RTOD;
	//remember to change float version if this function changed

}

void eulerZXZ(double rmap[3][3], double phi1, double Phi, double phi2)
{
    double a[3][3],b[3][3],c[3][3];
	double eps=1.0e-7;
    int i,j;

	if (fabs(Phi)<eps) Phi = eps;
    a[0][0]=cos(phi1);
    a[0][1]=sin(phi1);
    a[0][2]=0.0;
    a[1][0]=-sin(phi1);
    a[1][1]=cos(phi1);
    a[1][2]=0.0;
    a[2][0]=0.0;
    a[2][1]=0.0;
    a[2][2]=1.0;

    b[0][0]=1;
    b[0][1]=0.0;
    b[0][2]=0.0;
    b[1][0]=0.0;
    b[1][1]=cos(Phi);
    b[1][2]=sin(Phi);
    b[2][0]=0.0;
    b[2][1]=-sin(Phi);
    b[2][2]=cos(Phi);

    c[0][0]=cos(phi2);
    c[0][1]=sin(phi2);
    c[0][2]=0.0;
    c[1][0]=-sin(phi2);
    c[1][1]=cos(phi2);
    c[1][2]=0.0;
    c[2][0]=0.0;
    c[2][1]=0.0;
    c[2][2]=1.0;

    matmult(b,a,rmap);
    matmult(c,rmap,rmap);

}

/*!
 * uneulerZXZ returns 0<*phi1<360, 0<*Phi<180, 0<*phi2<360
 * input is orientation matrix
 */
void uneulerZXZ(double rmap[3][3],double *phi1, double *Phi, double *phi2)
{
	double eps=1.0e-10;
	double sb=0.0;
	double signPhi=-1.0;
	/* for rh ZXZ orientation matrix is
     * row 0
     * cos(phi1)*cos(phi2)-sin(phi1)*sin(phi2)*cos(Phi);
     * sin(phi1)*cos(phi2)+cos(phi1)*sin(phi2)*cos(Phi);
     * sin(phi2)*sin(Phi);
     * row 1
     * -cos(phi1)*sin(phi2)-sin(phi1)*cos(phi2)*cos(Phi);
     * -sin(phi1)*sin(phi2)+cos(phi1)*cos(phi2)*cos(Phi);
     * cos(phi2)*sin(Phi);
     * row 2
     * sin(phi1)*sin(Phi);
     * -cos(phi1)*sin(Phi);
     * cos(Phi);
	 */

	if (rmap[2][1]<0)
		signPhi=1.0;

	/*
	 * acos returns 0 -> PI
	 * asin, atan return -PI/2 -> PI/2
	 * atan2 returns -PI -> PI
    *Phi= acos(rmap[2][2])*signPhi;
	 */
    *Phi= acos(rmap[2][2]);

    if (fabs(*Phi) > eps && fabs(*Phi-PI) > eps) {
        *phi1= atan2(rmap[2][0],-rmap[2][1]);
    	*phi2= atan2(rmap[0][2],rmap[1][2]);
    }
    else {
// AVOID DIV BY 0
    // *Phi is 0 or PI and cos(*Phi)=1
		// cannot calc *phi2
		// can only calculate *phi1 if we assume *phi2==0
      *phi2 = 0.0;
      *phi1 =  atan2(rmap[0][1],rmap[0][0]);
    }
    double tmpori[3], newori[3];
    tmpori[0]=newori[0]=*phi1;
    tmpori[1]=newori[1]=*Phi;
    tmpori[2]=newori[2]=*phi2;

    eulerRange(tmpori,newori);

	*phi1 = newori[0] * RTOD;
	*Phi = newori[1] * RTOD;
	*phi2 = newori[2] * RTOD;
}

/*!
 * based on orientation lib (orilib) function ol_e_e_rad
 * first and third entries in range 0->2*PI
 * second entry in range 0->PI
 * input and output in radians
 */
void eulerRange(double *e, double *e2)
{
    int i;
    double eps=1.0e-10;

    for (i = 0; i < 3; i++)
      e2[i] = e[i];

    /* phi within [0,(2*PI)] */
    e2[1] -= ((int) (e2[1] / (2 * PI))) * (2 * PI);
    if (e2[1] < 0)
      e2[1] += (2 * PI);

    /* 
     * if phi>PI, applying reflection
     * so phi within [0,PI]
     */
    if (e2[1] > PI)
    {
      e2[1] = (2 * PI) - e2[1];
      e2[0] += PI;
      e2[2] += PI;
    }

    /* treating the std case where phi != 0 */
    if (fabs (e2[1]) > eps && fabs (e2[1] - PI) > eps)
    {
      /* phi2 within [0,(2*PI)] */
      e2[2] -= ((int) (e2[2] / (2 * PI))) * (2 * PI);
      if (e2[2] < 0)
        e2[2] += (2 * PI);
    }
    /* treating degeneracy: phi = 0: phi1 += phi2 and phi2 = 0. */
    else if (fabs (e2[1]) > eps)
    {
      e2[0] += e2[2];
      e2[2] = 0;
    }
    else /* the same at phi = 180 */
    {
      e2[0] -= e2[2];
      e2[2] = 0;
    }

    /* phi1 within [0,(2*PI)] */
    e2[0] -= ((int) (e2[0] / (2 * PI))) * (2 * PI);
    if (e2[0] < 0)
      e2[0] += (2 * PI);

    return;
}
#if XY
/*!
 * uneulerZXZ returns -90<*phi1<90, -180<*Phi<180, -90<*phi2<90
 */
void uneulerZXZ(double rmap[3][3],double *phi1, double *Phi, double
*phi2)
{
	double eps=1.0e-10;
	double sb=0.0;
	double signPhi=-1.0;
	/* for above rh ZXZ cba matrix is
     * row 0
     * cos(phi1)*cos(phi2)-sin(phi1)*sin(phi2)*cos(Phi);
     * sin(phi1)*cos(phi2)+cos(phi1)*sin(phi2)*cos(Phi);
     * sin(phi2)*sin(Phi);
     * row 1
     * -cos(phi1)*sin(phi2)-sin(phi1)*cos(phi2)*cos(Phi);
     * -sin(phi1)*sin(phi2)+cos(phi1)*cos(phi2)*cos(Phi);
     * cos(phi2)*sin(Phi);
     * row 2
     * sin(phi1)*sin(Phi);
     * -cos(phi1)*sin(Phi);
     * cos(Phi);
	 */

	if (rmap[2][1]<0)
		signPhi=1.0;

	/*
	 * acos returns 0 -> PI
	 * asin, atan return -PI/2 -> PI/2
	 * atan2 returns -PI -> PI
	 */
    *Phi= acos(rmap[2][2])*signPhi;

    if (fabs(*Phi) > eps) {
        *phi1= atan(-rmap[2][0]/rmap[2][1]);
    	*phi2= atan(rmap[0][2]/rmap[1][2]);
    }
    else {
// AVOID DIV BY 0
    // *Phi is 0 or PI and cos(*Phi)=1
		// cannot calc *phi2
		// can only calculate *a if we assume *g==0
      *phi2 = 0.0;
      *phi2 = asin(rmap[1][0]);
    }
	*Phi *= RTOD;
	*phi1 *= RTOD;
	*phi2 *= RTOD;
	//remember to change float version if this function changed

}
#endif

void eulerZXZ2cax(double a, double b, double c, double *cax)
{
    double rmap[3][3];

    a=a*DTOR;
    b=b*DTOR;
    c=c*DTOR;

    //calculate equivalent rotation tensor
    orientmatZXZ(rmap,a,b,c);

    cax[0]= rmap[2][0];
    cax[1]= rmap[2][1];
    cax[2]= rmap[2][2];
}

/*
 * for backward compatibility
 */
void Euler2cax(double a, double b, double c, double *cax)
{
	 euler2cax(a, b, c, cax);
}

void euler2cax(double a, double b, double c, double *cax)
{
    double rmap[3][3];

    a=a*DTOR;
    b=b*DTOR;
    c=c*DTOR;

    //calculate equivalent rotation tensor
    euler(rmap,a,b,c);

    cax[0]= rmap[2][0];
    cax[1]= rmap[2][1];
    cax[2]= rmap[2][2];
}

void uneuler(double rmap[3][3],float *a, float *b, float *g)
{
	double eps=1.0e-10;
	double tmpb, sb=0.0;
	double signb=1.0;

	if (rmap[2][0]<0)
		signb=-1.0;

	/*
	 * acos returns 0 -> PI
	 * asin, atan return -PI/2 -> PI/2
	 * atan2 returns -PI -> PI
	 */
    tmpb= acos(rmap[2][2])*signb;

    if (fabs(tmpb) > eps) {
		sb = sin(tmpb);
        *a= (float)(atan( (rmap[2][1]/(-sb))/(rmap[2][0]/sb)));
    	*g= (float)(atan((rmap[1][2]/(-sb))/(rmap[0][2]/(-sb))));
    }
    else {
    // *b is 0 or PI and cos(*b)=1
		// cannot calc *g
		// can only calculate *a if we assume *g==0
      *g = 0.0;
      *a = (float)asin(rmap[1][0]);
    }
	*b = (float)(tmpb*RTOD);
	*a *= RTOD;
	*g *= RTOD;
}


/*
 * for backward compatibility
 */
void Euler2cax(float a, float b, float c, float *cax)
{
	 euler2cax(a, b, c, cax);
}

void euler2cax(float a, float b, float c, float *cax)
{
    double rmap[3][3];

    a=a*DTOR;
    b=b*DTOR;
    c=c*DTOR;

    //calculate equivalent rotation tensor
    euler(rmap,(double)a,(double)b,(double)c);

    cax[0]=(float) rmap[2][0];
    cax[1]=(float) rmap[2][1];
    cax[2]=(float) rmap[2][2];
}

void orient(double rmap[3][3])
{
    double ang1,ang2,ang3;
    double zmag;

    do
    {
        ang1=2.0*angran.randouble()-1.0;
        ang2=2.0*angran.randouble()-1.0;
        ang3=2.0*angran.randouble()-1.0;
        zmag=sqrt(ang1*ang1+ang2*ang2+ang3*ang3);
    }
    while(zmag < 0.00001 || zmag > 1.00);

    ang1=angran.randouble()*2.0*M_PI;
    ang2=acos(ang3/zmag);
    ang3=angran.randouble()*2.0*M_PI;
    euler(rmap,ang1,ang2,ang3);
}

void orientZXZ(double rmap[3][3])
{
    double ang1,ang2,ang3;
    double zmag;

    do
    {
        ang1=2.0*angran.randouble()-1.0;
        ang2=2.0*angran.randouble()-1.0;
        ang3=2.0*angran.randouble()-1.0;
        zmag=sqrt(ang1*ang1+ang2*ang2+ang3*ang3);
    }
    while(zmag < 0.00001 || zmag > 1.00);

    ang1=angran.randouble()*2.0*M_PI;
    ang2=acos(ang3/zmag);
    ang3=angran.randouble()*2.0*M_PI;
    eulerZXZ(rmap,ang1,ang2,ang3);
}

void matmult(double a[3][3], double b[3][3], double ab[3][3])
{                                         
   /*!
     Parameters:
     order: row or column major
     TransA: transpose A
     TransB: transpose B
     M:      rows in A
     N:      cols in B
     K:      cols in A
     alpha:  scalar factor
     A:      matrix A
     lda:    major stride for A
     B:      matrix B
     ldb:    major stride for B
     beta:  scalar factor
     C:      matrix C
     ldc:    major stride for C
     Performs operation C = alpha*A*B + beta*C
    */
   int i;
   double tmp[3][3], *p1, *p2;
   /* use a tmp array as ab may be the same address as a or b */
   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0,
               &a[0][0], 3, &b[0][0], 3, 0.0, &tmp[0][0], 3);
   for (p1=&tmp[0][0],p2=&ab[0][0],i=0; i<9; i++,p1++,p2++) *p2 = *p1;
}

void matinverse(double a[3][3], double ab[3][3])
{
    int s, i, j;

    gsl_matrix *ludecomp = gsl_matrix_calloc(3,3);
    gsl_matrix *inverse = gsl_matrix_calloc(3,3);
    gsl_permutation *perm = gsl_permutation_alloc(3);
    for (i=0; i<3; i++)
      for (j=0; j<3; j++)
        gsl_matrix_set(ludecomp,i,j,a[i][j]);
    gsl_linalg_LU_decomp(ludecomp,perm,&s);
    gsl_linalg_LU_invert(ludecomp,perm,inverse);
    for (i=0; i<3; i++)
      for (j=0; j<3; j++)
        ab[i][j] = gsl_matrix_get(inverse,i,j);
    gsl_matrix_free(ludecomp);
    gsl_matrix_free(inverse);
    gsl_permutation_free(perm);
}

void mataxismult(double a[3][3], double ax[3], double y[3])
{                                         
   /*!
     Parameters:
     order: row or column major
     TransA: transpose A
     M:      rows in A
     N:      cols in ax
     alpha:  scalar factor
     A:      matrix A
     lda:    major stride for A
     ax:      vector ax //1x3
     ldb:    major stride for ax
     beta:  scalar factor
     y:      vector y 
     ldc:    major stride for ax
     Performs operation y = alpha*A'*ax + beta
    */
   int i;
   double tmp[3], *p1, *p2;
   /* use a tmp array as y may be the same address as x */
   //Call below will return A[2] if ax={0,0,1}  ie 1x3
   cblas_dgemv(CblasRowMajor, CblasTrans, 3, 3, 1.0,
               &a[0][0], 3, ax, 1, 0.0, tmp, 1);
   for (p1=&tmp[0],p2=&y[0],i=0; i<3; i++,p1++,p2++) *p2 = *p1;
}

void matvecmult(double a[3][3], double x[3], double y[3])
{                                         
   /*!
     Parameters:
     order: row or column major
     TransA: transpose A
     M:      rows in A
     N:      cols in x
     alpha:  scalar factor
     A:      matrix A
     lda:    major stride for A
     x:      vector x
     ldb:    major stride for x
     beta:  scalar factor
     y:      vector y
     ldc:    major stride for y
     Performs operation y = alpha*A*x + beta
    */
   int i;
   double tmp[3], *p1, *p2;
   /* use a tmp array as y may be the same address as x */
   cblas_dgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0,
               &a[0][0], 3, x, 1, 0.0, tmp, 1); //assumes x is 3x1
   for (p1=&tmp[0],p2=&y[0],i=0; i<3; i++,p1++,p2++) *p2 = *p1;
}

/*!
 * assumes starting at 0 ie rowl=coll=0
 * last indices for each dimension are rowh, colh
 */
int **imatrix(long rowl, long rowh, long coll, long colh)
{
    long i,j,nrows=rowh+1,ncols=colh+1;
    int **im=0;
                                                                                
    if (rowl!=0 || coll!=0 )
        OnError("imatrix start indices should be zero",INDEX_ERR);
    im= new int *[nrows];
    if (im) {
      if ((im[0]= new int[nrows*ncols])==0) {
            delete [] im; im=0;
      }
      else
        /* set up the array mapping */
        for(i=1;i<nrows;i++)  im[i]=im[i-1]+ncols;
    }
    return(im);
}

void free_imatrix(int **im,long rowl,long rowh,long coll,long colh)
{
      if (im) {
          if (im[0]) { delete [] im[0]; im[0]=0; }
          delete [] im;
      }
}

/*!
 * assumes starting at 0 ie rowl=coll=0
 * last indices for each dimension are rowh, colh
 */
double **dmatrix(long rowl, long rowh, long coll, long colh)
{
    long i,j,nrows=rowh+1,ncols=colh+1;
    double **d=0;
                                                                                
    if (rowl!=0 || coll!=0 )
        OnError("dmatrix start indices should be zero",INDEX_ERR);
    d=(double **)malloc(nrows*sizeof(double));
    if (d) {
      if (((d[0]=
            (double *)malloc(nrows*ncols*sizeof(double)))==0)) {
            free(d); d=0;
      }
      else
        /* set up the array mapping */
        for(i=1;i<nrows;i++)  d[i]=d[i-1]+ncols;
    }
    return(d);
}

void free_dmatrix(double **d,long rowl,long rowh,long coll,long colh)
{
      if (d) {
          if (d[0]) { free(d[0]); d[0]=0; }
          free(d);
      }
}


/*!
 * assumes starting at 0 ie rowl,coll,depthl are all 0
 * last indices for each dimension are rowh, colh, depthh
 */
double ***d3tensor(long rowl, long rowh,
                   long coll, long colh,
                   long depthl, long depthh)
{
    long i,j,nrows=rowh+1,ncols=colh+1,ndepth=depthh+1;
    double ***m3t=0;
                                                                                
    if (rowl!=0 || coll!=0 || depthl!=0)
        OnError("d3tensor start indices should be zero",INDEX_ERR);
    m3t=(double ***)malloc(nrows*sizeof(double));
    if (m3t) {
      if (((m3t[0]=
            (double **)malloc(nrows*ncols*sizeof(double)))==0) ||
          ((m3t[0][0]=
            (double *)malloc(nrows*ncols*ndepth*sizeof(double)))==0)) {
          if (m3t[0]) {
            free(m3t[0]); m3t[0]=0;
            free(m3t); m3t=0;
          }
      }
      else {
        /* set up the array mapping */
        for(i=1;i<ncols;i++) m3t[0][i]=m3t[0][i-1]+ndepth;
        for(i=1;i<nrows;i++) {
            m3t[i]=m3t[i-1]+ncols;
            m3t[i][0]=m3t[i-1][0]+ncols*ndepth;
            for(j=1;j<ncols;j++) m3t[i][j]=m3t[i][j-1]+ndepth;
        }
      }
    }
    return m3t;
}

void free_d3tensor(double ***d3t,long rowl,long rowh,long coll,long colh,
                   long depthl,long depthh)
{
      if (d3t) {
          if (d3t[0][0]) { free(d3t[0][0]); d3t[0][0]=0; }
          if (d3t[0]) { free(d3t[0]); d3t[0]=0; }
          free(d3t);
      }
}

#if XY
void matmult(double a[3][3], double b[3][3], double ab[3][3])
{                                         
    double d[3][3],z;
    int i,j,k;
                                
    
                                
    for(i=0;i<3;i++)
    {
	for(j=0;j<3;j++)
	{
	    z=0.0;
	    for(k=0;k<3;k++)
	    {
		z=z+a[i][k]*b[k][j];
	    }
	    d[i][j]=z;
	}
    }
    
    for(i=0;i<3;i++)
    {
	for(j=0;j<3;j++)
	{
	    ab[i][j]=d[i][j];
	}
    }
}	

void matinverse(double a[3][3], double ab[3][3])
{                                         
    //double d[3][3],z,aa;
    
    float  **fa, **fd, **d;
    int i,j,k;
    
    
    d=matrix(1,3,1,3);
    fa=matrix(1,3,1,3);
    fd=matrix(1,3,1,3);

    
    for(i=1;i<=3;i++)
    {
	for(j=1;j<=3;j++)
	{
	    fa[i][j]=(float)a[i-1][j-1];
            d[i][j]=(float)a[i-1][j-1];
	}
    }
    
    gaussj(fa,3,d,3);
   
    

    for(i=1;i<=3;i++)
    {
	for(j=1;j<=3;j++)
	{
	    ab[i-1][j-1]=(double)fa[i][j];
            
	}
    }
    free_matrix(d,1,3,1,3);
    free_matrix(fa,1,3,1,3);
    free_matrix(fd,1,3,1,3);
   
  /*                              
    matdeterm(a, &aa);

    matadj(a, d);

    for(i=0;i<3;i++)
    {
	for(j=0;j<3;j++)
	{
	    z=0.0;
	    for(k=0;k<3;k++)
	    {
		d[i][k]=d[i][k]/(aa);
	    }
	    //d[i][k]=z;
	}
    }
    
    for(i=0;i<3;i++)
    {
	for(j=0;j<3;j++)
	{
	    ab[i][j]=d[i][j];
	}
    }
    
  */
     
}
	
void matdeterm(double a[3][3], double *da)
{                                         
    //double da;
    
                                
    *da=-a[0][2]*a[1][1]*a[2][0]-a[0][0]*a[1][2]*a[2][1]-a[0][1]*a[1][0]*a[2][2]+
        a[0][0]*a[1][1]*a[2][2]+a[0][1]*a[1][2]*a[2][0]+a[0][2]*a[1][0]*a[2][1];
  
}
void matadj(double a[3][3], double ia[3][3])
{                                         
    double d[3][3],b[3][3],z;
    int i,j,k;

    b[0][0]=(a[1][1]*a[2][2]-a[1][2]*a[2][1]);
    b[0][1]=-((a[1][0]*a[2][2]-a[1][2]*a[2][0]));
    b[0][2]=(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
    b[1][0]=-((a[0][1]*a[2][2]-a[0][2]*a[2][1]));
    b[1][1]=(a[0][0]*a[2][2]-a[0][2]*a[2][0]);
    b[1][2]=-((a[0][0]*a[2][1]-a[0][1]*a[2][0]));
    b[2][0]=(a[0][1]*a[1][2]-a[0][2]*a[1][1]);
    b[2][1]=-((a[0][0]*a[1][2]-a[0][2]*a[1][0]));
    b[2][2]=(a[0][0]*a[1][1]-a[0][1]*a[1][0]);
                                
    

    //transpose

    for(i=0;i<3;i++)
    {
	for(j=0;j<3;j++)
	{
	    ia[i][j]=b[j][i];
	}
    }

}		
#endif
