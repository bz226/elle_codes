#ifndef E_mat_h
#define E_mat_h
void euler(double rmap[3][3], double phi, double rho, double g);
void eulerZXZ(double rmap[3][3], double phi, double rho, double g);
void uneuler(double rmap[3][3], double *a, double *b, double *g);
void uneulerZXZ(double rmap[3][3], double *a, double *b, double *g);
void euler2cax(double a, double b, double c, double *cax);
void eulerZXZ2cax(double a, double b, double c, double *cax);
void uneuler(double rmap[3][3], float *a, float *b, float *g);
void euler2cax(float a, float b, float c, float *cax);
void orient(double rmap[3][3]);
void orientZXZ(double rmap[3][3]);
void orientmat(double a[3][3], double phi, double rho, double g);
void orientmatZXZ(double a[3][3], double phi, double rho, double g);
void eulerRange(double *e, double *e2);
void matmult(double a[3][3], double b[3][3], double ab[3][3]);
void matinverse(double a[3][3], double ab[3][3]);
void mataxismult(double a[3][3], double ax[3], double y[3]);
void matvecmult(double a[3][3], double x[3], double y[3]);
void matdeterm(double a[3][3], double *da);
void matadj(double a[3][3], double ia[3][3]);
double **dmatrix(long nrl, long nrh, long ncl, long nch);
int **imatrix(long nrl, long nrh, long ncl, long nch);
void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch);
void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch);
double ***d3tensor(long nrl, long nrh, long ncl, long nch, long ndl,
long ndh);
void free_d3tensor(double ***t,long nrl, long nrh, long ncl, long nch,
long ndl, long ndh);
/*
 * Below this line are functions included for backward compatibility
 * only
 */
void Euler2cax(double a, double b, double c, double *cax);
void Euler2cax(float a, float b, float c, float *cax);
#endif
