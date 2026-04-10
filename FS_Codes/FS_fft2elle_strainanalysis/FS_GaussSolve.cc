#include "FS_GaussSolve.h"

/* 
 * How to solve any system of equations with N unknowns
 */
/*!
// Allocating array memory for coefficient matrix
int N = 6;
double **dCoef = 0;
dCoef=dmatrix(0,N-1,0,N);
// Counting 1st value as rows and 2nd as columns

dCoef[0][0] = 3; dCoef[0][1] = 2; dCoef[0][2] = 5; dCoef[0][3] = 6; dCoef[0][4] = 2; dCoef[0][5] = 3; dCoef[0][6] = 1;
dCoef[1][0] = 2; dCoef[1][1] = 7; dCoef[1][2] = 1; dCoef[1][3] = 3; dCoef[1][4] = 7; dCoef[1][5] = 3; dCoef[1][6] = 8;
dCoef[2][0] = 4; dCoef[2][1] = 9; dCoef[2][2] = 8; dCoef[2][3] = 2; dCoef[2][4] = 6; dCoef[2][5] = 6; dCoef[2][6] = 4;
dCoef[3][0] = 1; dCoef[3][1] = 4; dCoef[3][2] = 5; dCoef[3][3] = 3; dCoef[3][4] = 9; dCoef[3][5] = 1; dCoef[3][6] = 2; 
dCoef[4][0] = 7; dCoef[4][1] = 5; dCoef[4][2] = 7; dCoef[4][3] = 2; dCoef[4][4] = 4; dCoef[4][5] = 6; dCoef[4][6] = 6; 
dCoef[5][0] = 3; dCoef[5][1] = 2; dCoef[5][2] = 1; dCoef[5][3] = 4; dCoef[5][4] = 7; dCoef[5][5] = 5; dCoef[5][6] = 3;
printf("\nStarting coefficient matrix:\n");
SolverDisplayDmatrix(dCoef,N);
printf("\n");

SolveEquationSystem(dCoef,N);


printf("\nFinal:\n");
SolverDisplayDmatrix(dCoef,N);
printf("\n");
    
free_dmatrix(dCoef,0,N,0,N); 
*/ 


int SolveEquationSystem(double **dCoef,int N)
{
    /*
     * The following function solves a system of N equations with N unknowns
     * The result is stored in the input array following the Gaussian scheme 
     * (if x,y,z are unknown):
     * 1 0 0 2.8 x = 2.8
     * 0 1 0 3.4 y = 3.4
     * 0 0 1 1.2 z = 1.2
     * 
     * Input is the coefficient matrix created using dmatrix() and the number of 
     * unknowns
     */
    int err = 0;
    double dTmpValue = 0.0; // for any temporarily used value
    
    // 1. Check that the initial diagonal elements are not zero:
    if(SolverSwitchRows(dCoef,N))
    {
        printf("ERROR: Not solvable, too many zero coefficients\n");
        return (1);
    }
    
    // 1. Iteratively set all diagonal elements to 1, row by row:
    for (int row=0;row<N;row++)
    {    
        // 1.1 Again check that the initial diagonal elements are not zero:
        if(SolverSwitchRows(dCoef,N))
        {
            printf("ERROR: Not solvable, too many zero coefficients\n");
            return (1);
        }
        // 1.1 Divide all values in row by the element that should be 1:
        dTmpValue = dCoef[row][row];
        for (int col=0;col<=N;col++) dCoef[row][col] /= dTmpValue;
        
        // 1.2 Subtract all elements of rows further down by this row times a 
        // factor to set all elements below the element that should be 1 to 0:
        for (int row2=row+1;row2<N;row2++)
        {
            dTmpValue = dCoef[row2][row]; // will be the factor             
            for (int col=0;col<=N;col++) 
                dCoef[row2][col] -= dTmpValue*dCoef[row][col];   
        }                                    
    }
    
    // 3. Now the last unknown (uk) is known and we can compute all others
    for (int uk=N-1;uk>0;uk--)
    {
        for (int row=N-2;row>=0;row--)
        {
            if (uk>row)
            {
                dCoef[row][N] -= dCoef[row][uk]*dCoef[uk][N];
                dCoef[row][uk] = 0.0; // we may set this to zero now 
            }
        }
    }
        
    //dCoef[1][N] -= dCoef[1][2]*dCoef[2][N];
    //dCoef[1][2] = 0.0; // we may set this to zero now 
    
    //dCoef[0][N] -= dCoef[0][2]*dCoef[2][N];
    //dCoef[0][2] = 0.0; // we may set this to zero now 
    
    //dCoef[0][N] -= dCoef[0][1]*dCoef[1][N];
    //dCoef[0][1] = 0.0; // we may set this to zero now 
    
    return (err);
}

/*
 * Switching rows while solving coefficient matrix to make sure that no diagonal 
 * element is zero
 */
int SolverSwitchRows(double **dCoef, int N)
{    
    double dValTmp = 0.0; // Temporarily storing information
    for (int i=0;i<N;i++)
    {
        if (dCoef[i][i]==0.0)
        {
            // Switch with any suitable row j 
            // --> only suits if after switch [j][j]!=0
            bool bFound = false;
            int j=0;
            while(!bFound) 
            {
                if (dCoef[j][i]!=0.0 && dCoef[i][j]!=0.0) bFound = true;
                else j++;
                
                if (j>=N)  return(1);
            }
            
            // Actually switch the rows:
            for (int k=0;k<=N;k++) 
            {
                dValTmp = dCoef[i][k];
                dCoef[i][k] = dCoef[j][k];
                dCoef[j][k] = dValTmp;
            }
        }
    }
    return (0);
}

void SolverDisplayDmatrix(double **dMatrix,int N) //N is number of rows
{    
    for (int i=0;i<N;i++)
    {
        for (int j=0;j<=N;j++)
        {
            printf("%.2f ",dMatrix[i][j]);
        }
        printf("\n");
    }
}

void TestSolver()
{
    // Allocating array memory for coefficient matrix
    int N = 6;
    double **dCoef = 0;
    dCoef=dmatrix(0,N-1,0,N);
    // Counting 1st value as rows and 2nd as columns

    dCoef[0][0] = 1; dCoef[0][1] = 3; dCoef[0][2] = 2.5; dCoef[0][3] = 0; dCoef[0][4] = 0; dCoef[0][5] = 0; dCoef[0][6] = 2;
    dCoef[1][0] = 1; dCoef[1][1] = 3; dCoef[1][2] = 1.4; dCoef[1][3] = 0; dCoef[1][4] = 0; dCoef[1][5] = 0; dCoef[1][6] = 1;
    dCoef[2][0] = 1; dCoef[2][1] = 2; dCoef[2][2] = 1.4; dCoef[2][3] = 0; dCoef[2][4] = 0; dCoef[2][5] = 0; dCoef[2][6] = 4;
    dCoef[3][0] = 0; dCoef[3][1] = 0; dCoef[3][2] = 0; dCoef[3][3] = 1; dCoef[3][4] = 3; dCoef[3][5] = 2.5; dCoef[3][6] = 7; 
    dCoef[4][0] = 0; dCoef[4][1] = 0; dCoef[4][2] = 0; dCoef[4][3] = 1; dCoef[4][4] = 3; dCoef[4][5] = 1.4; dCoef[4][6] = 6; 
    dCoef[5][0] = 0; dCoef[5][1] = 0; dCoef[5][2] = 0; dCoef[5][3] = 1; dCoef[5][4] = 2; dCoef[5][5] = 1.4; dCoef[5][6] = 2;
    
    // Good web-page to validate:
    // http://www.arndt-bruenner.de/mathe/scripts/gleichungssysteme.htm
    
    printf("\nStarting coefficient matrix:\n");
    SolverDisplayDmatrix(dCoef,N);
    printf("\n");

    SolveEquationSystem(dCoef,N);


    printf("\nFinal:\n");
    SolverDisplayDmatrix(dCoef,N);
    printf("\n");
        
    free_dmatrix(dCoef,0,N,0,N); 
}
