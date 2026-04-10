#ifndef FS_GaussSolve_h
#define FS_GaussSolve_h

#include <stdio.h>
#include "mat.h"

using namespace std;

void TestSolver();

// For solver for systems of equations:
int SolveEquationSystem(double **dCoef,int N);
    int SolverSwitchRows(double **dCoef, int N);
    
// Just to display the coefficient matrix, N is number of unknowns:
void SolverDisplayDmatrix(double **dMatrix,int N);

#endif
