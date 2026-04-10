
#ifndef phasefield_h
#define phasefield_h

#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "parseopts.h"
#include "init.h"
#include "runopts.h"
#include "file.h"
#include "setup.h"


void Reference();

/*
 * indices for User data values for this process
 */
#define uRESET 	0   // reset unode values (1 = reset)
#define uK 		1	// Latent heat
#define uEPS 	2   // interfacial width
#define uDELTA 	3   // modulation of the interfacial width
#define uANGLE0 4   // orientation of the anisotropy axis
#define uANISO 	5   // anisotropy 2*PI/ANISO
#define uALPHA 	6   // m(T) = ALPHA/PI * atan(GAMMA*(TEQ-T))
#define uGAMMA 	7   //

#define NX 	 	300 // size of the mesh NX*NY 
#define NY 	 	300	// currently assumes NX=NY 

#define PI 		3.14159265358
#define	NR_END	1

#endif
