
/******************************************************
* Spring Code Mike 2.0
*
* header for phase_base class in phase_base.h
* 
* inherits the lattice class
*
* Expansion of Spring Code for Reactions
*
*
*
*
* Daniel Koehn and Jochen Arnold March 2003
* Mainz 
*
* Daniel Dec. 2003
*
* Daniel Koehn and Till Sachau Dec. 2004
*
* We thank Anders Malthe-Srenssen for his enormous help
******************************************************/

#ifndef PHASE_BASE_H
#define PHASE_BASE_H

#include "phase_base.h"
#include "lattice.h"

class Phase_Base : public Lattice
	{
	public:

		double time;				// time
		double area;				// area of the box
		float dif_const;			// diffusion constant
		float gas_const;			// universal gas constant
		float the_size;				// scaling variable for space
		float temp;        		 	// Temperature
		float rate_constant;			// external variable for rate constant
		float grav_grad;			// gravitational gradient
		float fluid_pressure;			// fluid pressure
		float pois;				// poisson number
		float initial_diameter; // for the heat distribution

		Phase_Base();

		~Phase_Base();

		void Adjust_Hole_List();					// cleaner for make holes etc.
                void Update_Fluid_Neighbour_List();				// oriented list gives right and left 

		//------------------------------------------------------------------
		// usr functions for the intialization
		//------------------------------------------------------------------
		void MakeHole(int nb);						// make a hole
		void Set_Mineral_Parameters(int what);				// set parameters of minerals
		void Set_Absolute_Box_Size(float size);				// set absolute size
		void Set_Time(float intime, int intimescale);			// set absolute time and timescale
		void Set_Mass_Distribution(float mass_size, float factor);	// set a distribution on mass
                
        };

#endif
