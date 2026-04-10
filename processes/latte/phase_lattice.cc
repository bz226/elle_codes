/******************************************************
* Spring Code Mike 2.0
*
* Functions for phase_lattice class in phase_lattice.cc
*
* Function inherits phase_base class which itself 
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
* We thank Anders Malthe-Soerenssen for his enormous help
******************************************************/







//------------------------------------
// system headers
//------------------------------------

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <math.h>

//------------------------------------
// elle headers
//------------------------------------

#include "lattice.h"         // include lattice class
#include "phase_lattice.h"   // include phase_lattice class that inherits lattice class
#include "unodes.h"          // include unode funct. plus undodesP.h for c++
#include "attrib.h"
#include "attribute.h"
#include "attribarray.h"
#include "nodes.h"
#include "nodesP.h"
#include "interface.h"
#include "file.h"
#include "error.h"
#include "general.h"
#include "runopts.h"
#include "polygon.h"
#include "tripoly.h"
#include "display.h"
#include "check.h"
#include "convert.h"
#include "update.h"

using std::cout;
using std::endl;

//  CONSTRUCTOR
/*******************************************************
 * Constructor for the phase_lattice class
 * Does not do much at the moment, defines some variables
 * Phase lattice inherits the lattice class that does
 * most of the intialization work. 
 *
 *
 * Daniel March 2003
 ********************************************************/

//---------------------------------------------------------------
// Constructor of Phase_Lattice class
//---------------------------------------------------------------

Phase_Lattice::Phase_Lattice():

water_compres(0.0000000005),
count_time(0.0)

{


}

// Set_Fluid_Pressure
/***********************************************************
* flunction sets the fluid pressure
*
* pressure is read in MPa and scaled to model pressure
* assuming that 1.0 in the model is 10 GPa
*
* daniel 2004
***********************************************************/

//---------------------------------------------------------
//  read pressure in MPa
//  scale pressure 
//  and call Update_Fluid_Neighbour_List in 
//  the phase_base class to give particles along the 
//  boundary their fluid pressure
//---------------------------------------------------------

void Phase_Lattice::Set_Fluid_Pressure(float pressure)
{
	fluid_pressure = pressure * -0.0001;
	//Update_Fluid_Neighbour_List();
	Adjust_Hole_List();
}

// Set_RateFactor
/*********************************************************************
* Sets a distribution on the rate of the reaction. 
* pics a pseudorandom number. New rate is from the original rate 
* to factor times the original rate. Linear distribution 
*
* daniel 2004
*********************************************************************/
 
void Phase_Lattice::Set_RateFactor(float factor)
{
    int i;     			// counter
    float ran_nb;		// pseudorandom number

    runParticle = &refParticle;

    for (i = 0 ; i < numParticles ; i ++) // loop through particles
    {
        ran_nb = rand() / (float) RAND_MAX;  	// pic pseudorandom from 0 to 1.0 
        ran_nb = ran_nb * (factor-1.0);		// distribution from 0 to factor - 1.0
        ran_nb = ran_nb + 1.0;			// 0.0 is now 1.0 times the rate

        runParticle->rate_factor = runParticle->rate_factor * ran_nb;

        runParticle = runParticle->nextP;
    }
}
 
// Set_Rate_Two_Phase
/**********************************************************************************
* for a bimodal distribution of dissolution constants. 
* input ratio gives the ratio of the second to the first value of the 
* rate constants or youngs moduli. 1.0 is 100 percent for the second phase relative
* to the original phase, 0.01 1 percent. 
* factor gives the percentage that the original rate will change. 1.0 is no 
* change. 
* elastic is the same, changes the spring constants of particles and 
* the youngs modulus (for the repulsion) 
* which particle is in the second distribution is determined using 
* pseudorandom numbers as probability factors. 
*
* Gives a distribution that is Gaussian through the picking of random numbers 
*
*
* Daniel 2004
**********************************************************************************/

//--------------------------------------------------------------------------
// called in phase_lattice.cc, defined in phase_lattice.h
//
// receives ratio which is the percentage of the second phase
// and a factor for the reaction constants of the second phase
// and a factor (elastic) for the youngs modulus and spring constants
// of the second phase
//-------------------------------------------------------------------------
 
 void Phase_Lattice::Set_Rate_Two_Phase(float ratio, float factor, float elastic)
{
    int i,j;     			// counter
    float ran_nb;		// pseudorandom number

    srand(std::time(0));

    for (i = 0 ; i < numParticles ; i ++) // loop through particles
    {
        ran_nb = rand() / (float) RAND_MAX;  // pic pseudorandom from 0 to 1.0 for grains

        if (ran_nb < ratio)  // in the second distribution ? 
        {
        
          runParticle->rate_factor = runParticle->rate_factor * factor; // change rate constant
          runParticle->young = runParticle->young * elastic; // change youngs modulus

          for (j = 0 ; j < 8 ; j++)
          {
            if (runParticle->neigP[j])  // and change the spring constants
            {
              runParticle->springf[j] = runParticle->springf[j] * elastic;
            }
          }

        }

        runParticle = runParticle->nextP;
    }
}

// Active_Grain_Boundaries
/****************************************************************
* This function activates all the particels on grain boundaries
* All these can then dissolve as well (or react) 
* relates elle grains to reactions of mike
*
* daniel 2004
****************************************************************/

//--------------------------------------------------------
// called from the mike.elle file by the user
// in phase_lattice class, defined in phase_lattice.h
//
// receives nothing but activates all grian boundaries 
// from the initial elle file as reactive (particles
// along that boundary). 
// elle grain boundary particles have the flag is_boundary
// turned on. Reactive means that the flag isHoleBoundary
// is also turned on. 
//--------------------------------------------------------

void Phase_Lattice::Active_Grain_Boundaries()
{
  int i;

  for (i = 0 ; i < numParticles ; i++)  // loop through particles
  {
     if (!runParticle->isHole) // if not in a hole 
     {
       if (runParticle->is_boundary) // is boundary if boundary of an elle grain
       {
       
           runParticle->isHoleBoundary = true; // I am defined as a hole boundary
       }
     }
     runParticle = runParticle->nextP;
  }
}

// All_Active
/*****************************************************************
* All particles are now defined as grain boundaries and can 
* react. 
*
* daniel 2004
*****************************************************************/

void Phase_Lattice::All_Active()
 {
  int i;

  for (i = 0 ; i < numParticles ; i++)  // loop through particles
  {
     if (!runParticle->isHole)
     {
        //-------------------------------------------------
        // all except hole particles are boundaries 
        // and can react
        //-------------------------------------------------
        
           runParticle->isHoleBoundary = true; 
     }
     runParticle = runParticle->nextP;
  }
}



// Set_Mass_Distribution
/***********************************************************
* Set a distribution for the mass (area ) of particles
* for the reaction. 
*
* function receives two variables, the size of the mass 
* distribution, distribution is +- mass size, 
* and a factor to make the distribution larger. 
* mass_size should be number between 0.0 and 2.0, dont
* use 2.0 otherwise some particles have zero mass. 
* factor can increase the distribution, however this also 
* changes the overall mean mass in the system. 
* factor of 1.0 does not change anything
*
* Daniel, March 2003, December 2003
***********************************************************/

 void Phase_Lattice::Set_Mass_Distribution(float mass_size, float factor)
 {
    int i;     			// counter
    float ran_nb;		// pseudorandom number
    float mass_nb;		// distribution for mass 
    
    runParticle = &refParticle;
    
    for (i = 0 ; i < numParticles ; i ++) // loop through particles
    {
        ran_nb = rand() / (float) RAND_MAX;  // pic pseudorandom from 0 to 1.0 for grains
        mass_nb = ran_nb  * mass_size;  // now from 0 to distribution size
        
        //-------------------------------------------
        // set the mass distribution
        //-------------------------------------------
        
        runParticle->area = runParticle->area * (1.0+mass_nb-(mass_size/2.0));
        
        //--------------------------------------------
        // multiply by factor
        //--------------------------------------------
	
		runParticle->area = runParticle->area * factor;
        
        runParticle = runParticle->nextP;
    }
 }
 
// Make_Circular_Hole
/*********************************************************************
* Function make a round hole in the lattice. Radius is radius in 
* particles ? 
* if shift is 0.0 then the hole is in the center or the box. 
* with shift it can be shifted towards the left or right along
* the x direction. 
* box dimension in x is 1.0, 0.0 to 0.5 is shift to the right
* -0.0 to - 0.5 shift to the left. 
*
* Daniel 2004
**********************************************************************/
 
 void Phase_Lattice::Make_Circular_Hole(float radius, float shift)
 {
	int i,j;		// counters
	float help;

    runParticle = &refParticle;   // start with first particle
    
    for (i = 0 ; i < numParticles ; i++)  // loop through all particles
    {
        help = (runParticle->ypos*(particlex )-(particlex/2.0))*(runParticle->ypos*(particlex)-(particlex/2.0));
        help = sqrt(help + (runParticle->xpos*(particlex)-((particlex+shift)/2.0))*(runParticle->xpos*(particlex)-((particlex+shift)/2.0)));
		
        if (help < radius)    // if particle is in the hole
        {
            runParticle->isHole = true;  // set flag-> particle is now part of a hole
	    runParticle->young = 0.0; 	// young modulus for repulsion is set to zero
            
            //----------------------------------------------------
            // in the case that this particle was the boundary 
            // of some hole it is not a boundary anymore but now
            // in the hole
            //----------------------------------------------------
            
            
            if (runParticle->isHoleBoundary)  
            {
                runParticle->isHoleBoundary = false;
            }
            
            //-------------------------------------------------------
            // now set all the spring constants of the particle 
            // to zero
            //-------------------------------------------------------
            
            for (j=0 ; j<8 ; j++)    // loop through neighbours of particle 
            {
                if (runParticle->neigP[j])    // if there is a neighbour
                {
                    runParticle->springf[j] = 0.0;  // set springconstant to zero
                }
            }
        }
        runParticle = runParticle->nextP;  // and go on looping
    }
    
    //--------------------------------------------------------
    // now we need a cleaning function in order to set the 
    // new hole boundary
    //--------------------------------------------------------
    
    Adjust_Hole_List();   // this is the cleaner
	 
 }

// Make_Elliptical_Hole
/*********************************************************************
* Function makes an elliptical hole in the lattice. 
* height is the height ot the ellipse in particles, ratio 
* the ratio of the two radii, ratio x to y, number larger than 1.0 
* is a flat lying ellipse 
* if shift is 0.0 then the hole is in the center or the box. 
* with shift it can be shifted towards the left or right along
* the x direction. 
* box dimension in x is 1.0, 0.0 to 0.5 is shift to the right
* -0.0 to - 0.5 shift to the left. 
*
* Daniel 2004
**********************************************************************/ 
 
 void Phase_Lattice::Make_Elliptical_Hole(float height, float ratio, float shift)
 {
	int i,j;		// counters
	float help;

    runParticle = &refParticle;   // start with first particle
    
    for (i = 0 ; i < numParticles ; i++)  // loop through all particles
    {
    help = (runParticle->ypos*(particlex )-(particlex/2.0))*(runParticle->ypos*(particlex)-(particlex/2.0));
    help = sqrt(help + ((runParticle->xpos+shift)*(particlex/ratio)-((particlex/ratio)/2.0))*((runParticle->xpos+shift)*(particlex/ratio)-((particlex/ratio)/2.0)));
		
        if (help < height)    // if particle is in the grain
        {
            runParticle->isHole = true;  // set flag-> particle is now part of a hole
	    runParticle->young = 0.0; 	// young modulus for repulsion is set to zero
            
            //----------------------------------------------------
            // in the case that this particle was the boundary 
            // of some hole it is not a boundary anymore but now
            // in the hole
            //----------------------------------------------------
            
            
            if (runParticle->isHoleBoundary)  
            {
                runParticle->isHoleBoundary = false;
            }
            
            //-------------------------------------------------------
            // now set all the spring constants of the particle 
            // to zero
            //-------------------------------------------------------
            
            for (j=0 ; j<8 ; j++)    // loop through neighbours of particle 
            {
                if (runParticle->neigP[j])    // if there is a neighbour
                {
                    runParticle->springf[j] = 0.0;  // set springconstant to zero
                }
            }
        }
        runParticle = runParticle->nextP;  // and go on looping
    }
    
    //--------------------------------------------------------
    // now we need a cleaning function in order to set the 
    // new hole boundary
    //--------------------------------------------------------
    
    Adjust_Hole_List();   // this is the cleaner
	 
 }
 
// Set Distribution Surface Energy
/*********************************************************************
* Function sets a distribution on the surface free energy of 
* particles 
*
* input in percent, 1.0 = 100; 0.01 = 1.0 percent
*
* variation +- input percent
*
* does two things, distribution is a distribution of the surface 
* free energy whereas springdis is a distribution on the number 
* of open springs of the particle
*
* Daniel 2004
***********************************************************************/

void Phase_Lattice::Set_Distribution_Surface_Energy(float distribution, float springdis)
{
    int i; 
    float ran_nb;
    
    for (i = 0; i < numParticles; i++)
    {
        ran_nb = rand() / (float) RAND_MAX;  // pic random number
    
        ran_nb = (ran_nb - 0.5)  * 2.0 * springdis;   //  +- springdis
        
        runParticle->spring_var = ran_nb;  // apply spring variation 
        
        ran_nb = ran_nb * distribution / springdis; // apply distribution of surface free energy
    
        runParticle->surf_free_E = runParticle->surf_free_E + (runParticle->surf_free_E * ran_nb);
    
        runParticle = runParticle->nextP;
    }
}

// Set Distribution Particle MV
/*************************************************************************
* Set a distribution on the molecular volume of particles
*
* Daniel 2004
*************************************************************************/

void Phase_Lattice::Set_Distribution_Particle_MV(float distribution)
{
    int i;     		// counter
    float ran_nb;	// random number
    
    for (i = 0; i < numParticles; i++)
    {
        ran_nb = rand() / (float) RAND_MAX;
    
        ran_nb = (ran_nb - 0.5) * distribution * 2.0; // from - to + distribution
    
        runParticle->mV = runParticle->mV + (runParticle->mV * ran_nb);  // and apply
    
        runParticle = runParticle->nextP;
    }
}

// DissolveXRow
/***********************************************************************
* Function dissolves all particles between x min and x max
* 
* Daniel 2004
***********************************************************************/


void Phase_Lattice::DissolveXRow(float xmin, float xmax)
{
	int i,j;	// counter
	
	for (i = 0; i<numParticles; i++)
	{
			
		if (runParticle->xpos >= xmin && runParticle->xpos <= xmax)
		{
			runParticle->isHole = true; 	// is a hole
			runParticle->young = 0.0;	// youngs modulus is zero
			if (runParticle->isHoleBoundary) 
                            runParticle->isHoleBoundary = false;  // not a solid anymore
			for (j = 0; j < 8; j++)
			{
                            if (runParticle->neigP[j])
                                    runParticle->springf[j] = 0.0; // spring constants zero
			}
		}
		runParticle = runParticle->nextP;
	}
	Adjust_Hole_List(); // clean up 
}

// DissolveYRow
/***********************************************************************
* Function dissolves all particles between y min and y max
* 
* Daniel 2004
***********************************************************************/


void Phase_Lattice::DissolveYRow(float ymin, float ymax, bool remove)
{
	int i,j; 	// counter
	
	for (i = 0; i<numParticles; i++)
	{
					
		if (runParticle->ypos >= ymin && runParticle->ypos <= ymax)
		{
			runParticle->isHole = true;	// is a hole
			runParticle->young = 0.0;	// youngs modulus zero
			if (remove)
			{
				runParticle->ypos = -0.1;
				ElleRemoveUnodeFromFlynn(runParticle->p_Unode->flynn(),runParticle->p_Unode->id());
			}
			if (runParticle->isHoleBoundary) 
			
                 runParticle->isHoleBoundary = false; // not solid boundary
							
			
			for (j = 0; j < 8; j++)
			{
                            if (runParticle->neigP[j])
                                runParticle->springf[j] = 0.0; // spring constants zero
			}
		}
                
                //-------------------------------------------------------------------
                // all particles are initially defined to be in the upper box 
                // for one of the deformation routines. Now everything that is 
                // below the dissolved row is defined to be in a lower box, not in
                // the upper box. One deformation routine just moves all particles 
                // in the upper box downwards and all particles in the lower box 
                // upwards. Particles that then meet in the middle build up local 
                // stresses. Routine for Stylolites (Dissolution_Stylos(), in 
                // phase_lattice.cc) 
                // All other routines should ignore this flag. 
                // breaks down if more than one y row dissolves.
                //-------------------------------------------------------------------
                
                else if (runParticle->ypos < ymin)
                        runParticle->isUpperBox = false; 
		runParticle = runParticle->nextP;
	}
	Adjust_Hole_List();	// cleaner
}

 //DissolveYRow
/***********************************************************************
* Function dissolves all particles between y min and y max
* 
* Daniel 2004
***********************************************************************/


void Phase_Lattice::DissolveYRowSinus(float ymin, float ymax, bool remove)
{
	int i,j; 	// counter
	
	for (i = 0; i<numParticles; i++)
	{
					
		if (runParticle->ypos >= (ymin + sin(runParticle->xpos*8)/8) && runParticle->ypos <= (ymax+sin(runParticle->xpos*8)/8))
		{
			runParticle->isHole = true;	// is a hole
			runParticle->young = 0.0;	// youngs modulus zero
			if (remove)
			{
				runParticle->ypos = -0.1;
				ElleRemoveUnodeFromFlynn(runParticle->p_Unode->flynn(),runParticle->p_Unode->id());
			}
			if (runParticle->isHoleBoundary) 
			
                 runParticle->isHoleBoundary = false; // not solid boundary
							
			
			for (j = 0; j < 8; j++)
			{
                            if (runParticle->neigP[j])
                                runParticle->springf[j] = 0.0; // spring constants zero
			}
		}
                
                //-------------------------------------------------------------------
                // all particles are initially defined to be in the upper box 
                // for one of the deformation routines. Now everything that is 
                // below the dissolved row is defined to be in a lower box, not in
                // the upper box. One deformation routine just moves all particles 
                // in the upper box downwards and all particles in the lower box 
                // upwards. Particles that then meet in the middle build up local 
                // stresses. Routine for Stylolites (Dissolution_Stylos(), in 
                // phase_lattice.cc) 
                // All other routines should ignore this flag. 
                // breaks down if more than one y row dissolves.
                //-------------------------------------------------------------------
                
                else if (runParticle->ypos < (ymin + sin(runParticle->xpos*8)/8))
                        runParticle->isUpperBox = false; 
		runParticle = runParticle->nextP;
	}
	Adjust_Hole_List();	// cleaner
}

// Set_Concentration
/****************************************************************
* Set a basic concentration for particles
****************************************************************/

void Phase_Lattice::Set_Concentration()
	{
	int i;

	for (i = 0; i < numParticles; i++)
		{
		if (runParticle->isHole)
			runParticle->left_conc = runParticle->conc;
		else if (runParticle->isHoleBoundary)
			runParticle->left_conc = (1/runParticle->mV) ;
		else
			runParticle->left_conc = 1/runParticle->mV;

		runParticle = runParticle->nextP;
		}
	}
        
// Set_Concentration_Hole
/*******************************************************************
* give particles in fluid concentration defined by concentration
*
* In this routine all particles that are within the a hole 
* (at this moment only one hole in the box or all holes the 
* same concentration) get the same concentration defined by the 
* input parameter concentration 
*******************************************************************/

void Phase_Lattice::Set_Concentration_Hole(float concentration)
	{
	int i;

	for (i = 0 ; i < numParticles ; i++)
		{
		if (runParticle->isHole)
			runParticle->conc = concentration;

		runParticle = runParticle->nextP;
		}
	}

// SetGaussianRateDistribution
/*******************************************************************
* puts a quenched noise on the reaction constants of the particles
* in the box. This is a gaussian distribution of rate constants
* with a mean g_mean and a deviation of g_sigma
* 
* The distribution is determined as follows: Pic a pseudorandom number (from mean 
* plus/minus 2 times sigma)  to give a particle a dissolution constant
* Then determine the probability for that constant to appear from
* the Gauss function. Then pic a second pseudo random number (from 0 to sigma) to 
* determine whether or not the grain will actually get this spring constant. If 
* the random number is below the probability the spring contant is accepted, 
* if it is higher it is rejected and a new one is drawn. 
*
* Function writes also a text file with the set spring constant distribution
* calld gauss.txt 
*******************************************************************/

void Phase_Lattice::SetGaussianRateDistribution(double g_mean,double g_sigma)
{
    FILE *stat;   // for the outfile
    
    float prob;     // probability from gauss
    float k_spring;   // pict spring
    int i;       // counters
    float ran_nb;    // rand number
	
	srand(std::time(0));

    stat=fopen("gauss.txt","a");  // open statistic output append file 

    // grain_counter counter how many grain there were intially
    // times two is for security, not all grains may be there 
    // this can certainly be made nicer

    runParticle = &refParticle;
    
    for (i = 0; i < numParticles ; i++)    // loop through the particles now
    {
		do
		{
			k_spring = rand() / (float) RAND_MAX;    // pic a pseudorandom float between 0 and 1

            //----------------------------------------------------------------
            // now convert this distribution to a distribution from 
            // zero minus 8 times sigma to plus 8 times sigma
            //-----------------------------------------------------------------

			k_spring = (k_spring - 0.5)* 8.0 * (8.0 * g_sigma);  

            //--------------------------------------------------------------
            // and shift it to mean plus minus 2 times sigma
            //--------------------------------------------------------------

			k_spring = k_spring + g_mean;
            
            //---------------------------------------------------------------
            // now apply gauss function to determine a probability for this
            // reaction constant to occur
            //---------------------------------------------------------------

			prob = (1/(g_sigma*sqrt(2.0*3.1415927))); // part one (right part) of function

			prob = prob*(exp(-0.5*(((k_spring)-g_mean)/g_sigma)*(((k_spring)-g_mean)/g_sigma))); // rest
            
            //--------------------------------------------------------------
            // now adjust probablity to run from 0 to 1.0
            //--------------------------------------------------------------

			prob = prob * sqrt(2.0 * 3.1415927) * g_sigma;
            
            //---------------------------------------------------------------
            // pic the second pseudorandom number
            //---------------------------------------------------------------

			ran_nb = rand() / (float) RAND_MAX;

            //--------------------------------------------------------------
            // if the number picted is smaller or the same as the probability 
            // from the gauss function accept the rate constant, if not go on 
            // looping. 
            //-------------------------------------------------------------------

			if (ran_nb <= prob)   // if smaller
			{
				if (k_spring <= 0.0)  // if not 0.0 or smaller (dont want that)
					;
				else
				{
                    // set the constant in the list !
                    runParticle->rate_factor = k_spring;
                    
                    // and dump some data
                    
					fprintf(stat,"%f",k_spring);
					fprintf(stat,"\n");
                    
                    // and break the while loop ! 
                    
					break;
				}
			}
		}while(1);  // loop until break
		
		runParticle = runParticle->nextP;
    }
    
    
}

//-----------------------------------------------------------------------------
//		REACTION ROUTINES 
//-----------------------------------------------------------------------------


// Dissolution_Strain
/****************************************************
* first reaction function in Phase-Lattice class
*
* dissolution of quartz in pure water at 200 degrees
* no change in temperature so far. 
* simple dissolution as function of elastic and 
* surface energies. 
*
* receives the number of particles that have to 
* dissolve before a picture is dumped -> no time
* dependence of this picture dump -> action dumps
*
* Daniel March 2003, December 2003
*
* Missing long range surface energy effects
*****************************************************/

void Phase_Lattice::Dissolution_Strain(int dump)
	{
	int i,j;		// counters
	int dump_pict;	// counter for pictdump
	float pois;		// poisson ratio
	float lame1;	// Lame constant one, for elastic energy equation
	float lame2;	// second Lame constant for elastic energy equation
	float uxx;		// strain tensor component xx
	float uyy;		// strain tensor component yy
	float uxy;		// strain tensor component xy
	float rate;		// reaction rate (quartz dissolution at 200 degrees)
	float highest_rate; 	// counter for highest dissolution probability
	float internal_time;	// internal time scale counter
	float surfE;		// surface energery dummy
        float surface_right[40];	// help array for surface energy
        float surface_left[40];	// help array for surface energy
        Particle *help;		// help pointer

	//---------------------------------------------------------
	// initialization of time step, internal time is time,
	// dump_pict is zero. Time is the real size of the
	// time steps. This function will loop around within
	// this time step and dissolve particles until time is
	// over.
	//---------------------------------------------------------

	internal_time = time;
	dump_pict = 0;

	//--------------------------------------
	// start time loop
	//--------------------------------------

	do
		{
		cout << internal_time << endl;     // output how internal time goes down

		highest_rate = 0.0;	// initialization of probability

		pois = 0.3333333;	// Poisson ratio in triangular lattice is 0.3333

		temp = 200;         	// set temperature in degrees for quartz dissolution
		rate_constant = 0.000001;    // rate constant for quartz at 200C

		//------------------------------------------
		// loop through the particles
		//------------------------------------------

		runParticle = &refParticle;     // for particle loops

		for (i = 0 ; i < numParticles ; i++)  // and loop
			{
			if (runParticle->isHoleBoundary)   // only dissolve hole boundary particles
				{
				//if (!runParticle->is_lattice_boundary)  // dont dissolve lattice boundaries
					{
					//-----------------------------------------
					// set some local counters in particles
					// to zero
					//-----------------------------------------

					runParticle->open_springs = 0;
                                        runParticle->fluid_neig_count = 0;
                                        
                                        //-----------------------------------------------
                                        // lattice boundaries need an extra spring
                                        // for their surface energy because in the 
                                        // following algorithm only the fluid is 
                                        // considered and not the pressing pistons
                                        //-----------------------------------------------
                                        
                                        if (runParticle->is_lattice_boundary)
                                            runParticle->open_springs = 1;
					

					//------------------------------------------------
					// now loop through the springs of the particle
					//------------------------------------------------

					for (j = 0; j < 8; j++)
						{
						if (runParticle->neigP[j])  // if spring is active
							{
							//------------------------------------------------------
							// now count springs that are open towards the hole
							//------------------------------------------------------

							if (runParticle->neigP[j]->isHole)
								{
								runParticle->open_springs = runParticle->open_springs + 1;
								}
							//------------------------------------------------------
							// and count neighbour particles that are also part of
							// the bounary of the hole and not part of the lattice
							// boundary
							// neighbours are placed in a pointer list and number
							// is counted so that they can be accessed easily
							//------------------------------------------------------

							if (runParticle->neigP[j]->isHoleBoundary)
								{
								//if (!runParticle->neigP[j]->is_lattice_boundary)
									{
									//-------------------------------------------------
									// set particle into list
									//-------------------------------------------------

									runParticle->fluid_neigP[runParticle->fluid_neig_count] = runParticle->neigP[j];

									//----------------------------------------------
									// and update counter for place in the list
									// for the next particle
									//----------------------------------------------

									runParticle->fluid_neig_count = runParticle->fluid_neig_count + 1;
									}
								}
							}
						}

					//-------------------------------------------------------------
					// calculate the surface energy for a single particle
					// using the real radius of the particle and a function
					// that uses the number of open springs to calculate the
					// surface curvature.
					// the_size scales the non-dimensional radius to a real value
					// two open springs are zero curvature in the spring model
					// in the end the value is multiplied by the molecular volume
					//--------------------------------------------------------------

					runParticle->surfE = runParticle->surf_free_E*2.0/(runParticle->radius*the_size);

					runParticle->surfE = runParticle->surfE*((runParticle->open_springs-2.0)/4.0);

					runParticle->newsurfE = runParticle->surfE * runParticle->mV;

					}
				}
			runParticle = runParticle->nextP;
			}

			for (i = 0; i < numParticles; i++)
			{
				if (runParticle->isHoleBoundary)
				{
					if(runParticle->fluid_neig_count > 1)
					{
						runParticle->surfE = (runParticle->newsurfE + ((runParticle->fluid_neigP[0]->newsurfE+runParticle->fluid_neigP[1]->newsurfE)*0.5))*0.5;
					}
					else if (runParticle->fluid_neig_count > 0)
					{
						runParticle->surfE = (runParticle->newsurfE + runParticle->fluid_neigP[0]->newsurfE)*0.5;
					}
				}
				
				runParticle = runParticle->nextP;
			}
		//---------------------------------------------------------
		// loop a second time through all particles
		//---------------------------------------------------------

		for (i = 0 ; i < numParticles ; i++)
			{
			if (runParticle->isHoleBoundary)   // if the particle is on a boundary of the hole
				{
				//if (!runParticle->is_lattice_boundary)     // if its not part of the lattice boundary
					{

					//-------------------------------------------------------------------
					// first calculate the two lame constants from the youngs modulus of
					// a particle and the poisson number of the model
					//-------------------------------------------------------------------

					lame1 = runParticle->young * pois / (1.0-(pois*pois)*(1.0+pois));
					lame2 = runParticle->young / (2.0 + pois);

					//------------------------------------------------------------------------
					// calculate the components of the infinitesimal strain tensor from the
					// components of the stress tensor for the particle
					//------------------------------------------------------------------------

					uxy = ((1+pois)/runParticle->young)*runParticle->sxy;
					uxx = (1/runParticle->young)*(runParticle->sxx-(runParticle->syy*pois));
					uyy = (1/runParticle->young)*(runParticle->syy-(runParticle->sxx*pois));

					//------------------------------------------------------------------------
					// and now calculate the elastic energy for a single particle using the
					// two lame constants and the components of the infinitesimal strain
					// tensor of each particle
					//------------------------------------------------------------------------

					runParticle->eEl = 0.5 * lame1 * (uxx+uyy)*(uxx+uyy);
					runParticle->eEl = runParticle->eEl+(lame2*((uxx*uxx)+(uyy*uyy)+2*(uxy*uxy)));

					//-------------------------------------------------------------------------
					// and multiply by the molecular volume plus a scaling parameter for the
					// stress. The stress scaling is set with the youngsmodulus of the material
					//--------------------------------------------------------------------------

					runParticle->eEl = runParticle->eEl * runParticle->mV * pressure_scale * pascal_scale;

                                        //---------------------------------------------------------
                                        // setting average surface energy
                                        //---------------------------------------------------------

                                        for (j = 0; j < 40; j++)
                                        {
                                            surface_left[j] = 0.0;
                                            surface_right[j] = 0.0;
                                        }
						
                                        if (runParticle->fluid_neig_count)
                                        {							
							
                           
                                            surfE = 0.0;
                            
                                        //-------------------------------------------
                                        // first run along surface to the left 
                                        // and write the surface energies of 
                                        // the single particles along this interface
                                        // in the array surface_left
                                        //-------------------------------------------
                            
                                            if (runParticle->leftNeighbour)  // if I have a left neighbour 
                                            {
                                                help = runParticle->leftNeighbour;  // set help pointer 
							
                                                for (j = 0; j<40; j++)  // do that for 10 neighbours 
                                                {
                                                    if (help->leftNeighbour)   // if there is somebody on the left 
                                                    {	
                                                        surface_left[j] = help->surfE;  // write surface energy of that particle 
                                                        help = help->leftNeighbour;     // set help pointer 
                                                    }
                                                    else
                                                        surface_left[j] = 0.0;   // no particle means zero surface energy = no influence 
                                                }
                                            }
                            
                                            //-------------------------------------------------------
                                            // do the same thing for the neighbours towards the right 
                                            //--------------------------------------------------------
                            
                                            if (runParticle->rightNeighbour)
                                            {
                                                help = runParticle->rightNeighbour;
                                                for (j = 0; j<40; j++)
                                                {
                                                    if (help->rightNeighbour)
                                                    {
                                                        surface_right[j] = help->surfE;
                                                        help = help->rightNeighbour;
                                                    }
                                                    else
                                                        surface_right[j] = 0.0;
                                                }
                                            }
                            
                            //------------------------------------------------------------
                            // now we add all these surface energies up
                            // 
                            // this is done by taking the average energy of 
                            // the particle itself and the particle plus succesive next 
                            // neighbours, first the first one right and left plus the 
                            // particle, then the next two in each direction plus the 
                            // particle etc. for twenty one particles. The the surface
                            // energies are scaled according to distance of furthest
                            // neighbours so the average of three particles is divided
                            // by two, the average for five particles by four etc. 
                            // Therefore surface energies scale with 1/curvature radius
                            //-------------------------------------------------------------
                            
                                            surfE = runParticle->surfE;  // particle itself
                            
                                            for (j = 0; j < 40; j++)  // next ten
                                            {
                                                surfE = surfE + surface_left[j] + surface_right[j];  // add all up 
                                                surface_left[j] = surfE;  // and put them in surface_left array 
                                            }
                                            surfE = 0.0;
                                            for (j = 0; j < 40; j++)  // and do mean and scaling 
                                            {
                                                surfE = surfE + (surface_left[j]/((((j+1)*2)+1)*(((j+1)*2)+1)));
                                            }
							
                                            runParticle->newsurfE = runParticle->surfE + surfE; // and add particle itself
                                        }
                                        else
                                            runParticle->newsurfE = 0.0;

					//---------------------------------------------------------
					// now calculate the rate of dissolution for that particle
					// using the elastic and surface energy, divide by the
					// gasconstant and temperature in Kelvin and use a
					// linear rate law
					//---------------------------------------------------------

					rate = rate_constant*(1-exp(-((runParticle->eEl+runParticle->newsurfE)/(gas_const*(temp+273)))));

					//cout << rate << endl;

					//------------------------------------------------------------------
					// now calculate the probability of the particle to dissolve in the
					// given time (internal_time) as a function of the speed that
					// the particle will dissolve with depending on its open surface to
					// the fluid and its "area" (two-dimensional)
					//------------------------------------------------------------------

					rate = rate * internal_time * runParticle->mV * runParticle->rate_factor;


					runParticle->prob = rate * (runParticle->open_springs/6.0) / sqrt(runParticle->area/Pi);
					//runParticle->prob = rate / sqrt(runParticle->area/Pi);

					//-------------------------------------------------------------------
					// save highest probability during the run -> gives the internal
					// time step used.
					//-------------------------------------------------------------------

					if (runParticle->prob > highest_rate)
						{
						highest_rate = runParticle->prob;   // highest probability
						preRunParticle = runParticle;       // this is the particle (point to it)
						}
					}
				}

			runParticle = runParticle->nextP;  // and loop on
			}
		//---------------------------------------
		// talk a bit to the user
		//---------------------------------------
		//cout << preRunParticle->open_springs << endl;
		//cout << preRunParticle->prob << endl;
		//cout << preRunParticle->area << endl;
		//cout << preRunParticle->eEl << endl;
		cout << preRunParticle->newsurfE << endl;
		//cout << preRunParticle->nb << endl;

		//---------------------------------------------------------------------------------
		// Now we went through all the particles and calculated their probabilities to
		// dissolve in this case. Now we have to check if a particle is dissolving in the
		// given time - probability is larger than 1 and dissolve it and shrink all
		// other particles or shrink all particles according to the current time step
		// and their probabilities.
		//---------------------------------------------------------------------------------

		if (preRunParticle->prob > 1.0)				// fastest particle dissolves in time step given
			{
			//--------------------------------------
			// dissolve particle totally
			//--------------------------------------

			for (j = 0; j < 8 ; j++)
				{
				if (preRunParticle->neigP[j])  			// check if spring is active
					{
					preRunParticle->springf[j] = 0.0;				// spring is zero
					preRunParticle->isHole = true;					// I am a hole now
					preRunParticle->isHoleBoundary = false;			// I am not a boundary anymore
					preRunParticle->young = 0.0;					// repulsion is zero

					//----------------------------------------------------
					// now the grain number has to change, I am now
					// part of another grain -> part of hole
					// so far no new numbers for new holes -> should
					// follow someday
					//----------------------------------------------------

					if (preRunParticle->neigP[j]->isHole)      // first find the hole
						{
						preRunParticle->grain = preRunParticle->neigP[j]->grain;   // now I am part of that hole
						}
					}
				}
			//--------------------------------------------------------------------------------------
			// now shrink all the other particles (shrinking is virtual with particle->area,
			// does not affect the particle->radius that is used for the elastic model. Elastic
			// model only sees if particle is totally gone or not.
			// time step for shrinking is determined by the time needed to dissolve the previous
			// particle with the highest probability
			//---------------------------------------------------------------------------------------

			for (i = 0 ; i < numParticles ; i++)	// check all particles
				{
				if (runParticle->isHoleBoundary)	// If I am along a hole boundary -> can react/dissolve
					{
					//if (!runParticle->is_lattice_boundary)		// if I am not a lattice boundary
						{
						//------------------------------------------------
						// shrink the area
						//------------------------------------------------

						runParticle->area=runParticle->area - (runParticle->area*runParticle->prob/preRunParticle->prob);
						}
					}

				runParticle = runParticle->nextP;		// next particle
				}
			//---------------------------------------------------------
			// now the internal_time has to be changed,
			// subtract time needed to dissolve particle with
			// highest probability
			//---------------------------------------------------------

			internal_time = internal_time * (preRunParticle->prob-1.0)/preRunParticle->prob;

			//----------------------------------------------------------------
			// and call in the cleaner to get rid of junk
			// adjusts springs along boundaries of hole...
			//----------------------------------------------------------------

			Adjust_Hole_List();

			}

		//-----------------------------------------------------------------------
		// if we go in here that means that in the given time step no particle
		// could dissolve completely. now all particles shrink according to
		// given time step.
		//------------------------------------------------------------------------

		else
			{
			for (i = 0 ; i < numParticles ; i++)
				{
				if (runParticle->isHoleBoundary)	// I am a hole boundary
					{
					//if (!runParticle->is_lattice_boundary)  // I am not a lattic boundary
						{
						//-----------------------------------------------------------
						// shrink particle according to given time step, that
						// is identical to the probability, we used the time step
						// to determine that ....
						//-----------------------------------------------------------

						runParticle->area = runParticle->area - (runParticle->area * runParticle->prob);
						}
					}

				runParticle = runParticle->nextP;
				}
			internal_time = 0.0;  // now the internal time is zero, used it all up !

			}


		Relaxation();  // and do a relaxation (because now a particle is gone...)

		//---------------------------------------------------------------------
		// this is an internal cleaner that gets rid of very small particles
		//---------------------------------------------------------------------

		for (i = 0 ; i < numParticles ; i++)
			{
			if (runParticle->isHoleBoundary)  // I am on a hole boundary
				{
				//if (!runParticle->is_lattice_boundary)   // I am not a lattice boundary
					{
					//--------------------------------------------------------------------
					// if my virtual area -> particle->area is very much smaller than my
					// original radius (this is then scale free...)
					// I will dissolve without changing the time...
					//--------------------------------------------------------------------

					if (runParticle->area < (runParticle->radius*runParticle->radius*the_size*the_size*0.00001))
						{
						for (j = 0; j < 8 ; j++)
							{
							if (runParticle->neigP[j])   // if spring is active
								{
								runParticle->springf[j] = 0.0;				// spring is zero
								runParticle->isHole = true;					// is hole
								runParticle->isHoleBoundary = false;		// no boundary anymore
								runParticle->young = 0.0;					// repulsion also zero

								if (runParticle->neigP[j]->isHole)
									{
									runParticle->grain = runParticle->neigP[j]->grain;  // and now part of the hole
									}
								}
							}

						dump_pict = dump_pict + 1;   // counter for pict action dump

						Adjust_Hole_List();   // Cleaner for the boundaries

						Relaxation();    // and relax

						}
					}
				}

			runParticle = runParticle->nextP;
			}


		dump_pict = dump_pict + 1;   // counter for pcit action dump

		cout << dump_pict << endl;

		if (dump > 0)    // if user wanted action picts
			{
			if (dump_pict >= dump)  // if time is up
				{
				UpdateElle();     // make a pict
				dump_pict = 0;    // and reset
				}
			}


		}
	while (internal_time > 0.0);   // as long as we can dissolve and have time
	}



/******************************************************
* Reaction for grain boundaries including the stress
*
* Daniel, September 2003
*
* Function shrinks particles 
*
* variable dump is for picture dumps after a certain
* amount of particles have dissolved specified by
* dump. This gives action pictures since the 
* processes can be highly non-linear
*******************************************************/

void Phase_Lattice::Dissolution_Strain_Stress(int dump)
	{
	int i,j,ii,jj;		// counters for loops
	int dump_pict;		// internal counter for picture dump
	int spring_count;	// number of open springs
	int box, pos;		// helper for position in rep box
	float pois;		// poisson number 
	float lame1;		// first lame constant
	float lame2;		// second lame constant
	float uxx;		// normal part of strain tensor in x
	float uyy;		// normal part of strain tensor in y
	float uxy;		// shear strain part of strain tensor in xy
	float dx,dy;		// displacement delta x and delta y 
	float alen, dd;		// equilibrium length and normalized length in a direction 
	float fn,fx,fy;		// forces normal, in x and in y 
	float uxi, uyi;		// normalized strain in x and y 
	float rate;		// rate of dissolution 
	float highest_rate;	// highest dissolution rate 
	float internal_time;	// internal time counter 
	float surfE;		// surface energy 
	float sum_Energy;	// sum of molar energies
	float thresh;		// threshold for reactions
	float bound_stress;	// boundary stress = normal stress on interface 
	float rep_constant;	// repulsion constant 
	bool new_grain;		// new grain or not 
	Particle *neig;		// help pointer 


	internal_time = time;
	dump_pict = 0;

	do
		{
		cout << internal_time << endl;

		highest_rate = 0.0;

		pois = 0.3333333;	// poisson ratio of triangular lattice 

		temp = 200;         	// for quartz dissolution
		rate_constant = 0.0000001;    // rate constant for quartz at 200C

		runParticle = &refParticle;     // for particle loops

		for (i = 0 ; i < numParticles ; i++)
			{
			if (runParticle->isHoleBoundary)      // if its a hole boundary
				{
				if (!runParticle->is_lattice_boundary)  // not lattice boundary
					{
					if (1)  //dummy
						{
						//----------------------------------------
						// zero out some values
						//----------------------------------------

						runParticle->open_springs = 0;
						runParticle->bound_stress = 0.0;
						runParticle->fluid_neig_count = 0;
						spring_count = 0;

						//------------------------------------------------
						// repulsion only works if we set flags
						// true for all particles that are connected
						// and the particle itself, otherwise
						// will count stresses twice or worse
						//------------------------------------------------

						runParticle->done = true;

						//--------------------------------------------------
						// now loop through possible connected neighbours
						//--------------------------------------------------

						for (j = 0; j < 8; j++)
							{
							//------------------------------------
							// if the neighbour exists
							//-----------------------------------

							if (runParticle->neigP[j])
								{
								runParticle->neigP[j]->done = true;    // set flag for repulsion

								spring_count = spring_count + 1;  // count number of active connections

								//------------------------------------------------------------
								// if the neighbour is of the same grain than myself
								//-----------------------------------------------------------

								if (runParticle->grain == runParticle->neigP[j]->grain)
									{

									//-------------------------------------------------
									// if the neighbour also sits on a Hole boundary
									// I just ran along the boundary and can memorize
									// this neighbours position for surface energy
									// and pressure on the boundary
									//-------------------------------------------------

									if (runParticle->neigP[j]->isHoleBoundary)
										{
										//---------------------------------------------------
										// do not include the lattice boundary particles,
										// they are fixed and have therefore not the right
										// stresses
										//---------------------------------------------------

										if (!runParticle->neigP[j]->is_lattice_boundary)
											{
											//------------------------------------------------------------
											// ok, now put this neighbour in the list so that we
											// can call it directly
											// and count the number of neighbours that we find along
											// the boundary.
											//------------------------------------------------------------

											runParticle->fluid_neigP[runParticle->fluid_neig_count] = runParticle->neigP[j];
											runParticle->fluid_neig_count = runParticle->fluid_neig_count + 1;
											}
										}

									}
								else
									{
									if (runParticle->neigP[j]->isHole)
										{
										//-----------------------------------------------------------
										// this is a boundary spring, add it to the number of springs
										//-----------------------------------------------------------

										runParticle->open_springs = runParticle->open_springs + 1;

										//----------------------------------------------------------------
										// and go through the whole monty of calculating the stresses
										// on that boundary spring (spring has only normal force in this
										// direction..
										//----------------------------------------------------------------

										// get distance in x and y directions

										dx = runParticle->neigP[j]->xpos - runParticle->xpos;
										dy = runParticle->neigP[j]->ypos - runParticle->ypos;

										// get distance parallel to spring and equilibrium distance (alen)

										dd = sqrt((dx*dx)+(dy*dy));
										alen = runParticle->radius + runParticle->neigP[j]->radius;

										// get unit length in x and y

										if (dd != 0.0)
											{
											uxi = dx/dd;
											uyi = dy/dd;
											}
										else
											{
											cout << " zero divide in Reaction2 " << endl;
											uxi = 0.0;
											uyi = 0.0;
											}

										//---------------------------------------------------------------
										// determine the force on particle = constant times strain
										//---------------------------------------------------------------

										fn = runParticle->springf[j] * (dd - alen);    // normal force
										fx = fn * uxi;                    // force x
										fy = fn * uyi;                    // force y

										fn = fx * uxi + fy * uyi;   // force in direction of spring

										fn = fn * 2.0 * runParticle->radius;    // scaling to particle size

										//-------------------------------------------------------------
										// scale normal force by particle area divided by six because
										// the open surface is now only for one spring i.e. 1/6 of
										// the particles circumference or area
										//-------------------------------------------------------------

										fn = fn /((runParticle->radius*runParticle->radius*Pi)/6.0);

										//-----------------------------------------------------------
										// and add that to the particles boundary stress
										//-----------------------------------------------------------

										runParticle->bound_stress = runParticle->bound_stress + fn;

										}
									}
								}
							}

						//-----------------------------------------------------------
						// check if particle has a broken spring
						// if yes look for repulsion stress and add broken spring
						// to the number of open springs.
						// only works for triangular lattice now with 6 springs
						//-----------------------------------------------------------

						if (1)
							{
							runParticle->open_springs = runParticle->open_springs + 6 - spring_count;

							//cout << "in" << endl;

							//----------------------------------------------------
							// now do the repulsion box check again
							//----------------------------------------------------


							for (ii=0;ii<3;ii++)
								{
								for (jj=0;jj<3;jj++)
									{
									//--------------------------------------------------
									// define box positions for different cases
									// box is square so you have to run through
									// 9 small boxes where the particle
									// is in the central box
									// box position is now a one dimensional array
									// where the number of the box in the array is
									// a function of x and y position of that box
									// in space. All particles are in this box so
									// that any not connected particles can be
									// found easily without looping around in the
									// box and we can check if they have a repulsive
									// force on our particle.
									// The repulsive force in this routine is important
									// on fractures where bonds are broken but particles
									// are pushed into each other. This pressure on the
									// open surface has to be included in the rate law
									//---------------------------------------------------

									if (jj==0) box = 2*particlex*(-1);   // lower row of boxes
									if (jj==1) box = 0;	          // middle row of boxes
									if (jj==2) box = 2*particlex;        // upper row of boxes

									pos = runParticle->box_pos+(ii-1)+box;  // define box position

									if (pos>0)
										{
										neig = 0;
										neig = repBox[pos];  // get particle at that position

										while (neig)
											{
											//-------------------------------------------------------
											// Done here means that the particle is either
											// the particle itself or its a connected neighbour
											// these cannot be used in this list because
											// they are either already used to calculate the
											// boundary stress or they are part of the internal
											// grain and not boundary particles.
											// Other particles that are further away but still
											// in the rep box dont matter, only compressive forces
											// are applied here
											//--------------------------------------------------------

											if (!neig->done)   // if not already done
												{
												if (neig->xpos)  // just in case
													{
													//--------------------------------------------------------------
													// get distance to neighbour
													//--------------------------------------------------------------

													dx = neig->xpos - runParticle->xpos;
													dy = neig->ypos - runParticle->ypos;

													dd = sqrt((dx*dx)+(dy*dy));

													//--------------------------------------------------------------
													// get equilibrium length
													//--------------------------------------------------------------

													alen = runParticle->radius + neig->radius;

													//--------------------------------------------------------------
													// determine the unitlengths in x and y
													//--------------------------------------------------------------

													if (dd != 0.0)
														{
														uxi = dx/dd;
														uyi = dy/dd;
														}
													else
														{
														cout << " zero divide in Relax " << endl;
														uxi = 0.0;
														uyi = 0.0;
														}

													//---------------------------------------------------------------
													// determine the force on particle = constant times strain
													//---------------------------------------------------------------

													//---------------------------------------------------------
													// first calculate the repulsion constant by using an
													// average of the youngs moduli of the two particles
													//--------------------------------------------------------

													rep_constant = (runParticle->young + neig->young)/2.0;
													if (neig->young == 0.0) rep_constant = 0.0;

													// scale back to spring constant

													rep_constant = rep_constant * sqrt(3.0)/2.0;

													fn = rep_constant * (dd - alen);    // normal" force per length" (pressure times length)
													fx = fn * uxi;                    //" force x"
													fy = fn * uyi;                    //" force y"

													if (fn < 0.0)    // if compressive
														{

														fn = fx * uxi + fy * uyi;   // force in direction of spring

														fn = fn * 2.0 * runParticle->radius;    // scaling to particle size

														//-------------------------------------------------------------
														// scale normal force by particle area divided by six because
														// the open surface is now only for one spring i.e. 1/6 of
														// the particles circumference or area
														//-------------------------------------------------------------

														fn = fn /((runParticle->radius*runParticle->radius*Pi)/6.0);

														//-----------------------------------------------------------
														// and add that to the particles boundary stress
														//-----------------------------------------------------------

														runParticle->bound_stress = runParticle->bound_stress + fn;

														cout << "repulsion stress" << runParticle->bound_stress << endl;
														}
													}
												}
											if (neig->next_inBox)   // if there is another one in box
												{
												neig = neig->next_inBox;  // move pointer
												}
											else
												{
												break;
												}
											}
										}
									}
								}
							}

						//---------------------------------------------------
						// now turn off the flags
						//---------------------------------------------------

						runParticle->done = false;

						for (j = 0; j < 8 ; j++)
							{
							if (runParticle->neigP[j])
								{
								runParticle->neigP[j]->done = false;
								}
							}


						runParticle->surfE = runParticle->surf_free_E*2.0/(runParticle->radius*the_size);

						runParticle->surfE = runParticle->surfE*((runParticle->open_springs-2.0)/4.0);

						runParticle->surfE = runParticle->surfE * runParticle->mV;

						}
					}
				}
			runParticle = runParticle->nextP;
			}
		for (i = 0 ; i < numParticles ; i++)
			{
			if (runParticle->isHoleBoundary)
				{
				if (!runParticle->is_lattice_boundary)
					{
					if (1)
						{

						//----------------------------------------------------------------------
						// setting elastic constants
						//----------------------------------------------------------------------

						lame1 = runParticle->young * pois / (1.0-(pois*pois)*(1.0+pois));
						lame2 = runParticle->young / (2.0 + pois);

						//-----------------------------------------------------------------------
						// getting strain tensor
						//-----------------------------------------------------------------------

						uxy = ((1+pois)/runParticle->young)*runParticle->sxy;
						uxx = (1/runParticle->young)*(runParticle->sxx-(runParticle->syy*pois));
						uyy = (1/runParticle->young)*(runParticle->syy-(runParticle->sxx*pois));

						//-----------------------------------------------------------------------
						// calculate elastic energy for particle
						//-----------------------------------------------------------------------

						runParticle->eEl = 0.5 * lame1 * (uxx+uyy)*(uxx+uyy);
						runParticle->eEl = runParticle->eEl+(lame2*((uxx*uxx)+(uyy*uyy)+2*(uxy*uxy)));

						//---------------------------------------------------------
						// scaling energy
						//---------------------------------------------------------

						runParticle->eEl = runParticle->eEl * runParticle->mV * pressure_scale * pascal_scale;

						//---------------------------------------------------------
						// setting average surface energy
						//---------------------------------------------------------

						surfE = 0.0;

						if (runParticle->fluid_neig_count)
							{

							for (j = 0 ; j < runParticle->fluid_neig_count ; j++)
								{
								surfE = surfE + runParticle->fluid_neigP[j]->surfE;
								}

							surfE = surfE / runParticle->fluid_neig_count;
							runParticle->surfE = (runParticle->surfE + surfE)/2.0;
							}

						//-----------------------------------------------------------
						// scale pressure
						//-----------------------------------------------------------

						bound_stress = 0.0;

						if (runParticle->fluid_neig_count)
							{

							for (j = 0; j < runParticle->fluid_neig_count ; j++)
								{
								bound_stress = bound_stress + runParticle->fluid_neigP[j]->bound_stress;
								}

							bound_stress = bound_stress / runParticle->fluid_neig_count;

							//runParticle->bound_stress = (runParticle->bound_stress + bound_stress)/2.0;
							}

						runParticle->bound_stress = -1.0 * runParticle->bound_stress;

						runParticle->bound_stress = runParticle->bound_stress * pressure_scale * pascal_scale * runParticle->mV;

						//--------------------------------------------------------------------
						// summation of Energy
						//--------------------------------------------------------------------

						sum_Energy = runParticle->eEl + runParticle->surfE + runParticle->bound_stress;



						//-----------------------------------------------------
						// rate law
						//-----------------------------------------------------

						rate = rate_constant*(1-(exp(0.002)/exp((sum_Energy)/(gas_const*(temp+273)))));

						//cout << rate << endl;

						rate = rate * internal_time * runParticle->mV;

						//----------------------------------------------------
						// get probability for particle to react
						//----------------------------------------------------


						runParticle->prob = rate * (runParticle->open_springs/6.0) / sqrt(runParticle->area/Pi);

                                                //------------------------------------------------------
                                                // determine particle that dissolves fastest 
                                                //------------------------------------------------------

						if (runParticle->prob > highest_rate)
							{
							highest_rate = runParticle->prob;
							preRunParticle = runParticle;
							}
						}
					}
				}

			runParticle = runParticle->nextP;
			}
                        
                // and talk a bit 
                
		cout << preRunParticle->open_springs << endl;
		cout << preRunParticle->prob << endl;
		cout << preRunParticle->bound_stress << endl;
		cout << preRunParticle->eEl << endl;
		cout << preRunParticle->surfE << endl;

		if (preRunParticle->prob > 1.0) // particle dissolves completely 
			{

			for (j = 0; j < 8 ; j++)
				{
				if (preRunParticle->neigP[j])   // kill connections and dissolve 
					{
					preRunParticle->springf[j] = 0.0;
					preRunParticle->young = 0.0;
					preRunParticle->isHole = true;
					preRunParticle->isHoleBoundary = false;

					if (preRunParticle->neigP[j]->isHole)
						{
						preRunParticle->grain = preRunParticle->neigP[j]->grain;
						}
					}
				}


			for (i = 0 ; i < numParticles ; i++)  // and shrink the rest according to time used to dissolve fastest particle 
				{
				if (runParticle->isHoleBoundary)
					{
					if (!runParticle->is_lattice_boundary)
						{
                                                //------------------------------------------------------------------------
                                                // and shrink, area is virtual only for dissolution not for elastic part 
                                                //------------------------------------------------------------------------
						runParticle->area=runParticle->area - (runParticle->area*runParticle->prob/preRunParticle->prob);
						}
					}

				runParticle = runParticle->nextP;
				}
                        //------------------------------------
                        // adjust internal time 
                        //------------------------------------

			internal_time = internal_time * (preRunParticle->prob-1.0)/preRunParticle->prob;

			Adjust_Hole_List();
			}
		else  // in this case all particles shrink according to the given time, none dissolves completely 
			{
			for (i = 0 ; i < numParticles ; i++)
				{
				if (runParticle->isHoleBoundary)
					{
					if (!runParticle->is_lattice_boundary)
						{
						runParticle->area = runParticle->area - (runParticle->area * runParticle->prob);

						}
					}

				runParticle = runParticle->nextP;
				}
			internal_time = 0.0;

			}

		Relaxation();


		dump_pict = dump_pict + 1;

		cout << dump_pict << endl;

		if (dump > 0)
			{
			if (dump_pict >= dump)
				{
				UpdateElle();
				dump_pict = 0;
				}
			}

		}
	while (internal_time > 0.0);
    }


/******************************************************
* Reaction for grain boundaries including the stress
*
* includes reactions at grain boundaries, not only around holes
*
* includes long range surface energy effects
*
* input parameter dump determines after how many dissolved
* particles a picture is taken. 
* input parameter stress determines whether or not the 
* normal stress on the surface is the only driving
* force or if surface and elastic energy plus the normal 
* stress are the driving forces. 0 = all three, other 
* number is only the normal stress (or traction)
* shrink = 0 means one particle dissolves at a time
* and the others remain the same
* shrink = 1 means that the others shrink according
* to their dissolution rate and the time given 
* by the fastest particle
*  
* Daniel, September 2003
*******************************************************/

void Phase_Lattice::Dissolution_Stylos(int dump, int stress, int shrink)
{
    int i,j,ii,jj;	// counters for loops
    int dump_pict;	// counter for pict dumps
    int spring_count;	// spring counter
    int box, pos;	// box positions for repulsion box
    float pois;		// poison ratio
    float lame1;	// lame constant one 
    float lame2;	// second lame constant
    float uxx;		// strain tensor xx
    float uyy;		// strain tensor yy
    float uxy;		// strain tensor xy
    float dx,dy;	// lengths dx and dy
    float alen, dd;	// equilibrium length and unit length 
    float fn,fx,fy;	// forces, normal and in x and y direction
    float uxi, uyi;	// unit length x and y 
    float rate;		// reaction rate
    float highest_rate;		// highest reaction rate (fastest)
    float internal_time;	// internal time steps
    float surfE;		// surface energy counter 
    float sum_Energy;		// sum of all energies 
    float bound_stress;		// normal stress on the boundary 
    float rep_constant;		// repulsion constant 
    bool new_grain;		// created a new grain (or hole) 
    Particle *neig;		// help pointer for neighbours
    float surface_right[10];	// help array for surface energy
    float surface_left[10];	// help array for surface energy
    Particle *help;		// help pointer


    internal_time = time; 	// set internal time
    dump_pict = 0;		// initialize particle counter for picture dump

    do{
        cout << internal_time << " time left" << endl;

        highest_rate = 0.0;	// initialize highest rate

        pois = 0.3333333;	// fixed poisson number

        temp = 200;         	// for quartz dissolution
        rate_constant = 0.0000001;    // rate constant for quartz at 200ÁC

        runParticle = &refParticle;     // for particle loops

        for (i = 0 ; i < numParticles ; i++)
        {
            if (runParticle->isHoleBoundary)      // if its a hole boundary
            {
                        //----------------------------------------
			// zero out some values
			//----------------------------------------

                    runParticle->open_springs = 0;
                    runParticle->bound_stress = 0.0;
                    runParticle->fluid_neig_count = 0;
                    spring_count = 0;

			//------------------------------------------------
			// repulsion only works if we set flags
			// true for all particles that are connected
			// and the particle itself, otherwise
			// will count stresses twice or worse
			//------------------------------------------------

                    runParticle->done = true;

			//--------------------------------------------------
			// now loop through possible connected neighbours
			//--------------------------------------------------

			for (j = 0; j < 8; j++)
			{
			    //------------------------------------
			    // if the neighbour exists
			    //-----------------------------------

			    if (runParticle->neigP[j])
			    {
				runParticle->neigP[j]->done = true;    // set flag for repulsion

                                    //-------------------------------------------------
				    // if the neighbour also sits on a Hole boundary
				    // I just ran along the boundary and can memorize
				    // this neighbours position for surface energy
				    // and pressure on the boundary
				    //-------------------------------------------------

				    if (runParticle->neigP[j]->isHoleBoundary)
				    {					
					
					    //------------------------------------------------------------
					    // ok, now put this neighbour in the list so that we
					    // can call it directly
					    // and count the number of neighbours that we find along
					    // the boundary.
					    //------------------------------------------------------------
              
                                        runParticle->fluid_neigP[runParticle->fluid_neig_count] = runParticle->neigP[j];
                                        runParticle->fluid_neig_count = runParticle->fluid_neig_count + 1;
                                    }

                                    //---------------------------------------------------
                                    //	count number of open springs for surface energy
                                    //---------------------------------------------------
                                
                                    if (runParticle->neigP[j]->isHole)
                                        runParticle->open_springs = runParticle->open_springs + 1;
                               
			    }
			}
			
                        //----------------------------------------------------
                        // now do the repulsion box check again
                        //----------------------------------------------------

			    for (ii=0;ii<3;ii++)
			    {
				for (jj=0;jj<3;jj++)
				{
				    //--------------------------------------------------
				    // define box positions for different cases
				    // box is square so you have to run through
				    // 9 small boxes where the particle
				    // is in the central box
				    // box position is now a one dimensional array
				    // where the number of the box in the array is
				    // a function of x and y position of that box
				    // in space. All particles are in this box so
				    // that any not connected particles can be
				    // found easily without looping around in the
				    // box and we can check if they have a repulsive
				    // force on our particle.
				    // The repulsive force in this routine is important
				    // on fractures where bonds are broken but particles
				    // are pushed into each other. This pressure on the
				    // open surface has to be included in the rate law
				    //---------------------------------------------------

				    if (jj==0) box = 2*particlex*(-1);   // lower row of boxes
				    if (jj==1) box = 0;	          // middle row of boxes
				    if (jj==2) box = 2*particlex;        // upper row of boxes

				    pos = runParticle->box_pos+(ii-1)+box;  // define box position

				    if (pos>0)
				    {
					neig = 0;
					neig = repBox[pos];  // get particle at that position

					while (neig)
					{
					    //-------------------------------------------------------
					    // Done here means that the particle is either
					    // the particle itself or its a connected neighbour
					    // these cannot be used in this list because
					    // they are either already used to calculate the
					    // boundary stress or they are part of the internal
					    // grain and not boundary particles.
					    // Other particles that are further away but still
					    // in the rep box dont matter, only compressive forces
					    // are applied here
					    //--------------------------------------------------------

					    if (!neig->done)   // if not already done
					    {
						if (neig->xpos)  // just in case
						{
						    //--------------------------------------------------------------
						    // get distance to neighbour
						    //--------------------------------------------------------------

						    dx = neig->xpos - runParticle->xpos;
						    dy = neig->ypos - runParticle->ypos;

						    dd = sqrt((dx*dx)+(dy*dy));

						    //--------------------------------------------------------------
						    // get equilibrium length
						    //--------------------------------------------------------------

						    alen = runParticle->radius + neig->radius;

						    //--------------------------------------------------------------
						    // determine the unitlengths in x and y
						    //--------------------------------------------------------------

						    if (dd != 0.0)
						    {
							uxi = dx/dd;
							uyi = dy/dd;
						    }
						    else
						    {
							cout << " zero divide in Relax " << endl;
							uxi = 0.0;
							uyi = 0.0;
						    }

						    //---------------------------------------------------------------
						    // determine the force on particle = constant times strain
						    //---------------------------------------------------------------

						    //---------------------------------------------------------
						    // first calculate the repulsion constant by using an
						    // average of the youngs moduli of the two particles
						    //--------------------------------------------------------

						    rep_constant = (runParticle->young + neig->young)/2.0;
            
						    if (neig->young == 0.0) rep_constant = 0.0;

						    // scale back to spring constant

						    rep_constant = rep_constant * sqrt(3.0)/2.0;

						    fn = rep_constant * (dd - alen);    // normal" force per length" (pressure times length)
						    fx = fn * uxi;                    //" force x"
						    fy = fn * uyi;                    //" force y"

						    if (fn < 0.0)    // if compressive
						    {

							fn = fx * uxi + fy * uyi;   // force in direction of spring

							fn = fn * 2.0 * runParticle->radius;    // scaling to particle size

							//-------------------------------------------------------------
							// scale normal force by particle area divided by six because
							// the open surface is now only for one spring i.e. 1/6 of
							// the particles circumference or area
							//-------------------------------------------------------------

							fn = fn /((runParticle->radius*runParticle->radius*Pi)/6.0);

							//-----------------------------------------------------------
							// and add that to the particles boundary stress
							//-----------------------------------------------------------

							runParticle->bound_stress = runParticle->bound_stress + fn;

							//cout << "repulsion stress" << runParticle->bound_stress << endl;
						    }
						}
					    }
					    if (neig->next_inBox)   // if there is another one in box
					    {
						neig = neig->next_inBox;  // move pointer
					    }
					    else
					    {
						break;
					    }
					}
				    }
				}
			    }
			

			//---------------------------------------------------
			// now turn off the flags
			//---------------------------------------------------

			runParticle->done = false;

			for (j = 0; j < 8 ; j++)
			{
			    if (runParticle->neigP[j])
			    {
				runParticle->neigP[j]->done = false;
			    }
			}
                        //-------------------------------------------------------------------------
                        // determine surface energy for a single particle using the number of 
                        // open springs, scaled by particle radius and the box size
                        //-------------------------------------------------------------------------

			runParticle->surfE = runParticle->surf_free_E*2.0/(runParticle->radius*the_size);

                        //-------------------------------------------------------------------------
                        //  two open springs is defined to be a straight surface so it has 
                        // a curvature of zero
                        //-------------------------------------------------------------------------

			runParticle->surfE = runParticle->surfE*((runParticle->open_springs-2.0)/4.0);

			runParticle->surfE = runParticle->surfE * runParticle->mV;
			
			if (runParticle->is_lattice_boundary)
				runParticle->surfE = 0.0;  // lattice boundary particles are critical 
 
		    
                
            }
            runParticle = runParticle->nextP;
        }
        for (i = 0 ; i < numParticles ; i++)	// second loop through all particles
        {
            if (runParticle->isHoleBoundary)
            {
               
                        //----------------------------------------------------------------------
                        // setting elastic constants
                        //----------------------------------------------------------------------

                        lame1 = runParticle->young * pois / (1.0-(2.0*pois)*(1.0+pois));
                        lame2 = runParticle->young / (2.0 + 2.0*pois);

                        //-----------------------------------------------------------------------
                        // getting strain tensor
                        //-----------------------------------------------------------------------

                        uxy = ((1+pois)/runParticle->young)*runParticle->sxy;
                        uxx = (1/runParticle->young)*(runParticle->sxx-(runParticle->syy*pois));
                        uyy = (1/runParticle->young)*(runParticle->syy-(runParticle->sxx*pois));

                        //-----------------------------------------------------------------------
                        // calculate elastic energy for particle
                        //-----------------------------------------------------------------------

                        runParticle->eEl = 0.5 * lame1 * (uxx+uyy)*(uxx+uyy);
                        runParticle->eEl = runParticle->eEl+(lame2*((uxx*uxx)+(uyy*uyy)+2*(uxy*uxy)));

                        //---------------------------------------------------------
                        // scaling energy
                        //---------------------------------------------------------

                        runParticle->eEl = runParticle->eEl * runParticle->mV * pressure_scale * pascal_scale;

                        //---------------------------------------------------------
                        // setting average surface energy
                        //---------------------------------------------------------

                        for (j = 0; j < 10; j++)
                        {
                            surface_left[j] = 0.0;
                            surface_right[j] = 0.0;
                        }
						
                        if (runParticle->fluid_neig_count)
                        {							
							
                           
                            surfE = 0.0;
                            
                            //-------------------------------------------
                            // first run along surface to the left 
                            // and write the surface energies of 
                            // the single particles along this interface
                            // in the array surface_left
                            //-------------------------------------------
                            
                            if (runParticle->leftNeighbour)  // if I have a left neighbour 
                            {
                                help = runParticle->leftNeighbour;  // set help pointer 
							
                                for (j = 0; j<10; j++)  // do that for 10 neighbours 
                                {
                                    if (help->leftNeighbour)   // if there is somebody on the left 
                                    {	
                                        surface_left[j] = help->surfE;  // write surface energy of that particle 
                                        help = help->leftNeighbour;     // set help pointer 
                                    }
                                    else
                                        surface_left[j] = 0.0;   // no particle means zero surface energy = no influence 
                                }
                            }
                            
                            //-------------------------------------------------------
                            // do the same thing for the neighbours towards the right 
                            //--------------------------------------------------------
                            
                            if (runParticle->rightNeighbour)
                            {
                                help = runParticle->rightNeighbour;
                                for (j = 0; j<10; j++)
                                {
                                    if (help->rightNeighbour)
                                    {
                                        surface_right[j] = help->surfE;
                                        help = help->rightNeighbour;
                                    }
                                    else
                                        surface_right[j] = 0.0;
                                }
                            }
                            
                            //------------------------------------------------------------
                            // now we add all these surface energies up
                            // 
                            // this is done by taking the average energy of 
                            // the particle itself and the particle plus succesive next 
                            // neighbours, first the first one right and left plus the 
                            // particle, then the next two in each direction plus the 
                            // particle etc. for twenty one particles. The the surface
                            // energies are scaled according to distance of furthest
                            // neighbours so the average of three particles is divided
                            // by two, the average for five particles by four etc. 
                            // Therefore surface energies scale with 1/curvature radius
                            //-------------------------------------------------------------
                            
                            surfE = runParticle->surfE;  // particle itself
                            
                            for (j = 0; j < 10; j++)  // next ten
                            {
                                surfE = surfE + surface_left[j] + surface_right[j];  // add all up 
                                surface_left[j] = surfE;  // and put them in surface_left array 
                            }
                            surfE = 0.0;
                            for (j = 0; j < 10; j++)  // and do mean and scaling 
                            {
                                surfE = surfE + (surface_left[j]/((((j+1)*2)+1)*(((j+1)*2)+1)));
                            }
							
                            runParticle->newsurfE = runParticle->surfE + surfE; // and add particle itself
                        }
			else
                            runParticle->newsurfE = 0.0;

                        //-----------------------------------------------------------
                        // scale pressure
                        //-----------------------------------------------------------

                        runParticle->bound_stress = -1.0 * runParticle->bound_stress;

                        runParticle->bound_stress = runParticle->bound_stress * pressure_scale * pascal_scale * runParticle->mV;

                        //--------------------------------------------------------------------
                        // summation of Energy
                        //--------------------------------------------------------------------
                        
                        if (stress==0)
                            sum_Energy = runParticle->eEl + runParticle->newsurfE + runParticle->bound_stress;
                        else
                            sum_Energy = runParticle->bound_stress;

                        //-----------------------------------------------------
                        // rate law
                        //-----------------------------------------------------

                        rate = rate_constant*(1.0-(1.0/exp((sum_Energy)/(gas_const*(temp+273.0)))));

                        rate = rate * internal_time * runParticle->mV * runParticle->rate_factor;
                        
                             
                        //----------------------------------------------------
                        // get probability for particle to react
                        //----------------------------------------------------

                        runParticle->prob = rate / sqrt(runParticle->area/Pi);
                        
                        if (runParticle->prob > highest_rate)
                        {
                            highest_rate = runParticle->prob; // particle that dissolves fastest
                            preRunParticle = runParticle;
                        }
                    }

            runParticle = runParticle->nextP;
        }
        cout << preRunParticle->open_springs << " open springs" <<  endl;
        cout << preRunParticle->prob <<  " dissolution rate " << endl;
        cout << preRunParticle->bound_stress << " stress " << endl;
        cout << preRunParticle->eEl << " elastic energy " << endl;
        cout << preRunParticle->surfE << " surface energy " <<  endl;
        //cout << preRunParticle->area << endl;

        if (preRunParticle->prob > 1.0)		// particle dissolves completely
        {
            preRunParticle->young = 0.0;		// young modulus is zero 
            preRunParticle->isHole = true;		// its in the hole (fluid)
            preRunParticle->isHoleBoundary = false;	// not a solid boundary anymore
            
            for (j = 0; j < 8 ; j++)			// loop through connections
            {
                if (preRunParticle->neigP[j])		// if connection
                {
                    preRunParticle->springf[j] = 0.0;	// that spring constant is also zero                   
                    
                    if (preRunParticle->neigP[j]->isHole) // change number of grain
                    {
                        preRunParticle->grain = preRunParticle->neigP[j]->grain;
                    }
                }
            }

            if (shrink == 1) // shrink means shrink the other particles according to time of fastest particle
            {
                for (i = 0 ; i < numParticles ; i++) // loop all others
                {
                    if (runParticle->isHoleBoundary) // if reactive surface
                    {
                    
                        runParticle->area=runParticle->area - (runParticle->area*runParticle->prob/preRunParticle->prob);
                        if (runParticle->area <= 0.0)
                        cout << runParticle->area    << "negative ARea !"  << endl;
                    
                    }

                    runParticle = runParticle->nextP;
                }
            }

            //----------------------------------------------------------------------------------------
            // now subtract the time needed to dissolve the fastest particle form the internal time
            //----------------------------------------------------------------------------------------

            internal_time = internal_time * (preRunParticle->prob-1.0)/preRunParticle->prob;
            preRunParticle = preRunParticle->nextP;
            
            preRunParticle->prob = -1.0;	// just in case

		
	    Adjust_Hole_List(); // and clean up 
      
      
        }
        else // if no particle dissolves in the given time step
        {
          if (shrink) // if we shrink particles
          {
             for (i = 0 ; i < numParticles ; i++)
            {
                if (runParticle->isHoleBoundary)
                {
                    //------------------------------------------------------------------------------
                    // shrink particles according to the internal time step and their dissolution 
                    // rates 
                    //------------------------------------------------------------------------------
                    
                    runParticle->area = runParticle->area - (runParticle->area * runParticle->prob);                    
                }

                runParticle = runParticle->nextP;
            }
          }
          internal_time = 0.0;  // time is used up 

        }

        Relaxation();  // and relax (something changed in the model if a particle is dissolved ! ) 


        dump_pict = dump_pict + 1;  // dissolved on particle 

        cout << dump_pict << " particles dissolved in step " << endl;

        if (dump > 0)
        {
            if (dump_pict >= dump)
            {
                UpdateElle();  // draw a picture 
                dump_pict = 0;
            }
        }


    }while (internal_time > 0.0);  // use up all the time given by the external step 
}


/******************************************************
* Reaction for grain boundaries including the stress
*
* includes reactions at grain boundaries, not only around holes
*
* Daniel, September 2003
*******************************************************/

void Phase_Lattice::Dissolution_StylosII(int dump, int stress, int use_grains, int shrink, int sizex)
{
    int i,j,ii,jj;	// counters for loops
    int dump_pict;
    int spring_count, count;
    int box, pos;
    float pois;
    float lame1;
    float lame2;
    float uxx;
    float uyy;
    float uxy;
    float dx,dy;
    float alen, dd;
    float fn,fx,fy;
    float uxi, uyi;
    float rate;
    float highest_rate;
    float internal_time;
    float surfE;
    float sum_Energy;
    float thresh;
    float bound_stress;
    float rep_constant;
    bool new_grain;
    Particle *neig;
	float surface_right[10];
	float surface_left[10];
    Particle *help;
	float meansurfstress;


    internal_time = time;
    dump_pict = 0;

    do{
        cout << internal_time << endl;

        highest_rate = 0.0;

        pois = 0.3333333;

        temp = 200;         	// for quartz dissolution
        rate_constant = 0.0000001;    // rate constant for quartz at 200¡C
		
		count = 0;
		meansurfstress = 0;

        runParticle = &refParticle;     // for particle loops

        for (i = 0 ; i < numParticles ; i++)
        {
			
			//----------------------------------------
			// zero out some values
			//----------------------------------------

               runParticle->open_springs = 0;
                  runParticle->bound_stress = 0.0;
                  runParticle->fluid_neig_count = 0;
                  spring_count = 0;
            runParticle->increase_rate = false;
			runParticle->surfcount = 0;
            
            if (runParticle->isHoleBoundary)      // if its a hole boundary
            {
               
			

			//------------------------------------------------
			// repulsion only works if we set flags
			// true for all particles that are connected
			// and the particle itself, otherwise
			// will count stresses twice or worse
			//------------------------------------------------

                    runParticle->done = true;

			//--------------------------------------------------
			// now loop through possible connected neighbours
			//--------------------------------------------------

			for (j = 0; j < 8; j++)
			{
			    //------------------------------------
			    // if the neighbour exists
			    //-----------------------------------

			    if (runParticle->neigP[j])
			    {
					runParticle->neigP[j]->done = true;    // set flag for repulsion

				

				//------------------------------------------------------------
				// if the neighbour is of the same grain than myself
				//-----------------------------------------------------------


				    //-------------------------------------------------
				    // if the neighbour also sits on a Hole boundary
				    // I just ran along the boundary and can memorize
				    // this neighbours position for surface energy
				    // and pressure on the boundary
				    //-------------------------------------------------

						if (runParticle->neigP[j]->isHoleBoundary)
						{
					//---------------------------------------------------
					// do not include the lattice boundary particles,
					// they are fixed and have therefore not the right
					// stresses
					//---------------------------------------------------

					
					    //------------------------------------------------------------
					    // ok, now put this neighbour in the list so that we
					    // can call it directly
					    // and count the number of neighbours that we find along
					    // the boundary.
					    //------------------------------------------------------------
           
							runParticle->fluid_neigP[runParticle->fluid_neig_count] = runParticle->neigP[j];
							runParticle->fluid_neig_count = runParticle->fluid_neig_count + 1;
						}			
						if (runParticle->neigP[j]->isHole)
							  runParticle->open_springs = runParticle->open_springs + 1;
						if (!runParticle->neigP[j]->isHole)
						{
						  if (runParticle->is_boundary && use_grains != 0)
						  {
							if (runParticle->neigP[j]->grain != runParticle->grain || use_grains == 0)
							{
					//-----------------------------------------------------------
					// this is a boundary spring, add it to the number of springs
					//-----------------------------------------------------------

					//runParticle->open_springs = runParticle->open_springs + 1;

					//----------------------------------------------------------------
					// and go through the whole monty of calculating the stresses
					// on that boundary spring (spring has only normal force in this
					// direction..
					//----------------------------------------------------------------

					// get distance in x and y directions

								dx = runParticle->neigP[j]->xpos - runParticle->xpos;
								dy = runParticle->neigP[j]->ypos - runParticle->ypos;

					// get distance parallel to spring and equilibrium distance (alen)

								dd = sqrt((dx*dx)+(dy*dy));
								alen = runParticle->radius + runParticle->neigP[j]->radius;

					// get unit length in x and y

								if (dd != 0.0)
								{
									uxi = dx/dd;
									uyi = dy/dd;
								}
								else
								{
									cout << " zero divide in Reaction2 " << endl;
									uxi = 0.0;
									uyi = 0.0;
								}

					//---------------------------------------------------------------
					// determine the force on particle = constant times strain
					//---------------------------------------------------------------

								fn = runParticle->springf[j] * (dd - alen);    // normal force
								fx = fn * uxi;                    // force x
								fy = fn * uyi;                    // force y

								fn = fx * uxi + fy * uyi;   // force in direction of spring

								fn = fn * 2.0 * runParticle->radius;    // scaling to particle size

					//-------------------------------------------------------------
					// scale normal force by particle area divided by six because
					// the open surface is now only for one spring i.e. 1/6 of
					// the particles circumference or area
					//-------------------------------------------------------------

								fn = fn /((runParticle->radius*runParticle->radius*Pi)/6.0);

					//-----------------------------------------------------------
					// and add that to the particles boundary stress
					//-----------------------------------------------------------

					//runParticle->bound_stress = runParticle->bound_stress + fn;
						  }
						 }
						}
					}
			    }
			
			//-----------------------------------------------------------
			// check if particle has a broken spring
			// if yes look for repulsion stress and add broken spring
			// to the number of open springs.
			// only works for triangular lattice now with 6 springs
			//-----------------------------------------------------------

			    //----------------------------------------------------
			    // now do the repulsion box check again
			    //----------------------------------------------------


			    for (ii=0;ii<3;ii++)
			    {
					for (jj=0;jj<3;jj++)
					{
				    //--------------------------------------------------
				    // define box positions for different cases
				    // box is square so you have to run through
				    // 9 small boxes where the particle
				    // is in the central box
				    // box position is now a one dimensional array
				    // where the number of the box in the array is
				    // a function of x and y position of that box
				    // in space. All particles are in this box so
				    // that any not connected particles can be
				    // found easily without looping around in the
				    // box and we can check if they have a repulsive
				    // force on our particle.
				    // The repulsive force in this routine is important
				    // on fractures where bonds are broken but particles
				    // are pushed into each other. This pressure on the
				    // open surface has to be included in the rate law
				    //---------------------------------------------------

						if (jj==0) box = 2*particlex*(-1);   // lower row of boxes
						if (jj==1) box = 0;	          // middle row of boxes
						if (jj==2) box = 2*particlex;        // upper row of boxes

						pos = runParticle->box_pos+(ii-1)+box;  // define box position

						if (pos>0)
						{
							neig = 0;
							neig = repBox[pos];  // get particle at that position

							while (neig)
							{
					    //-------------------------------------------------------
					    // Done here means that the particle is either
					    // the particle itself or its a connected neighbour
					    // these cannot be used in this list because
					    // they are either already used to calculate the
					    // boundary stress or they are part of the internal
					    // grain and not boundary particles.
					    // Other particles that are further away but still
					    // in the rep box dont matter, only compressive forces
					    // are applied here
					    //--------------------------------------------------------

							if (!neig->done)   // if not already done
							{
								if (neig->xpos)  // just in case
								{
						    //--------------------------------------------------------------
						    // get distance to neighbour
						    //--------------------------------------------------------------

						    dx = neig->xpos - runParticle->xpos;
						    dy = neig->ypos - runParticle->ypos;

						    dd = sqrt((dx*dx)+(dy*dy));

						    //--------------------------------------------------------------
						    // get equilibrium length
						    //--------------------------------------------------------------

						    alen = runParticle->radius + neig->radius;

						    //--------------------------------------------------------------
						    // determine the unitlengths in x and y
						    //--------------------------------------------------------------

						    if (dd != 0.0)
						    {
							uxi = dx/dd;
							uyi = dy/dd;
						    }
						    else
						    {
							cout << " zero divide in Relax " << endl;
							uxi = 0.0;
							uyi = 0.0;
						    }

						    //---------------------------------------------------------------
						    // determine the force on particle = constant times strain
						    //---------------------------------------------------------------

						    //---------------------------------------------------------
						    // first calculate the repulsion constant by using an
						    // average of the youngs moduli of the two particles
						    //--------------------------------------------------------

						    rep_constant = (runParticle->young + neig->young)/2.0;
             
						    if (neig->young == 0.0) rep_constant = 0.0;

						    // scale back to spring constant

						    rep_constant = rep_constant * sqrt(3.0)/2.0;

						    fn = rep_constant * (dd - alen);    // normal" force per length" (pressure times length)
						    fx = fn * uxi;                    //" force x"
						    fy = fn * uyi;                    //" force y"

						    if (fn < 0.0)    // if compressive
						    {

							fn = fx * uxi + fy * uyi;   // force in direction of spring

							fn = fn * 2.0 * runParticle->radius;    // scaling to particle size

							//-------------------------------------------------------------
							// scale normal force by particle area divided by six because
							// the open surface is now only for one spring i.e. 1/6 of
							// the particles circumference or area
							//-------------------------------------------------------------

							fn = fn /((runParticle->radius*runParticle->radius*Pi)/6.0);

							//-----------------------------------------------------------
							// and add that to the particles boundary stress
							//-----------------------------------------------------------
								
								meansurfstress = meansurfstress + fn; 
								count = count + 1; 
								runParticle->surfcount = runParticle->surfcount + 1;

								runParticle->bound_stress = runParticle->bound_stress + fn;

								 runParticle->increase_rate = true;

								//cout << "repulsion stress" << runParticle->bound_stress << endl;
							
								}
							}
					    }
					    if (neig->next_inBox)   // if there is another one in box
					    {
							neig = neig->next_inBox;  // move pointer
					    }
					    else
					    {
							break;
					    }
					}
				    
				}
			   }
			}

			//---------------------------------------------------
			// now turn off the flags
			//---------------------------------------------------

			runParticle->done = false;

			for (j = 0; j < 8 ; j++)
			{
			    if (runParticle->neigP[j])
			    {
					runParticle->neigP[j]->done = false;
			    }
			}


			runParticle->surfE = runParticle->surf_free_E*2.0/(runParticle->radius*the_size);

			runParticle->surfE = runParticle->surfE*((runParticle->open_springs-2.0)/4.0);

			runParticle->surfE = runParticle->surfE * runParticle->mV;
			
			if (runParticle->is_lattice_boundary)
				runParticle->surfE = 0.0;

			}
            runParticle = runParticle->nextP;
        }
        for (i = 0 ; i < numParticles ; i++)
        {
            if (runParticle->isHoleBoundary)
            {
              

                        //----------------------------------------------------------------------
                        // setting elastic constants
                        //----------------------------------------------------------------------

                        lame1 = runParticle->young * pois / (1.0-(2.0*pois)*(1.0+pois));
                        lame2 = runParticle->young / (2.0 + 2.0*pois);

                        //-----------------------------------------------------------------------
                        // getting strain tensor
                        //-----------------------------------------------------------------------

                        uxy = ((1+pois)/runParticle->young)*runParticle->sxy;
                        uxx = (1/runParticle->young)*(runParticle->sxx-(runParticle->syy*pois));
                        uyy = (1/runParticle->young)*(runParticle->syy-(runParticle->sxx*pois));

                        //-----------------------------------------------------------------------
                        // calculate elastic energy for particle
                        //-----------------------------------------------------------------------

                        runParticle->eEl = 0.5 * lame1 * (uxx+uyy)*(uxx+uyy);
                        runParticle->eEl = runParticle->eEl+(lame2*((uxx*uxx)+(uyy*uyy)+2*(uxy*uxy)));

                        //---------------------------------------------------------
                        // scaling energy
                        //---------------------------------------------------------

                        runParticle->eEl = runParticle->eEl * runParticle->mV * pressure_scale * pascal_scale;

                        //---------------------------------------------------------
                        // setting average surface energy
                        //---------------------------------------------------------

                        for (j = 0; j < 10; j++)
						{
							surface_left[j] = 0.0;
							surface_right[j] = 0.0;
						}
						
                        if (runParticle->fluid_neig_count)
                        {							
							surfE = 0.0;
							if (runParticle->leftNeighbour)
							{
								help = runParticle->leftNeighbour;
							
								for (j = 0; j<10; j++)
								{
									if (help->leftNeighbour)
									{	
										surface_left[j] = help->surfE;
										help = help->leftNeighbour;
									}
									else
										surface_left[j] = 0.0;
								}
							}
							if (runParticle->rightNeighbour)
							{
								help = runParticle->rightNeighbour;
								for (j = 0; j<10; j++)
								{
									if (help->rightNeighbour)
									{
										surface_right[j] = help->surfE;
										help = help->rightNeighbour;
									}
									else
										surface_right[j] = 0.0;
								}
							}
							surfE = runParticle->surfE;
							for (j = 0; j < 10; j++)
							{
								surfE = surfE + surface_left[j] + surface_right[j];
								surface_left[j] = surfE;
							}
							surfE = 0.0;
							for (j = 0; j < 10; j++)
							{
								surfE = surfE + (surface_left[j]/((((j+1)*2)+1)*(((j+1)*2)+1)));
							}
							//cout << surfE << endl;
							runParticle->newsurfE = runParticle->surfE + surfE;

							//runParticle->surfE = (runParticle->surfE * (1-surface_area_factor)) + (surfE * surface_area_factor);
                        }
						else 
							runParticle->newsurfE = 0.0;
						

                        //-----------------------------------------------------------
                        // scale pressure
                        //-----------------------------------------------------------

                       /* bound_stress = 0.0;

                        if (runParticle->fluid_neig_count)
                        {

                            for (j = 0; j < runParticle->fluid_neig_count ; j++)
                            {
                                bound_stress = bound_stress + runParticle->fluid_neigP[j]->bound_stress;
                            }

                            bound_stress = bound_stress / runParticle->fluid_neig_count;

                            //runParticle->bound_stress = (runParticle->bound_stress + bound_stress)/2.0;
                        }*/

                        runParticle->bound_stress = -1.0 * runParticle->bound_stress;
						
						if (stress == 5) 
						{
							if (count != 0)
							{
							runParticle->bound_stress = runParticle->bound_stress + ((meansurfstress/count)*runParticle->surfcount);
								//cout << "mean stress" << meansurfstress/count << endl;
							}
							else
								runParticle->bound_stress = 0.0;
						}
						//cout << runParticle->bound_stress << endl;
                        runParticle->bound_stress = runParticle->bound_stress * pressure_scale * pascal_scale * runParticle->mV;
						//cout << runParticle->bound_stress << endl;
                        //--------------------------------------------------------------------
                        // summation of Energy
                        //--------------------------------------------------------------------
                        if (stress==0)
							sum_Energy = runParticle->eEl + runParticle->newsurfE + runParticle->bound_stress;
                        else if (stress == 1)
							sum_Energy = runParticle->bound_stress;
						else if (stress == 2)
							sum_Energy = runParticle->bound_stress + runParticle->eEl;
						else if (stress == 3)
						{
							if (runParticle->bound_stress > 0.0)
								sum_Energy = 1000000.0;
							else 
								sum_Energy = 0.0;
						}
						else if (stress == 4)
							sum_Energy = runParticle->eEl + runParticle->newsurfE;
						else if (stress == 5)
							sum_Energy = runParticle->eEl + runParticle->newsurfE + runParticle->bound_stress;



                        //-----------------------------------------------------
                        // rate law
                        //-----------------------------------------------------

                        rate = rate_constant*(1-(1/exp((sum_Energy)/(gas_const*(temp+273))))); // killed exp(0.002)

                        //cout << rate << endl;

                        rate = rate * internal_time * runParticle->mV * runParticle->rate_factor;

                       //if (runParticle->increase_rate)
                         // rate = rate * increase_rate_factor;

                        //----------------------------------------------------
                        // get probability for particle to react
                        //----------------------------------------------------

                        //if (use_grains == 0)
                         // runParticle->open_springs = 6.0;

                        //runParticle->prob = rate * (runParticle->open_springs/6.0) / sqrt(runParticle->area/Pi);
                        runParticle->prob = rate / sqrt(runParticle->area/Pi);
                        
                        if (runParticle->prob > highest_rate)
                        {
                            highest_rate = runParticle->prob;
                            preRunParticle = runParticle;
                        }
                    }
            

            runParticle = runParticle->nextP;
        }
      // cout << preRunParticle->open_springs << endl;
       //cout << preRunParticle->prob << endl;
       //cout << preRunParticle->bound_stress << endl;
       //cout << preRunParticle->eEl << endl;
      //cout << preRunParticle->newsurfE << endl;
       // cout << preRunParticle->area << endl;

        if (preRunParticle->prob > 1.0)
        {
            preRunParticle->young = 0.0;
            preRunParticle->isHole = true;
            preRunParticle->isHoleBoundary = false;

            
            for (j = 0; j < 8 ; j++)
            {
                if (preRunParticle->neigP[j])
                {
                    preRunParticle->springf[j] = 0.0;
                    
                    
                    if (preRunParticle->neigP[j]->isHole)
                    {
                        preRunParticle->grain = preRunParticle->neigP[j]->grain;
                    }
                }
            }
			ElleRemoveUnodeFromFlynn(preRunParticle->p_Unode->flynn(),preRunParticle->p_Unode->id());
			preRunParticle->ypos = -0.1;

            if (shrink == 1)
            {
            for (i = 0 ; i < numParticles ; i++)
            {
                if (runParticle->isHoleBoundary)
                {
                    //if (!runParticle->is_lattice_boundary)
                    {
                        //runParticle->area=runParticle->area - (runParticle->area*runParticle->prob/preRunParticle->prob);
                        if (runParticle->area <= 0.0)
                        cout << runParticle->area    << "negative ARea !"  << endl;
                    }
                }

                runParticle = runParticle->nextP;
            }
            }

            internal_time = internal_time * (preRunParticle->prob-(1.0/(particlex/sizex)))/preRunParticle->prob;
            preRunParticle = preRunParticle->nextP;
            preRunParticle->prob = -1.0;

		
	    Adjust_Hole_List();
      //Active_Grain_Boundaries();
      
        }
        else
        {
          if (shrink)
          {
             for (i = 0 ; i < numParticles ; i++)
            {
                if (runParticle->isHoleBoundary)
                {
                    //if (!runParticle->is_lattice_boundary)
                    {
                        runParticle->area = runParticle->area - (runParticle->area * runParticle->prob);

                    }
                }

                runParticle = runParticle->nextP;
            }
            }
            internal_time = 0.0;

        }

        Relaxation();


        dump_pict = dump_pict + 1;

        cout << dump_pict << endl;

        if (dump > 0)
        {
            if (dump_pict >= dump)
            {
                UpdateElle(false);
                dump_pict = 0;
            }
        }


    }while (internal_time > 0.0);
}


// GrowthDissolution 
/****************************************************************************
* Function for growth and dissolution of particles along a fluid-solid 
* interface. Needs concentration in the fluid. Only particles that are 
* in the fluid can precipitate so far 
* Particles cannot shrink or grow, they dissolve or grow totally ! 
* If a particle on the interface is supposed to grow a new particle
* of the fluid is reactivated and becomes a solid. 
*
* dump means take a picture after "dump" particles have dissolved. 
* int use_Box means use the repulsion box for the concentration if 1
* and the dissolved particles themself if 0
*
* Daniel October-December 2004
*****************************************************************************/
    
void Phase_Lattice::GrowthDissolution(int dump, int use_Box, int change, int experiment_time)
{
	

    int i,j,ii,jj;			// counters for loops
    int dump_pict;			// counter for picture dumps
    int spring_count;			// count springs
    int box, pos;			// for repulsion box positions
    int neig_count;			// count neighbours along interface
    int count;				// general counter
    float abs_prob;			// absolute probability to react 
    float pois;				// poison number 
    float lame1;			// first lame constant
    float lame2;			// second lame constant
    float uxx;				// strain tensor xx
    float uyy;				// strain tensor yy
    float uxy;				// strain tensor xy
    float dx,dy;			// distance in x and y 
    float alen, dd;			// equilibrium length and distance
    float fn,fx,fy;			// normal force and force in x and y direction
    float uxi, uyi;			// normalized length in x and y 
    float rate;				// reaction rate 
    float highest_rate;			// highest reaction rate 
    float internal_time;		// internal time 
    float surfE;			// surface energy 
    float sum_Energy;			// all energies 
    float bound_stress;			// normal stress on boundary 
    float rep_constant;			// repulsion constant 
    float time_increment;		// smaller time increment 
    float internal_time_step;		// internal time step
    float conc_div;			// 
    bool react_full;			// flag to react full or in steps
    float react_factor;			// changes reaction constant 
    bool new_grain;			// flag for new grains (or holes)
    bool done;				// flag for done stress determination
    bool particle_dis;			// does particle dissolve 
    float surface_right[40];		// surface energy array
    float surface_left[40];		// second surface energy array 
    Particle *neig;			// help pointer for neighbours
    Particle *help;			// general help pointer 
    int fluidchange;
	bool found;
    
    fluidchange = change;

    internal_time = time; 	// set internal time = external time step given by time 
    dump_pict = 0;

    do{
        highest_rate = 0.0;		// rate to zero 

        pois = 0.3333333;		// fixed poisson ratio 

        //temp = 25;         		//room temperature for NaCl
        //rate_constant = 0.0008;		// rate_constant 
        
        temp = 200;
        rate_constant =  0.000001;

        runParticle = &refParticle;     // for particle loops
    	react_factor = 1.0; 		
        particle_dis = false;
		
        for (i = 0 ; i < numParticles ; i++)  // and loop first time
        {
            runParticle->prob = 0.0;
			
            if (runParticle->isHoleBoundary)      // if its a hole boundary
            {
           
			//----------------------------------------
			// zero out some values
			//----------------------------------------

			runParticle->open_springs = 0;
			runParticle->bound_stress = 0.0;
			runParticle->fluid_neig_count = 0;
			spring_count = 0;

			//------------------------------------------------
			// repulsion only works if we set flags
			// true for all particles that are connected
			// and the particle itself, otherwise
			// will count stresses twice or worse
			//------------------------------------------------

			runParticle->done = true;

			//--------------------------------------------------
			// now loop through possible connected neighbours
			//--------------------------------------------------

			for (j = 0; j < 8; j++)
			{
			    //------------------------------------
			    // if the neighbour exists
			    //-----------------------------------

			    if (runParticle->neigP[j])
			    {
					runParticle->neigP[j]->done = true;    // set flag for repulsion

					spring_count = spring_count + 1;  // count number of active connections

                                

				    //-------------------------------------------------
				    // if the neighbour also sits on a Hole boundary
				    // I just ran along the boundary and can memorize
				    // this neighbours position for surface energy
				    // and pressure on the boundary
				    //-------------------------------------------------

				    if (runParticle->neigP[j]->isHoleBoundary)
				    {
					//---------------------------------------------------
					// do not include the lattice boundary particles,
					// they are fixed and have therefore not the right
					// stresses
					//---------------------------------------------------

						if (!runParticle->neigP[j]->is_lattice_boundary)
						{	
					    //------------------------------------------------------------
					    // ok, now put this neighbour in the list so that we
					    // can call it directly
					    // and count the number of neighbours that we find along
					    // the boundary.
					    //------------------------------------------------------------

							runParticle->fluid_neigP[runParticle->fluid_neig_count] = runParticle->neigP[j];
							runParticle->fluid_neig_count = runParticle->fluid_neig_count + 1;
						}
				    }

				
				    if (runParticle->neigP[j]->isHole)
				    {
					//-----------------------------------------------------------
					// this is a boundary spring, add it to the number of springs
					//-----------------------------------------------------------

						runParticle->open_springs = runParticle->open_springs + 1;

                                        //--------------------------------------------------------------
                                        // normal stress on interface is proportional to fluid pressure 
                                        //--------------------------------------------------------------
					
						if (runParticle->fluid_P < 0.0)
                                            fn =  runParticle->fluid_P;

						runParticle->bound_stress = fn;  

				    }
				}
			}

			//-----------------------------------------------------------
			// do the repulsion box routine now. if two particles 
                        // on either side of the fluid are pushed into each 
                        // other they feel an additional normal stress component 
                        // in addition to the fluid pressure part. 
			//-----------------------------------------------------------

			    //----------------------------------------------------
			    // now do the repulsion box check again
			    //----------------------------------------------------


			    for (ii=0;ii<3;ii++)
			    {
				for (jj=0;jj<3;jj++)
				{
				    //--------------------------------------------------
				    // define box positions for different cases
				    // box is square so you have to run through
				    // 9 small boxes where the particle
				    // is in the central box
				    // box position is now a one dimensional array
				    // where the number of the box in the array is
				    // a function of x and y position of that box
				    // in space. All particles are in this box so
				    // that any not connected particles can be
				    // found easily without looping around in the
				    // box and we can check if they have a repulsive
				    // force on our particle.
				    // The repulsive force in this routine is important
				    // on fractures where bonds are broken but particles
				    // are pushed into each other. This pressure on the
				    // open surface has to be included in the rate law
				    //---------------------------------------------------

				    if (jj==0) box = 2*particlex*(-1);   // lower row of boxes
				    if (jj==1) box = 0;	          // middle row of boxes
				    if (jj==2) box = 2*particlex;        // upper row of boxes

				    pos = runParticle->box_pos+(ii-1)+box;  // define box position
	
				    if (pos>0)
				    {
					neig = 0;
					neig = repBox[pos];  // get particle at that position

					while (neig)
					{
					    //-------------------------------------------------------
					    // Done here means that the particle is either
					    // the particle itself or its a connected neighbour
					    // these cannot be used in this list because
					    // they are either already used to calculate the
					    // boundary stress or they are part of the internal
					    // grain and not boundary particles.
					    // Other particles that are further away but still
					    // in the rep box dont matter, only compressive forces
					    // are applied here
					    //--------------------------------------------------------

					    if (!neig->done)   // if not already done
					    {
						if (neig->xpos)  // just in case
						{
						    //--------------------------------------------------------------
						    // get distance to neighbour
						    //--------------------------------------------------------------

						    dx = neig->xpos - runParticle->xpos;
						    dy = neig->ypos - runParticle->ypos;

						    dd = sqrt((dx*dx)+(dy*dy));

						    //--------------------------------------------------------------
						    // get equilibrium length
						    //--------------------------------------------------------------

						    alen = runParticle->radius + neig->radius;

						    //--------------------------------------------------------------
						    // determine the unitlengths in x and y
						    //--------------------------------------------------------------

						    if (dd != 0.0)
						    {
							uxi = dx/dd;
							uyi = dy/dd;
						    }
						    else
						    {
							cout << " zero divide in Relax " << endl;
							uxi = 0.0;
							uyi = 0.0;
						    }

						    //---------------------------------------------------------------
						    // determine the force on particle = constant times strain
						    //---------------------------------------------------------------

						    //---------------------------------------------------------
						    // first calculate the repulsion constant by using an
						    // average of the youngs moduli of the two particles
						    //--------------------------------------------------------

						    rep_constant = (runParticle->young + neig->young)/2.0;
						    if (neig->young == 0.0) rep_constant = 0.0;

						    // scale back to spring constant

						    rep_constant = rep_constant * sqrt(3.0)/2.0;

						    fn = rep_constant * (dd - alen);    // normal" force per length" (pressure times length)
						    fx = fn * uxi;                    //" force x"
						    fy = fn * uyi;                    //" force y"
			
						    if (fn < 0.0)    // if compressive
						    {

							fn = fx * uxi + fy * uyi;   // force in direction of spring

							fn = fn * 2.0 * runParticle->radius;    // scaling to particle size

							//-------------------------------------------------------------
							// scale normal force by particle area divided by six because
							// the open surface is now only for one spring i.e. 1/6 of
							// the particles circumference or area
							//-------------------------------------------------------------

							fn = fn /((runParticle->radius*runParticle->radius*Pi)/6.0);

							//-----------------------------------------------------------
							// and add that to the particles boundary stress
							//-----------------------------------------------------------

							runParticle->bound_stress = runParticle->bound_stress + fn;

							cout << "repulsion stress" << runParticle->bound_stress << endl;
						    }				
						}
					    }
					    if (neig->next_inBox)   // if there is another one in box
					    {
						neig = neig->next_inBox;  // move pointer
					    }
					    else
					    {
						break;
					    }
					
				    }
				  }
			    }
			}

			//---------------------------------------------------
			// now turn off the flags
			//---------------------------------------------------

			runParticle->done = false;

			for (j = 0; j < 8 ; j++)
			{
			    if (runParticle->neigP[j])
			    {
				runParticle->neigP[j]->done = false;
			    }
			}

                        //--------------------------------------------------------------------------
                        // determine surface energy for a single particle 
                        // depends on amount of open springs to the fluid 
                        //--------------------------------------------------------------------------

			runParticle->surfE = runParticle->surf_free_E*2.0/(runParticle->radius*the_size);

			runParticle->surfE = runParticle->surfE*(((runParticle->open_springs)-2.0)/4.0);

			runParticle->newsurfE = runParticle->surfE * runParticle->mV;

		    }
          
            runParticle = runParticle->nextP;
        }

		
		for (i = 0; i < numParticles; i++)
			{
				if (runParticle->isHoleBoundary)
				{
					if(runParticle->fluid_neig_count > 1)
					{
						runParticle->surfE = (runParticle->newsurfE + ((runParticle->fluid_neigP[0]->newsurfE+runParticle->fluid_neigP[1]->newsurfE)*0.5))*0.5;
					}
					else if (runParticle->fluid_neig_count > 0)
					{
						runParticle->surfE = (runParticle->newsurfE + runParticle->fluid_neigP[0]->newsurfE)*0.5;
					}
				}
				
				runParticle = runParticle->nextP;
			}
    //--------------------------------------------------------------------------
    //	this second large loop adds all the energies, scales surface 
    //	energy and determines the particle with the highest driving
    //	force to dissolve or grow
    //--------------------------------------------------------------------------

        for (i = 0 ; i < numParticles ; i++)
        {
            if (runParticle->isHoleBoundary)
            {
                
                        //----------------------------------------------------------------------
                        // calculate the elastic energy first 
                        //----------------------------------------------------------------------
                        // setting elastic constants
                        //----------------------------------------------------------------------

                        lame1 = runParticle->young * pois / (1.0-(pois*pois)*(1.0+pois));
                        lame2 = runParticle->young / (2.0 + pois);

                        //-----------------------------------------------------------------------
                        // getting strain tensor
                        //-----------------------------------------------------------------------

                        uxy = ((1+pois)/runParticle->young)*runParticle->sxy;
                        uxx = (1/runParticle->young)*(runParticle->sxx-(runParticle->syy*pois));
                        uyy = (1/runParticle->young)*(runParticle->syy-(runParticle->sxx*pois));

                        //-----------------------------------------------------------------------
                        // calculate elastic energy for particle
                        //-----------------------------------------------------------------------

                        runParticle->eEl = 0.5 * lame1 * (uxx+uyy)*(uxx+uyy);
                        runParticle->eEl = runParticle->eEl+(lame2*((uxx*uxx)+(uyy*uyy)+2*(uxy*uxy)));

                        //---------------------------------------------------------
                        // scaling energy
                        //---------------------------------------------------------

                        runParticle->eEl = runParticle->eEl * runParticle->mV * pressure_scale * pascal_scale;

                        //---------------------------------------------------------
                        // setting average surface energy
                        //---------------------------------------------------------
                        // in this case the surface energy of ten particles 
                        // in each direction along the interface is averaged and 
                        // added to the particle s surface energy. The influence 
                        // of the surface curvature decays with 1/distance between
                        // particles that are furthest away. 
                        //---------------------------------------------------------


                        if (runParticle->fluid_neig_count) // if there are neighbours 
                        {
                                        
                            surfE = 0.0;
                            
                            //-------------------------------------------
                            // first run along surface to the left 
                            // and write the surface energies of 
                            // the single particles along this interface
                            // in the array surface_left
                            //-------------------------------------------
                            
                            if (runParticle->leftNeighbour)  // if I have a left neighbour 
                            {
                                help = runParticle->leftNeighbour;  // set help pointer 
							
                                for (j = 0; j<40; j++)  // do that for 10 neighbours 
                                {
                                    if (help->leftNeighbour)   // if there is somebody on the left 
                                    {	
                                        surface_left[j] = help->surfE;  // write surface energy of that particle 
                                        help = help->leftNeighbour;     // set help pointer 
                                    }
                                    else
                                        surface_left[j] = 0.0;   // no particle means zero surface energy = no influence 
                                }
                            }
                            
                            //-------------------------------------------------------
                            // do the same thing for the neighbours towards the right 
                            //--------------------------------------------------------
                            
                            if (runParticle->rightNeighbour)
                            {
                                help = runParticle->rightNeighbour;
                                for (j = 0; j<40; j++)
                                {
                                    if (help->rightNeighbour)
                                    {
                                        surface_right[j] = help->surfE;
                                        help = help->rightNeighbour;
                                    }
                                    else
                                        surface_right[j] = 0.0;
                                }
                            }
                            
                            //------------------------------------------------------------
                            // now we add all these surface energies up
                            // 
                            // this is done by taking the average energy of 
                            // the particle itself and the particle plus succesive next 
                            // neighbours, first the first one right and left plus the 
                            // particle, then the next two in each direction plus the 
                            // particle etc. for twenty one particles. The the surface
                            // energies are scaled according to distance of furthest
                            // neighbours so the average of three particles is divided
                            // by two, the average for five particles by four etc. 
                            // Therefore surface energies scale with 1/curvature radius
                            //-------------------------------------------------------------
                            
                            surfE = runParticle->surfE;  // particle itself
                            
                            for (j = 0; j < 40; j++)  // next ten
                            {
                                surfE = surfE + surface_left[j] + surface_right[j];  // add all up 
                                surface_left[j] = surfE;  // and put them in surface_left array 
                            }
                            surfE = 0.0;
                            for (j = 0; j < 40; j++)  // and do mean and scaling 
                            {
                                surfE = surfE + (surface_left[j]/((((j+1)*2)+1)*(((j+1)*2)+1)));
                            }
							
                            runParticle->newsurfE = runParticle->surfE + surfE; // and add particle itself

							
                        }
                        else 
                            runParticle->newsurfE = 0.0;
						

                        //-----------------------------------------------------------
                        // scale pressure
                        //-----------------------------------------------------------

                    
                        runParticle->bound_stress = -1.0 * runParticle->bound_stress;

                        runParticle->bound_stress = runParticle->bound_stress * pressure_scale * pascal_scale * runParticle->mV;

                        //--------------------------------------------------------------------
                        // summation of Energy
                        //--------------------------------------------------------------------

                        sum_Energy = runParticle->eEl + runParticle->newsurfE + runParticle->bound_stress;


                    if (use_Box == 0)  // if the box index is zero use the fluid particles for the concentration 
                    {
                        for (j = 0; j<8; j++)
                        {
                            if (runParticle->neigP[j])
                            {
                                if (runParticle->neigP[j]->isHole)
                                {
                                    runParticle->conc = runParticle->neigP[j]->conc; // get concentration from particle in fluid 
                                    break;
                                }
                            }
                        }
                    }
                    else  // use the repulsion box concentration 
                    {
                        runParticle->conc = 0.0;

                        //-----------------------------------------------
                        // use the repulsion box, find box for particle
                        // routine is looking for a box with concentration
                        // at the moment. Is ok if concentration is constant
                        // has to be changed if diffusion is applied  
                        //-----------------------------------------------

                        for (jj=0;jj<3;jj++)
                        {
                            for (j=0;j<3;j++)
                            {

                                //------------------------------------------
                                //  case  upper  or  lower  row (+-size)
                                //------------------------------------------

                                if (j==0) box = 2*particlex*(-1);
                                if (j==1) box = 0;
                                if (j==2) box = 2*particlex;

                                //-----------------------------------------
                                //  now  check  this  position
                                //  all  particles  in  this  position !!
								//  concBox is zero if there is a particle
								//  if it is a fluid it should have #
								//  a positive concentration
                                //-----------------------------------------

                                pos = runParticle->box_pos+(jj-1)+box;

                                if (concBox[pos])
                                {
                                    if (concBox[pos] > 0.0)
                                    {
                                        runParticle->conc = concBox[pos];
                                        break;
                                    }
                                }
                            }
                        }
					


                        //-----------------------------------------------------
                        // rate law
                        //
                        // basic equilibrium concentration is 5416 for 
                        // NaCl at room temperature 
                        //-----------------------------------------------------
                       // if (!runParticle->isHole)
                        {
                            rate = rate_constant*(1-((runParticle->conc/(5416))/exp((sum_Energy)/(gas_const*(temp+273)))));
                            if (rate < 0.0)
                            {
                                rate = -rate_constant* (1-((5416*exp(sum_Energy/(gas_const*(temp+273))))/runParticle->conc));
								
                            }
                        }
                		
                        rate = rate * internal_time * runParticle->mV * runParticle->rate_factor;

                        //----------------------------------------------------
                        // get probability for particle to react
                        //----------------------------------------------------
                       
                        if (rate > 0.0)
                        {
                            runParticle->prob = rate * (runParticle->open_springs/6.0) / sqrt(runParticle->area/Pi);
                        }
                        else
                        {
                            runParticle->prob = rate * 0.5 / sqrt(runParticle->area/Pi);
                        }
						              
                                                              
                      //  runParticle->prob = rate /sqrt(runParticle->area/Pi);


                        abs_prob = sqrt(runParticle->prob * runParticle->prob); // in order to get highest rate 
                        
                        //-----------------------------------------------------
                        // dis_time is a damping part for particles 
                        // to react, means particle can only react once 
                        // each dis_time 
                        // dis_time is set by function set dis_time. 
                        // otherwise discrete reaction produces too much 
                        // flickering 
                        //-----------------------------------------------------

                      //  if (runParticle->conc > 0.0)
                        {
                            if (runParticle->dis_time <= 0)
                            {
                                if (abs_prob > highest_rate)
                                {
                                    highest_rate = abs_prob;
                                    preRunParticle = runParticle;
                                }
                            }
                            else
                            {
                                runParticle->dis_time = runParticle->dis_time - 1;  // count down 
                            }
                           }
                          //  else
                            //{
                               // runParticle->dis_time = runParticle->dis_time - 1;  // count down 
                           // }

                    }
            
            }

            runParticle = runParticle->prevP;
        }
        
        //---------------------------------------------
        // talk a bit to the public 
        //---------------------------------------------
        
        cout << " open springs " << preRunParticle->open_springs << endl;
        cout <<  " probability " << preRunParticle->prob << endl;
        cout << " boundary stress " << preRunParticle->bound_stress << endl;
        cout << " elastic energy " << preRunParticle->eEl << endl;
        cout << " surface energy " << preRunParticle->newsurfE << endl;

        react_full = true;  // is not really used at the moment
		
        if (preRunParticle->prob > 1.0)  // dissolution
        {
			cout << "dissolution" << endl;
            if (react_full) // dummy at the moment
			
            {
                particle_dis = true;  // dissolve
                if (use_Box == 0)  // concentration from particles, not box 
                {
                    preRunParticle->conc = 1/preRunParticle->mV; // concentration at that particle
                }
                else // use box 
                {
                    if (concBox[preRunParticle->box_pos] == 0.0)  // if this box is not yet fluid 
                        old_fl_vol = old_fl_vol + 1;  // make it fluid without changing fluid volume

                    concBox[preRunParticle->box_pos] = 1/preRunParticle->mV; // set concentration of that box 
                }

                neig_count = 0;
            	for (j = 0; j < 8 ; j++) // go through neighbours 
            	{
                    if (preRunParticle->neigP[j])
                    {
                    	preRunParticle->springf[j] = 0.0;  // set spring to zero 
                        preRunParticle->young = 0.0;  // youngs modulus is zero 
                    	preRunParticle->isHole = true;  // is now part of hole (fluid)
                        preRunParticle->isHoleBoundary = false;  // no boundary anymore
                        preRunParticle->dis_time = overall_dis_time;  // damping value

                    	if (preRunParticle->neigP[j]->isHole)
                    	{
                           preRunParticle->grain = preRunParticle->neigP[j]->grain;

                    	}
                    }
            	}
            }
		
            //--------------------------------------------------------
            // adjust the internal time accordingly
            //--------------------------------------------------------

           internal_time = internal_time * (sqrt(preRunParticle->prob*preRunParticle->prob)-(1.0/react_factor))/sqrt(preRunParticle->prob*preRunParticle->prob);

            Adjust_Hole_List();  // and clean up 
        }
        
        else if (preRunParticle->prob<-1.0 )  // growth
        {
			cout << " in growth " << endl;

          
                particle_dis = true;
				found = false;
                for (j = 0;j<8;j++)
                {
                    if (preRunParticle->neigP[j])
                    {
                        if (preRunParticle->neigP[j]->isHole)
                        {
							cout << "found " << endl;
							found = true;
                            break;   // find one neighbour in the hole 
							
                        }
                    }
                }
			 if (found)
			 {
                preRunParticle->neigP[j]->isHole = false;  // and activate that neighour (grows)
			
                for (jj = 0; jj<8; jj++) // loop around the neighbours neighbours 
                {
                    if (preRunParticle->neigP[j]->neigP[jj])
                    {
                        if (preRunParticle->neigP[j]->neigP[jj]->young > 0.0)
                        {
                            //------------------------------------------------------------------------------
                            // now activate that particle, give it a youngs modulus and a spring constant
                            //------------------------------------------------------------------------------
                        
                            preRunParticle->neigP[j]->young = preRunParticle->neigP[j]->neigP[jj]->young;
                            preRunParticle->neigP[j]->springf[jj] = preRunParticle->neigP[j]->young;
                            
                            //-------------------------------------------------------------------------------
                            // and adjust the concentration 
                            //-------------------------------------------------------------------------------
                            
                            if (use_Box == 0)
                            {
                                preRunParticle->neigP[j]->conc = 1/preRunParticle->mV;
                            }
                            else // use the box for the concentration
                            {
                                if (concBox[preRunParticle->neigP[j]->box_pos] != 0.0)
                                    old_fl_vol = old_fl_vol - 1;
                                concBox[preRunParticle->neigP[j]->box_pos] = 0.0;  // this is now solid
                            }
							preRunParticle->neigP[j]->dis_time = overall_dis_time; // set damping value
							break;
                        }
					
                    }
				}
			
                
                if (use_Box == 0)  // dont use the box, use the particles for the concentration
                {
                    for (jj = 0; jj<8; jj++)
                    {
                        if (preRunParticle->neigP[j]->neigP[jj])
                        {
                            if (preRunParticle->neigP[j]->neigP[jj]->isHole) // find a neighbour in the fluid
                            {
                                // and subtract the concentration used to grow the particle
                            
                                preRunParticle->neigP[j]->neigP[jj]->conc = preRunParticle->neigP[j]->neigP[jj]->conc - (1/preRunParticle->neigP[j]->mV);
                                break;
                            }
                        }
					}
					
                    
                }
                else	// we use the box 
                {

                    //----------------------------------------------------------
                    // we now try to find a neighbouring box to the box position
                    // that just became part of the solid. 
                    // Once a neighbour is found the concentration is subtracted
                    // from the fluid that was needed to grow the particle 
                    //----------------------------------------------------------

                    done = false;

                    for (i=0;i<3;i++)
                    {
                        for (jj=0;jj<3;jj++)
                        {

                            //------------------------------------------
                            //  case  upper  or  lower  row (+-size)
                            //------------------------------------------

                            if (jj==0) box = 2*particlex*(-1);
                            if (jj==1) box = 0;
                            if (jj==2) box = 2*particlex;

                            //-----------------------------------------
                            //  now  check  this  position
                            //  all  particles  in  this  position !!
                            //-----------------------------------------

                            pos = preRunParticle->neigP[j]->box_pos+(i-1)+box;

                            if (concBox[pos])  // if this is active 
                            {
                                if (concBox[pos] > 0.0 && done == false)  // if it is fluid 
                                {
                                    // subtract the concentration 
                                    
                                    concBox[pos] = concBox[pos] + concBox[pos] - (1/preRunParticle->neigP[j]->mV);
                                    done = true;  // and we are done 
                                    break;
                                }
                            }

                        }
                   
					}
				

            }
			}
					else
						cout << "big problem" << endl;
		cout << "out growth" << endl;
            Adjust_Hole_List();  // and clean 
			preRunParticle->prob = 0.0;
        }
        else
        {
            internal_time = 0.0;  // nothing happend
        }
		
       		 
        preRunParticle->prob = 0.0; // just in case
	
        count_time = count_time + time - internal_time;
		
        if (particle_dis)
            Relaxation();   // only relax if something changed 
                        	

        if (use_Box == 0)
        {
            Adjust_Concentration(); // adjust concentrations of particles
        }
        else
        {
            Adjust_Conc_Box(fluidchange);  // adjust concentration of background box 
        }

        dump_pict = dump_pict + 1; 

        if (dump > 0)
        {
            if (dump_pict >= dump)
            {
                UpdateElle();  // give out a picture 
                dump_pict = 0;
				DumpStatisticSurface ((time-internal_time) + time*experiment_time);
            }
        }


    }while (internal_time > 0.0); // time is up 
}

// Set_Diffusion_Constant
/****************************************************************
* set a diffusion constant
****************************************************************/


void Phase_Lattice::Set_Diffusion_Constant(float dif_constant)
	{
	dif_const = dif_constant;
	}

// Set_Grav_Grad
/****************************************************************
* set a gravitiation gradient
****************************************************************/

void Phase_Lattice::Set_Grav_Grad(float read_grav_grad)
	{
	grav_grad = read_grav_grad;
	}

// Adjust_Concentration
/****************************************************************
* adjust concentration of particles in the fluid, just takes
* an average value for all particles, diffusion is thougth
* to be fast 
****************************************************************/        
        
void Phase_Lattice::Adjust_Concentration()
{
    int i;			// loop counter 
    int count;			// counter for average 
    float concentration;	// end concentration 
     FILE *stat;    		// outfile
    
    count = 0;
    concentration = 0.0;
    
    for (i = 0; i < numParticles ; i++)
    {
        if (runParticle->isHole)
        {
            concentration = concentration + runParticle->conc; // and count 
            count = count + 1;
        }
        runParticle = runParticle->nextP;
    }
    concentration = concentration / count;   // and average
    cout << "concentration" << concentration << endl; 
     for (i = 0; i < numParticles ; i++)
    {
        if (runParticle->isHole)
        {
            runParticle->conc = concentration;   // and apply concentration 
        }
        runParticle = runParticle->nextP;
    }
    stat=fopen("ConcStat.txt","a");    // open append file 
         
    fprintf(stat,";%f\n",concentration);  // and write concentration 
         
    fclose(stat);  // close file   
}

// Adjust_Conc_Box
/**************************************************************************
* function adjusts the concentration of the background box used 
* for the concentration of the fluid. 
* function also determines the fluid pressure if the fluid volume changed
* using the compressibility of water 
**************************************************************************/

void Phase_Lattice::Adjust_Conc_Box(int change)
{
    int i;			// loop counter 
    float average_conc;		// concentration average 
    Particle *help_P;		// help pointer 
    float help;			// help variable 
    float delta_p;		// pressure difference 
    FILE *stat;    		// file pointer 
    
    fluid_volume = 0;
    average_conc = 0.0;
    delta_p = 0.0;
    
    
    for (i = 0; i < max_y; i++)
    {
        if (concBox[i])
        {
            if (concBox[i] != 0.0)
            {
                //-------------------------------------------------------
                // run through the box and add up the concentrations 
                // and the number of boxes that are fluid 
                //-------------------------------------------------------
            
                average_conc = average_conc + concBox[i];
                fluid_volume = fluid_volume + 1;
            }
        }
    }
    average_conc = average_conc / fluid_volume;  // average concentration
    
    //-----------------------------------------------------------------
    // run again through the box and check if the fluid volume 
    // changed due to the relaxation and deformation routine
    //-----------------------------------------------------------------
    
    fluid_volume = 0;
    for (i = 0; i < max_y; i++)
    {
        if (concBox[i])
        {
            concBox[i] = 0.0;
            if (repBox[i])
            {
                if (repBox[i]->isHole) // if first particle in box is hole
                {
                    concBox[i] = average_conc;  // apply concentration
                    fluid_volume = fluid_volume + 1;  // this is fluid 
                }
                else
                {
                    help_P = repBox[i];  // loop through tail particles in box 
                    while (help_P->next_inBox)  // as long as there is a tail
                    {
                        if (help_P->next_inBox->isHole)  // is that fluid ? 
                        {
                            concBox[i] = average_conc;  // apply concentration
                            fluid_volume = fluid_volume + 1; // is fluid 
                            break;  // and exit 
                        }
                        help_P = help_P->next_inBox;  // keep running tail 
                    }
                }
            }
        }
    }

    cout << "oldfluid" << old_fl_vol << endl;
    cout << "fluidvolume" << fluid_volume << endl;
    cout << "time" << count_time << endl;
	cout << " average concentration  " << average_conc << endl;
    
    //--------------------------------------------------------------------------
    // calculate change in fluid pressure using the fluid volume, the change 
    // in fluid volume and the compressibility
    //--------------------------------------------------------------------------
    
    if (change == 1)
    {
    
    delta_p = float(((float)old_fl_vol - (float)fluid_volume)/(float)fluid_volume) ;
        cout << "delta_p" << delta_p << endl;
    help = (water_compres * pressure_scale * pascal_scale);
        cout << help << endl;
    delta_p = delta_p / help;
        cout << delta_p  << endl;
     
    //---------------------------------------------------
    // calculate new fluid pressure 
    //---------------------------------------------------
              
    fluid_pressure = fluid_pressure - delta_p;
        if (fluid_pressure > 0.0)
            fluid_pressure = -0.0001;
            
    }
            
    //--------------------------------------------
    // and apply to the particles on the boundary
    //--------------------------------------------
    
    for (i = 0; i < numParticles ; i++)
    {
        if (runParticle->isHoleBoundary & !runParticle->is_lattice_boundary)
        {
            runParticle->fluid_P = fluid_pressure;
        }
        runParticle = runParticle->nextP;
    }
    
    //---------------------------------------------
    // and write out statistic
    //---------------------------------------------
          
    stat=fopen("ConcStat.txt","a"); 

    fprintf(stat,"%f",count_time);
    fprintf(stat,";%f",fluid_pressure);
    fprintf(stat,";%f\n",average_conc);
         
    fclose(stat);  // close file 

     old_fl_vol = fluid_volume;

}

// Set_Dis_Time
/************************************************************
* sets a damping for the reaction of particles to avoid
* too much flickering due to the discrete changes
* int time particle will not dissolve once its precipitated
*
* Daniel 2004
************************************************************/

void Phase_Lattice::Set_Dis_Time(int time)
{
    overall_dis_time = time;
}

// Make_Concentration_Box
/***************************************************************
* function initializes the concentration box in the background
* so that the fluid is treated in another layer that is cubic
*
* input is the y max size of the box and the initial 
* concentration of the fluid in the box 
*
* Daniel 2004
***************************************************************/

void Phase_Lattice::Make_Concentration_Box(float ymax, float concentration)
{
    int i;
    Particle *help_P;
        
    box_conc = concentration;
    
    max_y = int(ymax * particlex * 2) * particlex;  // size of box 
    
    fluid_volume = 0;
    
    for (i = 0; i < max_y; i++) // run through box 
    {
        //--------------------------------------------------------
        // concBox is the concentration box 
        // repBox is the repulsion box that contains pointers 
        // to particles that are in the box. 
        // concBox only contains concentrations for that site
        //--------------------------------------------------------
        
        concBox[i] = 0.0;
        if (repBox[i])  // if there is someone in the Repulsion box 
        {
            if (repBox[i]->isHole)  // if that is a hole 
            {
                concBox[i] = box_conc;  // concbox is also hole->give concentration
                fluid_volume = fluid_volume + 1;  // and add to fluid volume
            }
            else  // if not go along tail 
            {
                help_P = repBox[i];  // helper for the loop 
                while (help_P->next_inBox)  // as long as there is somebody along the tail
                {
                    if (help_P->next_inBox->isHole)  // if that is a hole particle 
                    {
                        concBox[i] = box_conc;  // this is fluid 
                        fluid_volume = fluid_volume + 1;
                        break;
                    }
                    help_P = help_P->next_inBox;
                }
            }
        }
        
    }
    old_fl_vol = fluid_volume;
	cout << fluid_volume << endl;
}

// DumpStressRow
/*****************************************************************
* function does dump the stress tensor, x or y position of particle
* the mean stress, differential stress and elastic energy
* within the box between xmin,xmax, ymin and ymax. 
*
* xrow = 1 means plot x coordinate, xrow = 0 plot y coordinate
*
* daniel 2004
*****************************************************************/

void Phase_Lattice::DumpStressRow(double y_box_min,double y_box_max, double x_box_min,double x_box_max, int xrow)
{
    FILE *stat;       // file pointer
    int i,j,jj, p_counter; // counters
    float mean,differential; // mean stress, differential stress, pressure
    float m_sxx,m_syy,m_sxy;  // stress tensor
    float finite_strain; // finite strain within the box of interest
    float sxx,syy,sxy,smax,smin;  // stress tensor and eigenvalues
	float uxx,uyy,uxy,lame1,lame2,pois;
 
    // set counters to zero
          
    p_counter = 0;

    mean = 0.0;
    differential = 0.0;
    m_sxx = 0.0;
    m_syy = 0.0;
    m_sxy = 0.0;
	
    pois = 0.3333333333333333333;

    //finite_strain = strain * (def_time_u + def_time_p);   // calculate finite strain

   stat=fopen("RowStat.txt","a");  // open statistic output append file 
 
    runParticle = &refParticle;         // start

    for (i=0 ; i<numParticles ; i++)   // look which particles are in the box
    {
		if (runParticle->ypos > y_box_min)  // larger ymin
		{
			if (runParticle->ypos < y_box_max)  // smaller ymax
			{
				if (runParticle->xpos > x_box_min)  // larger xmin
				{
					if (runParticle->xpos < x_box_max)  // smaller xmax
					{
						if (!runParticle->is_lattice_boundary)
						{
							if (!runParticle->isHole)
							{
								if (!runParticle->isHoleBoundary)
								{
						
                        // set stress tensor

						sxx = runParticle->sxx;
						syy = runParticle->syy;
						sxy = runParticle->sxy;

						// eigenvalues

						smax = ((sxx+syy)/2.0) + sqrt(((sxx-syy)/2.0)*((sxx-syy)/2.0) + sxy*sxy);

						smin = ((sxx+syy)/2.0) - sqrt(((sxx-syy)/2.0)*((sxx-syy)/2.0) + sxy*sxy);
                        
                        // values added for particles 

						m_sxx = sxx;

						m_syy = syy;

						m_sxy = sxy;

						mean = (smax + smin)/2.0;
						
						if (runParticle->young != 0.0)
						{
	    //-------------------------------------------------------------------
                    // first calculate the two lame constants from the youngs modulus of 
                    // a particle and the poisson number of the model
                    //-------------------------------------------------------------------
            
                    lame1 = runParticle->young * pois / (1.0-(pois*pois)*(1.0+pois));
                    lame2 = runParticle->young / (2.0 + pois);
                    
                    //------------------------------------------------------------------------
                    // calculate the components of the infinitesimal strain tensor from the 
                    // components of the stress tensor for the particle
                    //------------------------------------------------------------------------
                    
                    uxy = ((1+pois)/runParticle->young)*runParticle->sxy;
                    uxx = (1/runParticle->young)*(runParticle->sxx-(runParticle->syy*pois));
                    uyy = (1/runParticle->young)*(runParticle->syy-(runParticle->sxx*pois));
                    
                    //------------------------------------------------------------------------
                    // and now calculate the elastic energy for a single particle using the 
                    // two lame constants and the components of the infinitesimal strain 
                    // tensor of each particle
                    //------------------------------------------------------------------------
                    
                    runParticle->eEl = 0.5 * lame1 * (uxx+uyy)*(uxx+uyy);
                    runParticle->eEl = runParticle->eEl+(lame2*((uxx*uxx)+(uyy*uyy)+2*(uxy*uxy)));
            
                    //-------------------------------------------------------------------------
                    // and multiply by the molecular volume plus a scaling parameter for the 
                    // stress. The stress scaling is set with the youngsmodulus of the material
                    //--------------------------------------------------------------------------
            
                    runParticle->eEl = runParticle->eEl * runParticle->mV * pressure_scale * pascal_scale;
                	}
					else 
						runParticle->eEl = 0.0;
				
						differential = (smax - smin);
						if (xrow == 1)
						 fprintf(stat,"%f",runParticle->xpos);
						else 
							fprintf(stat,"%f",runParticle->ypos);
						fprintf(stat,";%f",runParticle->eEl);
    					fprintf(stat,";%f",mean);
   						 fprintf(stat,";%f",m_sxx);
    					fprintf(stat,";%f",m_syy);
    					fprintf(stat,";%f",m_sxy);
   						 fprintf(stat,";%f\n",differential);
					}
				}
			}
					}
				}
			}
		}
		runParticle = runParticle->nextP;
    }
    
    

   
    // and dump the data 
          
  
    
    fclose(stat);  // close file
}   

// DumpYStressRow
/*****************************************************************
* function does dump the stress tensor, x or y position of particle
* the mean stress, differential stress and elastic energy
* within the box between xmin,xmax, ymin and ymax. 
*
* xrow = 1 means plot x coordinate, xrow = 0 plot y coordinate
*
* outfile has a different name than Dump Stress Row.. 
*
* daniel 2004
*****************************************************************/


void Phase_Lattice::DumpYStressRow(double y_box_min,double y_box_max, double x_box_min,double x_box_max, int xrow)
{
    FILE *stat;       // file pointer
    int i,j,jj, p_counter; // counters
    float mean,differential; // mean stress, differential stress, pressure
    float m_sxx,m_syy,m_sxy;  // stress tensor
    float finite_strain; // finite strain within the box of interest
    float sxx,syy,sxy,smax,smin;  // stress tensor and eigenvalues
	float uxx,uyy,uxy,lame1,lame2,pois;
 
    // set counters to zero
          
    p_counter = 0;

    mean = 0.0;
    differential = 0.0;
    m_sxx = 0.0;
    m_syy = 0.0;
    m_sxy = 0.0;
	
	pois = 0.3333333333333333333;

    //finite_strain = strain * (def_time_u + def_time_p);   // calculate finite strain

   stat=fopen("RowYStat.txt","a");  // open statistic output append file 
 
    runParticle = &refParticle;         // start

    for (i=0 ; i<numParticles ; i++)   // look which particles are in the box
    {
		if (runParticle->ypos > y_box_min)  // larger ymin
		{
			if (runParticle->ypos < y_box_max)  // smaller ymax
			{
				if (runParticle->xpos > x_box_min)  // larger xmin
				{
					if (runParticle->xpos < x_box_max)  // smaller xmax
					{
						if (!runParticle->is_lattice_boundary)
						{
							if (!runParticle->isHole)
							{
								//if (!runParticle->isHoleBoundary)
								//{
						
                        // set stress tensor

						sxx = runParticle->sxx;
						syy = runParticle->syy;
						sxy = runParticle->sxy;

						// eigenvalues

						smax = ((sxx+syy)/2.0) + sqrt(((sxx-syy)/2.0)*((sxx-syy)/2.0) + sxy*sxy);

						smin = ((sxx+syy)/2.0) - sqrt(((sxx-syy)/2.0)*((sxx-syy)/2.0) + sxy*sxy);
                        
                        // values added for particles 

						m_sxx = sxx;

						m_syy = syy;

						m_sxy = sxy;

						mean = (smax + smin)/2.0;
						
						if (runParticle->young != 0.0)
						{
	    //-------------------------------------------------------------------
                    // first calculate the two lame constants from the youngs modulus of 
                    // a particle and the poisson number of the model
                    //-------------------------------------------------------------------
            
                    lame1 = runParticle->young * pois / (1.0-(2.0*pois)*(1.0+pois));
                    lame2 = runParticle->young / (2.0 + 2.0*pois);
                    
                    //------------------------------------------------------------------------
                    // calculate the components of the infinitesimal strain tensor from the 
                    // components of the stress tensor for the particle
                    //------------------------------------------------------------------------
                    
                    uxy = ((1+pois)/runParticle->young)*runParticle->sxy;
                    uxx = (1/runParticle->young)*(runParticle->sxx-(runParticle->syy*pois));
                    uyy = (1/runParticle->young)*(runParticle->syy-(runParticle->sxx*pois));
                    
                    //------------------------------------------------------------------------
                    // and now calculate the elastic energy for a single particle using the 
                    // two lame constants and the components of the infinitesimal strain 
                    // tensor of each particle
                    //------------------------------------------------------------------------
                    
                    runParticle->eEl = 0.5 * lame1 * (uxx+uyy)*(uxx+uyy);
                    runParticle->eEl = runParticle->eEl+(lame2*((uxx*uxx)+(uyy*uyy)+2*(uxy*uxy)));
            
                    //-------------------------------------------------------------------------
                    // and multiply by the molecular volume plus a scaling parameter for the 
                    // stress. The stress scaling is set with the youngsmodulus of the material
                    //--------------------------------------------------------------------------
            
                    runParticle->eEl = runParticle->eEl * runParticle->mV * pressure_scale * pascal_scale;
                	}
					else 
						runParticle->eEl = 0.0;
				
						differential = (smax - smin);
						if (xrow == 1)
						 fprintf(stat,"%f",runParticle->xpos);
						else 
							fprintf(stat,"%f",runParticle->ypos);
						fprintf(stat,";%f",runParticle->eEl);
    					fprintf(stat,";%f",mean);
   						 fprintf(stat,";%f",m_sxx);
    					fprintf(stat,";%f",m_syy);
    					fprintf(stat,";%f",m_sxy);
   						 fprintf(stat,";%f\n",differential);
					
				}
			}
					}
				}
			}
		}
		runParticle = runParticle->nextP;
    }
    
    

   
    // and dump the data 
          
  
    
    fclose(stat);  // close file
}

void Phase_Lattice::SetFluidLatticeGasRandom(float percent)
{
	int i, j; 
	float ran_nb;
	
	srand(std::time(0));
	
	for (i = 0; i < numParticles; i++)
	{
		

		for (j = 0; j < 6; j ++)
		{
			if (runParticle->neigP[j])
			{
				ran_nb = rand() / (float) RAND_MAX;  // pic pseudorandom from 0 to 1.0 for grains
				
				if (ran_nb < percent)
				{
					runParticle->fluid_particle[j] = 1;
				}
			}
		}
		
		runParticle = runParticle->nextP;
	}
}

void Phase_Lattice::InsertFluidLatticeGas(float x, float percent)
{
	int i, j; 
	float ran_nb;
	
	srand(std::time(0));
	
	for (i = 0; i < numParticles; i++)
	{
		if (runParticle->xpos < x)
		{
			if (!runParticle->nofluid)
			{
				for (j = 0; j < 6; j ++)
				{
					if (runParticle->neigP[j])
					{
						ran_nb = rand() / (float) RAND_MAX;  // pic pseudorandom from 0 to 1.0 for grains
				
						if (ran_nb < percent)
						{
							runParticle->fluid_particle[j] = 1;
						}
					}
				}
			}
		}
		
		runParticle = runParticle->nextP;
	}
}

void Phase_Lattice::RemoveFluidLatticeGas(float x)
{
	int i, j; 
	

	
	for (i = 0; i < numParticles; i++)
	{
		if (runParticle->xpos > x)
		{

			for (j = 0; j < 6; j ++)
			{
				if (runParticle->neigP[j])
				{
					
						runParticle->fluid_particle[j] = 0;
					
				}
			}
		}
		
		runParticle = runParticle->nextP;
	}
}

void Phase_Lattice::SetWallsLatticeGas(int grain)
{
	int i, j; 
	
	for (i = 0; i < numParticles; i++)
	{
		if (runParticle->grain == grain)
		{
			runParticle->nofluid = true;
			runParticle->fluid_particles = 8;
			
			for (j = 0; j < 6; j ++)
			{
				if (runParticle->neigP[j])
				{
					
						runParticle->fluid_particle[j] = 0;
					
				}
			}
		}
		
		runParticle = runParticle->nextP;
	}
}

void Phase_Lattice::SetFluidLatticeGasRandomGrain(float percent,int nbr)
{
	int i, j; 
	float ran_nb;
	
	srand(std::time(0));
	
	for (i = 0; i < numParticles; i++)
	{
		if (runParticle->grain == nbr)
		{

			for (j = 0; j < 6; j ++)
			{
				if (runParticle->neigP[j])
				{
					ran_nb = rand() / (float) RAND_MAX;  // pic pseudorandom from 0 to 1.0 for grains
				
					if (ran_nb < percent)
					{
						runParticle->fluid_particle[j] = 1;
					}
				}
			}
		}
		
		runParticle = runParticle->nextP;
	}
}

void Phase_Lattice::UpdateFluidLatticeGas()
{
	int i, j, count; 
	bool done;
	float ran_nb;
	
	srand(std::time(0));
	
	for (i = 0; i < numParticles; i++)
	{
		if (!runParticle->nofluid)
			runParticle->fluid_particles = 0;
		count = 0;
		
		for (j = 0; j < 6; j ++)
		{
			if (runParticle->neigP[j])
			{
				runParticle->fluid_particle_new[j] = 0;
						
				if (runParticle->fluid_particle[j] == 1)
					count++;
				
			}
		}
		if (!runParticle->nofluid)
			runParticle->fluid_particles = count; 
		
			if (count ==1)
			{
				done = false;
					for (j = 0; j < 6; j ++)
					{
						
						if (runParticle->neigP[j])
						{
							if (runParticle->fluid_particle[j] == 1)
							{
								if (j>2)
								{
									if (runParticle->neigP[j-3])
									{
										if (!runParticle->neigP[j-3]->nofluid)
										{
											runParticle->fluid_particle_new[j-3] = 1;
											done = true;
										}
									}
								}
								else
								{
									if (runParticle->neigP[j+3])
									{
										if (!runParticle->neigP[j+3]->nofluid)
										{
											runParticle->fluid_particle_new[j+3] = 1;
											done = true;
										}
									}
								}
								if (done)
									break;
								else 
								{
									runParticle->fluid_particle_new[j] = 1;
									break;
								}
							}
								
				
						}
					}
			}
			else if (count == 2) 
			{
					ran_nb = rand() / (float) RAND_MAX;  // pic pseudorandom from 0 to 1.0 for grains
				
					for (j = 0; j < 6; j ++)
					{
						if (runParticle->neigP[j])
						{
							if (runParticle->fluid_particle[j] == 1)
							{
								done = false;
								
								if (ran_nb > 0.5)
								{
									if (j>0)
									{
										if (runParticle->neigP[j-1])
										{
											if (!runParticle->neigP[j-1]->nofluid)
											{
												runParticle->fluid_particle_new[j-1] = 1;
												done = true;
											}
										}
									}
									else
									{
										if (runParticle->neigP[5])
										{
											if (!runParticle->neigP[5]->nofluid)
											{
												runParticle->fluid_particle_new[5] = 1;
												done = true; 
											}
										}
									}
									if (!done)
										runParticle->fluid_particle_new[j] = 1;
								}
								else
								{
								
									if (j<5)
									{
										if (runParticle->neigP[j+1])
										{
											if (!runParticle->neigP[j+1]->nofluid)
											{
												runParticle->fluid_particle_new[j+1] = 1;
												done = true;
											}
										}
									}
									else
									{
										if (runParticle->neigP[0])
										{
											if (!runParticle->neigP[0]->nofluid)
											{
												runParticle->fluid_particle_new[0] = 1;
												done = true; 
											}
										}
									}
									if (!done)
										runParticle->fluid_particle_new[j] = 1;
								}
							
							}
								
				
						}
					}
			}
			else if (count >= 3)
			{
					for (j = 0; j < 6; j ++)
					{
						if (runParticle->neigP[j])
						{
							if (runParticle->fluid_particle[j] == 1)
							{
								done = false;
								if (j>0)
								{
									if (runParticle->neigP[j-1])
									{
										if (!runParticle->neigP[j-1]->nofluid)
										{
											runParticle->fluid_particle_new[j-1] = 1;
											done = true;
										}
									}
								}
								else
								{
									if (runParticle->neigP[j+1])
									{
										if (!runParticle->neigP[j+1]->nofluid)
										{
											runParticle->fluid_particle_new[j+1] = 1;
											done = true; 
										}
									}
								}
								if (!done)
									runParticle->fluid_particle_new[j] = 1;
							
							}
								
				
						}
					}
			}
			
				
		
		
		runParticle = runParticle->nextP;
	}
	
	for (i = 0; i < numParticles; i++)
	{
		for (j = 0; j < 6; j++)
		{
			if (runParticle->neigP[j])
			{
				runParticle->fluid_particle[j] = runParticle->neigP[j]->fluid_particle_new[runParticle->neig_spring[j]];
			}
		}
		
		runParticle = runParticle->nextP;
	}
	for (i = 0; i < numParticles; i++)
	{
		for (j = 0; j < 6; j++)
		{
			if (runParticle->neigP[j])
			{
				runParticle->fluid_particle_new[j] = 0;
			}
		}
		
		runParticle = runParticle->nextP;
	}
	UpdateElle();
}
