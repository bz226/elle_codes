
/******************************************************
 * Spring Code Mike 2.0
 *
 * Functions for phase_base class in phase_base.cc
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

//------------------------------------
// elle headers
//------------------------------------

#include "phase_base.h"
#include "lattice.h"


// have to define these in the new Elle version

using std::cout;
using std::endl;
using std::vector;


//  CONSTRUCTOR
/*******************************************************
 * Constructor for the phase_base class
 * Does not do much at the moment, defines some variables
 * Phase base inherits the lattice class that does
 * most of the intialization work. 
 *
 *
 * Daniel 2004
 ********************************************************/

//---------------------------------------------------------------
// Constructor of Phase_Base class
//---------------------------------------------------------------

Phase_Base::Phase_Base():
    // -----------------------------------------------------------------------------
    // default values for some variables. Variables are declared in lattice
    // header
    // -----------------------------------------------------------------------------

    area (1.0), 		// area of the whole box
    the_size (1.0),	// scale of x dimension of box in m
    gas_const (8.314),    // set universal gas constant
    dif_const (0.0),
    grav_grad(0.0),
    time (0.0)		// time in seconds
{
  gas_const = 8.314;
}


Phase_Base::~Phase_Base()
{}



/***************************************************************
 * This is a cleaner function in Phase_base class
 * It adjusts the boundary lists of holes. 
 * Hole boundaries have to be known because these particles 
 * have surface energies and can react
 *
 * Daniel March 2003
 ****************************************************************/

void Phase_Base::Adjust_Hole_List()
{
  int i,j;   	     // counters

  //-------------------------------------------------------------
  // This loop does not start with the first particle
  // necessarily. The list of particles is circular so
  // that you can start anywhere and come out at the same
  // starting position. Therefore loops can be called
  // within loop without disturbing the sequence of particles
  // in earlier loops.
  //--------------------------------------------------------------


  for (i = 0 ; i < numParticles ; i++)  // loop through particles
    {
      if (!runParticle->isHole)    // if its not particle of the hole
        {
          //--------------------------------------------------
          // check if its a hole boundary and if yes set it
          // also have to set its springs to the hole to 0.0
          //--------------------------------------------------

          runParticle->isHoleInternalBoundary = false;
          runParticle->isHoleBoundary = false;
          for (j=0 ; j<8 ; j++)
            {
              if (runParticle->neigP[j])
                {
                  if (runParticle->neigP[j]->isHole)   // if neighour in the hole
                    {
                      runParticle->isHoleBoundary = true;   // its a boundary
                      runParticle->springf[j] = 0.0;        // spring to hole
                    }
                  else
                    {
                      if (runParticle->springf[j]==0.0)
                        {
                          runParticle->springf[j] = runParticle->young;
                        }
                    }
                }
            }
        }
      /*else		// in the hole
      {
      	runParticle->isHoleBoundary = false;
      	runParticle-> isHoleInternalBoundary = false;

      	for (j=0;j<8;j++)
      	{
      		if (runParticle->neigP[j])
      		{
      			if (!runParticle->neigP[j]->isHole)
      			{	
      				runParticle->isHoleInternalBoundary = true;
      			}
      		}
      	}
      }*/
      runParticle = runParticle->nextP;
    }
  Update_Fluid_Neighbour_List();

}


// Update_Fluid_Neighbour_List
/************************************************************************************
 * Function that creates an oriented list around holes. Boundary particles
 * have to be defined as hole boundaries (isHoleBoundary = true)
 * particle becomes right and left neighbour along the interface where 
 * right and left means looking from the solid towards the fluid. 
 *
 * the oriented list is important for fluid pressure and for long range surface
 * energy effects. 
 *
 * daniel 2004
 *************************************************************************************/

//---------------------------------------------------------------------
// function in phase_base.cc
//
// no real variables but the function changes particle pointers in the
// particle objects -> right and left neighbour
// function also takes fluid pressure away and gives it back to
// boundary particles.
//
// function writes out a warning if a particle has only one or no
// neighbours along the interface
//----------------------------------------------------------------------

void Phase_Base::Update_Fluid_Neighbour_List()
{
  int i,j,count;
  double slope,slope_a,average_x,average_y;

  for (i=0; i<numParticles; i++)
    {
      runParticle->fluid_P = 0.0;
      runParticle->rightNeighbour = NULL;
      runParticle->leftNeighbour = NULL;

      if (runParticle->isHoleBoundary)
        {
          average_x = 0.0;
          average_y = 0.0;
          count = 0;

          //-------------------------------------------------------
          // first set right and left neighbour without checking
          //-------------------------------------------------------

          for (j=0;j<8;j++)
            {
              if (runParticle->neigP[j])
                {
                  if (runParticle->neigP[j]->isHoleBoundary)
                    {
                      runParticle->rightNeighbour= runParticle->neigP[j];
                      break;
                    }
                }
            }
          for (j=0;j<8;j++)
            {
              if (runParticle->neigP[j])
                {
                  if (runParticle->neigP[j]->isHoleBoundary)
                    {
                      if (runParticle->neigP[j]->nb != runParticle->rightNeighbour->nb)
                        runParticle->leftNeighbour= runParticle->neigP[j];
                    }
                }
            }

          //----------------------------------------------------
          // now come the warnings
          //----------------------------------------------------

          if (!runParticle->rightNeighbour)
            //cout << "right not set" << endl; // if particle is totally alone
            ;
          else if (!runParticle->leftNeighbour)
            {
              //cout <<" left not set" << endl;   // happens at edges

              for (j=0;j<8;j++)
                {
                  if (runParticle->neigP[j])
                    {
                      if (runParticle->neigP[j]->isHole)
                        {
                          average_x = average_x + runParticle->neigP[j]->xpos;
                          average_y = average_y + runParticle->neigP[j]->ypos;
                          count = count + 1;
                        }
                    }
                }
              average_x = average_x / count;
              average_y = average_y / count;

              if (runParticle->xpos < runParticle->rightNeighbour->xpos)
                {
                  if (average_y < runParticle->ypos)
                    {
                      runParticle->leftNeighbour = runParticle->rightNeighbour;
                      runParticle->rightNeighbour = NULL;
                    }
                }
              else if (runParticle->xpos > runParticle->rightNeighbour->xpos)
                {
                  if (average_y > runParticle->ypos)
                    {
                      runParticle->leftNeighbour = runParticle->rightNeighbour;
                      runParticle->rightNeighbour = NULL;
                    }
                }

            }
          else if (runParticle->leftNeighbour->nb == runParticle->rightNeighbour->nb)
            cout <<"left and right same !"<< endl;  // well this should not happen

          //-------------------------------------------------------
          // now we have to check which is the real right and
          // left neighbour
          //
          // take an average of the open spring positions
          // in order to guess where the fluid is
          //-------------------------------------------------------

          else
            {
              count = 0;
              average_x = 0;
              average_y = 0;

              for (j=0;j<8;j++)
                {
                  if (runParticle->neigP[j])
                    {
                      if(runParticle->neigP[j]->isHole)
                        {
                          average_x = average_x + runParticle->neigP[j]->xpos;
                          average_y = average_y + runParticle->neigP[j]->ypos;
                          count = count + 1;
                        }
                    }
                }
              average_x = average_x /count;
              average_y = average_y /count;

              //---------------------------------------------------------
              // check where the neighbours are in space and compare to
              // the average that tells where the fluid is
              //---------------------------------------------------------

              if (runParticle->rightNeighbour->xpos < (runParticle->leftNeighbour->xpos - (runParticle->radius)))
                {
                  if (runParticle->rightNeighbour->ypos < (runParticle->leftNeighbour->ypos- (runParticle->radius)))
                    {
                      if (average_x < runParticle->xpos)
                        {

                          runParticle->neig = runParticle->leftNeighbour;
                          runParticle->leftNeighbour = runParticle->rightNeighbour;
                          runParticle->rightNeighbour = runParticle->neig;
                        }
                    }
                  else if (runParticle->rightNeighbour->ypos > (runParticle->leftNeighbour->ypos+ (runParticle->radius)))
                    {
                      if (runParticle->xpos < average_x)
                        {

                          runParticle->neig = runParticle->leftNeighbour;
                          runParticle->leftNeighbour = runParticle->rightNeighbour;
                          runParticle->rightNeighbour = runParticle->neig;
                        }
                    }
                  else
                    {
                      if (runParticle->ypos < average_y)
                        {
                          runParticle->neig = runParticle->leftNeighbour;
                          runParticle->leftNeighbour = runParticle->rightNeighbour;
                          runParticle->rightNeighbour = runParticle->neig;
                        }
                    }

                }
              else if (runParticle->rightNeighbour->xpos > (runParticle->leftNeighbour->xpos + (runParticle->radius)))
                {
                  if (runParticle->rightNeighbour->ypos < (runParticle->leftNeighbour->ypos - (runParticle->radius)))
                    {
                      if (average_x < runParticle->xpos)
                        {

                          runParticle->neig = runParticle->leftNeighbour;
                          runParticle->leftNeighbour = runParticle->rightNeighbour;
                          runParticle->rightNeighbour = runParticle->neig;
                        }
                    }
                  else if (runParticle->rightNeighbour->ypos > (runParticle->leftNeighbour->ypos + (runParticle->radius)))
                    {
                      if(average_x > runParticle->xpos)
                        {

                          runParticle->neig = runParticle->leftNeighbour;
                          runParticle->leftNeighbour = runParticle->rightNeighbour;
                          runParticle->rightNeighbour = runParticle->neig;
                        }
                    }
                  else   // y the same
                    {
                      if (average_y < runParticle->ypos)
                        {
                          runParticle->neig = runParticle->leftNeighbour;
                          runParticle->leftNeighbour = runParticle->rightNeighbour;
                          runParticle->rightNeighbour = runParticle->neig;
                        }
                    }

                }
              else   // x the same
                {

                  if (runParticle->rightNeighbour->ypos > runParticle->leftNeighbour->ypos)
                    {
                      if (average_x > runParticle->xpos)
                        {
                          runParticle->neig = runParticle->leftNeighbour;
                          runParticle->leftNeighbour = runParticle->rightNeighbour;
                          runParticle->rightNeighbour = runParticle->neig;

                        }
                    }
                  else
                    {
                      if (average_x < runParticle->xpos)
                        {
                          runParticle->neig = runParticle->leftNeighbour;
                          runParticle->leftNeighbour = runParticle->rightNeighbour;
                          runParticle->rightNeighbour = runParticle->neig;

                        }
                    }
                }
            }
          runParticle->fluid_P = fluid_pressure; // and apply fluid pressure
        }

      runParticle = runParticle->nextP;
    }
}


// Set_Time
/********************************************************************
 * base function in BAse_Phase class that is used to set different
 * basic timescales for the reaction/dissolution processes
 *
 * function receives two variables: intime is the real time
 * in numbers and intimescale a scaling variable to scale that
 * time, otherwise numbers would be too large. 
 * depending on mineral, temperature and stress timescales
 * will be extremely variable. 
 *
 * Daniel March, December 2003
 ************************************************************************/

void Phase_Base::Set_Time(float intime, int intimescale)
{
  if (intimescale == 1) // hours
    {
      time = intime * 60.0 * 60.0;
    }
  else if(intimescale == 2) // days
    {
      time = intime * 60.0 * 60.0 * 24.0;
    }
  else if (intimescale == 3) // months
    {
      time = intime * 60.0 * 60.0 * 24.0 * 30.0;
    }
  else if (intimescale == 4) // years
    {
      time = intime * 60.0 * 60.0 * 24.0 * 365.0;
    }
  else if (intimescale == 0) //seconds
    {
      time = intime;
    }
}

// MakeHole
/******************************************************************************
 * This function simply makes a whole somewhere where an Elle grain was. 
 * 
 * Input is the number of the Elle grain
 *
 * Youngs modulus of particles is then zero. 
 * Surface can then react and fluid can have a fluid pressure
 *******************************************************************************/

void Phase_Base::MakeHole(int nb)
{
  int i,j;		// counters

  runParticle = &refParticle;   // start with first particle

  for (i = 0 ; i < numParticles ; i++)  // loop through all particles
    {
      if (runParticle->grain == nb)    // if particle is in the grain
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


// Set_Mineral_Parameter
/*******************************************************************
 * function in Phase_Lattice class that is used to set mineral 
 * parameters molecular volume and surface free Energy
 *
 * function receives an index for mineral
 *
 * Daniel March, December 2003
 *******************************************************************/

void Phase_Base::Set_Mineral_Parameters(int what)
{
  int i;     // counter

  runParticle = &refParticle;

  for (i = 0 ; i < numParticles ; i++)
    {
      if (what == 1)
        {
          // quartz
          runParticle->mV = 0.000022688;
          runParticle->surf_free_E = 0.35;
        }
      else if (what == 2)
        {
          // NaCl
          runParticle->mV = 0.000024;  // change this !
          runParticle->surf_free_E = 0.05;
        }
      else if (what == 3)
        {
          // Olivine
          runParticle->mV = 0.0000437;  // check this !
          runParticle->surf_free_E = 0.6;

        }
      else
        {
          //cout << "Don't know this number" << endl;
          //cout << "Molecular Volume not set !" << endl;
          break;
        }

      runParticle = runParticle->nextP;
    }
}


// Set_Absolute_Box_Size
/**************************************************************
 * Function in Phase_Lattice class that sets absolute size. 
 *
 * we size of the Elle box is scaled because reaction processes
 * use surface energies that have absolute scales. 
 *
 * receives the size of the x dimension of the modeling box 
 * in meters
 * function scales the area of the particles 
 *
 * Daniel March, December 2003
 ****************************************************************/

void Phase_Base::Set_Absolute_Box_Size(float size)
{
  int i;          // counter

  the_size = size;   // size in meters, size of x dimension of box

  runParticle = &refParticle;

  for (i = 0 ; i < numParticles ; i++)  // and loop
    {
      runParticle->area = runParticle->area * size * size;  // scale particles
      runParticle->real_radius = runParticle->radius * size;

      runParticle = runParticle->nextP;
    }

  initial_diameter = runParticle->real_radius * 2;
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

void Phase_Base::Set_Mass_Distribution(float mass_size, float factor)
{
  int i;			// counter
  float ran_nb;		// pseudorandom number
  float mass_nb;		// distribution for mass

  runParticle = &refParticle;

  for (i = 0; i < numParticles; i++)	// loop through particles
    {
      ran_nb = rand() / (float) RAND_MAX;	// pic pseudorandom from 0
      // to 1.0 for grains
      mass_nb = ran_nb * mass_size;	// now from 0 to distribution size

      // -------------------------------------------
      // set the mass distribution
      // -------------------------------------------

      runParticle->area =
        runParticle->area * (1.0 + mass_nb - (mass_size / 2.0));

      // --------------------------------------------
      // multiply by factor
      // --------------------------------------------

      runParticle->area = runParticle->area * factor;

      runParticle = runParticle->nextP;
    }
}
