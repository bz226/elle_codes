/******************************************************
 * Spring Code Mike 2.0
 *
 * Functions for min_trans_lattice class in min_trans_lattice.cc
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
 * We thank Anders Malthe-Srenssen for his enormous help
 *
 * Daniel Koehn and Till Sachau 2004/5
 ******************************************************/


//srand(std::time(0));   Problem on the mac ! (line 201)

//runParticle->temperature = round(runParticle->temperature); Problem on Mac line 2074


// ------------------------------------
// system headers
// ------------------------------------

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <stdio.h>
#include <math.h>

// ------------------------------------
// elle headers
// ------------------------------------

#include "lattice.h"				// include lattice class
#include "min_trans.h"
#include "phase_base.h"
// inherits lattice class
#include "unodes.h"				// include unode funct. plus undodesP.h
// for c++
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



// CONSTRUCTOR
/*******************************************************
 * Constructor for the phase_lattice class
 * Does not do much at the moment, defines some variables
 * Phase lattice inherits the lattice class that does
 * most of the intialization work. 
 *
 *
 * Daniel March 2003
 ********************************************************/

// ---------------------------------------------------------------
// Constructor of Min_Trans_Lattice class
// ---------------------------------------------------------------
Min_Trans_Lattice::Min_Trans_Lattice()
{
	transition = true;
}

void Min_Trans_Lattice::Activate_MinTrans()
{
  int i;
  // --------------------------------------------------------------------
  // Set some more starting values for particles
  // --------------------------------------------------------------------

  Activate_Lattice();

  for (i=0; i<numParticles; i++)
    {
      if (runParticle->mineral == 1)
        {
          runParticle->surf_free_E = 0.6;

          //set the initial temperature and the molare volume
          runParticle->temperature = 1000.0;

          runParticle->mV = Adjust_Molare_Volume();	// check this !

          runParticle->previous_prob = 0;
          runParticle->spinel_content = 0;
          runParticle->previous_spinel_content = 0;

          runParticle->real_radius = the_size*runParticle->radius;

          runParticle->nucleus = false;

          runParticle->area = pow(double (runParticle->real_radius), 2.0) * 3.1415;
        }
      runParticle = runParticle->nextP;
    }

  //Mineral_Transformation();
  initial_diameter = 2*runParticle->real_radius;

  //fuer die zeit-minimierung
  logicalflag = true;
  timeflag = false;

  //set initial parameters for heat-flow
  runParticle = &refParticle;	// start of list
  runParticle = runParticle->prevP;	// go back one
  org_wall_pos_y = runParticle->ypos;

  runParticle = runParticle->prevP;
  org_wall_pos_x = runParticle->xpos;

  heat_distribution.heat_flow_enabled = false; // in standard settings heat-flow is turned off
  heat_distribution.numParticles = numParticles;

  old_time = time;

  no_heat_flag = false;

  // activation-energy set by user? default == no
  set_act_energy = false;
  act_energy = 400000.0;

  // pressure barrier ste by user ? if not->
  set_pressure_barrier = false;
  pressure_barrier = 12e9;

}



//concept stolen from Update_Fluid_Neighbour_List()
void
Min_Trans_Lattice::Update_Surf_Neighbour_List()
{
  int i,j,count;
  double slope,slope_a,average_x,average_y;


  for (i=0; i<numParticles; i++)
    {
      runParticle->fluid_P = 0.0;
      runParticle->rightNeighbour = NULL;
      runParticle->leftNeighbour = NULL;

      if (runParticle->is_boundary && !runParticle->is_lattice_boundary)
        {
          average_x = 0.0;
          average_y = 0.0;
          count = 0;
          for (j=0;j<6;j++)
            {
              if (runParticle->neigP[j])
                {
                  if (runParticle->neigP[j]->grain == runParticle->grain)
                    {
                      runParticle->rightNeighbour= runParticle->neigP[j];
                      break;
                    }
                }
            }
          for (j=0;j<6;j++)
            {
              if (runParticle->neigP[j])
                {
                  if (runParticle->neigP[j]->grain == runParticle->grain)
                    {
                      if (runParticle->neigP[j]->nb != runParticle->rightNeighbour->nb)
                        runParticle->leftNeighbour= runParticle->neigP[j];

                    }
                }
            }
          if (!runParticle->rightNeighbour)
            {
              //cout <<" right not set" << endl;   // if particle is totally alone
            }
          else if (!runParticle->leftNeighbour)
            {
              //cout << "left not set" << endl;    // can happen at edges, happens quite often.
              for (j=0;j<6;j++)
                {
                  if (runParticle->neigP[j])
                    {
                      if(runParticle->neigP[j]->grain == runParticle->grain)
                        {
                          average_x = average_x + runParticle->neigP[j]->xpos;
                          average_y = average_y + runParticle->neigP[j]->ypos;
                          count = count + 1;
                        }
                    }
                }
              average_x = average_x /count;
              average_y = average_y /count;

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
            {
              //cout <<"left and right same !"<< endl;  //should be ok;
              ;
            }
          else
            {
              count = 0;
              average_x = 0;
              average_y = 0;
              for (j=0;j<8;j++)
                {
                  if (runParticle->neigP[j])
                    {
                      if(runParticle->neigP[j]->grain == runParticle->grain)
                        {
                          average_x = average_x + runParticle->neigP[j]->xpos;
                          average_y = average_y + runParticle->neigP[j]->ypos;
                          count = count + 1;
                        }
                    }
                }
              average_x = average_x /count;
              average_y = average_y /count;

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
                          //cout << "in" << endl;
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
                          //cout << " y same " << endl;
                          runParticle->neig = runParticle->leftNeighbour;
                          runParticle->leftNeighbour = runParticle->rightNeighbour;
                          runParticle->rightNeighbour = runParticle->neig;
                        }
                    }

                }
              else   // x the same
                {
                  //cout << "in x same" << endl;
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
        }

      runParticle = runParticle->nextP;
    }
}



// "main" function for phase-transitions. is called in experiment.cc,
// coordinates the rest.

void Min_Trans_Lattice::Start_Reactions()
{

  int   i,                     // for loop
  j;
  float  interim_time,                           // stores time during gbm
  pressure,
  factor;

  cout << "mole_per_particle: " << mole_per_particle << endl;
  cout << "numParticles: " << numParticles << " " << endl;
  cout << "rate_constant: " << rate_constant << endl;

  pressure = Pressure();

  cout << "Pressure: " << pressure << endl;

  /************************************
  * preset variables:
  ************************************/

  runParticle = &refParticle;                 // for particle loops

  pois = 0.3333333;               // poisson number for the triangular lattice

  Make_Phase_Boundaries();

  actual_time = time;

  for (i=0; i<numParticles; i++)
    {
      runParticle->mV = Adjust_Molare_Volume();

      runParticle->real_radius = runParticle->radius * the_size;
      initial_diameter = runParticle->real_radius * 2;
      runParticle->area = runParticle->real_radius * runParticle->real_radius * Pi;

      runParticle = runParticle->nextP;
    }

  mole_per_particle = pow(refParticle.real_radius, 3.0) * 1.33333333 * 3.14 / refParticle.mV;

  Change_Particle();

  time_interval = actual_time;

  nucleus = NULL;

  Relaxation();

  if (pressure > pressure_barrier) // start if barrier is oversteped
    {

      if (logicalflag == true)
        {
          timeflag = true;
          Change_Timestep();
        }
      else
        {
          do
            {
              for (i = 0; i < numParticles; i++)
                {
                  runParticle->spinel_content = 0.0;
                  runParticle->rate = 0.0;

                  runParticle->local_maximum = false;

                  runParticle->prob = 0.0;

                  runParticle->neig3 = NULL;
                  runParticle->right_now_transformed = false;
                  runParticle->previously_transformed = false;


                  runParticle = runParticle->nextP;
                }

              nucleation_occured = false;
              //Nucleation();

              //if (nucleation_occured)
              {

                cout << "time interval:" << time_interval << endl;

                do
                  {
                    highest_prob = 0.0;
                    std::srand(std::time(0));

                    for (i = 0; i < numParticles; i++)
                      {
                        runParticle->phase_change_required = false;
                        runParticle->previous_sigman = 0;

                        runParticle->rate = 0.0;

                        runParticle->prob = 0.0;
                        runParticle->previous_deltaG = 0.0;
                        runParticle->neig3 = NULL;

                        runParticle->mV = Adjust_Molare_Volume();

                        runParticle = runParticle->nextP;
                      }

                    Gbm();

                    if (highest_prob > 1.0)
                      Relaxation();

                    UpdateElle();

                  }
                while (highest_prob > 1.0);
              }
            }
          while (nucleation_occured);
        }
    }
}



// insert "nuclei", from which the grainboundary migration will start
void
Min_Trans_Lattice::Change_Particle()
{
  int i, one, two, row, column1, column2;

  column1 = int (sqrt(numParticles));
  row = int (column1 / 2);

  column1 = int(column1/3);
  column2 = column1 * 2;

  one = row * int(sqrt(numParticles)) + column1;
  two = row * int(sqrt(numParticles)) + column2;

  for (i=0; i<numParticles; i++)
    {
      if (runParticle->nb == one || runParticle->nb == two)
        {
          if (runParticle->mineral == 1)
            {
              runParticle->prob = 2.0;
            }
        }
      runParticle = runParticle->nextP;
    }

  no_heat_flag = true;
  Mineral_Transformation();
  no_heat_flag = false;

  // adjust changed springs on grain boundaries
  AdjustConstantGrainBoundaries();
  Make_Phase_Boundaries();
  // and relax
  Relaxation();
  //Exchange_Probabilities();
  UpdateElle();

}




// calculates the energy difference for boundary-particles due to a phase-change
void Min_Trans_Lattice::Reaction_Two()
{
  int             i, j;

  // neue variablen im wesentlichen fr rate-law / normal stress
  float           merker1,        // multipurpose
  merker2,            // dito
  numerator,
  denominator,
  molareVolume,
  deltaf,             // delta elastic energy
  sigman,
  deltav,             // volumendifferenz, molare
  inverse_mean_stress,
  time2;              //fr zeitzwischenspeicherung

  // variablen fr boundary-steigung
  bool            neigh_found; // fr abfrage in prob-calculation

  double rate_constant,
  rate,
  rate2,
  exponent,
  N,
  k;

  double delta_Ev_pos, delta_Ev_neg, delta_Ed_pos, delta_Ed_neg, delta_surf_pos, delta_surf_neg, deltaG_pos, deltaG_neg;
  //if (runParticle->neig3) runParticle->time = runParticle->neig3->time;

  double eEl_dif_pos, eEl_dif_neg;

  N=6e23;
  k=1.5 * 1e-23;

  Particle *helper;

  /******************************
   * erste energy
  ******************************/

  for (i=0; i < numParticles; i++)
    {
      Energy_Calculation(runParticle);

      runParticle->previous_eEl = runParticle->eEl;
      runParticle->previous_eEl_dilatation = runParticle->eEl_dilatation;
      runParticle->previous_eEl_distortion = runParticle->eEl_distortion;
      runParticle->previous_surfE = runParticle->surfE;

      runParticle->previous_mV = runParticle->mV;

      runParticle = runParticle->nextP;
    }

  for (i = 0; i < numParticles; i++)
    {
      if (runParticle->phase_change_required)
        {
          for (j = 0; j < 6; j++)
            {
              runParticle->dir_rate[j] = 0.0;

              if (runParticle->neigP[j])
                {
                  if (runParticle->mineral != runParticle->neigP[j]->mineral)
                    {
                      runParticle->dir_rate[j] = Normal_Stress(runParticle->neigP[j],j);
                    }
                }
              if (std::isnan(runParticle->dir_rate[j]))
                cout << "isnogud" << endl;
            }
        }

      runParticle = runParticle->nextP;
    }
  //second deltaG calculation

  //vollstaendige aenderung, um gesamtaenderung der eEl zu berechnen
  Provisoric_Mineral_Transformation();

  //save current values, avoid a second relaxation!
  for (i=0; i<numParticles; i++)
    {
      // instead of relaxation 1
      runParticle->prev_xpos = runParticle->xpos;
      runParticle->prev_ypos = runParticle->ypos;
      runParticle->prev_sxx = runParticle->sxx;
      // instead of relaxation 2
      runParticle->prev_syy = runParticle->syy;
      runParticle->prev_sxy = runParticle->sxy;

      runParticle = runParticle->nextP;
    }

  Relaxation();

  for (i=0; i<numParticles; i++)
    {
      Energy_Calculation(runParticle);

      runParticle = runParticle->nextP;
    }

  for (i = 0; i < numParticles; i++)
    {
      if (runParticle->phase_change_required)
        {

          for (j = 0; j < 6; j++)
            {
              if (runParticle->neigP[j])
                {
                  if (runParticle->dir_rate[j] != 0.0)
                    {

                      sigman = 0.5*(-runParticle->sxx - runParticle->syy) + 0.5*(-runParticle->sxx + runParticle->syy)*cos(2.0*runParticle->boundary_angle[j]) - runParticle->sxy*sin(2.0*runParticle->boundary_angle[j]);

                      sigman *= pressure_scale * pascal_scale;

                      runParticle->dir_rate[j] = sigman - runParticle->dir_rate[j];
                    }
                }
              if (std::isnan(runParticle->dir_rate[j]))
                cout << "isnogud" << endl;
            }
        }

      runParticle = runParticle->nextP;
    }

  for (i = 0; i < numParticles; i++)
    {
      if (runParticle->phase_change_required)
        {

          runParticle->eEl_dif = (runParticle->eEl-runParticle->previous_eEl);

          if (runParticle->eEl_dif > 0)
            {
              delta_Ed_pos = runParticle->eEl_dif;
              delta_Ed_neg = 0;
            }
          else
            {
              delta_Ed_pos = 0;
              delta_Ed_neg = runParticle->eEl_dif;
            }

          for (j=0; j<6; j++)
            {
              deltaG_pos = (runParticle->dir_rate[j]) * (runParticle->mV - runParticle->previous_mV) * mole_per_particle;
              deltaG_neg = 0.0;

              if (deltaG_pos < 0.0)
                {
                  deltaG_neg = deltaG_pos;
                  deltaG_pos = 0.0;
                }

              if (runParticle->dir_rate[j] != 0.0)
                {
                  runParticle->dir_rate[j] = 162000.0*(runParticle->temperature+273.0)*(exp(-(Activation_Energy()/N + (deltaG_neg+delta_Ed_neg)/(N*mole_per_particle))/(k*(runParticle->temperature+273.0))) -  exp(-(Activation_Energy()/N - (deltaG_pos+delta_Ed_pos)/(N*mole_per_particle))/(k*(runParticle->temperature+273.0))));
                }
            }
        }

      runParticle = runParticle->nextP;
    }

  Invert_Provisoric_Mineral_Transformation();

  // instead of a second relaxation, use the saved values
  for (i=0; i<numParticles; i++)
    {
      runParticle->xpos = runParticle->prev_xpos;
      runParticle->ypos = runParticle->prev_ypos;
      runParticle->sxx = runParticle->prev_sxx;
      runParticle->syy = runParticle->prev_syy;
      runParticle->sxy = runParticle->prev_sxy;

      runParticle->phase_change_required = false;

      runParticle = runParticle->nextP;
    }
}



// function calculates normal stress on grain boundaries, where it uses
// neighbouring gb-particles to estimate the true grain-boundary direction
float Min_Trans_Lattice::Normal_Stress(Particle * fixed_neig,int neignb)
{
  float      merker2,
  merker1,
  mean_stress,
  r,
  sigman,
  grain_angle,
  boundary_angle,
  phi,
  dd_neigh,
  dd_neigh_max, // zur Berechnung der Entfernung zweier Nachbarpartikeln
  neigh1_x,
  neigh1_y,        // speichert pos des vorg�ger-neighbours
  neigh2_x,
  neigh2_y,
  boundary_slope;

  Particle   *neig1,
  *neig2;

  int i,
  j,
  numNeig; //number of neighbours of other phase

  bool done,ok;

  //-------------------------------------------------------------------------------------------
  //find out about number of neig-particles of other phase, to distinguish types of boundary
  //-------------------------------------------------------------------------------------------
  numNeig = 0;
  neig1 = NULL;
  neig2 = NULL;
  done = false;

  for (i=0; i<6; i++)
    {
      if (fixed_neig->neigP[i])
        {
          if (fixed_neig->neigP[i]->is_phase_boundary && fixed_neig->neigP[i]->grain == runParticle->grain)
            {
              if (fixed_neig->neigP[i] != runParticle)
                {
                  for (j=0; j<6; j++)
                    {
                      if (fixed_neig->neigP[i]->neigP[j] == runParticle)
                        {
                          if (!done)
                            {
                              neig1 = fixed_neig->neigP[i];
                              done = true;
                            }
                          else
                            {
                              neig2 = fixed_neig->neigP[i];
                            }
                        }
                    }
                }
            }
        }
    }

  if (!neig2)
    {
      done = false;
      for (i=0; i<6; i++)
        {
          if (fixed_neig->neigP[i])
            {
              if (fixed_neig->neigP[i]->is_phase_boundary && fixed_neig->neigP[i]->grain != runParticle->grain)
                {
                  for (j=0; j<6; j++)
                    {
                      if (fixed_neig->neigP[i]->neigP[j] == runParticle)
                        {
                          if (!done)
                            {
                              neig1 = fixed_neig->neigP[i];
                              done = true;
                            }
                          else
                            {
                              neig2 = fixed_neig->neigP[i];
                            }
                        }
                    }
                }
            }
        }
    }

  if (!neig2)
    {
      done = false;
      //find connected neighbours
      for (i=0; i<6; i++)
        {
          if (fixed_neig->neigP[i])
            {
              if (fixed_neig->neigP[i] != runParticle)
                {
                  if (fixed_neig->neigP[i]->is_phase_boundary && fixed_neig->neigP[i]->grain == runParticle->grain)
                    {
                      neig1 = fixed_neig->neigP[i];
                      done = true;
                      neig2 = runParticle;
                    }
                }
            }
        }

      if (!done)
        {
          for (i=0; i<6; i++)
            {
              if (fixed_neig->neigP[i])
                {
                  if (fixed_neig->neigP[i]->is_phase_boundary && fixed_neig->neigP[i]->grain == fixed_neig->grain)
                    {
                      neig1 = fixed_neig->neigP[i];
                      done = true;
                      neig2 = fixed_neig;
                    }
                }
            }
        }
    }

  if (neig2)
    {
      neigh1_x = neig1->xpos;
      neigh2_x = neig2->xpos;
      neigh1_y = neig1->ypos;
      neigh2_y = neig2->ypos;

      //-------------------------------------------------------------------------------------------
      // slope-determination
      //-------------------------------------------------------------------------------------------

      //vermeidet 0 im nenner:
      if (neigh1_x == neigh2_x)
        {
          neigh2_x = neigh1_x + 0.0000000001;
        }

      boundary_slope =(neigh1_y - neigh2_y) / (neigh1_x - neigh2_x);

      // winkelabweichung von boundary auf horizontale
      boundary_angle = atan(boundary_slope);
      boundary_angle += Pi/2;

      runParticle->boundary_angle[neignb] = boundary_angle;

      sigman = 0.5*(-runParticle->sxx - runParticle->syy) + 0.5*(-runParticle->sxx + runParticle->syy)*cos(2.0*boundary_angle) - runParticle->sxy*sin(2.0*boundary_angle);

      sigman *= pressure_scale * pascal_scale;

      if (std::isnan(sigman))
        cout << "isnogud" << endl;

      return(sigman);
    }
  else
    {
      return (0); //dummy-value if no result was obtained
    }
}




// does basic energy-calculaton, i.e. elastic energy,
// surface energy
void Min_Trans_Lattice::Energy_Calculation(Particle * partikel)
{
  int              j,
  ii,
  jj,                    // counters for loops
  spring_count,    // counter of open springs
  box,
  pos;                // dummies for relaxation box position
  float           lame1,          // first lame constant
  lame2,              // second lame constant
  uxx, // x component of infinitesimal strain tensor
  uyy, // y component of infinitesimal strain tensor
  uxy, // xy component of infinitesimal strain tensor
  dx,
  dy,     // delta x, delta y
  alen,
  dd;     // equilibrium length, normal length
  float           fn,
  fx,
  fy,     // normal force, force in x, force in y
  uxi,
  uyi; // specific components of strain tensor
  float           rate,  // rate for reaction
  surfE,     // surface energy
  rep_constant, // repulsion constant
  eEl_distortion,
  eEl_dilatation,
  bulk;
  float sum_Energy;
  float bound_stress;

  Particle      *neig;   // particle object pointer

  float surface_right[10];
  float surface_left[10];
  Particle *help;

  //for the surface-energy
  Update_Surf_Neighbour_List();

  // ------------------------------------------------------------------------------
  // Each particle has a number for its mineral now so that
  // the model knows
  // if we have already changed phase or not..
  // ------------------------------------------------------------------------------

  // -------------------------------------------------------------------
  // first we calculate the boundary stress of the particle
  // maybe this routine should move to the particle class
  // -------------------------------------------------------------------

  // ----------------------------------------
  // zero out some values
  // ----------------------------------------

  partikel->open_springs = 0;
  partikel->bound_stress = 0.0;
  partikel->fluid_neig_count = 0;
  spring_count = 0;

  // ------------------------------------------------
  // repulsion only works if we set flags
  // true for all particles that are connected
  // and the particle itself, otherwise
  // will count stresses twice or worse
  // ------------------------------------------------

  partikel->done = true;

  // --------------------------------------------------
  // now loop through possible connected neighbours
  // --------------------------------------------------

  for (j = 0; j < 6; j++)
    {
      // ------------------------------------
      // if the neighbour exists
      // -----------------------------------

      if (partikel->neigP[j])
        {
          partikel->neigP[j]->done = true; // set flag for repulsion

          spring_count = spring_count + 1; // count number of active connections

          // ------------------------------------------------------------
          // if the neighbour is of the same grain than
          // myself
          // -----------------------------------------------------------

          if (partikel->grain == partikel->neigP[j]->grain)
            {

              // -------------------------------------------------
              // if the neighbour also sits on a grain
              // boundary
              // I just run along the boundary and can
              // memorize
              // this neighbours position for surface energy
              //
              // and pressure on the boundary
              // -------------------------------------------------

              if (partikel->neigP[j]->is_boundary)
                {
                  // ---------------------------------------------------
                  // do not include the lattice boundary
                  // particles,
                  // they are fixed and have therefore not
                  // the right
                  // stresses
                  // ---------------------------------------------------

                  if (!partikel->neigP[j]->is_lattice_boundary)
                    {
                      // ------------------------------------------------------------
                      // ok, now put this neighbour in the
                      // list so that we
                      // can call it directly
                      // and count the number of neighbours
                      // that we find along
                      // the boundary.
                      // ------------------------------------------------------------

                      partikel->fluid_neigP[partikel->
                                            fluid_neig_count]
                      = partikel->neigP[j];
                      partikel->fluid_neig_count =
                        partikel->fluid_neig_count + 1;
                    }
                }

            }
          //else
          {
            //
            if (partikel->neigP[j]->isHole)
              partikel->open_springs = partikel->open_springs + 1;
            if (!partikel->neigP[j]->isHole)
              {
                if (partikel->is_boundary)
                  {
                    if (partikel->neigP[j]->grain != partikel->grain)
                      {

                        //-----------------------------------------------------------
                        // this is a boundary spring, add it to the
                        // number of springs
                        // -----------------------------------------------------------

                        partikel->open_springs += 1;

                        // ----------------------------------------------------------------
                        // and go through the whole monty of
                        // calculating the stresses
                        // on that boundary spring (spring has only
                        // normal force in this
                        // direction..
                        // ----------------------------------------------------------------

                        // get distance in x and y directions

                        dx = partikel->neigP[j]->xpos - partikel->xpos;
                        dy = partikel->neigP[j]->ypos - partikel->ypos;

                        // get distance parallel to spring and
                        // equilibrium distance (alen)

                        dd = sqrt((dx * dx) + (dy * dy));
                        alen = partikel->radius + partikel->neigP[j]->radius;

                        // get unit length in x and y

                        if (dd != 0.0)
                          {
                            uxi = dx / dd;
                            uyi = dy / dd;
                          }
                        else
                          {
                            cout << " zero divide in Reaction2 " << endl;
                            uxi = 0.0;
                            uyi = 0.0;
                          }

                        // ---------------------------------------------------------------
                        // determine the force on particle = constant
                        // times strain
                        // ---------------------------------------------------------------

                        fn = partikel->springf[j] * (dd - alen);  // normal force
                        fx = fn * uxi;   // force x
                        fy = fn * uyi;   // force y

                        fn = fx * uxi + fy * uyi; // force in
                        // direction of
                        // spring

                        fn = fn * 2.0 * partikel->radius;  // scaling
                        // to
                        // particle
                        // size

                        // -------------------------------------------------------------
                        // scale normal force by particle area divided
                        // by six because
                        // the open surface is now only for one spring
                        // i.e. 1/6 of
                        // the particles circumference or area
                        // -------------------------------------------------------------

                        fn = fn / ((partikel->radius * partikel->radius * Pi) / 6.0);

                        // -----------------------------------------------------------
                        // and add that to the particles boundary
                        // stress
                        // -----------------------------------------------------------

                        //partikel->bound_stress = partikel->bound_stress + fn;

                      }
                  }
              }
          }
        }
    }

  // -----------------------------------------------------------
  // check if particle has a broken spring
  // if yes look for repulsion stress and add broken spring
  // to the number of open springs.
  // only works for triangular lattice now with 6 springs
  // -----------------------------------------------------------

  if (spring_count < 6)
    {
      partikel->open_springs = partikel->open_springs + 6 - spring_count;

      // cout << "in" << endl;

      // ----------------------------------------------------
      // now do the repulsion box check again
      // ----------------------------------------------------


      for (ii = 0; ii < 3; ii++)
        {
          for (jj = 0; jj < 3; jj++)
            {
              // --------------------------------------------------
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
              // box and we can check if they have a
              // repulsive
              // force on our particle.
              // The repulsive force in this routine is
              // important
              // on fractures where bonds are broken but
              // particles
              // are pushed into each other. This pressure
              // on the
              // open surface has to be included in the rate
              // law
              // ---------------------------------------------------

              if (jj == 0)
                box = particlex * (-1);  // lower row of
              // boxes
              if (jj == 1)
                box = 0;  // middle row of boxes
              if (jj == 2)
                box = particlex;   // upper row of
              // boxes

              pos = partikel->box_pos + (ii - 1) + box;   // define
              // box
              // position
              //

              if (pos > 0)
                {
                  neig = 0;
                  neig = repBox[pos];   // get particle at
                  // that position

                  while (neig)
                    {
                      // -------------------------------------------------------
                      // Done here means that the particle
                      // is either
                      // the particle itself or its a
                      // connected neighbour
                      // these cannot be used in this list
                      // because
                      // they are either already used to
                      // calculate the
                      // boundary stress or they are part of
                      // the internal
                      // grain and not boundary particles.
                      // Other particles that are further
                      // away but still
                      // in the rep box dont matter, only
                      // compressive forces
                      // are applied here
                      // --------------------------------------------------------

                      if (!neig->done)
                        { // if not already done
                          if (neig->xpos)
                            { // just in case
                              // --------------------------------------------------------------
                              // get distance to neighbour
                              // --------------------------------------------------------------

                              dx = neig->xpos - partikel->xpos;
                              dy = neig->ypos - partikel->ypos;

                              dd = sqrt((dx * dx) + (dy * dy));

                              // --------------------------------------------------------------
                              // get equilibrium length
                              // --------------------------------------------------------------

                              alen = partikel->radius + neig->radius;

                              // --------------------------------------------------------------
                              // determine the unitlengths
                              // in x and y
                              // --------------------------------------------------------------

                              if (dd != 0.0)
                                {
                                  uxi = dx / dd;
                                  uyi = dy / dd;
                                }
                              else
                                {
                                  cout << " zero divide in Relax "  << endl;
                                  uxi = 0.0;
                                  uyi = 0.0;
                                }

                              // ---------------------------------------------------------------
                              // determine the force on
                              // particle = constant times
                              // strain
                              // ---------------------------------------------------------------

                              // ---------------------------------------------------------
                              // first calculate the
                              // repulsion constant by using
                              // an
                              // average of the youngs
                              // moduli of the two particles
                              // --------------------------------------------------------

                              rep_constant = (partikel->young +  neig->young) / 2.0;

                              // scale back to spring
                              // constant

                              rep_constant = rep_constant * sqrt(3.0) / 2.0;

                              fn = rep_constant * (dd - alen);  // normal"force per length" (pressure times length)
                              fx = fn * uxi;  // " force x"
                              fy = fn * uyi;  // " force y"

                              if (fn < 0.0)   // if
                                // compressive
                                {

                                  fn = fx * uxi + fy * uyi;  // force in direction of spring

                                  fn = fn * 2.0 * partikel->radius;   // scaling to particle size

                                  // -------------------------------------------------------------
                                  // scale normal force by
                                  // particle area divided
                                  // by six because
                                  // the open surface is now
                                  // only for one spring
                                  // i.e. 1/6 of
                                  // the particles
                                  // circumference or area
                                  // -------------------------------------------------------------

                                  fn = fn / ((partikel->radius * partikel->radius * Pi) / 6.0);

                                  // -----------------------------------------------------------
                                  // and add that to the
                                  // particles boundary
                                  // stress
                                  // -----------------------------------------------------------

                                  partikel->bound_stress = partikel->bound_stress + fn;
                                }
                            }
                        }
                      if (neig->next_inBox)
                        {        // if there is another one in box
                          neig = neig->next_inBox;     // move pointer
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
  // ---------------------------------------------------
  // now turn off the flags
  // ---------------------------------------------------

  partikel->done = false;

  for (j = 0; j < 6; j++)
    {
      if (partikel->neigP[j])
        {
          partikel->neigP[j]->done = false;
        }
    }
  // ----------------------------------------------------------------------
  // now we have the boundary stress and a list with
  // pointers to the
  // neighbours of our particle and need the surface energy
  // of the
  // particle as a function of its size and its open springs
  // ->
  // curvature of the surface
  // ----------------------------------------------------------------------

  partikel->surfE = partikel->surf_free_E * 2.0 / (partikel->radius * the_size);

  partikel->surfE = partikel->surfE * ((partikel->open_springs - 2.0) / 4.0);

  partikel->surfE = partikel->surfE * partikel->mV;

  // -----------------------------------
  // first calculate the elastic energy
  // -----------------------------------
  // setting elastic constants
  // ----------------------------------

  lame1 = partikel->young * pois / ((1.0 - 2.0 * pois) * (1.0 + pois));
  lame2 = partikel->young / (2.0 + 2.0*pois);

  // -----------------------------------------------------------------------
  // getting strain tensor
  // -----------------------------------------------------------------------

  uxy = ((1 + pois) / partikel->young) * partikel->sxy;
  uxx = (1 / partikel->young) * (partikel->sxx - (partikel->syy * pois));
  uyy = (1 / partikel->young) * (partikel->syy - (partikel->sxx * pois));

  // -----------------------------------------------------------------------
  // calculate elastic energy for particle
  // -----------------------------------------------------------------------

  partikel->eEl = (0.5 * lame1 * (uxx + uyy) * (uxx + uyy) + (lame2 * ((uxx * uxx) + (uyy * uyy) + 2 * (uxy * uxy))));
  //scaling:
  partikel->eEl *= partikel->mV * mole_per_particle  * pressure_scale * pascal_scale;

  bulk = partikel->young/(3*(1-2*pois));
  //alt:
  //eEl_distortion = lame2 * ((uxx-uyy)*(uxx-uyy)+(6*uxy*uxy));
  //daniel versuch:
  eEl_distortion = lame2 * (4*uxy*uxy+2*uxx*uxx+2*uyy*uyy-((4/3)*(uxx*uxx+uyy*uyy)));
  eEl_dilatation = (bulk/2) * (uxx+uyy) * (uxx+uyy);

  partikel->eEl_distortion = eEl_distortion * partikel->mV * mole_per_particle  * pressure_scale * pascal_scale;
  partikel->eEl_dilatation = eEl_dilatation * partikel->mV * mole_per_particle  * pressure_scale * pascal_scale;

  // ---------------------------------------------------------
  // setting average surface energy
  // look at the list of neighbours and add them
  // to the surface energy of the particle and take average
  // ---------------------------------------------------------

  if (partikel->is_boundary)
    {
      //if (!partikel->is_lattice_boundary)
      {
        if (1)
          {
            for (j = 0; j < 10; j++)
              {
                surface_left[j] = 0.0;
                surface_right[j] = 0.0;
              }

            if (partikel->fluid_neig_count)
              {
                surfE = 0.0;
                if (partikel->leftNeighbour)
                  {
                    help = partikel->leftNeighbour;

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
                if (partikel->rightNeighbour)
                  {
                    help = partikel->rightNeighbour;
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
                surfE = partikel->surfE;
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
                partikel->newsurfE = partikel->surfE + surfE;

                //partikel->surfE = (partikel->surfE * (1-surface_area_factor)) + (surfE * surface_area_factor);
              }
            else
              {
                partikel->newsurfE = 0.0;
              }

            //partikel->surfE = partikel->newsurfE;
            //newsurfE statt surfE

            //-----------------------------------------------------------
            // scale pressure
            //-----------------------------------------------------------

            bound_stress = 0.0;

            if (partikel->fluid_neig_count)
              {

                for (j = 0; j < partikel->fluid_neig_count ; j++)
                  {
                    bound_stress = bound_stress + partikel->fluid_neigP[j]->bound_stress;
                  }

                bound_stress = bound_stress / partikel->fluid_neig_count;

                //partikel->bound_stress = (partikel->bound_stress + bound_stress)/2.0;
              }

            partikel->bound_stress = -1.0 * partikel->bound_stress;

            partikel->bound_stress = partikel->bound_stress * pressure_scale * pascal_scale * partikel->mV;
          }
      }
    }

}



// called by reaction_two(), to restore the original spinel-distribution
// without having to relax again. saves some time, costs some memory.
void Min_Trans_Lattice::Invert_Provisoric_Mineral_Transformation()
{
  int           i,j;

  //firstly, restore the old heat-distribution
  //Invert_Heat_Lattice();

  //secondly, restore old particle-volumes, radii and young-moduli
  for (i=0; i<numParticles; i++)
    {

      runParticle->mineral = runParticle->previous_mineral;
      runParticle->mV = runParticle->previous_mV;      // restore the molecular volume
      runParticle->young = runParticle->previous_young;      // restore young's modulus

      runParticle->original_springf = runParticle->previous_original_springf;
      for (j=0; j<6; j++)    // and of course the spring force
        {
          runParticle->springf[j] = runParticle->previous_springf[j];
        }

      runParticle->grain = runParticle->previous_grain;

      runParticle->radius = runParticle->previous_radius;             // and change radius
      runParticle->real_radius = runParticle->previous_real_radius;
      runParticle->area = runParticle->previous_area;

      //cout << "temperature (reverse): " << runParticle->temperature << endl;

      runParticle->phase_change_required = false;

      runParticle = runParticle->nextP;
    }

  Make_Phase_Boundaries();
}



// void Min_Trans_Lattice::Save_Heat_Lattice()
// {
//   int i;
//
//   for (i = 0; i < numParticles; i++)
//     {
//       //cout << "Partikel-temperature (save heat): " << runParticle->temperature << endl;
//       runParticle->previous_temperature = runParticle->temperature;
//       runParticle = runParticle->nextP;
//     }
//
//   heat_distribution.save_heat_distribution();
// }



// void Min_Trans_Lattice::Invert_Heat_Lattice()
// {
//   int i;
//
//   for (i = 0; i < numParticles; i++)
//     {
//       runParticle->temperature = runParticle->previous_temperature;
//       //cout << "Partikel-temperature (invert heat): " << runParticle->temperature << endl;
//       runParticle = runParticle->nextP;
//     }
//
//   heat_distribution.reset_heat_distribution();
// }



// if a phase-changed occured, this function will be called, which sets the new
// attributes for the particle in question.
// the condition for a transformation: particle->prob > 1.0
void Min_Trans_Lattice::Mineral_Transformation()
{
  int             i,
  j,
  counter;

  bool            new_grain;                  // flag for new grain or not
  float           factor;

  for (i = 0; i < numParticles; i++)
    {
      if (runParticle->prob >= 1.0)
        {

          for (j=0; j<4; j++)
            runParticle->l[j] = 0.0;

          cout << "particle-number: " << runParticle->nb << endl;
          cout << "prob: " << runParticle->prob << endl;

          //fuer die zeit-anpassung nach erster umwandlung

          switch (runParticle->mineral)
            {
            case 1:
              new_grain = true;               // first we assume its a new grain of spinell
              runParticle->mineral = 2;            // change mineral index

              if (heat_distribution.heat_flow_enabled && !no_heat_flag)
                {
                  runParticle->temperature += Latent_Heat_Release();
                  Set_Heat_Lattice(runParticle);
                }

              runParticle->young = runParticle->young * 1.2; // change repulsion constant
              runParticle->original_springf *= 1.2;
              runParticle->previous_original_springf = runParticle->original_springf;

              for (j = 0; j < 8; j++)
                {
                  if (runParticle->neigP[j])
                    {         // if spring is active
                      runParticle->springf[j] *= 1.2; // change spring constant

                      // -----------------------------------------------------------
                      // now check what the neighbours are, if they are
                      // of a
                      // different mineral spring is a boundary, if not
                      // this is
                      // not a new grain of spinell in olivine.
                      // -----------------------------------------------------------

                      if (runParticle->neigP[j]->mineral == 1)
                        {
                          runParticle->neigP[j]->is_boundary = true;  // real boundary
                        }
                      else
                        {
                          new_grain = false;
                          runParticle->grain = runParticle->neigP[j]->grain;// give it the name of the other grain of spinell
                        }
                    }
                }

              if (new_grain)
                {                   // if this is a new grain
                  runParticle->grain = highest_grain + 1; // new number for grain
                  highest_grain = highest_grain + 1;      // and adjust grain counter
                }

              break;


            case 2:
              new_grain = true;                              // first we assume its a new grain of olivine
              runParticle->mineral = 1;                      // change mineral index0

              if (heat_distribution.heat_flow_enabled && !no_heat_flag)
                {
                  runParticle->temperature += Latent_Heat_Release();
                  Set_Heat_Lattice(runParticle);
                }

              runParticle->young = runParticle->young * 0.833333333;  // change repulsion constant
              runParticle->original_springf *= 0.833333333;
              runParticle->previous_original_springf = runParticle->original_springf;

              for (j = 0; j < 8; j++)
                {
                  if (runParticle->neigP[j])           // if spring is active
                    {
                      runParticle->springf[j] *= 0.833333333; // change spring constant

                      // -----------------------------------------------------------
                      // now check what the neighbours are, if they are
                      // of a
                      // different mineral spring is a boundary, if not
                      // this is
                      // not a new grain of spinell in olivine.
                      // -----------------------------------------------------------

                      if (runParticle->neigP[j]->mineral == 2)
                        {
                          runParticle->neigP[j]->is_boundary = true;           // real boundary
                          runParticle->neigP[j]->is_phase_boundary = true;
                        }
                      else
                        {
                          new_grain = false;
                          runParticle->grain = runParticle->neigP[j]->grain;   // give it the name of the other grain of spinell
                        }
                    }
                }

              if (new_grain)
                {                                  // if this is a new grain
                  runParticle->grain = highest_grain + 1;              // new number for grain
                  highest_grain = highest_grain + 1;         // and adjust grain counter
                }

              break;
            }
          //zur zeit-minimierung (s.a. Set_Mineral_Parameters, Time-change funktion)
          timeflag = true;
        }

      //weil die uhren jetzt auch fuer addition zurckgedreht werden wollen:
      runParticle->previous_prob = 0.0;
      runParticle->prob = 0.0;

      runParticle = runParticle->nextP;
    }

  counter = 0;

  for (i=0; i<numParticles; i++)
    {
      if (runParticle->grain != beware_grain1 && runParticle->grain != beware_grain2 && runParticle->grain != beware_grain3)
        {
          runParticle->mV = Adjust_Molare_Volume();
          runParticle->previous_real_radius = runParticle->real_radius;
          runParticle->real_radius = pow(((3.0 * mole_per_particle * runParticle->mV)/(4.0 * 3.1415)), (1.0 / 3.0));// and change radius
          factor = (runParticle->real_radius / runParticle->previous_real_radius);
          runParticle->radius *= factor;

          runParticle->area = pow(double(runParticle->real_radius), 2.0) * 3.1415;
        }

      runParticle = runParticle->nextP;
    }

  //Heat_Flow (4);

}



// function sets the "is_phase_boundary"-flag of
// all particles.
void Min_Trans_Lattice::Make_Phase_Boundaries()
{
  int i, j, ii, jj;
  Particle *neig;
  int box, pos;		// dummies for relaxation box position tensor


  for (i = 0; i < numParticles; i++)
    {	// and loop through particles
      for (j = 0; j < 8; j++)
        {	// loop through neighbours of particle
          if (runParticle->neigP[j])
            {	// if there is a neighbour
              if (runParticle->neigP[j]->mineral != runParticle->mineral)
                {	// if I am a boundary
                  runParticle->is_phase_boundary = true;
                  break;
                }
              else
                {
                  runParticle->is_phase_boundary = false;
                }
            }
        }
      runParticle = runParticle->nextP;	// go on looping
    }
}



// function returns mean stress over all particles

float Min_Trans_Lattice::Pressure()
{
  int i, p_counter;
  float smax, smin, mean_stress, sxx, sxy, syy;
  // set values

  // first find out about stress: mean_stress
  p_counter = 0;
  mean_stress = 0.0;
  runParticle = &refParticle;	// start
  for (i = 0; i < numParticles; i++)
    {	// look which particles are in the box
      if (runParticle->ypos > 0.1)
        {	// larger ymin
          if (runParticle->ypos < 0.9)
            {	// smaller ymax
              if (runParticle->xpos > 0.1)
                {	// larger xmin
                  if (runParticle->xpos < 0.9)
                    {	// smaller xmax
                      p_counter = p_counter + 1;	// count particles
                      // set stress tensor
                      sxx = runParticle->sxx;
                      syy = runParticle->syy;
                      sxy = runParticle->sxy;
                      // eigenvalues
                      smax =
                        ((sxx + syy) / 2.0) +
                        sqrt(((sxx - syy) / 2.0) * ((sxx -
                                                     syy) / 2.0) +
                             sxy * sxy);
                      smin =
                        ((sxx + syy) / 2.0) -
                        sqrt(((sxx - syy) / 2.0) * ((sxx -
                                                     syy) / 2.0) +
                             sxy * sxy);
                      mean_stress += (smax + smin) / 2.0;
                    }
                }
            }
        }
      runParticle = runParticle->nextP;
    }
  // and divide by number of particles
  mean_stress = mean_stress / p_counter;

  return (-mean_stress * pascal_scale * pressure_scale);
}



// calculates the time it takes for the fastest particle to transform
// called in start_reactions()
void Min_Trans_Lattice::Gbm()
{
  int            i,j;

  float                         rate,
  time2,
  prob,
  lowest_time,
  lowest_time2,
  local_time,
  v,
  t,
  time_interval2,
  dd;

  float random_number;

  Particle *neig;

  neig = NULL;

  lowest_time = time_interval;   //zeit nach nukleiierung, wenn keine nukleiierung, dann actual_time
  highest_prob = 0.0;

  //cout << "grainboundary migration, time = " << actual_time << ", highest_prob = " << highest_prob << endl;

  cout << "gbm 1" << endl;

  for (i=0; i<numParticles; i++)
    {
      runParticle->neig3 = NULL;
      runParticle->prob = 0.0;
      runParticle->spinel_content = 0.0;

      runParticle->delta_eEl_distortion = 0;
      runParticle->delta_eEl_dilatation = 0;

      runParticle = runParticle->nextP;
    }


  cout << "time_interval: " << time_interval << " refParticle->time: " << refParticle.time << endl;

  //transform boundary-particles, calculate delta G
  Call_Reaction_Two();

  //initialize random-number generator:
  std::srand(std::time(0));

  //set a standard
  runParticle = &refParticle;
  for (i=0; i<numParticles;i++)
    {
      if (runParticle->is_phase_boundary && !runParticle->is_lattice_boundary)
        {
          lowest_time = CalcTime();
          neig = runParticle;
          break;
        }
      runParticle=runParticle->nextP;
    }

  for (i=0; i<numParticles; i++)
    {
      if (runParticle->is_phase_boundary && !runParticle->is_lattice_boundary)
        {
          if (runParticle->grain != beware_grain1 && runParticle->grain != beware_grain2 && runParticle->grain != beware_grain3)
            {
              t = CalcTime();

              if (t < lowest_time)
                {
                  lowest_time = t;
                  neig = runParticle;
                }

            }
        }
      runParticle = runParticle->nextP;
    }

  if (lowest_time == 0.0)
    cout << "warning: lowest time == 0.0" << endl;

  cout << "lowest time: " << lowest_time << endl;

  if (lowest_time < time_interval && lowest_time > 0.0 && neig)
    {
      neig->prob = 2.0;
      highest_prob = 2.0;

      DumpDiffStressAndTime(neig,lowest_time,neig->eEl_dif);
    }
  else
    highest_prob = 0.0;

  if (neig)
    {
      for (i=0; i<4; i++)
        neig->l[i] = 0.0;
    }

  for (i=0; i<numParticles; i++)
    {
      if (runParticle->is_phase_boundary && !runParticle->is_lattice_boundary && runParticle != neig)
        {
          CalcArea(lowest_time);
        }
      runParticle = runParticle->nextP;
    }

  //time setting
  time_interval-=lowest_time;
  if (time_interval <= 0.0)
    {
      for (i=0; i<numParticles; i++)
        {
          runParticle->prob = 0.0;
          runParticle = runParticle->nextP;
        }
      time_interval = 0.0;
      highest_prob = 0.0;
    }
  cout << time_interval << endl;

  Mineral_Transformation();
  // adjust changed springs on grain boundaries
  AdjustConstantGrainBoundaries();
  Make_Phase_Boundaries();

  if (heat_distribution.heat_flow_enabled)
    Heat_Flow(lowest_time);

  // and relax
  Relaxation();
  //Exchange_Probabilities();
  UpdateElle();
}



// calculates the energy for different grain-growth scenarios
// (only spinel grows vs only olvine growths)
// called in gbm()
void Min_Trans_Lattice::Call_Reaction_Two()
{
  int i,
  j;

  for (i=0; i<numParticles; i++)
    {
      if (runParticle->mineral == 1 && runParticle->is_phase_boundary && !runParticle->is_lattice_boundary)
        {
          runParticle->phase_change_required = true;
        }
      else
        {
          runParticle->phase_change_required = false;
        }

      runParticle = runParticle->nextP;
    }

  Reaction_Two();

  cout << " __________________________1_________________________" << endl;

  for (i=0; i<numParticles; i++)
    {

      if (runParticle->mineral == 2 && runParticle->is_phase_boundary && !runParticle->is_lattice_boundary)
        {
          runParticle->phase_change_required = true;
        }
      else
        {
          runParticle->phase_change_required = false;
        }
      runParticle = runParticle->nextP;

    }

  Reaction_Two();
}



// calculate the area, that will be overgrown during a timestep,
// which is defined in the CalcTime()-function
// called in gbm()
float
Min_Trans_Lattice::CalcArea(float lowest_time)
{
  int i;
  float rate[4];        //indicate directions of rate in the rectangle
  float l,l1,l2;        //the length of a basic square size
  float x1, x2, F0, F1; //transformed area variables (F0=partikel-area, F1=remaining area after timestep, x1,x2=new side-lengthes

  F0 = runParticle->real_radius*runParticle->real_radius*Pi;
  l = sqrt(F0);

  l1 = l-runParticle->l[0]-runParticle->l[2];
  l2 = l-runParticle->l[1]-runParticle->l[3];

  for (i=0; i<4; i++)
    rate[i] = 0.0;

  //pass hexagonaly directed rates over to rates in a square
  for (i=0; i<6; i++)
    {
      if (runParticle->dir_rate[i] < 0.0 && runParticle->mineral == 2)
        runParticle->dir_rate[i] = 0.0;

      switch (i)
        {
        case 0:
          rate[0] += runParticle->dir_rate[0];
          break;

        case 1:
          rate[0] += sin(Pi/6) * runParticle->dir_rate[1];
          rate[1] += cos(Pi/6) * runParticle->dir_rate[1];
          break;

        case 2:
          rate[1] += cos(Pi/6) * runParticle->dir_rate[2];
          rate[2] += sin(Pi/6) * runParticle->dir_rate[2];
          break;

        case 3:
          rate[2] += runParticle->dir_rate[3];
          break;

        case 4:
          rate[2] += sin(Pi/6) * runParticle->dir_rate[4];
          rate[3] += cos(Pi/6) * runParticle->dir_rate[4];
          break;

        case 5:
          rate[3] += cos(Pi/6) * runParticle->dir_rate[5];
          rate[0] += sin(Pi/6) * runParticle->dir_rate[5];
          break;
        }

    }

  runParticle->l[0] += lowest_time*rate[0];
  runParticle->l[1] += lowest_time*rate[1];
  runParticle->l[2] += lowest_time*rate[2];
  runParticle->l[3] += lowest_time*rate[3];

}



//this function calculates the time, that it takes to overgrow the
//area of a particle, depending on the number of neighbours (the
//contact area) and the rates into this direction
//idea: calculate the time for either two sides to meet at a certain
//point, choose the lowest calculated time
// called in gbm()
float
Min_Trans_Lattice::CalcTime()
{
  int i,j, buffer1, buffer2;
  float rate[4];         //indicate directions of rate in the rectangle
  float l1,l2;               //the length of a basic square size
  float a;               //the transformed area
  float lowest_time, t;

  l1 = l2 = sqrt(runParticle->real_radius*runParticle->real_radius*Pi);

  l1 -= (runParticle->l[0] + runParticle->l[2]);
  l2 -= (runParticle->l[1] + runParticle->l[3]);

  for (i=0; i<4; i++)
    rate[i] = 0.0;

  //pass hexagonaly directed rates over to rates in a square
  for (i=0; i<6; i++)
    {
      //if (runParticle->dir_rate[i] < 0.0 && runParticle->mineral == 2)
      //runParticle->dir_rate[i] = 0.0;
      if (std::isnan(runParticle->dir_rate[i]))
        cout << "isnogud" << endl;

      switch (i)
        {
        case 0:
          rate[0] += runParticle->dir_rate[0];
          break;

        case 1:
          rate[0] += sin(Pi/6) * runParticle->dir_rate[1];
          rate[1] += cos(Pi/6) * runParticle->dir_rate[1];
          break;

        case 2:
          rate[1] += cos(Pi/6) * runParticle->dir_rate[2];
          rate[2] += sin(Pi/6) * runParticle->dir_rate[2];
          break;

        case 3:
          rate[2] += runParticle->dir_rate[3];
          break;

        case 4:
          rate[2] += sin(Pi/6) * runParticle->dir_rate[4];
          rate[3] += cos(Pi/6) * runParticle->dir_rate[4];
          break;

        case 5:
          rate[3] += cos(Pi/6) * runParticle->dir_rate[5];
          rate[0] += sin(Pi/6) * runParticle->dir_rate[5];
          break;
        }

    }

  //now calculate the time of growth to a common meeting point
  lowest_time = l1 / (rate[0]+rate[2]);
  if (lowest_time < 0)
    lowest_time = 10*time_interval;

  t = l2 / (rate[1]+rate[3]);
  if (t<0)
    t = 10*time_interval;

  if (t < lowest_time)
    lowest_time = t;

  //cout << lowest_time << endl;

  return (lowest_time);
}



// calculates the latent-heat realease after a phase-change. called in
// fctn "mineral_transformation"
float Min_Trans_Lattice::Latent_Heat_Release()
{
  int           i,
  j;
  float            heat_capacity,
  latent_heat,
  temperature;


  if (runParticle->mineral == 1)
    {
      heat_capacity = 200.2 - 1037.0*pow(double(runParticle->temperature), -0.5) - 1320000.0*pow(double(runParticle->temperature), -2.0)- 492000000.0*pow(double(runParticle->temperature), -3.0);
      temperature = -39000.0 / heat_capacity;
      //cout << runParticle->temperature << endl;
    }
  else if (runParticle->mineral == 2)
    {
      heat_capacity = 217.8 - 1408.0*pow(double(runParticle->temperature), -0.5) - 526000000.0*pow(double(runParticle->temperature), -3.0);
      temperature = 39000.0 / heat_capacity;
      //cout << runParticle->temperature << endl;
    }

  return(temperature);
}



// called in reaction_two(). changes boundary-particles
void
Min_Trans_Lattice::Provisoric_Mineral_Transformation()
{

  int           i,j;
  float factor;

  //here the old distribution of heat within the 'thin-section' will be saved
  //Save_Heat_Lattice();

  //flag (=condition) to perform a provisoric phase-change for each particle:=phase_change_required

  //make it a loop through all particles, to built the new configuration of particles at ones:

  for (i=0; i<numParticles; i++)
    {

      runParticle->previous_mineral = runParticle->mineral;
      runParticle->previous_mV = runParticle->mV;            // save the molecular volume
      runParticle->previous_young = runParticle->young;                     // save young
      runParticle->previous_grain = runParticle->grain;

      for (j=0; j<6; j++)
        {
          runParticle->previous_springf[j] = runParticle->springf[j];
        }

      runParticle = runParticle->nextP;
    }

  runParticle = &refParticle;

  for (i=0; i<numParticles; i++)
    {
      //ask for phase_change-condition
      //if (runParticle->grain != beware_grain1 && runParticle->grain != beware_grain2 && runParticle->grain != beware_grain3) {
      if (runParticle->phase_change_required == true)
        {
          if (runParticle->mineral == 1)
            {

              runParticle->mineral = 2;                // change mineral index
              runParticle->young = runParticle->young * 1.2;       // change repulsion constant
              runParticle->original_springf *= 1.2;
              for (j=0; j<6; j++)
                {
                  runParticle->springf[j] *= 1.2;
                }

              for (j=0; j<6; j++)
                {
                  if (runParticle->neigP[j])
                    {
                      if (runParticle->neigP[j]->mineral == runParticle->mineral)
                        {
                          runParticle->grain = runParticle->neigP[j]
                                               ->grain;
                        }
                    }
                }
              if (runParticle->neig3)
                runParticle->grain = runParticle->neig3->grain;

            }
          else if (runParticle->mineral == 2)
            {
              runParticle->mineral = 1;                   // change mineral index
              runParticle->young = runParticle->young * 0.8333333333;       // change repulsion constant
              runParticle->original_springf *= 0.8333333333;

              for (j=0; j<6; j++)
                {
                  runParticle->springf[j] *= 0.8333333333;
                }

              for (j=0; j<6; j++)
                {
                  if (runParticle->neigP[j]->mineral == runParticle->mineral)
                    {
                      runParticle->grain = runParticle->neigP[j]->grain;
                    }
                }
              if (runParticle->neig3)
                runParticle->grain = runParticle->neig3->grain;

            }
          //}
        }

      runParticle = runParticle->nextP;
    }

  //here, finally, the new molare volume & radius for the particle will be calculated:
  for (i=0; i<numParticles; i++)
    {
      //if (runParticle->grain != beware_grain1 && runParticle->grain != beware_grain2 && runParticle->grain != beware_grain3) {
      runParticle->mV = Adjust_Molare_Volume();  //geaenderte druckbedingungen

      runParticle->previous_real_radius = runParticle->real_radius;
      runParticle->real_radius = pow(((3.0 * mole_per_particle * runParticle->mV)/(4.0 * 3.1415)), (1.0 / 3.0));   // and change radius

      //factor to adjust (internal) radius of particle
      factor = runParticle->real_radius / runParticle->previous_real_radius;

      runParticle->previous_radius = runParticle->radius;
      runParticle->radius *= factor;

      runParticle->previous_area = runParticle->area;
      runParticle->area = pow(double(runParticle->real_radius), 2.0) * 3.1415;
      //}

      runParticle = runParticle->nextP;
    }

  Make_Phase_Boundaries();

  AdjustConstantGrainBoundaries();
}



// set heat distribution in class heat_lattice by
// passing x/y-position of particles + their temperatures
// performed only in the beginning for all particles and
// if a particle actually changed the phase.
// called in heat_flow
void Min_Trans_Lattice::Set_Heat_Lattice(Particle *prtl)
{
  double delta_x, delta_y,Time_Step_Heat;

  Time_Step_Heat = 5.0; //dummy-value

  runParticle = &refParticle;
  runParticle = runParticle->prevP;
  delta_y = runParticle->ypos /  org_wall_pos_y;

  runParticle = runParticle->prevP;
  delta_x = runParticle->xpos / org_wall_pos_x;

  heat_distribution.initial_diameter = initial_diameter;

  heat_distribution.set_Parameters(delta_y, delta_x,
                                   Time_Step_Heat);

  runParticle = prtl;

  heat_distribution.set_cell_temperature(runParticle->xpos,runParticle->ypos,runParticle->temperature);
}



// read the actual temperature-distribution, apply it
// to the particles. called in "heat-flow()" function
void Min_Trans_Lattice::Read_Heat_Lattice()
{
  int i;

  for (i = 0; i < numParticles; i++)
    {
      runParticle->temperature = heat_distribution.set_particle_temperature(runParticle->xpos, runParticle->ypos);
      //cout << "temperature of particle: " << runParticle->temperature << endl;
      runParticle = runParticle->nextP;
    }
}



// "main"-function for the heat flow
// is called in the experiment class or
// - if turned on - in the gbm-routine
void Min_Trans_Lattice::Heat_Flow(float Time_Step_Heat)
{
  int p_counter, i, col;
  float delta_t, wall_pos, delta_x, delta_y, density, specific_heat, lambda, highest_temp;

  if (Time_Step_Heat > 0.0)
    {

      runParticle = &refParticle;
      runParticle = runParticle->prevP;
      delta_y = runParticle->ypos /  org_wall_pos_y;

      runParticle = runParticle->prevP;
      delta_x = runParticle->xpos / org_wall_pos_x;

      heat_distribution.initial_diameter = initial_diameter;

      heat_distribution.set_Parameters(delta_y, delta_x,
                                       Time_Step_Heat);

      //perform heat flow, read and set particle-properties
      //Set_Heat_Lattice();		//temperaturen der partikel uebergeben
      heat_distribution.heat_flow();	//ausfhren der waermeleitungs-routine
      Read_Heat_Lattice();	//rueckuebergabe der temperaturen an partikel
    }
}



// A function to adjust the molare volume, using the
// temperature. It does NOT include the compressibility.
// Called after a phase-change of a particle.
float Min_Trans_Lattice::Adjust_Molare_Volume()
{

  float            alpha,
  mean_stress,
  mV,
  t;

  mean_stress = mean_stress = ((-runParticle->sxx) + (-runParticle->syy)) / 2;
  mean_stress *= pascal_scale * pressure_scale;

  switch (runParticle->mineral)
    {
    case 1:
      t = runParticle->temperature + 273;
      //after temperature
      //runParticle->temperature = 1000.0;
      alpha = 0.00003052*t + 0.000000004 * pow(double(t), 2.0) + 0.5824 * pow(double(t), -1.0);
      t=298;
      alpha -= 0.00003052*t + 0.000000004 * pow(double(t), 2.0) + 0.5824 * pow(double(t), -1.0);

      mV = 0.00004367 * exp(alpha);

      //after pressure
      //adjustment occurs using the bulk-modulus
      //t = runParticle->temperature;  //formel nach °C
      //mV -= (1.0/(135000000000.0 + 16000000.0 * (t))) * mV * fabs(mean_stress);//s.Fei et al.

      break;

    case 2:
      t = runParticle->temperature + 273;
      //after temperature
      //runParticle->temperature = 1000.0;

      alpha = 0.00002711*t + 0.00000000344 * pow(double(t), 2.0) + 0.5767 * pow(double(t), -1.0);
      t = 298;
      alpha -= 0.00002711*t + 0.00000000344 * pow(double(t), 2.0) + 0.5767 * pow(double(t), -1.0);

      mV = 0.00003953 * exp(alpha);

      //after pressure
      //adjustment occurs using the bulk-modulus
      //t = runParticle->temperature - 27;//formel nach Raumtemperatur in °K
      //mV -= (1.0/(174000000000.0 - 27000000.0 * t)) * mV * fabs(mean_stress);//s.Fei et al.

      break;
    }

  mV = fabs(mV);
  return(mV);

}



// The activation energy used for the calculation of the
// rate of reaction. Currently it is fixed to a certain value,
// commenting this out will calculate the activation energy depending
// including the activation-volume.
float Min_Trans_Lattice::Activation_Energy()
{
  float      activation_energy,
  activation_volume,
  activation_enthalpy,
  mean_stress;

  // default for set_act_energy == false
  if (!set_act_energy)
    {

      mean_stress = ((-runParticle->sxx) + (-runParticle->syy)) / 2.0;
      mean_stress *= pascal_scale * pressure_scale;

      activation_enthalpy = 159.0 * 1805.0; //schmelztemperatur richtig?
      activation_volume = 1.0/(1.19e4 * (1.0 + mean_stress/8000000000.0));

      activation_energy = activation_enthalpy + activation_volume * mean_stress;
      activation_energy = 350000;

    }

  else
    activation_energy = act_energy;

  //cout << "activation energy: " << activation_energy << endl;
  return (activation_energy);
}



// presets an ectivation-energy and the pressure-barrier where gbm starts
// for olivine->spinel!
void Min_Trans_Lattice::SetReactions(double energy, double barrier)
{
  act_energy = energy;
  set_act_energy = true;

  pressure_barrier = barrier;
  set_pressure_barrier = true;

  heat_distribution.SetHeatFlowParameters(4.2, 1005.0, 0.000001, 1000.0);
}



// changes the phase of complete grain as basis for grain-growth
// works with up to 3 grains
void Min_Trans_Lattice::Change_Grain_Mineral(int grain1, int grain2,
    int grain3)
{
  // aufruf in main():
  // uses grain-number to identify, changes phase to spinel

  int i;
  float factor;

  beware_grain1 = grain1;
  beware_grain2 = grain2;
  beware_grain3 = grain3;

  //**********************************************
  //define grain:
  //**********************************************

  for (i = 0; i < numParticles; i++)
    {
      if (runParticle->grain == beware_grain1
          || runParticle->grain == beware_grain2
          || runParticle->grain == beware_grain3)
        {
          runParticle->mineral = 2;
          runParticle->young *= 1.2;

          runParticle->mV = Adjust_Molare_Volume();
          runParticle->previous_real_radius = runParticle->radius;
          runParticle->real_radius = pow(((3.0 * mole_per_particle * runParticle->mV) / (4.0 * 3.1415)), (1.0 / 3.0));	// and change radius
          factor =
            (runParticle->real_radius /
             runParticle->previous_real_radius);
          runParticle->radius *= factor;

          runParticle->area =
            pow(double (runParticle->real_radius), 2.0) * 3.1415;

        }

      runParticle = runParticle->nextP;
    }
}


/*****************************************************
* Convenience-function, to approach the relevant pressure
* with large deformation- and time-steps and to continue
* with smaller steps.
*****************************************************/
void Min_Trans_Lattice::Change_Timestep()
{
  int i;

  if (timeflag && logicalflag)
    {
      time /= 20.0;
      logicalflag = false;
    }
}



void Min_Trans_Lattice::Exchange_Probabilities()
{

  // used for the addition of probabilities to change phase

  int i;

  for (i = 0; i < numParticles; i++)
    {
      if (runParticle->prob < 1.0)
        {
          runParticle->previous_spinel_content +=
            pow(double (runParticle->rate * time_interval), 3);
          //cout << "prev. prob: " << runParticle->previous_prob << endl;
        }
      else
        {
          runParticle->previous_spinel_content =
            runParticle->spinel_content;
        }

      runParticle = runParticle->nextP;
    }
}



void Min_Trans_Lattice::Change_Young()
{
  if (runParticle->mineral == 1)
    {
      runParticle->young *= 1.2;
    }
  else
    {
      runParticle->young *= 0.8333333;
    }
}



void Min_Trans_Lattice::Restore_Young()
{
  if (runParticle->mineral == 1)
    {
      runParticle->young *= 0.8333333;
    }
  else
    {
      runParticle->young *= 1.2;
    }
}



/**********************************************************************************************************
******************************* Functions for Nucleation, not in use **************************************
**********************************************************************************************************/

// function to nucleate. still experimental
void Min_Trans_Lattice::Nucleation()
{
  int i,
  j;
  double rate,
  I,
  Y,
  v,
  lowest_time,
  buffer,
  prob;

  //voreinstellungen, vielleicht sind nicht alle davon nötig...
  //damit der partikel beim nucleation_list_pointer nicht mitgezählt wird
  for (i=0; i<numParticles; i++)
    {
      runParticle->prob = 0;
      runParticle = runParticle->nextP;
    }

  runParticle = &refParticle;

  lowest_time = actual_time;   //zeit nach wärmeleitung

  nucleation_occured = false;

  help_pointer = NULL;

  for (i=0; i<numParticles; i++)
    {
      if (!runParticle->is_lattice_boundary && runParticle->mineral == 1)
        {
          Y = Growth_Rate_For_Nucleation();
          I = Nucleation_Rate();

          v = pow(double(actual_time), 4.0) * pow(double(Y), 3.0) * I * Pi / 3.0;
          runParticle->rate = pow(double(Y), 3.0) * I;

          //standardisiert für ein mol, daher kein faktor
          runParticle->spinel_content = v;
        }
      runParticle = runParticle->nextP;
    }

  //sortiere nukleiierte partikel entsprechend der hoechsten volumina
  Find_Local_Maxima();

  for (i=0; i<numParticles; i++)
    {
      if (runParticle->mineral == 1 && !runParticle->is_lattice_boundary)
        {
          if (runParticle->prob >= 1.0 && runParticle != nucleus)
            {
              buffer = pow((mole_per_particle*runParticle->mV) / (runParticle->rate * Pi) * 3.0,0.25);

              if (buffer < actual_time)
                {
                  nucleation_occured = true;
                  runParticle->time = buffer;
                }
              else
                runParticle->time = actual_time;
            }
          else
            runParticle->time = actual_time;
        }
      else
        runParticle->time = actual_time;

      runParticle = runParticle->nextP;
    }

  if (!nucleation_occured)
    help_pointer = NULL;

  if (nucleation_occured)
    {
      if (!nucleus)
        {
          time = actual_time;

          for (i=0; i<numParticles; i++)
            {
              if (runParticle->time < time && runParticle->time > 0.0 && runParticle->prob > 1.0)
                {
                  nucleus = runParticle;
                  time = runParticle->time;
                }
              runParticle = runParticle->nextP;
            }
        }

      time = actual_time;

      for (i=0; i<numParticles; i++)
        {
          if (runParticle->time < time && runParticle != nucleus && runParticle->time > 0.0 && runParticle->prob > 1.0)
            {
              help_pointer = runParticle;
              time = runParticle->time;
            }
          runParticle = runParticle->nextP;
        }

      if (help_pointer)
        {
          time_interval = help_pointer->time - nucleus->time;
          actual_time -= time_interval;
        }
      else
        {
          time_interval = 0;
          actual_time -= nucleus->time;
        }
    }

  if (help_pointer)
    {
      for (i=0; i<numParticles; i++)
        {
          if (runParticle != nucleus)
            {
              runParticle->prob = 0.0;
            }
          runParticle = runParticle->nextP;
        }

      //nucleus = help_pointer;

      nucleus = NULL;

      Mineral_Transformation();

      AdjustConstantGrainBoundaries();

      Make_Phase_Boundaries();

      Relaxation();
    }

  UpdateElle();
}



// calculates energy barrier for nucleation

float Min_Trans_Lattice::Energy_Barrier()
{
  float prefactor,
  surface_energy,
  driving_force,
  molare_volume,
  energy_barrier,
  mean_stress;

  mean_stress = ((-runParticle->sxx) + (-runParticle->syy)) / 2;
  mean_stress *= pascal_scale * pressure_scale;

  if (runParticle->mineral == 1)
    {
      molare_volume = 0.00004367;
    }
  else if (runParticle->mineral==2)
    {
      molare_volume = 0.000039;
    }

  prefactor = (16.0 * Pi) / 3.0;
  surface_energy = 0.6;
  driving_force = Driving_Force();

  surface_energy = pow(double(surface_energy), 3.0);
  molare_volume = pow(double(molare_volume), 2.0);
  driving_force = pow(double(driving_force), 2.0);

  energy_barrier = prefactor * ((surface_energy * molare_volume) / driving_force);

  return (energy_barrier);
}



float Min_Trans_Lattice::Excess_Pressure()
{

  float delta_P, //differenz to the equilibrium pressure for the given temperature (akaogie, daessler)
  P,          //at phase boundary with temperature T
  T,
  mean_stress;

  float clapeyron_constant,
  enthalpy_differenz,
  volume_differenz;

  enthalpy_differenz = 29970.0;
  volume_differenz = 0.0000316;  //volumen ist KORRIGIERT fr J/(pa*mol)!!!!! vgl. stosch, akaogie table 3
  T = runParticle->temperature + 273;

  P = 14400000000.0 + (enthalpy_differenz/volume_differenz) * log(T/1473);  // P(1) und T(1) = 14.4 GPa, 1473 K

  mean_stress = ((-runParticle->sxx) + (-runParticle->syy)) / 2;
  mean_stress *= pascal_scale * pressure_scale;

  delta_P = P - mean_stress;

  //cout << "deltaP: " << delta_P << endl;
  //cout << "mean stress: " << mean_stress << endl;

  return (delta_P);
}



float Min_Trans_Lattice::Undercooling()
{

  float delta_T, //differenz to the equilibrium temperature for the given pressure (akaogie, daessler)
  P,          //at phase boundary with temperature T
  T,
  mean_stress;

  float clapeyron_constant,
  enthalpy_differenz,
  volume_differenz;

  enthalpy_differenz = 29970.0;
  volume_differenz = 0.00000316;   //volumen ist KORRIGIERT fr J/(pa*mol)!!!!! vgl. stosch, akaogie table 3
  //TODO: volumenkorrektur auf beta-phasen umwandeln!!!

  mean_stress = ((-runParticle->sxx) + (-runParticle->syy)) / 2;
  mean_stress *= pascal_scale * pressure_scale;

  T = 1473 * exp((mean_stress - 14400000000.0) * (volume_differenz/enthalpy_differenz));

  delta_T = T - runParticle->temperature;

  //cout << "deltaT: " << delta_T << endl;

  return (delta_T);
}



float Min_Trans_Lattice::Driving_Force()
{
  float driving_force,
  entropy_change,
  mean_stress,
  mean_stress_change,
  volume_change,
  excess_pressure,
  undercooling;

  excess_pressure = Excess_Pressure();
  undercooling = Undercooling();

  //cout << "excess P: " << excess_pressure << " Delta T: " << undercooling << endl;

  mean_stress = ((-runParticle->sxx) + (-runParticle->syy)) / 2;
  mean_stress *= pascal_scale * pressure_scale;

  if (runParticle->mineral == 1)
    {
      volume_change = 0.00000316;  //hier: v in m^3
      entropy_change = 7.7;
    }
  else if (runParticle->mineral == 2)
    {
      volume_change = 0.00000316;  //hier: v in m^3
      entropy_change = 7.7;
    }

  //runParticle->mV * mole_per_particledriving_force = excess_pressure * volume_change - undercooling * entropy_change;
  driving_force = mean_stress * volume_change - runParticle->temperature * (entropy_change);

  //cout <<"driving force: " << driving_force << endl;

  return (driving_force);
}



float Min_Trans_Lattice::Growth_Rate_For_Nucleation()
{
  float rate,
  mean_stress,
  kelvin;

  kelvin = runParticle->temperature + 273;

  mean_stress = mean_stress = ((-runParticle->sxx) + (-runParticle->syy)) / 2;
  mean_stress *= pascal_scale * pressure_scale;

  runParticle->delta_pV = Driving_Force();

  rate = 162000 * kelvin * exp(-(Activation_Energy() / (gas_const * kelvin)));
  rate *= (1 - exp(-(runParticle->delta_pV) / (gas_const * kelvin)));

  return (rate);
}



double Min_Trans_Lattice::Nucleation_Rate()
{
  //eEL UND SURFe ÜERARBEITEN: EINHEITEN
  double k,
  rate,
  deltaG_hom,
  kelvin;

  double I_zero;

  int i;

  kelvin = runParticle->temperature + 273;

  I_zero = 1.0e+40;

  k = 1.38 * pow(10.0, -23); //boltzmann-constant

  deltaG_hom = Energy_Barrier(); //geeicht auf spinel-nukleiierung in olivin!!
  //cout << "Energy Barrier: " << deltaG_hom << endl;

  rate = I_zero * kelvin * exp(-(450000 / (gas_const * kelvin)));
  //cout << "rate: " << rate << endl;
  //cout << "deltaG_hom / (k * kelvin): " << deltaG_hom / (k * kelvin) << endl;

  rate *= exp(-(deltaG_hom / (k * kelvin)));
  //cout << "rate: " << rate << endl;

  return(rate);
}



void Min_Trans_Lattice::Find_Local_Maxima()
{
  int 				i,
  j;

  for (i=0; i < numParticles; i++)
    {      // loop through particles and find out local maxima among remaining particles
      if (runParticle->spinel_content > 0.0)
        {
          for (j=0; j<8; j++)
            {
              if (runParticle->neigP[j] && runParticle->neigP[j]->spinel_content > 0.0)
                {
                  if (runParticle->neigP[j]->spinel_content > runParticle->spinel_content)
                    {
                      runParticle->local_maximum = false;
                      break;
                    }
                  else
                    {
                      runParticle->local_maximum = true;
                    }
                }
            }
        }
      runParticle = runParticle->nextP;
    }

  for (i=0; i < numParticles; i++)
    {
      if (!runParticle->local_maximum)
        {
          runParticle->spinel_content = runParticle->previous_spinel_content;
          runParticle->prob = 0.0;
        }
      else
        runParticle->prob = 2.0;

      runParticle=runParticle->nextP;
    }
}



/**********************************************************************************************************
************************************ Additional transition-stuff ******************************************
**********************************************************************************************************/



// dumps diff-stress and time for the last overgrown particle
void
Min_Trans_Lattice::DumpDiffStressAndTime(Particle *particle,float time, float eEl_diff)
{
  FILE *stat;
  float sxx, syy, sxy, differential, smax, smin;

  sxx = particle->sxx;
  sxy = particle->sxy;
  syy = particle->syy;

  smax = ((sxx + syy) / 2.0) + sqrt(((sxx - syy) / 2.0) * ((sxx - syy) / 2.0) + sxy * sxy);
  smin = ((sxx + syy) / 2.0) - sqrt(((sxx - syy) / 2.0) * ((sxx - syy) / 2.0) + sxy * sxy);
  differential = smax - smin;

  stat = fopen("Transition", "a");

  fprintf(stat, " differential stress:");
  fprintf(stat, " %f", differential);
  fprintf(stat, " time:");
  fprintf(stat, " %f\n", time);
  fprintf(stat, " elast. energy diff.:");
  fprintf(stat, " %f\n", eEl_diff);

  fclose(stat);		// close file
}



// gaussian distribution for the youngs-moduli of grains, additional function
void
Min_Trans_Lattice::SetGaussianYoungDistribution_2(double g_mean,double g_sigma)
{

  float prob;     // probability from gauss
  float k_spring;   // pict spring
  int i,j;       // counters
  float ran_nb;    // rand number

  // grain_counter counter how many grain there were intially
  // times two is for security, not all grains may be there
  // this can certainly be made nicer

  runParticle = &refParticle;

  //initialize random-number generator:
  std::srand(std::time(0));

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

                  break;
                }
            }
        }
      while(1);  // loop until break

      runParticle = runParticle->nextP;
    }

  for (i=0; i<numParticles; i++)
    {
      runParticle->young *= runParticle->rate_factor;

      for (j=0; j<6; j++)
        {
          if (runParticle->springf[j])
            runParticle->springf[j] *= runParticle->rate_factor;
        }

      runParticle = runParticle->nextP;
    }
}



// convieniance function for heat flow-example
void Min_Trans_Lattice::HeatGrain(int T, int nb)
{
  int i;

  runParticle = &refParticle;

  for (i=0; i<numParticles; i++)
    {
      if (runParticle->grain == nb)
        {
          runParticle->temperature = T;
          Set_Heat_Lattice(runParticle);
        }

      runParticle = runParticle->nextP;
    }
}



// convieniance function for heat flow-example
void Min_Trans_Lattice::SetHeatLatticeHeatFlowExample()
{
  double delta_x, delta_y,Time_Step_Heat;
  int i;

  Time_Step_Heat = 0.5;

  runParticle = &refParticle;
  runParticle = runParticle->prevP;
  delta_y = runParticle->ypos /  org_wall_pos_y;

  runParticle = runParticle->prevP;
  delta_x = runParticle->xpos / org_wall_pos_x;

  heat_distribution.initial_diameter = initial_diameter;

  heat_distribution.set_Parameters(delta_y, delta_x,
                                   Time_Step_Heat);

  HeatGrain(1500, 3);

  HeatGrain(1500, 5);

  runParticle = &refParticle;

  for (i=0; i<numParticles; i++)
    {
      heat_distribution.set_cell_temperature(runParticle->xpos,runParticle->ypos,runParticle->temperature);
      //       if (runParticle->temperature == 1500)
      //       	cout << "1500" << endl;

      runParticle = runParticle->nextP;
    }

}
