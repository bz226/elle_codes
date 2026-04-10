/******************************************************
 * New Experiment Class for Elle/Latte 
 *
 * works now mainly with the lattice spring code
 *
 * Daniel and Till 2005
 * 
 * Latte Version 2.0 
 * koehn_uni-mainz.de
 ******************************************************/

// ------------------------------------
// system headers
// ------------------------------------

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <locale.h>
// ------------------------------------
// elle headers
// ------------------------------------

#include "experiment.h"
#include "unodes.h"		// include unode funct. plus undodesP.h
// for c++
#include "attrib.h"		// enth�lt struct coord (hier: xy)
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

// have to define these in the new Elle version

using std::cout;
using std::endl;
using std::vector;

// CONSTRUCTOR
/*******************************************************
 * Dont really construct anything here at the moment
 *
 * The experiment class is directly called from the 
 * Elle main function
 ********************************************************/

// ---------------------------------------------------------------
// Constructor of Experiment class
// ---------------------------------------------------------------

Experiment::Experiment()

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------


{
  cout << "Oh, my experiment starts !" << endl;
setlocale(LC_ALL,"en_US");
  experiment_time = 0;
}

/*************************************************************
* Now we start with the initialization function 
* This function initialized some Elle basics
* reads input file if one is there, 
* opens the interface
* and starts the local initialization functions for the 
* desired process
* 
*************************************************************/

void Experiment::Init()
{
  //-------------------------------------------------
  // local variables
  //-------------------------------------------------

  char *infile;  // input
  int err=0;     // pass errors
  int i, j;         // counter
  UserData udata;  // Elle Structure for data from input
  int process;  // variable for the data (which process)

  //*-----------------------------------------------
  //* clear the data structures
  //*-----------------------------------------------

  ElleReinit();

  //*------------------------------------------------
  //* read the data
  //*-----------------------------------------------

  infile = ElleFile();	// input file specified by -i
  file = infile;
  ElleUserData(udata);  // reads in data from the initial call of the program behind -u
  process = (int)udata[0];  // get this data from Elle

  //----------------------------------------------------
  // processes are defined by simple integers
  //----------------------------------------------------


  if (strlen(infile)>0)
    {
      //--------------------------------------------
      // only go in here if an input file is there
      //--------------------------------------------

      if (err=ElleReadData(infile))
        OnError(infile,err);

      //--------------------------------------------------
      // the switch function is used for manual input.
      // The processes can all be called from the
      // new interface.
      //
      // Without interface specify a process using the
      // -u command after calling the experiment.
      // ./my_experiment -u 1
      // -u 1 fracturing
      // -u 2 fracture boudinage
      // -u 3 expanding inclusions
      // -u 4 shrinkage cracks
      // -u 5 viscoelastic deformation
      // -u 6 grooves
      // -u 7 Stylolites
      // -u 8 combine graingrowth and fractures
      // -u 9 solid solid phase change
      // -u 10 heat flow
      // -u 11 pure grain growth
      // -u 12 Lattice gas diffusion
      // -u 13 Lattice gas fluid flow
      //--------------------------------------------------

      switch (process)
        {

          /*********************************************************************
          *    
          *	FRACTURE PROCESSES 
          * 
          **********************************************************************/
        case 0:
          SetUpFromFile();
          break;

        case 1: // fracturing
          cout << "Fracturing" << endl;
          cout << "Lattice Version 2.0, 2004/5" << endl;
          Activate_Lattice();  // construct the lattice
          SetPhase(0.0,0.0,2.0,1.2);	// set a distribution of breaking strengths (linear)
          SetGaussianSpringDistribution(1.0, 0.5); // set a distribution of Youngs Moduli
          MakeGrainBoundaries(1.0, 0.5);		// Make grain boundaries weaker
          SetFracturePlot(50,1); 	// plot fractures after 50 bonds broken
          break;

        case 2:  // fracture boudinage
          cout << "Fracture Boudinage" << endl;
          cout << "Lattice Version 2.0, 2004/5" << endl;
          Activate_Lattice(); // contruct the lattice
          SetGaussianStrengthDistribution(2.0,0.8);  // set a distribution of breaking strengths (gaussian)
          WeakenAll(0.1,1.0,1.0); // Lower Youngs modulus of all grains
          WeakenHorizontalParticleLayer(0.20,0.60,10.0,1.0,1.0); // Make a layer at (ymin, ymax, Youngs Modulus, Viscosity, breaking strength)
          WeakenHorizontalParticleLayer(0.9,0.92,10.0,1.0,1.0); // same as above
          SetFracturePlot(50,1); // plot fractures after 50 bonds are broken
          break;

        case 3:  // expanding inclusions
          cout << "Expanding Inclusions" << endl;
          cout << "Lattice Version 2.0, 2004/5" << endl;
          Activate_Lattice(); // construct the lattice
          SetPhase(0.0,0.0,0.8,1.2); // Linear distribution of breaking strengths
          SetFracturePlot(20,1); // plot fractures after 5 bonds are broken
          break;

        case 4: // shrinkage patterns
          cout << "Shrinkage Patterns" << endl;
          cout << "Lattice Version 2.0, 2004/5" << endl;
          Activate_Lattice(); // construct the lattice
          WeakenAll(2.0,1,1.0); // Weaken Youngs modulus of all grains
          SetPhase(0.0,0.0,1.0,1.0); // Distribution of breaking strengths (linear)
          //SetSinAnisotropy(20, 1.2); // Sinusoidal anisotropy (vertical) of Youngs Moduli
          //SetGaussianStrengthDistribution(1.0,0.35);  // Distribution of breaking strengths (gaussian)
          SetFracturePlot(20,1); // plot fractures after 5 bonds are broken
          break;
          /**************************************************************************
          *
          *		VISCOELASTIC DEFORMATION AND FRACTURING
          *
          *************************************************************************/

        case 5: // viscoelastic
          cout << "visco-elastic" << endl;
          cout << "Lattice Version 2.0, 2004/5" << endl;
          Activate_Lattice();   // Construct the lattice
          //--------------------------------------------------
          // Set external walls around the box. No particles
          // can now leave the box.
          // External walls are only compressive
          //--------------------------------------------------
          SetWallBoundaries(1,1.0);
          //--------------------------------------------------
          // Set a distribution on the breaking strengths of
          // all springs. Distribution can be Gaussian
          // around a mean strength or linear between
          // two endmembers. Values are distributed randomly
          //--------------------------------------------------
          //SetGaussianStrengthDistribution(2.5,0.8); // Set a Gaussian distribution with a mean and a variance
          SetPhase(0.0,0.0,1,0.6); // Set a linear distribution around a mean value and with a certain width
          //--------------------------------------------------
          // Change youngs modulus, Viscosity and breaking
          // Strength of all particles
          // Youngs modulus of 1 is default, Scales as
          // 10 GPa in the model (this value is then a
          // reference).
          // Viscosity is by default e^20 if Youngs modulus 1
          // in the model is 10 GPa in reality.
          //--------------------------------------------------
          WeakenAll(0.1,1,1.0);
          //----------------------------------------------------------------------------------
          // Change some initial values in order to have an anisotropy
          // a) Change values of one grain (or more)
          // b) induce a horizontal anisotropy in the form of hard flakes with varying Moduli
          // c) Insert a horizontal weaker or stronger layer
          //----------------------------------------------------------------------------------
          //WeakenGrain(7,10,10,1); // Change Youngs modulus and viscosity of grain nb 7
          //SetAnisotropyRandom(5,5); // insert a horizontal anisotropy of mica grains
          WeakenHorizontalParticleLayer(0.47,0.52,5.0,5.0,1.0); // Insert a horizontal layer
          //------------------------------------------------------------------------------
          //ChangeRelaxThreshold(0.1);//factor, 1=normal
          //SetFracturePlot(5,1);
          break;
          /*****************************************************************************
          *
          *		REACTIONS 
          *
          ******************************************************************************/

        case  6:  // grooves on free surfaces
          cout << "Dissolution Grooves" << endl;
          cout << "Phase_Lattice Version 2.0, 2004/5"<< endl;
          Activate_Lattice(); // construct the lattice
          SetPhase(0.0,0.0,500.0,0.8);  // impossible to break (strength * 500)
          SetGaussianRateDistribution(2.0,0.3); // Distribution on rate constants of reaction (gaussian)
          WeakenAll(8.0,1.0,1.0);  // Change Youngs modulus of all particles
          Set_Mineral_Parameters(1);  // define a mineral here Quartz
          Set_Absolute_Box_Size(0.00008);  // Set the absolute Elle box size in meters
          Set_Time(6000.0,4);  // set the time (here 6000 years)
          DissolveXRow(0.95,1.1);   // dissolve particles > xpos 0.95 to have a free interface on the right side
          SetWallBoundaries(0,15.0);
          break;

        case 7: // Stylolites
          cout << "Stylolite Roughening" << endl;
          cout << "Phase_Lattice Version 2.0, 2004/5"<< endl;
          Activate_Lattice();
          //----------------------------------------------------
          // give bonds very high strength (*500) to avoid
          // in this case fracturing during Stylolite growth
          //----------------------------------------------------
          SetPhase(0.0,0.0,500.0,0.8);
          //----------------------------------------------------
          // Set a Gaussian distribution on the rate constants
          // of single particles, first mean value, second
          // deviation
          //----------------------------------------------------
          //SetGaussianRateDistribution(2.0,0.1);
          Set_Rate_Two_Phase(0.05,0.6,1.0);
          //----------------------------------------------------
          // change the Youngs modulus of particles to make
          // them stiffer (*4.0). Second and third number are
          // breaking strength and viscosity, 1.0 means no
          // change of these parameters
          //----------------------------------------------------
          WeakenAll(4.0,1.0,1.0);
          //----------------------------------------------------
          // Set some mineral paramters, 1 means quartz as
          // mineral, sets the molecular volume and the
          // surface free energy
          //----------------------------------------------------
          Set_Mineral_Parameters(1);
          //----------------------------------------------------
          // gives the x dimension of the Elle box in meters
          //----------------------------------------------------
          Set_Absolute_Box_Size(0.1); //was 0.1
          //----------------------------------------------------
          // set the time for one deformation step. 6000 years
          // 4 means years
          //----------------------------------------------------
          Set_Time(40.0,4);  //was 40,4
          //----------------------------------------------------
          // dissolve initially one horizontal row of particles
          // in the middle of the Elle box
          // numbers are min and max y value
          //----------------------------------------------------
          //DissolveYRowSinus(0.49,0.5,true); //was 0.496
		  DissolveYRow(0.49,0.5,true); //was 0.496
		  ChangeRelaxThreshold(0.1);
		  //ChangeYoung(2);
          break;

        case 8:
          cout << "Combine Latte and GrainGrowth" << endl;
          cout << "Phase_Lattice Version 2.0, 2004/5"<< endl;
          Activate_Lattice();  // Construct the lattice
          SetPhase(0.0,0.0,200.0,1.0);  // Set breaking strength distributio
          SetGaussianSpringDistribution(0.5, 0.5);  // set distribution on youngs moduli
          MakeGrainBoundaries(1.0, 0.5);  // define grain boundaries
          SetFracturePlot(1,0);  // plot every fracture
          ElleAddDoubles();  // add some doubles for grain growth (Elle function)
          break;

        case 9: //phase change
          cout << "solid solid phase transformation, slow reaction, no distribution" << endl;
          Activate_MinTrans();
          SetReactions(450000.0, 12e9); // sets the activation energy (J/mol) for the reaction and the pressure barrier where the grain-boundary migration will start (Pa). Also calls setheatflowparameters() for olivine
          heat_distribution.SetHeatFlowParameters(4.2, 1005.0, 0.000001, 1000.0); // rho, c, diffusivity, boundary_condition (in °K)
          WeakenAll(20.0, 0.0, 5000.0);
          AdjustConstantGrainBoundaries();
          MakeGrainBoundaries(1.0,0.8);
          Set_Mineral_Parameters(3);
          Set_Absolute_Box_Size(0.005);
          Set_Time(120, 3); //(x,2):0=sek, 1=Stunden, 2=Tage, 3=Monate, 4=Jahre
          break;

        case 10:
          cout << "Heat Flow" << endl;
          Activate_MinTrans();   // contruct lattice for phase transformations
          SetPhase(0.0,0.0, 500.0,0.8); //spring constant wurde justiert (500.0 statt 0.0)
          Set_Absolute_Box_Size(0.01);  // Absolute box size in meters
          //SetHeatLatticeHeatFlowExample();
          heat_distribution.SetHeatFlowParameters(4.2, 1005.0, 0.000001, 1000.0); // rho, c, diffusivity, boundary_condition (in °K)
          HeatGrain(1500,3);
          break;

        case 11:
          cout << "pure grain growth" << endl;
          ElleAddDoubles();  // add doubles (Elle functions)
          // ElleUpdate();      make a picture
          ElleUpdateDisplay();      // make a picture
          break;

        case 12:
          cout << " Lattice Gas" << endl;
          Activate_Lattice();  // contruct the lattice
          SetFluidLatticeGasRandom(0.01);  // background contains some particles randomly distributed (1%)
          /*********************************************************
          * Specify some grains that have a concentration of 70 %
          *********************************************************/
          for (j = 0; j < HighestGrain(); j++)
            {
              SetFluidLatticeGasRandomGrain(0.4,j*10);
            }
          break;

        case 13:
          cout << " Lattice Gas Flow" << endl;
          Activate_Lattice();  // contruct the lattice
          SetFluidLatticeGasRandom(0.005);  // fluid density in the background (0.5 %)
          /*********************************************************
          * Specify some grains that are fracture walls 
          *********************************************************/
          for (j = 0; j < HighestGrain(); j++)
            {
              SetWallsLatticeGas(j*5);
            }
          break;

        case 15:
          Activate_Lattice();
          SetWallBoundaries(1,1);
          SetGaussianRateDistribution(2.0,0.3);
          //----------------------------------------------------
          // change the Youngs modulus of particles to make
          // them stiffer (*4.0). Second and third number are
          // breaking strength and viscosity, 1.0 means no
          // change of these parameters
          //----------------------------------------------------
          WeakenAll(8.0,1.0,100.0);
          //----------------------------------------------------
          // Set some mineral paramters, 1 means quartz as
          // mineral, sets the molecular volume and the
          // surface free energy
          //----------------------------------------------------
          Set_Mineral_Parameters(1);
          //----------------------------------------------------
          // gives the x dimension of the Elle box in meters
          //----------------------------------------------------
          Set_Absolute_Box_Size(0.0001);
          //----------------------------------------------------
          // set the time for one deformation step. 6000 years
          // 4 means years
          //----------------------------------------------------
          Set_Time(60.0,4);
          DissolveYRow(0.0,0.2,false);
          //DissolveXRow(0.8,1.1);
          //----------------------------------------------------
          // dissolve initially one horizontal row of particles
          // in the middle of the Elle box
          // numbers are min and max y value
          //----------------------------------------------------
          Set_Fluid_Pressure(10);
          Set_Concentration();
          Make_Concentration_Box(1.0, 2);
          Set_Dis_Time(40);
          break;

        case 16:
          Activate_Lattice();
          SetWallBoundaries(0,0.1);
          SetGaussianRateDistribution(2.0,0.01);
          //----------------------------------------------------
          // change the Youngs modulus of particles to make
          // them stiffer (*4.0). Second and third number are
          // breaking strength and viscosity, 1.0 means no
          // change of these parameters
          //----------------------------------------------------
          WeakenAll(8.0,1.0,100.0);
          //----------------------------------------------------
          // Set some mineral paramters, 1 means quartz as
          // mineral, sets the molecular volume and the
          // surface free energy
          //---------------------------------------------------
          Set_Mineral_Parameters(2);
          //----------------------------------------------------
          // gives the x dimension of the Elle box in meters
          //----------------------------------------------------
          Set_Absolute_Box_Size(0.0001);
          //----------------------------------------------------
          // set the time for one deformation step. 6000 years
          // 4 means years
          //----------------------------------------------------
          Set_Time(6.0,2);
          //DissolveYRow(0.0,0.2,false);
          DissolveXRow(0.8,1.1);
          //----------------------------------------------------
          // dissolve initially one horizontal row of particles
          // in the middle of the Elle box
          // numbers are min and max y value
          //----------------------------------------------------
          Set_Fluid_Pressure(0.01);
          Set_Concentration();
          Make_Concentration_Box(1.1, 1);
          Set_Dis_Time(40);
          break;

        case 17:
          cout << "solid solid phase transformation, fast reaction, no distribution" << endl;
          Activate_MinTrans();
          SetReactions(380000.0, 12e9); // sets the activation energy (J/mol) for the reaction and the pressure barrier where the grain-boundary migration will start (Pa). Also calls setheatflowparameters() for olivine
          heat_distribution.SetHeatFlowParameters(4.2, 1005.0, 0.000001, 1000.0); // rho, c, diffusivity, boundary_condition (in °K)
          WeakenAll(20.0, 0.0, 5000.0);
          AdjustConstantGrainBoundaries();
          MakeGrainBoundaries(1.0,0.8);
          Set_Mineral_Parameters(3);
          Set_Absolute_Box_Size(0.005);
          Set_Time(120, 3); //(x,2):0=sek, 1=Stunden, 2=Tage, 3=Monate, 4=Jahre
          break;

        case 18:
          cout << "solid solid phase transformation, high distribution" << endl;
          Activate_MinTrans();
          SetReactions(450000.0, 12e9); // sets the activation energy (J/mol) for the reaction and the pressure barrier where the grain-boundary migration will start (Pa). Also calls setheatflowparameters() for olivine
          heat_distribution.SetHeatFlowParameters(4.2, 1005.0, 0.000001, 1000.0); // rho, c, diffusivity, boundary_condition (in °K)
          SetPhase(20.0, 1.0, 5000.0, 0.000000001); //spring constant wurde justiert (500.0 statt 0.0)
          AdjustConstantGrainBoundaries();
          MakeGrainBoundaries(1.0,0.8);
          SetGaussianYoungDistribution_2(1.0,0.5);
          Set_Mineral_Parameters(3);
          Set_Absolute_Box_Size(0.005);
          Set_Time(120, 3); //(x,2):0=sek, 1=Stunden, 2=Tage, 3=Monate, 4=Jahre
          break;
		
        }
    }
  else
    cout << "no file open ! "<< endl; // no input file in function call, can be opened from the interface

  //UpdateElle();
	ElleUpdateDisplay();
  cout << "update display" << endl;
}


/******************************************************************
 * A runfunction for the Experiment
*
*	In this function each process (again specified by -u in a switch
*   command) is executed. 
*
*  Latte version 2.0, 2005/6
   ******************************************************************/



void Experiment::Run()
{
  //----------------------------------------------------
  // some local variables
  //----------------------------------------------------

  int i,j,k;        	// counter
  int time;     		// time
  int process;			// int for process
  UserData udata; 		// elle defined user data (-u )

  ElleCheckFiles();   	// Check the files

  ElleUserData(udata);		// get the usr data from elle
  process = (int)udata[0];	// process is user data 0

  ElleUpdateDisplay();

  //--------------------------------------------------
  // get the time from the interface
  //--------------------------------------------------

  time = EllemaxStages(); // number of stages


  //--------------------------------------------------
  // loop through the time steps
  //--------------------------------------------------

  for (i=0;i<time;i++)   // cycle through stages
    {
      cout << "time step" << experiment_time << endl;

      switch (process)
        {
        case 0:
          RunFromFile(experiment_time);
          break;

        case 1: // fracturing
          if (experiment_time < 15)
            {
              DeformLattice(0.001, 1);
            }
          else
            {
              DeformLatticePureShear(0.001,1);
            }
          break;

        case 2: // fracture boudinage
          DeformLatticePureShear(0.001,1);
          break;

        case 3: // expanding inclusions
          ShrinkGrain(5, -0.002, 1);
          break;

        case 4: // shrinkage patterns
          ShrinkBox(0.001, 1, 1);
          break;

        case 5: // viscos relax
          DeformLatticePureShear(0.001,1);
          ViscousRelax(1, 1e11); // strain 0.001 and 1e11 are strain rate 10^-12
          break;

        case 6:  // grooves
          DeformLattice(0.002,1);
          Dissolution_Strain(20);
          break;

        case 7: // Stylolites
          //-------------------------------------------------
          // Deform the lattice from upper and lower
          // boundaries. Upper and lower part of the lattice
          // are now just pressed together assuming
          // there is no resistance. Stresses build up once
          // the two sides meet. Deformation steps have to
          // be very small
          // The side walls are fixed. Movement is vertical
          // in steps of 0.00005 * y size (1.0)
          // second number means make a picture after the
          // movement. 0 means take no picture.
          //-------------------------------------------------
          DeformLatticeNewAverage2side(0.00005,1);
          //---------------------------------------------------
          // Dissolution routine. Particles are just dissolved
          // depending on stress, elastic and surface
          // energies. One particle is dissolved in a time
          // step. 100 means take a picture after 100 particles
          // have dissolved.
          //---------------------------------------------------
          Dissolution_StylosII(100000,0,0,0,1);
          break;

        case 8: // combined grain growth and fracturing
          DeformLatticePureShear(0.001,1);  // pure shear deformation strain in y direction is 0.1 %
          DoGrowth(i); // grain growth step, calls growth function in graingrowth.cc
          GetNewElleStructure();  // reread the Elle structure for Latte after grain growth
          break;

        case 9: //phase change
          DeformLattice(0.002, 1); // deform and relax //(0.0002,x): deformationsstep, ursprünglich 0.002
          heat_distribution.SetHeatFlowEnabled(1);
          //***float Lattice::DeformLattice(float move, int plot)***
          DumpStatisticStressBox(0.2,0.8,0.2,0.8,0.002);
          Start_Reactions();
          break;

        case 10:  // Heat Flow
          // elle->SetHeatFlowEnabled(1); // use this function only in context of phase change!!! Never together with the below called elle->Heat_Flow()
          // called in Min_Trans-class
          Heat_Flow(25);
          DeformLatticePureShear(0.005,1);
          break;

        case 11: // pure grain growth
          DoGrowth(i);  // do only grain growth (in graingrowth.cc)
          break;

        case 12: // diffusion
          UpdateFluidLatticeGas(); // run one lattice gas step (transport + collisions)
          break;

        case 13:	// fluid flow
          InsertFluidLatticeGas(0.03, 0.9); // pump in fluid at left boundary
          UpdateFluidLatticeGas();	// one lattice gas step (transport + collisions)
          RemoveFluidLatticeGas(0.97); // suck out fluid at right boundary
          break;

        case 15:
          //DeformLattice(0.001,1);
          DeformLatticeNoAverage(0.001,1);
          GrowthDissolution(20,1,1,experiment_time);
          break;

        case 16:
          DeformLattice(0.001,1);
          GrowthDissolution(1,1,0,experiment_time);
          break;

        case 17:
          DeformLattice(0.002, 1); // deform and relax //(0.0002,x): deformationsstep, ursprünglich 0.002
          heat_distribution.SetHeatFlowEnabled(1);
          //***float Lattice::DeformLattice(float move, int plot)***
          DumpStatisticStressBox(0.2,0.8,0.2,0.8,0.002);
          Start_Reactions();
          break;

		  case 18:
          DeformLattice(0.002, 1); // deform and relax //(0.0002,x): deformationsstep, ursprünglich 0.002
          heat_distribution.SetHeatFlowEnabled(1);
          //***float Lattice::DeformLattice(float move, int plot)***
          DumpStatisticStressBox(0.2,0.8,0.2,0.8,0.002);
          Start_Reactions();
          break;       
        }
      experiment_time ++;
      // LE I moved this from lattice
      // -----------------------------------------------------------------
      // call ElleUpdate() an Elle function that updates the interface
      // then the new values will be plotted
      // if security stop is set and max number of picts is reached dont
      // plot pict anymore.
      // -----------------------------------------------------------------
  if (!set_max_pict)
    {
      // cout << "interface" << endl;
      ElleUpdate ()
      ;
    }
  else if (num_pict < max_pict)
    {
      // cout << "interface" << endl;
      ElleUpdate ()
      ;
      max_pict = max_pict + 1;
    }
    }
}
