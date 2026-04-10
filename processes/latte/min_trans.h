/******************************************************
* Spring Code Mike 1.1
*
* Class Phase_Lattice in phase_lattice.h
*
* Basic header for Phase-transition Class Phase_Lattice
*
* Daniel and Jochen, March 2003
* Mainz
*
* Daniel Dec. 2003
******************************************************/
#ifndef _E_min_trans_lattice_h
#define _E_min_trans_lattice_h

#include "heat_lattice.h"
#include "phase_base.h"
#include "min_trans.h"
#include "phase_lattice.h"

class Min_Trans_Lattice : public Phase_Lattice
{
public:

  //till-variablen
  double actual_time,
  now,
  time_interval,
  old_time;
  bool                 timeflag,         //multipurpose
  reaction_finished,                  //flag for heat-conduction and time-management in gbm()
  logicalflag,
  nucleation_occured;
  float           mole_per_particle,
  highest_prob,
  lowest_prob;
  float pressure,
  org_wall_pos_y,
  org_wall_pos_x;

  int beware_grain1,
  beware_grain2,
  beware_grain3,
  count_steps;

  bool set_act_energy, set_pressure_barrier;   	// whether the activation-energy and the pressure-barrier for the transformation will be given by the user
  double act_energy, pressure_barrier;				// activation_energy and pressure barrier as set by user

  bool no_heat_flag;

  Heat_Lattice heat_distribution; // the boundary-condition

  Particle *nucleus, *help_pointer;

  Min_Trans_Lattice();					// Constructor
  ~Min_Trans_Lattice() {};				// Destructor

  void Activate_MinTrans();

  // ------------------------------------------------------------------
  // usr functions for the run, reactions
  // ------------------------------------------------------------------

  void Reaction_One();

  void Reaction_Two();

  void Start_Reactions();

  void Energy_Calculation(Particle *partikel);

  void Provisoric_Mineral_Transformation();

  void Mineral_Transformation();

  void Make_Phase_Boundaries();

  void Heat_Flow(float);

  void Find_Local_Maxima();

  void Read_Heat_Lattice();

  void Nucleation();

  void Gbm();

  float Pressure();

  float Latent_Heat_Release();

  void Set_Heat_Lattice(Particle*);

  float Adjust_Molare_Volume();

  void Invert_Provisoric_Mineral_Transformation();

  void Save_Heat_Lattice();

  void	Invert_Heat_Lattice();

  void Change_Grain_Mineral(int, int, int);

  void Change_Timestep();

  void Exchange_Probabilities();

  void Call_Reaction_Two();

  float Activation_Energy();

  float Growth_Rate_For_Nucleation();

  double Nucleation_Rate();

  float Normal_Stress(Particle *fixed_neig, int);

  void Change_Young();

  void Restore_Young();

  float Excess_Pressure();

  float Undercooling();

  float Energy_Barrier();

  float Driving_Force();

  void Check_Nucleus_Stability();

  void Update_Surf_Neighbour_List();

  float CalcTime();

  float CalcArea(float);

  void DumpDiffStressAndTime(Particle *,float,float);

  void SetGaussianYoungDistribution_2(double,double);

  void Change_Particle();

  void SetHeatFlowEnabled(int);

  void SetHeatFlowParameters(float,float,float,float);

  void HeatGrain(int,int);

  void SetHeatLatticeHeatFlowExample();

  void SetReactions(double, double);
  
};

#endif
