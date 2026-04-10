/******************************************************
* Spring Code Mike 2.0
*
* Class Phase_Lattice in phase_lattice.h
*
* Basic header for Phase-transition Class Phase_Lattice
* inherits Phase_Base that inherits the Lattic class
*
* Daniel and Jochen, March 2003
* Mainz
*
* Daniel Dec. 2003
*
* Daniel Koehn and Till Sachau 2004/5
******************************************************/
#ifndef _E_phase_lattice_h
#define _E_phase_lattice_h

#include "phase_base.h"

class Phase_Lattice : public Phase_Base
  {
  public:


    float increase_rate_factor;	// not used at the moment
    float water_compres;		// water compressibility
    float box_conc;			// concentration for the whole box

    float concBox[800000];		// concentration box in the background
    int fluid_volume;		// fluid volume
    int old_fl_vol;			// old fluid volume
    int max_y;			// size of conc Box
    float count_time;		// additional time counter

    int overall_dis_time;		// damping for reactions


    Phase_Lattice();                           // Constructor
    ~Phase_Lattice()
    {}
    ;                        // Destructor

    //------------------------------------------------------------------
    // usr functions for the intialization
    //------------------------------------------------------------------

    // set a fluid pressure in MPa

    void Set_Fluid_Pressure(float pressure);

    // dissolve a row between xmin and xmax

    void DissolveXRow(float xmin, float xmax);

    // dissolve a row between ymin and ymax

    void DissolveYRow(float ymin, float ymax, bool remove);
		

	void DissolveYRowSinus(float ymin, float ymax, bool remove);

    // create a circular hole with radius and shift it horizontally

    void Make_Circular_Hole(float radius, float shift);

    // create an ellipse and shift it horizontally, height in particles

    void Make_Elliptical_Hole(float height, float ratio, float shift);

    // set a linear distribution on the surface energies

    void Set_Distribution_Surface_Energy(float distribution, float springdis);

    // set a diffusion constant

    void Set_Diffusion_Constant(float dif_constant);

    // start to use a concentration

    void Set_Concentration();

    // set an initial concentration

    void Set_Concentration_Hole(float concentration);

    // set a gravitational gradient

    void Set_Grav_Grad(float read_grav_grad);

    // set a gaussian distribution of reaction rates

    void SetGaussianRateDistribution(double g_mean,double g_sigma);

    // change rate of boundaries

    void Set_RateFactor(float factor);

    // set a bimodal distribution of reaction constants and also elastic constant

    void Set_Rate_Two_Phase(float ratio, float factor, float elastic);

    // set a distribution on the molecular volume

    void Set_Distribution_Particle_MV(float distribution);

    // activate all elle grain boundaries

    void Active_Grain_Boundaries();

    // activate all particles

    void All_Active();
	
	void SetFluidLatticeGasRandom(float percent);
	
	void SetWallsLatticeGas(int grain);
	
	void InsertFluidLatticeGas(float x, float percent);
	
	void RemoveFluidLatticeGas(float x);
	
	void SetFluidLatticeGasRandomGrain(float percent,int nbr);
	
	void UpdateFluidLatticeGas();

    // set a linear distribution on the mass of particles

    void Set_Mass_Distribution(float mass_size, float factor);

    // adjust concentration of hole particles -> averages concentration

    void Adjust_Concentration();

    // average concentration of the concentration box

    void Adjust_Conc_Box(int change);

    // set a time for how often particles can change -> damping

    void Set_Dis_Time(int time);

    // Initialize the concentration box

    void Make_Concentration_Box(float ymax, float concentration);

    // dump statistics

    void DumpStressRow(double y_box_min,double y_box_max, double x_box_min,double x_box_max, int xrow);

    void DumpYStressRow(double y_box_min,double y_box_max, double x_box_min,double x_box_max, int xrow);


    //------------------------------------------------------------------
    // usr functions for the run, reactions
    //------------------------------------------------------------------

    // dissolution as a function of the elastic and surface energy (for free surfaces)

    void Dissolution_Strain(int dump);

    // dissolution as a function of elastic and surface energies and normal stress

    void Dissolution_Strain_Stress(int dump);

    // dissolution as a function of elastic,surface energy and normal stress, use for stylolites

    void Dissolution_Stylos(int dump, int stress, int shrink);
	
	
	void Dissolution_StylosII(int dump, int stress, int use_grains, int shrink, int sizex);

    // dissolution and growth as a function of normal stress, elastic energy, surface energy and concentration

    void GrowthDissolution(int dump, int use_Box, int change, int experiment_time);


  };

#endif
