/******************************************************
 * Lattice Spring Code 2.0
 *
 * Class Lattice in lattice.h
 *
 * Basic header for Fracture Class Lattice
 *
 * Daniel and Jochen oct. 2001
 * Daniel and Jochen Feb 2002 to Feb 2003
 * Oslo, Mainz
 *
 * Daniel Dec. 2003
 *
 * We thank Anders Malthe-Srenssen for his enormous help
 * and introduction to these codes
 *
 * new 2004/2005
 ******************************************************/
#ifndef _E_lattice_h
#define _E_lattice_h


#include "particle.h" // include the particle class so we can have pointers of type particle
#include "heat_lattice.h"  // header for the heat diffusion class

/******************************************************
 * the lattice class is the highest class at moment
 * is called from the main function (well, in elle the mike_elle.cc function
 * it will build the lattice in its constructor including
 * the particle list. This lists can be 
 * reached from lattice which has Runpointers and pointers 
 * pointing to the beginning and end of list. In order
 * to reach it have to refer to a refParticle.  The list is 
 * circular, each member points to the next in the list and 
 * last member points to the first one, and also backwards. 
 * the lattice class can also make a full relaxation and
 * can deform i.e. apply boundary conditions
 * Should also define the walls in this class, not done yet 
 * 
 * introduced several deformation times in order to preload
 * system, daniel March 4 2003
 *******************************************************/

class Lattice
{
    public:
	//------------------------------------------------------------------------------
	// first define the variables for the class that are important for all
	// the functions used for the class
	//------------------------------------------------------------------------------

	float dimx;      // lattice x dimension, 1.0 as default
	float dimy;      // lattice y dimension, 1.0 as default
	float xyratio;   // xy ratio depending on lattice type
	int nbBreak;       // counts broken springs in Lattice
	int local_nbBreak;  // local counter for fracture plot
	int internal_break;   // counter for dumping fracture pictures
	int def_time_p;       // time counter for steps of progressive deformation (pure shear)
	int def_time_u;       // time counter for steps of progressive def. (uniaxial compression)
	int fract_Plot_Numb;  // after how many fracture do I plot
	float relaxthres;  // relaxation threshold
	int layer[10000];  // numbers of grains or Flynns in a layer

	bool set_max_pict;  // set a maximum (default false)
	int max_pict;	// maximum number of pictures dumped
	int num_pict;	// current number of pictures dumped (counter)

	float Pi;  		// number pi
	
	int visc_rel;   //flag for viscous relax routines in relaxation

	float pressure_scale;  	// scale for pressure (at moment 1 = 10 GPa)
	float pascal_scale;		// scales MPa to Pascal

	double area_box;    // area of whole box for pure shear

	int particlex;   // number of particles in x
	int particley;   // number of particles in y
	
	float boundary_strength;
	float boundary_constant;
	
	//--------------------------------------------------------------------------------
	// walls of the lattice 
	//--------------------------------------------------------------------------------
	
	bool walls;
	float wall_rep;
	float upper_wall_pos;
	float lower_wall_pos;
	float right_wall_pos;
	float left_wall_pos;

	int grain_counter; // initial grain number

	float polyg_area;	// area for viscous step
	float polyg_area8;       // area for the octagon (BoxRad())

	bool adjust; //for debugging only

	bool debugflag, heat;

	int debug_nb;
	//-----------------------------------------------------------------------------
	// and define some pointers
	//-----------------------------------------------------------------------------

	Particle *firstParticle;    //pointer to particle list begin
	Particle *lastParticle;     //pointer to last particle in list
	Particle *runParticle;     // run-pointer for list
	Particle *preRunParticle;  // 2. run-pointer

	//---------------------------------------------------------------------------
	// these arrays have to have the same dimensions !
	//---------------------------------------------------------------------------

	Particle *repBox[2500000];  // The neighbour box for Repulsion

	int node_Box[2500000];      // neighbour box for nodes
	
	float grain_young[100000];  // saves youngs moduli of grains

	//-----------------------------------------------------------------
	//  maximum Elle nodes
	//-----------------------------------------------------------------

	int next_inBox[1000000];   // vector for nodes to save node number for node box

	bool nodes_Check[1000000];  // node vector for node check if connected to lattice

	//-----------------------------------------------------------------------------
	// and define a reference particle. this will be the first particle in the list
	// note that this has to change if that particle is deleted for whatever
	// reason
	//-----------------------------------------------------------------------------

	Particle  refParticle;    // the reference particle

	int numParticles;      // number of particle

	int highest_grain;    // highest grain number

	bool visc_flag, transition;



	//------------------------------------------------------------------------------
	// now define the functions
	//------------------------------------------------------------------------------

	Lattice();                           // Constructor
	~Lattice()
	{}
	;                        // Destructor

	void    Activate_Lattice();
	void 	MakeLattice(int type);          // build lattice
	void 	InitiateGrainParticle();        // find grains,boundaries,nodes etc.
	void 	MakeRepulsionBox();             // build repulsion box
	void 	SetBoundaries();                // boundary conditions
	void 	FindUnodeElle();                // connect to Unodes
	bool 	FullRelax();                    // simple full relaxation(overrelaxation)
	void 	Relaxation();                   // Relaxation
	int 	HighestGrain();			// find highest grain in Elle
	void 	UpdateElle();                   // talk a bit to Elle
	void  	UpdateElle(bool movenode);
	int  	TimeStep;
	
	//------------------------------------------------------------------
	// functions for external use called from mike.elle
	//------------------------------------------------------------------
		
	// set wall boundaries 	
		
	void SetWallBoundaries(int both, float constant);
		
	// set a sinusoidal variation of youngs moduli
		
	void SetSinAnisotropy(float nb, float ampl);
	
	// reread the elle file 

	void GetNewElleStructure();

	// security stop for pict dump

	void Set_Max_Pict(int max);
	
	void ChangeYoung(float change);

	// changes grain constant and tensile strength

	void WeakenGrain(int nb,float constant, float visc, float break_strength);

	// changes grainboundary constant and tensile strength

	void MakeGrainBoundaries(float constant, float break_strength);

	// function to adjust modulus of grainboundary springs

	void AdjustConstantGrainBoundaries();

	// function to adjust constants of particles so that springs have mean value of neighbours

	void AdjustParticleConstants();

	// plot after certain amount of fractures or not

	void SetFracturePlot(int numbfrac, int local);

	// Boundary conditions deform from upper side, do an average (uniaxial compression)

	float DeformLattice(float move, int plot);

	// deform volume constant with average

	float DeformLatticePureShear(float move, int plot);

	// deform volume not constant

	float DeformLatticePureShearAreaChange(float move, float area_change, int plot);

	// Deform Lattice without an average

	float DeformLatticeNoAverage(float move, int plot);

	// Deform Lattice from upper and lower boundaries, no average

	float DeformLatticeNoAverage2side(float move, int plot);

	// Deform Lattice from upper and lower boundaries, do an average

	float DeformLatticeNewAverage2side(float move, int plot);

	// do a viscous retardation step

	void ViscousRelax(int dump, float timestep);

	// functions for the viscous retardation step

	void SetSpringRatio();

	double VirtualNeighbour(int spring, int xy);

	void Adjust_Radii();   // adjust radii so that area is kept constant

	void Copy_NeigP_List(); // help function

	void BoxRad();

	// change the relaxation threshold

	void ChangeRelaxThreshold(float change);

	/*********************************************************
	 * CHANGES AND DISTRIBUTIONS
	 *********************************************************/

	// shrink or grow one particle

	void ChangeParticle(int number, float shrink, int plot);

	// shrink particle that is stressed most (highest mean stress)

	void ChangeParticleStress(float shrink, int plot);

	// shrink or expand a whole grain

	void ShrinkGrain(int nb, float shrink, int plot);

	// shrink the whole box

	void ShrinkBox(float shrink, int plot, int yes);

	// define layer using Elle grains

	void WeakenHorizontalLayer(double y_min,double y_max,float constant, float break_strength);

	// horizontal layer using only particles, independent of elle grains

	void WeakenHorizontalParticleLayer(double y_min,double y_max,float constant, float vis,float break_strength);

	// a whole row of even horizontal layers

	void MakeHorizontalLayers(float young_a,float viscous_a, float break_a, float thick_a, float young_b, float viscous_b, float break_b, float thick_b, int av);

	// make a tilted layer starting at ymin to ymax

	void WeakenTiltedParticleLayer (double y_min, double y_max, double shift,float constant, float vis,float break_strength);
	
	// make a tilted layer starting at xmin to xmax
	
	void WeakenTiltedParticleLayerX (double x_min, double x_max, double shift,
					float constant, float vis);

	// change constants of all particles

	void WeakenAll(float constant, float viscosity, float break_strength);

	// set a horizontal anisotropy using small particles rows of 1 - 3 particles

	void SetAnisotropy(float break_strength, float youngs_mod, float viscosity, float ratio, float length);
	
	// set a horizontal anisotropy, little mica flakes with 3 to 7 particle length and varying youngs moduli
	
	void SetAnisotropyRandom(float max, float boundary);

	// Change the Young modulus of grains by a factor

	void Change_Young(float young_factor);

	// sets a Gaussian distribution on spring constants of whole grains

	void SetGaussianSpringDistribution(double g_mean,double g_sigma);

	// set a Gaussian distribution on breaking stengths of all springs 

	void SetGaussianStrengthDistribution(double g_mean,double g_sigma);
	
	// set a Gaussian distribution on spring constant of all particles 
	
	void SetGaussianYoungDistribution (double g_mean, double g_sigma);

	// sets pseudorandom distribution of spring constants of grains and of breaking strength of all springs

	void SetPhase(float str_mean, float str_size,float br_mean, float br_size);

	/*********************************************************
	 *  STATISTICS DATADUMPS
	 *********************************************************/

	// write out the x and y coordinates plus strain of particles on the surface

	void DumpStatisticSurface(double strain);

	// write out x and y coordinates plus strain of particles on surface in a region

	void DumpTimeStatisticSurface(double strain, double xmin, double xmax);

	// function to dump statistics of a single grain

	void DumpStatisticStressGrain(double strain,int nb);

	// function to dump statistics of two grains

	void DumpStatisticStressTwoGrains(double strain,int nb1,int nb2);

	// function to dump statistics of a predefined box

	void DumpStatisticStressBox(double y_box_min,double y_box_max, double x_box_min,double x_box_max, double strain);

	// additional functions for the viscoelastic part, hidden 

	float Alen(int i,int j);
        
	void SetSpringAngle();

	void UpdateNeighList();

	void EraseNeighList();

	void UpdateSpringLength();

	void Tilt();
};

#endif
