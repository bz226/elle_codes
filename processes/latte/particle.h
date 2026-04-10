/******************************************************
 * Spring Code Latte 2.0
 *
 * Class Particle in particle.h
 *
 * Daniel Koehn and Jochen Arnold Feb. 2002
 * Oslo, Mainz
 *
 * Daniel Dec. 2003
 *
 * We thank Anders Malthe-Soerenssen for his enormous help
 * and introduction to these codes
 *
 * new Daniel  2004/5
 * koehn_uni-mainz.de
 ******************************************************/
#ifndef _E_particle_h
#define _E_particle_h

#include "unodesP.h" // include Unode class
#include <cmath>

/******************************************************
 * the particle class for the particles
 * contains pointers to neighbours and springs
 * contains function that can relax single particles
 * and function that can connect particle with neighbours
 * called from the lattice constructor 
 *
 * Daniel spring 2002
 *******************************************************/

class Particle
{
    public:
	//------------------------------------------------------------
	// variables
	//------------------------------------------------------------

	int nb;                // my number
	double radius;          // my radius
	double rad[8];		// different initial length for different springs
	double newrad[8];	// second length for springs 
	int neig_spring[8];	// find backwards pointers of neighbours 
	int boundary;          // what is my boundary if any (not yet active)
	float mbreak;          // what is my max breaking probability
	int neigh;             // counter for which neighbour breaks easiest
	bool fix_x;            // am I fixed in x ?
	bool fix_y;            // am I fixed in y ?

	float ratio_a, add[9], ratio[8];
	
	int fluid_particle[8];      	// fluid particles in lattice gas
	int fluid_particle_new[8];		// fluid particles in lattice gas
	int fluid_particles;			// counter for fluid particles in lattice gas 
	
	bool nofluid;					// bool for fracture walls

	float	rep_rad[9];				// repulsion radius

   float viscosity;					// viscosity of particle

	float xx,yy;

	float increase_rate;
	float rate_factor;
	float spring_var;
	float dis_time;

	bool spring_boundary[8];
	bool is_boundary;             	// particle of grain boundary
	bool is_lattice_boundary;     	// particle of box boundary
	bool no_break[8];				// not allowed to break
	bool is_phase_boundary;			// is a phase boundary
	bool isUpperBox; 				// upper part of box 
	bool isLowerBox;				// lower part of box 

	int elle_Node[32];            	// Elle nodes that are close by

	int box_pos;           			// pos  in repulsion box

	bool done;             			// already added in force list (direct neighbour)

	bool isHole;					// flag for Holes
	bool isHoleBoundary;			// flag for hole boundaries
	bool isHoleInternalBoundary; 	// flag for growth at boundaries

	float area;						// area of particle for reactions
	float mV; 						// Molecular Volume
	float surf_free_E;				// interfacial free energy
	float surfE;					// surface energy
	float newsurfE;					// mean surface energy across surface 

	float prob;          			// probability to react

	float sxx;             			// normal stress in x
	float syy;             			// normal stress in y
	float sxy;             			// shear stress in xy

	float bound_stress;    			// stress on boundary
	int mineral;		 			// number for mineral

	int adjust_area_p;

	float springf[8];      			// my spring constants 
	float springv[8];				// my viscosities
	float break_Str[8];    			// spring breaking strength (in normal stress)
	int grain;             			// number for my grain
	bool  break_Sp[8];     			// I break
	bool  draw_break;      			// extra flag for drawing particles broken
	Particle *neigP[8];    			// pointers to eight possible neighbours

	Particle *neigP2[8];    		// pointer to neighbours, not influenced by breaking springs

	Particle *fluid_neigP[8]; 		// neighbour list along fluid solid interface 
	int fluid_neig_count;    		// counter for neighbours 
	float young;					// youngs modulus
	int open_springs;				// how many open springs for surface energies 
	float fluid_P;					// fluid pressure 
	float movement;

	Particle *leftNeighbour;		// left neighbour along interface (looking at fluid)
	Particle *rightNeighbour;     	// right neighbour along interface (looking at fluid)
	Particle *nextP;       			//pointer to next in list
	Particle *prevP;       			//pointer backwards in list
	Particle *neig;        			//help pointer
	Particle *neig2;       			//second help pointer
	Particle *neig3;				// third help pointer 

	Particle *next_inBox;        	// in case there is more than one

	Unode    *p_Unode;     			//connected Unode pointer

	double xpos;            		// my x position
	double ypos;            		// my y position

	double oldxpos;					// old x position 
	double oldypos;					// old y position 

	double oldx; 					// second old x position 
	double oldy; 					// second old y position 

	short i;		 				// external counter ?

	//neue Variablen

	float conc;						// concentration 
	float new_conc;					// new concentration 
	float left_conc;				// left over  concentration 

	//new variables for phase change
	bool potential_nucleus;
	bool gbm_condition;
	bool nucleation_condition;
	bool already_done;
	bool phase_change_required;
	bool transformation_required;

	float 	eEl_volume;
	float eEl_distortion;
	float eEl;		// elastic Energy

	float 	time;
	float 	deltaG;
	float 	temperature;
	float 	deltaG_nucleation;
	float	deltaG_gbm;
	float 	deltaG_d;
	float 	deltaG_t;
	float 	mean_stress;
	float 	real_radius;

	//new variables for sorting nuclei
	bool	local_maximum;

	//new variables to remember previous settings
	int	previous_mineral;

	Particle   *fixed_neig;

	float 	previous_prob;
	float 	previous_mV;
	float	previous_temperature;
	float 	previous_young;
	float 	previous_deltaG;
	float 	previous_real_radius;
	float 	previous_radius;
	float	previous_area;
	float	last_prob;
	float	previous_eEl;
	float	previous_surfE;
	float	previous_mean_stress;
	float	previous_sigman;
	float	previous_eEl_volume;
	float	previous_eEl_distortion;
	float	previous_boundary_state;
	float	previous_phase_boundary_state;
	float	delta_eEl_distortion;
	float	delta_pV;
	float	delta_surfE;
	float	delta_eEl_volume;
	float	delta_Ed;  //strain-energy-change without volume-change
	float	Ed;
	float	delta_Ev;
	float	boundary_angle[6];
	float	spinel_content;
	float	previous_spinel_content;
	float	rate;
	float	phi;       //angle between boundary and sigman
	float   previous_eEl_dilatation;

	bool	previously_transformed;
	bool	nucleus;
	bool	right_now_transformed;

	float 	box_nb[6];   //associates springs to box_rad radii, to speed up repulsion

	float   dl[9];       //contains the difference between alen&dd in terms of boxrad-numbering
	float   rep[9];      //saves the according rep_constant from the repulsion-function
	double  spring_angle[6];
	double  area_factor;

	float   smallest_rad;
	int surfcount;

	float maxaxisangle, minaxisangle, maxaxislength, minaxislength;

	float   alen_list[88];
	int merk;
	Particle *neigh_list[88];

	bool  	visc_flag;   	//debugging
	bool 	no_spring_flag;	//debugging

	float 	rot_angle;      // the rotation angle for beams
	float a, b, fixangle, maxis1angle, maxis2angle, maxis1length, maxis2length;

	float dir_rate[6]; //rate with different directions for different neighbour-particles
	float l[4];        //sizes of the square

	float previous_springf[8],
	eEl_dilatation,
	original_springf,
	previous_original_springf,
	delta_eEl_dilatation,
	eEl_dif,
	prev_xpos,
	prev_ypos,
	prev_sxx,
	prev_syy,
	prev_sxy;
	
	int previous_grain;
	
	//------------------------------------------------
	// functions
	//------------------------------------------------

	Particle();          //constructor
	~Particle() {};      // destructor


	/************************************************
	* main relaxation routines
	*****************************************************/
		
	// the fast relaxation routine

	bool Relax(float relaxthresh,Particle **list, int size, bool wall,float rightwall,float leftwall,float lowerwall,float upperwall, float wallconstant, bool length, int debug_nb, int visc_rel); 

	bool Relax(Particle **list, int size, int visc_rel,bool wall,float rightwall,float leftwall,float lowerwall,float upperwall, float wallconstant);  // end relaxroutine, calculates stress and break
		
	float BreakBond();  // breaks a bond

	void ChangeBox(Particle **list, int size);  // in case particle out of box after move
		
		
	/*************************************
	* build lattice routines 
	**************************************/

	Particle *FindNeighbourF(int neigDist); // help routine for Connect

	Particle *FindNeighbourB(int neigDist); // help routine for Connect

	void Connect(int lType,int lParticlex,int lParticley);  // intial connections

	void SetSprings();   // sets springs to unit value at moment

	//****************************	
	// viscoelastic code	
	//****************************

	float Alen(int, float, float);   // change spring lengths of particles 

	void debug();

	void Tilt();		// tilt eliptical particles 

};
#endif
