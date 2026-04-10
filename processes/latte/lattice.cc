/******************************************************
 * Spring Code Lattice 2.0
 *
 * Functions for lattice class in lattice.cc
 *
 * Basic Spring Code for fracturing 
 *
 *
 *
 *
 * Daniel Koehn and Jochen Arnold feb. 2002 to feb. 2003
 * Oslo, Mainz 
 *
 * Daniel Koehn dec. 2003
 *
 * We thank Anders Malthe-S�renssen for his enormous help
 * and introduction to these codes
 *
 * Daniel Koehn and Till Sachau 2004/2005
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
#include <ctime>
#include <math.h>

// ------------------------------------
// elle headers
// ------------------------------------

#include "lattice.h"		// include lattice class
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
 * Constructor for lattice class 
 * calls the MakeLattice function and defines some variables 
 * at the moment. The Constructor then builds a lattice once
 * the lattice class is called (i.e. by the user in the "elle" 
 * function). 
 *
 * Daniel spring 2002
 *
 * add security stop for pictdumps
 *
 * Daniel march 2003
 *
 * make lattice is now taken out of the constructor
 * otherwise everything is made in the experiment class
 * has now to be called using Activate Lattice 
 ********************************************************/

// ---------------------------------------------------------------
// Constructor of Lattice class
// ---------------------------------------------------------------

Lattice::Lattice ():

    // -----------------------------------------------------------------------------
    // default values for some variables. Variables are declared in
    // lattice header
    // -----------------------------------------------------------------------------

    dimx (1.0),			// basic x dimension from Elle
    dimy (1.0),		// basic y dimension from Elle
    particlex (100),	// default size for particles in x
    nbBreak (0),		// number of broken springs in lattice
    def_time_p (0),		// deformation time zero (pure shear)
    def_time_u (0),		// deformation time zero (uniaxial
    // compression)
    grain_counter (0),	// initialize grain counter
    visc_rel (0),       // flag for viscous relax routines in relaxation (zero is no)
    Pi (3.1415927),		// number pi
    pressure_scale (10000),	// default 1.0 is 10 GPa
    pascal_scale (1000000),	// to get pascal from megapascal
    set_max_pict (false),	// no security stop for pict dump
    max_pict (1000),		// initialize maximum pics allowed
    num_pict (0),		// initialize pictcounter
    walls (false), relaxthres (0.00000000000002),	// relaxation threshold
    debug_nb (1400000000),
    transition(false)
{
  // nothing happens
}

/************************************************************+
*
*	Called from the experiment class to construct the 
*	Lattice if it is needed. Connects Elle and Latte
*
**************************************************************/

void Lattice::Activate_Lattice()
{

  int i, j;

  // --------------------------------------------------------------------
  // call the MakeLattice function and give it a number defining the
  // Lattice. 1 is triangular at the moment
  // --------------------------------------------------------------------


  MakeLattice (1);	// Make the Lattice

  Copy_NeigP_List ();	// for the viscous relaxation virtual
  // neighbour list

  highest_grain = HighestGrain ();	// get highest grain from Elle

  SetSpringRatio ();	// set the spring ratio for viscous part

  visc_flag = false;	// debugging

  for (i = 0; i < numParticles; i++) // default variables for the viscous routine
    {
      for (j = 0; j < 9; j++)
        runParticle->rep_rad[j] = runParticle->radius;

      runParticle->rot_angle = 0.0;

      runParticle->smallest_rad = runParticle->radius;

      runParticle->merk = 0;

      runParticle->maxis1length = runParticle->radius;
      runParticle->maxis2length = runParticle->radius;

      // this just for security
      runParticle->maxis1angle = 0.0;
      runParticle->maxis2angle = 0.0;

      runParticle = runParticle->nextP;
    }
}

// Copy_NeigP_List
/***************************************************************************************
 * function copies neigP list to NeigP2 list, which isn� subjected to breaking springs
 *
 * function is used for viscous retardation. This virtual neighbour list is important 
 * once springs break. The particle should still be able to change its shape. This 
 * is a problem since the normal repulsion routine assumes that the particles are round
 * 
 * written by Till Sachau 2004 
 ****************************************************************************************/
// ------------------------------------------------------------------
// function is called from the constructor of the lattice class
// ------------------------------------------------------------------

void
Lattice::Copy_NeigP_List ()
{
  int i, j;

  for (i = 0; i < numParticles; i++)
    {
      for (j = 0; j < 8; j++)
        {
          runParticle->neigP2[j] = runParticle->neigP[j];
        }
      runParticle = runParticle->nextP;
    }
}


// MAKE_LATTICE
/****************************************************************
 * Function that builds the lattice. is given the lattice type. 
 * does build its own lattice using particles. Creates particles, 
 * defines xy positions, connects all particles by calling connect in 
 * the particle class and connects to Unodes (pointer to them)
 * note that some variables are defined in the header lattice.h
 *
 * Note this is a bit simple at the moment. It uses xy coordinates and 
 * first finds these even though they are actually allready there in the 
 * Unodes. If this function is called and the Unode number and position 
 * and the particle number and position do not fit the code goes bananas. 
 * Therefore we have to write more advanced functions in the future. 
 *
 * The same goes for the connections. It uses also xy position at the 
 * moment but would be more clever to draw a circle or sphere around
 * a particle to find its neighbours. This will be especially important 
 * once we deal with random lattices. 
 *
 * daniel spring and summer 2002
 ******************************************************************/

// --------------------------------------------------------------------
// function MakeLattice of Lattice Class
// input variable int type
//
// function called from the constructor Lattice::Lattice()
// --------------------------------------------------------------------

void
Lattice::MakeLattice (int type)
{
  // -----------------------------------------------------------------------------
  // local variables
  // ----------------------------------------------------------------------------

  float xx;		// variable for x position in calculation
  float yy;		// variable for y position in calculation
  int irow, icol;		// variable for rows in x and y in
  // calculation
  int i, j, jj;		// counter
  Coords xy;		// unode position
  double oldy;		// help to check change in y row

  // ---------------------------------------------------------------------
  // zero some variables for checks
  // ---------------------------------------------------------------------

  for (i = 0; i < 1000000; i++)
    {
      nodes_Check[i] = false;
      next_inBox[i] = -1;	// no tail node in Box
    }

  // -------------------------------------------------------------------
  // the repulsion box is 2500000 squares wide now. This means that
  // the maximum resolution is a bit more than 1300 particles in
  // the x direction.
  // if larger resolution is wanted this number has to be increased !
  // however if there is not enough memory this will lead
  // to a segmentation fault once the program is started !
  // A segmentation fault even though the program compiles.
  // in addition to the initial definition of the size of the vector
  // repBox[] in the lattice.h header
  // -------------------------------------------------------------------

  for (i = 0; i < 2500000; i++)
    {
      repBox[i] = 0;
      node_Box[i] = -1;

    }
  local_nbBreak = 0;
  internal_break = 0;

  // talk to the public !

  cout << "H" << endl;
  cout << "E" << endl;
  cout << "L" << endl;
  cout << "L" << endl;
  cout << "O" << endl;
  cout << " " << endl;
  cout << " I am Latte " << endl;
  cout << " Version 2.0 2004/6 " << endl;
  cout << " " << endl;




  // ------------------------------------------------------------------------------
  // this function can be used in the future in another version of the
  // MakeLattice
  // function to get number of particles.
  // ------------------------------------------------------------------------------

  numParticles = ElleMaxUnodes ();	// call elle function to define
  // number of particles

  cout << numParticles << endl;	// talk a bit
  cout << "Constructing Lattice\n" << endl;

  // -----------------------------------------------------------------------------------
  // the triangular grid is defined by having rows of particles in x
  // that are shifted
  // by the radius of the particles. The y length is then found as
  // follows:
  // -----------------------------------------------------------------------------------
  // -----------------------------------------------------------------------------------
  // now everything is defined by ppm2elle and read in at the beginning.
  // the following loop finds out the particlex dimension of the
  // lattice. It just
  // runs along x row and checks when the y changes which gives you the
  // length of the
  // row and size of the lattice. Y is then just particlenumber divided
  // by x.
  // ------------------------------------------------------------------------------------

  if (type == 1)		// if it is a triangular grid apply ratio
    {
      // --------------------------------------------------------------------
      // ElleGetParticleUnode returns the object Unode with the index
      // that
      // you give the function. Then we can call directly functions
      // within
      // the Unode objects themselves. XY coordinates are not public in
      // the
      // Unode class !
      // ---------------------------------------------------------------------

      ElleGetParticleUnode (0)->getPosition (&xy);	// start with
      // first Unode

      oldy = xy.y;	// old position

      irow = int (sqrt ((double) numParticles));	// x always
      // smaller than y

      for (i = 0; i < irow; i++)	// run along first row
        {
          ElleGetParticleUnode (i)->getPosition (&xy);	// next particle
          // position

          if (xy.y != oldy)	// if it changes we are beginning a new
            // row
            {
              particlex = i;	// i is the x dimension now
              break;
            }
        }
      particley = numParticles / particlex;	// and define y
    }

  //cout << " px " << particlex << endl;	// talk a bit again
  //cout << " py " << particley << endl;	// particlex and particley
  // are dimensions in x,y

  numParticles = particlex * particley;	// mike definition of
  // particle number (has to
  // match Unodes)

  cout << " Num Particles " << numParticles << endl;

  /*************************************************************
   * initiate particle list  
   * put them in a connected list and give them numbers 
   * each particle is connected with the next one in the list
   * and connected backwards so that list is linked two ways
   * list is also circular so that a search can be started
   * anywhere in list by any particle in both directions
   * First and Last list particles are referenced in pointers
   *
   * Note that at the moment a refParticle is used in the lattice class to get
   * the beginning of the list. Pointers did not work so far. And passing pointers
   * from the outside makes things messy. Maybe there is a better trick for this. 
   * Note that now once particles may be deleted we have to check not to delete the
   * reference particle. 
   *************************************************************/

  runParticle = new Particle;	// pointer to first particle, make
  // particle (call particle constructor)
  refParticle = *runParticle;	// refParticle is the particle that I
  // point to

  firstParticle = preRunParticle = &refParticle;	// set additional
  // pointers

  // -----------------------------------------------------------------------------------
  // This loop does four things: it creates new particles, connects them
  // to the list
  // forwards and backwards and gives particles numbers
  // In the particle objects a pointer points to the next particle
  // (nextP), a pointer
  // points to the previous particle (prevP) and a number is given for
  // particles from
  // 0 to numParticles - 1 (nb)
  // two pointers are used in the loop to define particles (runParticle)
  // and the previous
  // particles (preRunParticle)
  // -----------------------------------------------------------------------------------

  for (i = 1; i < numParticles;)	// loop numParticles -1 times
    {
      runParticle = new Particle;	// make new object and point
      preRunParticle->nextP = runParticle;	// set pointer in prev
      // object
      runParticle->prevP = preRunParticle;	// set pointer to prev
      // object
      preRunParticle->nb = i - 1;	// my number is...start with 0
      preRunParticle = runParticle;	// change pointers
      i++;
    }
  lastParticle = runParticle;	// define end of list
  lastParticle->nb = numParticles - 1;	// give last particle a
  // number
  lastParticle->nextP = firstParticle;	// and connect a circular
  // list forwards
  firstParticle->prevP = lastParticle;	// and backwards

  /****************************************************************
   * set x and y positions of particles
   * Run through the list. then define x and y values 
   * Note that everything is done asuming the particle radius is 0.5
   * and then rescaled to the elle coordinates (or boxsize)
   * 
   ****************************************************************/

  // This routine makes a lattice itself. but the lattice is now
  // actually
  // already made in ppm2elle so that mike just reads the position of
  // Unodes
  // from an elle file. Still left this here, maybe need it some day

  /*
   * runParticle = firstParticle; // go to beginning of list for (i=0;i
   * < numParticles;) // loop through all particles {
   * //---------------------------------------------------------------------
   * // first find the y coordinate by dividing number by xsize and taking
   * the // integer value of the result
   * //---------------------------------------------------------------------
   * irow = (i)/particlex;
   * //---------------------------------------------------------------------
   * // then find x coordinate by subtracting the xrow(100) from the number
   * (i) // . 100 is irow * particlex. This is 0 for the first row and then
   * // minus 100 etc if particlex is 100.
   * //---------------------------------------------------------------------
   * icol = i - (irow * particlex); xx = icol;
   * //----------------------------------------------------------------------
   * // since the lattice is triangular each second row has to be shifted by
   * the // radius of the particles (radius 0.5 in this definition) // ok at
   * the moment since box has still absolute size of particlex
   * //----------------------------------------------------------------------
   * if (irow != (2*(irow/2))) { xx = xx + 0.5; // checks uneven numbers } yy 
   * = irow*1.0/xyratio; // scale y-rows for triangular runParticle->xpos =
   * xx/particlex; // scale to elle box runParticle->ypos = yy/particlex;
   * runParticle = runParticle->nextP; // grab next particle i++; } 
   */

  // ---------------------------------------------------------------
  // initialization of some Elle attributes that appear in the
  // interface and are written out in the function UpdateElle
  // ----------------------------------------------------------------

  ElleInitUnodeAttribute (U_FRACTURES);
  ElleInitUnodeAttribute (U_PHASE);
  ElleInitUnodeAttribute (U_TEMPERATURE);
  ElleInitUnodeAttribute (U_DIF_STRESS);
  ElleInitUnodeAttribute (U_MEAN_STRESS);
  ElleInitUnodeAttribute (U_DENSITY);
  ElleInitUnodeAttribute (U_YOUNGSMODULUS);
  ElleInitUnodeAttribute (U_ENERGY);

  /****************************************************************
   * just set the position of the particle equal to its unode
   * position. the positions are made triangular in the right 
   * way by ppm2elle now
   ****************************************************************/

  for (i = 0; i < numParticles; i++)
    {
      ElleGetParticleUnode (runParticle->nb)->getPosition (&xy);
      runParticle->xpos = xy.x;
      runParticle->ypos = xy.y;
      runParticle->oldx = xy.x;
      runParticle->oldy = xy.y;

      runParticle = runParticle->nextP;
    }

  /*******************************************************************
   * connect the particles
   * This just runs through the list again and calls a function 
   * within the particles themselves. (defined in particle.cc) 
   * Also set the springs in Particle class by call function SetSpring
   * defined in particle.cc
   ******************************************************************/

  runParticle = firstParticle;	// start
  for (i = 0; i < numParticles; i++)	// run
    {
      runParticle->Connect (type, particlex, particley);	// call
      // Connect
      // and
      // connect
      // (connect
      // in
      // particle.cc)
      runParticle->radius = runParticle->smallest_rad = runParticle->radius * 100.0 / particlex;	// rescale
      // radius

      runParticle->area = runParticle->radius * runParticle->radius * 3.1415927;	// set
      // area
      // for
      // reactions

      runParticle = runParticle->nextP;	// go to next particle

    }

  // ----------------------------------------------------------------
  // call set Springs in particle.cc after all particles are
  // connected. SetSprings give default spring constants,
  // youngs modulus and viscosity
  // ----------------------------------------------------------------

  for (i = 0; i < numParticles; i++)	// run
    {
      runParticle->SetSprings ();	// call SetSprings

      runParticle = runParticle->nextP;	// go to next particle

    }

  // -----------------------------------------------------------------
  // new function that finds the backward pointer of neighbours.
  // springs are always defined in two ways, by the particle
  // itself and by its neighbour. However the particle does not
  // know which spring of its neighbour points backwards.
  // this function finds the backward pointer and writes an
  // integer into neig_spring[] array that is the write counter in the
  // neigP[] array of the neighbour.
  // -----------------------------------------------------------------

  for (i = 0; i < numParticles; i++)	// loop
    {
      // a and b for the new radius-function (Alen(int) in particle-class !!!!!!!!!!!!!!!!
      runParticle->a = runParticle->b = runParticle->radius;
      runParticle->fixangle = 0.0;

      for (j = 0; j < 8; j++)	// run through neighbours
        {
          if (runParticle->neigP[j])	// if neighbour
            {
              for (jj = 0; jj < 8; jj++)	// run through neighbours
                // springs
                {
                  if (runParticle->neigP[j]->neigP[jj])	// if
                    // neighbour
                    {
                      if (runParticle->nb == runParticle->neigP[j]->neigP[jj]->nb)	// look
                        // backwards
                        // and
                        // find
                        // particle
                        // again
                        {
                          runParticle->neig_spring[j] = jj;	// thats
                          // the one
                        }
                    }
                }
            }
        }



      runParticle = runParticle->nextP;
    }


  // ------------------------------------------------------------------------------
  // now finish the MakeLattice call by doing some extra things
  // first call FindUnodeElle defined in Lattice.cc (below)
  // This function runs through the list of particles and calls an Elle
  // function
  // that was written for Mike and returns a pointer to the right Unode.
  // With this we can call functions in the Unode class directly and
  // in addition use all plotting routines. Also do not have to call
  // vector each
  // time we want a specific Unode, each particle has its Unode defined
  // now.
  // I think that makes things faster ?
  // -------------------------------------------------------------------------------

  FindUnodeElle ();	// new thing that finds pointer to Unode

  cout << "make RepulsionBox" << endl;

  MakeRepulsionBox ();	// fill box for Repulsion

  // ------------------------------------------------------------------------------
  // Now we define whatever boundaries we want for the particles at the
  // edges of
  // the box. Box is not wrapping at the moment !!
  // function SetBoundaries is defined below in lattice.cc
  // ------------------------------------------------------------------------------

  SetBoundaries ();	// apply some boundary conditions

  cout << "initiate Grains" << endl;

  InitiateGrainParticle ();	// find the grains, grain boundaries and
  // connect nodes

  // -----------------------------------------------------------------------------------------
  // Now we do a basic relaxation at the beginning. Should only run
  // through this
  // once if lattice is in equilibrium (which it should be at the
  // moment)
  // function Relaxation defined below in lattice.cc
  // thats the core of all this ! All the fun is happening there. This
  // we should
  // optimize as best as we can.
  // Relaxation calls FullRelax() in lattice.cc which returns a flag
  // depending on
  // whether or not things are relaxed. If things are not relaxed this
  // routine will
  // loop until they are relaxed. Therefore in this loop the program
  // runs forever if
  // too much happened or something goes wrong.
  // FullRelax calls Relax in the particle (defined in particle.cc)
  // which relaxes each
  // individual particle by looking at its neighbours plus whatever we
  // want (fluidpressure ?)
  // ------------------------------------------------------------------------------------------

  Relaxation ();		// do a basic relaxation

  // -------------------------------------------------------------------------------
  // and finally we call a routine that updates the interface and plots
  // the
  // Unode stresses (lots still to be done there, we have the full
  // tensor and can
  // do with that whatever we want !
  // UpdateElle also defined below in lattice.cc
  // returns stresses from particles to Unodes for the plotting
  // and calls an Elle routine to update interface
  // --------------------------------------------------------------------------------

  UpdateElle ();		// talk to Elle

}

// INITIATE_GRAIN_PARTICLE
/********************************************************************************
 * function is called from make mike function in the lattice class. 
 * 
 * this is a heavy one. Used to take ages of time. Now quite fast. 
 * finds the grains from the flynn and gives particles number of grain or flynn
 * that it is in. 
 * To do that fast it uses the relaxbox and gets the box position from 
 * the Unodes position and then finds the particle. So it only loops 
 * through the flynns and their nodes. 
 * the second part defines the grain boundary particles by checking which 
 * particles have neighbours with different grains. Rather straight forward. 
 * The last part finds the nodes in Elle and connects them with the particles. 
 * Loops through the particles and defines a region. Then it calls a 
 * EllePointInRegion routine with a node coord. In order not to loop through
 * all nodes each time (loops within loops are deadly) we first define a 
 * kind of second repulsion box with the same dimensions as the real 
 * repulsion box and fill all the nodes in the box. There is a second array
 * for single nodes that contains the tailparticles. Did not want to change
 * the node structure in Elle. 
 * Once the box is filled we can find nodes close to a particle (in same or 
 * neighbouring boxes) and can easily connect the nodes with particles. 
 * just have to make an additional boundary check in case (and that is so) 
 * that nodes are not surrounded by particles. 
 *
 * daniel summer 2002
 * daniel changed december 2002
 *
 * changed december 2004 -> all Elle nodes found now 
 *****************************************************************************/

// -------------------------------------------------------------------------
// function InitiateGrainParticle in lattice class
//
// called from Lattice::MakeLattice()
// -------------------------------------------------------------------------



void
Lattice::InitiateGrainParticle ()
{
  int i, j, ii, jj, iii, jjj;	// lots and lots of counter for the
  // thousand loops
  int nb;			// the important number
  int neig1;		// help neighbour
  int maxFlynn;		// maxFlynns (max in program not number of
  // flynns !)
  int count;		// a counter yeah !
  int max;		// for max of nodes (also not real node
  // number !)
  int x, y;		// conversion for box
  double fx, fy;		// more conversion for box

  Coords bbox[3], boundbox[4], xy;	// boxes and single coord structures
  // (attrib.h)
  NodeAttrib *p;		// Node structure (nodesP.h)


  // ------------------------------------------------------------------------------------
  // the first routine loops through the flynns, gets the Unode list for
  // flynns and
  // then defines a grain for each particle (an int for which flynn the
  // particle is
  // part of.
  // in order to find the particle the position of the Unode is taken
  // and converted to
  // the relaxation box where we look for the right particle
  // a bit around ten corners but very fast and seems to be efficient
  // ------------------------------------------------------------------------------------

  cout << "define grains" << endl;

  vector < int >unodelist;	// define unodelist vector for unodes

  maxFlynn = ElleMaxFlynns ();	// max number of Flynns (not real max
  // number)

  runParticle = &refParticle;	// for particle loops

  // -----------------------------------------------------------------
  // now loop through the flynns and get the Unode lists
  // -----------------------------------------------------------------

  for (j = 0; j < maxFlynn; j++)	// loop through flynns
    {
      if (ElleFlynnIsActive (j))	// use only active flynns
        {
          grain_counter = grain_counter + 1;

          ElleGetFlynnUnodeList (j, unodelist);	// just gets the
          // Unodelist for
          // flynn j

          count = unodelist.size ();	// gets size of Unodelist

          for (i = 0; i < count; i++)	// go through Unodelist
            {
              nb = unodelist[i];	// the real number of the Unode

              // -------------------------------------------------------------------------------
              // this routine returns the unode Object if you give it a
              // number so that we then
              // can directly call functions within the unode class like
              // getPosition (the
              // Position of type coord is not public in the Unode
              // class)
              // --------------------------------------------------------------------------------

              ElleGetParticleUnode (nb)->getPosition (&xy);

              // -----------------------------------------------------------------------------------
              // find position in Relax box to find Particle for Unode.
              // Particles point to Unodes
              // but Unodes dont point back. For the box we scale the
              // coord, convert them to ints
              // and make them one-dimensional
              // -----------------------------------------------------------------------------------

              fx = (xy.x * particlex);
              fy = (xy.y * particlex);
              x = int (fx);
              y = int (fy);

              // ---------------------------------------------------------
              // and make one-dimensional
              // ---------------------------------------------------------

              x = (y * particlex * 2) + x;

              // ------------------------------------------------------------
              // now we go to the box and look in it if particle is the
              // right one. if not we loop through the tailparticles
              // until
              // we find the right one
              // -------------------------------------------------------------

              if (nb == repBox[x]->nb)	// if the right particle
                // is directly in box
                {
                  if (repBox[x]->grain == -1)	// this strange only works
                    // like this
                    {
                      repBox[x]->grain = j;	// give particle grain
                      // number
                    }
                }
              else	// loop through tail
                {
                  runParticle = repBox[x];	// redefine pointer for
                  // looping

                  while (runParticle->next_inBox)	// if there is a
                    // tail (should
                    // be)
                    {
                      if (nb == runParticle->next_inBox->nb)	// if it
                        // is
                        // right
                        // particle
                        {
                          if (runParticle->next_inBox->grain == -1)	// this
                            // strange
                            // only
                            // works
                            // like
                            // this
                            {
                              runParticle->next_inBox->grain = j;	// give
                              // particle
                              // grain
                              // number
                            }
                          break;	// jump out of loop, all done
                        }
                      else
                        {
                          runParticle = runParticle->next_inBox;	// go
                          // to
                          // next
                          // in
                          // tail
                        }
                    }
                }
            }
        }
    }

  // -----------------------------------------------------------------------------------------
  // this defines the grain boundary particles by checking whether or
  // not particle has
  // a neighbour that is part of a different grain or flynn.
  // -----------------------------------------------------------------------------------------

  cout << " define grain boundaries " << endl;

  for (i = 0; i < numParticles; i++)	// and loop through particles
    {
      for (j = 0; j < 8; j++)	// loop through neighbours of particle
        {
          if (runParticle->neigP[j])	// if there is a neighbour
            {
              if (runParticle->neigP[j]->grain != runParticle->grain)	// if
                // not
                // of
                // same
                // grain
                {
                  runParticle->is_boundary = true;	// I am a boundary
                  // particle
                  break;
                }
            }
        }
      runParticle = runParticle->nextP;	// go on looping
    }

  // --------------------------------------------------------------------------------
  // This is the big one. Connects nodes with particles.
  // First put nodes in a Node list similar to the relaxation list so
  // that we can
  // find them easily if we know in what box we want to look
  // --------------------------------------------------------------------------------

  neig1 = -1;		// help definition

  // ----------------------------------------------
  // first zero all nodes (= -1)
  // ----------------------------------------------

  max = ElleMaxNodes ();	// gets max nodes

  // -----------------------------------------------------------------------------
  // this routine fills the box for nodes. It has the same dimension as
  // the relax box. Therefore it should be easy to find neighbouring
  // nodes
  // and particles using this box. Once it is defined it is rather fast.
  // still have to be careful with negative values.
  // nodes have additional attribute next_inBox (int) that contains tail
  // nodes
  // similar to the tail particles in the relaxbox.
  // the next_inBox array is used as virtual pointer for the nodes.
  // -----------------------------------------------------------------------------

  for (i = 0; i < max; i++)	// loop through nodes
    {


      if (ElleNodeIsActive (i))	// if node active
        {
          p = ElleNode (i);	// gets structure for the node

          // --------------------------------------------------
          // get position in the box first x and y and make
          // them ints. scale same as particle relax box
          // the box scales with the number of particles
          // --------------------------------------------------

          fx = (p->x * particlex);
          fy = (p->y * particlex);
          x = int (fx);
          y = int (fy);

          // ---------------------------------------------------------
          // and make one-dimensional
          // ---------------------------------------------------------

          x = (y * particlex*2) + x;

          // ----------------------------------------------------------
          // now fill the node box if it is empty (-1)
          // if it is full loop in the next_inBox array until
          // you find the end of the tail of nodes in that box
          // position
          // -----------------------------------------------------------

          if (node_Box[x] == -1)
            {
              node_Box[x] = i;	// fill the box
            }
          else	// now we make the tails
            {
              ii = node_Box[x];	// shift to find node with tail

              while (next_inBox[ii] != -1)	// loop until end of tail
                {
                  ii = next_inBox[ii];	// and shift and shift
                }
              next_inBox[ii] = i;	// found tail and fill in node i
            }
        }
    }

  // --------------------------------------------------------------------------------
  // now we come to the connection between particles and nodes
  // loop through particles until you find a boundary particle
  // then check all the neighbours of that particle
  // check the neighbour of the neighbour to find a triangle with the
  // original particle. Then find box position of the particles (plus
  // some
  // extra left and right box otherwise some nodes fall through the
  // net). Then
  // find the nodes with that boxposition in the node box. Then check if
  // that node
  // is really within the triangle that we look for.
  // In the end we take the particles that are boundary of the lattice
  // and
  // define a square around the particle in which we search for nodes on
  // the
  // boundaries.
  // --------------------------------------------------------------------------------

  cout << " and connect nodes to particles " << endl;

  for (i = 0; i < numParticles; i++)	// loop through the particles
    {
      neig1 = -1;	// flag -1 to find first neighbour

      {
        for (j = 0; j < 8; j++)	// loop through its neighbours
          {
            if (runParticle->neigP[j])	// if there is a neighbour
              // at that point
              {
                {
                  if (neig1 == -1)	// if the first neighbour
                    // is not yet defined
                    neig1 = j;	// i am the first

                  else	// else have to look for the second
                    // neighbour in triangle
                    {

                      for (ii = 0; ii < 8; ii++)	// loop through
                        // second
                        // neighbour
                        // neighbours
                        {
                          if (runParticle->neigP[neig1]->neigP[ii])	// if
                            // neighbour
                            // at
                            // that
                            // point
                            {
                              // -------------------------------------------------------
                              // if the number of the first
                              // neighbour is the same as
                              // the number of the neighbour of the
                              // second neighbour
                              // we have defined a closed triangle
                              // --------------------------------------------------------

                              if (runParticle->neigP[neig1]->neigP[ii]->nb == runParticle->neigP[j]->nb)
                                {
                                  // --------------------------------------------------------
                                  // the bbox is used for the
                                  // definition of a region for the
                                  // function EllePointInRegion that
                                  // gets a bbox, the dim of
                                  // the bbox (here three) and the
                                  // structure of type coord
                                  // with the coordinates of the
                                  // point (in this case the node)
                                  // ----------------------------------------------------------

                                  // ----------------------------------------------------------
                                  // DEFINE bbox[] by using x and y
                                  // of the particles of the
                                  // found triangle
                                  // -----------------------------------------------------------

                                  bbox[0].x = runParticle->xpos;
                                  bbox[0].y = runParticle->ypos;
                                  bbox[1].x = runParticle->neigP[neig1]->xpos;
                                  bbox[1].y = runParticle->neigP[neig1]->ypos;
                                  bbox[2].x = runParticle->neigP[j]->xpos;
                                  bbox[2].y = runParticle->neigP[j]->ypos;

                                  // -------------------------------------------------------------
                                  // now we loop through 9 different
                                  // box positions and all their
                                  // tail nodes and check if any of
                                  // these nodes fall into the
                                  // triangle
                                  // the box positions are the
                                  // positions from the three
                                  // particles
                                  // and the left and right box of
                                  // each particle. (thats where
                                  // we might loose nodes because
                                  // box is square lattice)
                                  //
                                  // x is always the final box
                                  // position
                                  // --------------------------------------------------------------

                                  for (iii = 0; iii < 9; iii++)
                                    {
                                      if (iii == 0)	// box
                                        // position
                                        // for
                                        // first
                                        // particle
                                        {
                                          // ---------------------------------------------------------
                                          // get positions, scale by
                                          // lattice and convert to
                                          // ints
                                          // ---------------------------------------------------------

                                          fx = (runParticle->xpos * particlex);
                                          fy = (runParticle->ypos * particlex);
                                          x = int (fx);
                                          y = int (fy);

                                          // ---------------------------------------------------------
                                          // and make
                                          // one-dimensional
                                          // ---------------------------------------------------------

                                          x = (y * particlex*2) + x;
                                        }
                                      else if (iii == 1)
                                        x = x + 1;	// box to
                                      // right
                                      else if (iii == 2)
                                        x = x - 2;	// box to
                                      // left
                                      else if (iii == 3)
                                        {
                                          // ---------------------------------------------------------
                                          // get positions, scale by
                                          // lattice and convert to
                                          // ints
                                          // ---------------------------------------------------------

                                          fx = (runParticle->neigP[neig1]->xpos * particlex);
                                          fy = (runParticle->neigP[neig1]->ypos * particlex);
                                          x = int (fx);
                                          y = int (fy);

                                          // ---------------------------------------------------------
                                          // and make
                                          // one-dimensional
                                          // ---------------------------------------------------------

                                          x = (y * particlex*2) + x;
                                        }
                                      else if (iii == 4)
                                        x = x + 1;	// box to
                                      // right
                                      else if (iii == 5)
                                        x = x - 2;	// box to
                                      // left
                                      else if (iii == 6)
                                        {
                                          // ---------------------------------------------------------
                                          // get positions, scale by
                                          // lattice and convert to
                                          // ints
                                          // ---------------------------------------------------------

                                          fx = (runParticle->neigP[j]->xpos * particlex);
                                          fy = (runParticle->neigP[j]->ypos * particlex);
                                          x = int (fx);
                                          y = int (fy);

                                          // ---------------------------------------------------------
                                          // and make
                                          // one-dimensional
                                          // ---------------------------------------------------------

                                          x = (y * particlex*2) + x;
                                        }
                                      else if (iii == 7)
                                        x = x + 1;	// box to
                                      // right
                                      else if (iii == 8)
                                        x = x - 2;	// box to
                                      // left

                                      jj = node_Box[x];	// define
                                      // node

                                      while (jj != -1)	// loop
                                        // through
                                        // tails
                                        {
                                          // ----------------------------------------------------------
                                          // if node is active and
                                          // has not been checked go
                                          // for it
                                          // ----------------------------------------------------------

                                          if (ElleNodeIsActive (jj) && !nodes_Check[jj])
                                            {
                                              p = ElleNode (jj);	// get
                                              // struct
                                              // of
                                              // node
                                              xy.x = p->x;	// convert
                                              // x
                                              // to
                                              // coord
                                              xy.y = p->y;	// convert
                                              // y
                                              // to
                                              // coord

                                              // ----------------------------------------------------
                                              // now make the region
                                              // check with the
                                              // predefined box,
                                              // the dimension of
                                              // the box and the xy
                                              // coord struct
                                              // of the node that
                                              // you found in the
                                              // node box
                                              // -----------------------------------------------------

                                              if (EllePtInRegion (bbox, 3, &xy))
                                                {
                                                  // --------------------------------------------------
                                                  // there are 16
                                                  // possible places
                                                  // for nodes in a
                                                  // particle. check
                                                  // if one is free
                                                  // and put node in
                                                  // do this for all
                                                  // three bounding
                                                  // particles
                                                  // --------------------------------------------------

                                                  for (jjj = 0; jjj < 16; jjj++)
                                                    {
                                                      // main
                                                      // particle

                                                      if (runParticle->elle_Node[jjj] == -1)
                                                        {
                                                          runParticle->
                                                          elle_Node
                                                          [jjj]
                                                          =
                                                            jj;
                                                          break;
                                                        }
                                                    }
                                                  for (jjj = 0; jjj < 16; jjj++)
                                                    {
                                                      // first
                                                      // neighbour

                                                      if (runParticle->neigP[neig1]->elle_Node[jjj] == -1)
                                                        {
                                                          runParticle->
                                                          neigP
                                                          [neig1]->
                                                          elle_Node
                                                          [jjj]
                                                          =
                                                            jj;
                                                          break;
                                                        }
                                                    }
                                                  for (jjj = 0; jjj < 16; jjj++)
                                                    {
                                                      // second
                                                      // neighbour

                                                      if (runParticle->neigP[j]->elle_Node[jjj] == -1)
                                                        {
                                                          runParticle->
                                                          neigP
                                                          [j]->
                                                          elle_Node
                                                          [jjj]
                                                          =
                                                            jj;
                                                          break;
                                                        }
                                                    }
                                                  // -----------------------------------------------
                                                  // now define flag
                                                  // that tells
                                                  // particles that
                                                  // this node has
                                                  // been connected
                                                  // otherwise we
                                                  // connect it at
                                                  // least three
                                                  // times
                                                  // to the same
                                                  // three particles
                                                  // ------------------------------------------------

                                                  nodes_Check
                                                  [jj]
                                                  =
                                                    true;
                                                }

                                            }
                                          // ---------------------------------
                                          // this is for the while
                                          // loop
                                          // go to next tail node in
                                          // box
                                          // ---------------------------------

                                          jj = next_inBox[jj];
                                        }
                                    }
                                }
                            }
                        }
                      // ---------------------------------------------
                      // reset neig1 to next neighbour
                      // ---------------------------------------------

                      neig1 = j;

                    }

                }
              }

          }
      }
      // ------------------------------------------------------------------------------------
      // now check the particles at the boundary of the lattice. Some
      // nodes might not be
      // within a triangle of particles. Therefore draw a square around
      // each boundary
      // particle and check if there is an unconnected node in this box.
      // -------------------------------------------------------------------------------------

      if (runParticle->is_lattice_boundary)	// if lattice boundary
        {
          // -------------------------------------
          // define a new bbox with 4 dimensions
          // particle is in the centre
          // -------------------------------------

          boundbox[0].x = runParticle->xpos - (runParticle->radius * 4.0);	// lower
          // left
          // corner
          boundbox[0].y =
            runParticle->ypos +
            (runParticle->radius * 4.0);

          boundbox[1].x = runParticle->xpos + (runParticle->radius * 4.0);	// upper
          // left
          // corner
          boundbox[1].y =
            runParticle->ypos +
            (runParticle->radius * 4.0);

          boundbox[2].x = runParticle->xpos + (runParticle->radius * 4.0);	// upper
          // right
          // corner
          boundbox[2].y =
            runParticle->ypos -
            (runParticle->radius * 4.0);

          boundbox[3].x = runParticle->xpos - (runParticle->radius * 4.0);	// lower
          // right
          // corner
          boundbox[3].y =
            runParticle->ypos -
            (runParticle->radius * 4.0);


          for (jj = 0; jj < max; jj++)	// loop through all nodes
            // again
            {
              // ----------------------------------------------------------
              // if node is active and has not been connected
              // ----------------------------------------------------------

              if (ElleNodeIsActive (jj) && !nodes_Check[jj])
                {
                  p = ElleNode (jj);	// get node struct and convert
                  // coordinates
                  xy.x = p->x;
                  xy.y = p->y;

                  if (EllePtInRegion (boundbox, 4, &xy))	// if in
                    // the box
                    {
                      for (jjj = 0; jjj < 16; jjj++)	// loop through
                        // possible nodes
                        // in particle
                        {
                          if (runParticle->elle_Node[jjj] == -1)	// if
                            // free
                            {
                              runParticle->elle_Node[jjj] = jj;	// fill
                              nodes_Check[jj] = true;	// flag up
                              break;
                            }
                          else if (runParticle->elle_Node[jjj] == jj)	// in
                            // case
                            // node
                            // is
                            // in
                            // there
                            {
                              break;	// we can exit
                            }
                        }
                    }
                }
            }
        }

      runParticle = runParticle->nextP;	// loop in particles
    }

  cout << " done with grain definitions " << endl;

  // ----------------------------------------------------------------------
  // just a test to see if all Elle nodes are connected now
  // ----------------------------------------------------------------------

  i = 0;

  for (jj = 0; jj < max; jj++)	// loop through Elle nodes
    {
      if (ElleNodeIsActive (jj) && !nodes_Check[jj])	// if node is
        // active and has
        // not been
        // checked
        {
          p = ElleNode (jj);	// define xy coordinates and pass out to
          // interface

          cout << "node not connected to lattice" << jj << " "
          << p->x << " " << p->y << endl;

          i++;
        }
    }
  if (i != 0)
    cout << i << endl;	// how many nodes were not connected
}

/****************************************************************************************
 * new function that is supposed to read in Elle data after things are deformed so 
 * that talking back and forth between Elle and mike works better. 
 *
 * Function now works. Only problems are fractures. We are now saving the youngs moduli
 * for single grains and also the values of the grainboundaries when they are set 
 * in the initialization functions. These are then taken off the particles and springs, 
 * the new geometry is read from Elle and the values are again applied to particles. 
 * Therefore grain boundaries can now really move including all their properties for the 
 * Lattice spring code. 
 *
 * You may have to check if all initialization function parameters are saved... 
 *
 * Function uses the new Unode attribute Flynn which has directly the grain number 
 *
 * Daniel 2005/6 
 *****************************************************************************************/

void
Lattice::GetNewElleStructure ()
{
  int i, j, ii, jj, iii, jjj;	// lots and lots of counter for the
  // thousand loops
  int nb;			// the important number
  int neig1;		// help neighbour
  int maxFlynn;		// maxFlynns (max in program not number of
  // flynns !)
  int count;		// a counter yeah !
  int max;		// for max of nodes (also not real node
  // number !)
  int x, y;		// conversion for box
  double fx, fy;		// more conversion for box

  Coords bbox[3], boundbox[4], xy;	// boxes and single coord structures
  // (attrib.h)
  NodeAttrib *p;		// Node structure (nodesP.h)


  // ------------------------------------------------------------------------------------
  // the first routine loops through the flynns, gets the Unode list for
  // flynns and
  // then defines a grain for each particle (an int for which flynn the
  // particle is
  // part of.
  // in order to find the particle the position of the Unode is taken
  // and converted to
  // the relaxation box where we look for the right particle
  // a bit around ten corners but very fast and seems to be efficient
  // ------------------------------------------------------------------------------------

  cout << "define grains" << endl;


  //-------------------------------------------------------
  // This is sooo simple now, Elle does all the work and
  // we just take the flynn number (flynn() function passes
  // that back as an integer)
  //-------------------------------------------------------

  for (j = 0;  j < numParticles; j++)
    {
      runParticle->grain = runParticle->p_Unode->flynn();
      runParticle = runParticle->nextP;
    }

  // -----------------------------------------------------------------------------------------
  // this defines the grain boundary particles by checking whether or
  // not particle has
  // a neighbour that is part of a different grain or flynn.
  // -----------------------------------------------------------------------------------------

  cout << " define grain boundaries " << endl;

  // first we have to kill the old values in order to define new ones
  // The particles have a flag is boundary if they are grain boundaries
  // and also the springs have flags. This is important since we have
  // to track the properties of grains and grain boundaries when these move
  // in Elle due to for example grain growth

  for (i=0;i<numParticles;i++)
    {
      if (runParticle->is_boundary == true)
        runParticle->is_boundary = false;

      // grain_young is saving the properties for grains

      runParticle->young = grain_young[runParticle->grain] *2.0/sqrt(3.0);

      for (j = 0;j<8;j++)
        {
          if (runParticle->neigP[j])	// if there is a neighbour
            {
              runParticle->springf[j] = grain_young[runParticle->grain];

              if (runParticle->spring_boundary[j] == true)
                {
                  runParticle->spring_boundary[j] = false;

                  runParticle->springf[j] = runParticle->springf[j] / boundary_constant;	// change
                  // spring
                  runParticle->break_Str[j] = runParticle->break_Str[j] / boundary_strength;	// change

                }
            }
        }
      runParticle = runParticle->nextP;
    }

  for (i = 0; i < numParticles; i++)	// and loop through particles
    {
      for (j = 0; j < 8; j++)	// loop through neighbours of particle
        {
          if (runParticle->neigP[j])	// if there is a neighbour
            {
              if (runParticle->neigP[j]->grain != runParticle->grain)	// if
                // not
                // of
                // same
                // grain
                {
                  runParticle->is_boundary = true;	// I am a boundary
                  // particle

                  for (j = 0; j < 8; j++)	// loop through neighbours
                    {
                      if (runParticle->neigP[j])	// if neighbour
                        {
                          if (runParticle->neigP[j]->grain != runParticle->grain)	// if
                            // neighbour
                            // is
                            // in
                            // different
                            // grain
                            {
                              runParticle->spring_boundary[j] = true;

                              runParticle->springf[j] = runParticle->springf[j] * boundary_constant;	// change
                              // spring
                              runParticle->break_Str[j] = runParticle->break_Str[j] * boundary_strength;	// change
                              // breakstrength
                            }
                        }
                    }

                  break;
                }
            }
        }
      runParticle = runParticle->nextP;	// go on looping
    }

  AdjustConstantGrainBoundaries(); // this is the cleaner to have gradients across grains.

  // --------------------------------------------------------------------------------
  // This is the big one. Connects nodes with particles.
  // First put nodes in a Node list similar to the relaxation list so
  // that we can
  // find them easily if we know in what box we want to look
  // --------------------------------------------------------------------------------

  neig1 = -1;		// help definition

  // ----------------------------------------------
  // first zero all nodes (= -1)
  // ----------------------------------------------

  for (i = 0; i < 2500000; i++)
    {

      node_Box[i] = -1;
    }


  max = ElleMaxNodes ();	// gets max nodes

  // -----------------------------------------------------------------------------
  // this routine fills the box for nodes. It has the same dimension as
  // the relax box. Therefore it should be easy to find neighbouring
  // nodes
  // and particles using this box. Once it is defined it is rather fast.
  // still have to be careful with negative values.
  // nodes have additional attribute next_inBox (int) that contains tail
  // nodes
  // similar to the tail particles in the relaxbox.
  // -----------------------------------------------------------------------------

  for (i = 0; i < max; i++)	// loop through nodes
    {
      next_inBox[i] = -1;	// no tail node in Box

      if (ElleNodeIsActive (i))	// if node active
        {
          p = ElleNode (i);	// gets structure for the node

          // --------------------------------------------------
          // get position in the box first x and y and make
          // them ints. scale same as particle relax box
          // the box scales with the number of particles
          // --------------------------------------------------

          fx = (p->x * particlex);
          fy = (p->y * particlex);
          x = int (fx);
          y = int (fy);

          // ---------------------------------------------------------
          // and make one-dimensional
          // ---------------------------------------------------------

          x = (y * particlex*2) + x;

          // ----------------------------------------------------------
          // now fill the node box if it is empty (-1)
          // if it is full loop in the next_inBox array until
          // you find the end of the tail of nodes in that box
          // position
          // -----------------------------------------------------------

          if (node_Box[x] == -1)
            {
              node_Box[x] = i;	// fill the box
            }
          else	// now we make the tails
            {
              ii = node_Box[x];	// shift to find node with tail

              while (next_inBox[ii] != -1)	// loop until end of tail
                {
                  ii = next_inBox[ii];	// and shift and shift
                }
              next_inBox[ii] = i;	// found tail and fill in node i
            }
        }
    }

  // --------------------------------------------------------------------------------
  // now we come to the connection between particles and nodes
  // loop through particles until you find a boundary particle
  // then check all the neighbours of that particle
  // check the neighbour of the neighbour to find a triangle with the
  // original particle. Then find box position of the particles (plus
  // some
  // extra left and right box otherwise some nodes fall through the
  // net). Then
  // find the nodes with that boxposition in the node box. Then check if
  // that node
  // is really within the triangle that we look for.
  // In the end we take the particles that are boundary of the lattice
  // and
  // define a square around the particle in which we search for nodes on
  // the
  // boundaries.
  // --------------------------------------------------------------------------------

  cout << " and connect nodes to particles " << endl;

  for (i = 0; i < numParticles; i++)	// loop through the particles
    {
      neig1 = -1;	// flag -1 to find first neighbour

      for (j = 0; j < 32; j++)	// maximum of 32 Elle_nodes per
        // particle at moment
        {
          runParticle->elle_Node[j] = -1;
        }


      {
        for (j = 0; j < 8; j++)	// loop through its neighbours
          {
            if (runParticle->neigP[j])	// if there is a neighbour
              // at that point
              {
                {
                  if (neig1 == -1)	// if the first neighbour
                    // is not yet defined
                    neig1 = j;	// i am the first

                  else	// else have to look for the second
                    // neighbour in triangle
                    {

                      for (ii = 0; ii < 8; ii++)	// loop through
                        // second
                        // neighbour
                        // neighbours
                        {
                          if (runParticle->neigP[neig1]->neigP[ii])	// if
                            // neighbour
                            // at
                            // that
                            // point
                            {
                              // -------------------------------------------------------
                              // if the number of the first
                              // neighbour is the same as
                              // the number of the neighbour of the
                              // second neighbour
                              // we have defined a closed triangle
                              // --------------------------------------------------------

                              if (runParticle->neigP[neig1]->neigP[ii]->nb == runParticle->neigP[j]->nb)
                                {
                                  // --------------------------------------------------------
                                  // the bbox is used for the
                                  // definition of a region for the
                                  // function EllePointInRegion that
                                  // gets a bbox, the dim of
                                  // the bbox (here three) and the
                                  // structure of type coord
                                  // with the coordinates of the
                                  // point (in this case the node)
                                  // ----------------------------------------------------------

                                  // ----------------------------------------------------------
                                  // DEFINE bbox[] by using x and y
                                  // of the particles of the
                                  // found triangle
                                  // -----------------------------------------------------------

                                  bbox[0].x = runParticle->xpos;
                                  bbox[0].y = runParticle->ypos;
                                  bbox[1].x = runParticle->neigP[neig1]->xpos;
                                  bbox[1].y = runParticle->neigP[neig1]->ypos;
                                  bbox[2].x = runParticle->neigP[j]->xpos;
                                  bbox[2].y = runParticle->neigP[j]->ypos;

                                  // -------------------------------------------------------------
                                  // now we loop through 9 different
                                  // box positions and all their
                                  // tail nodes and check if any of
                                  // these nodes fall into the
                                  // triangle
                                  // the box positions are the
                                  // positions from the three
                                  // particles
                                  // and the left and right box of
                                  // each particle. (thats where
                                  // we might loose nodes because
                                  // box is square lattice)
                                  //
                                  // x is always the final box
                                  // position
                                  // --------------------------------------------------------------

                                  for (iii = 0; iii < 9; iii++)
                                    {
                                      if (iii == 0)	// box
                                        // position
                                        // for
                                        // first
                                        // particle
                                        {
                                          // ---------------------------------------------------------
                                          // get positions, scale by
                                          // lattice and convert to
                                          // ints
                                          // ---------------------------------------------------------

                                          fx = (runParticle->xpos * particlex);
                                          fy = (runParticle->ypos * particlex);
                                          x = int (fx);
                                          y = int (fy);

                                          // ---------------------------------------------------------
                                          // and make
                                          // one-dimensional
                                          // ---------------------------------------------------------

                                          x = (y * particlex*2) + x;
                                        }
                                      else if (iii == 1)
                                        x = x + 1;	// box to
                                      // right
                                      else if (iii == 2)
                                        x = x - 2;	// box to
                                      // left
                                      else if (iii == 3)
                                        {
                                          // ---------------------------------------------------------
                                          // get positions, scale by
                                          // lattice and convert to
                                          // ints
                                          // ---------------------------------------------------------

                                          fx = (runParticle->neigP[neig1]->xpos * particlex);
                                          fy = (runParticle->neigP[neig1]->ypos * particlex);
                                          x = int (fx);
                                          y = int (fy);

                                          // ---------------------------------------------------------
                                          // and make
                                          // one-dimensional
                                          // ---------------------------------------------------------

                                          x = (y * particlex*2) + x;
                                        }
                                      else if (iii == 4)
                                        x = x + 1;	// box to
                                      // right
                                      else if (iii == 5)
                                        x = x - 2;	// box to
                                      // left
                                      else if (iii == 6)
                                        {
                                          // ---------------------------------------------------------
                                          // get positions, scale by
                                          // lattice and convert to
                                          // ints
                                          // ---------------------------------------------------------

                                          fx = (runParticle->neigP[j]->xpos * particlex);
                                          fy = (runParticle->neigP[j]->ypos * particlex);
                                          x = int (fx);
                                          y = int (fy);

                                          // ---------------------------------------------------------
                                          // and make
                                          // one-dimensional
                                          // ---------------------------------------------------------

                                          x = (y * particlex*2) + x;
                                        }
                                      else if (iii == 7)
                                        x = x + 1;	// box to
                                      // right
                                      else if (iii == 8)
                                        x = x - 2;	// box to
                                      // left

                                      jj = node_Box[x];	// define
                                      // node

                                      while (jj != -1)	// loop
                                        // through
                                        // tails
                                        {
                                          // ----------------------------------------------------------
                                          // if node is active and
                                          // has not been checked go
                                          // for it
                                          // ----------------------------------------------------------

                                          if (ElleNodeIsActive (jj) && !nodes_Check[jj])
                                            {
                                              p = ElleNode (jj);	// get
                                              // struct
                                              // of
                                              // node
                                              xy.x = p->x;	// convert
                                              // x
                                              // to
                                              // coord
                                              xy.y = p->y;	// convert
                                              // y
                                              // to
                                              // coord

                                              // ----------------------------------------------------
                                              // now make the region
                                              // check with the
                                              // predefined box,
                                              // the dimension of
                                              // the box and the xy
                                              // coord struct
                                              // of the node that
                                              // you found in the
                                              // node box
                                              // -----------------------------------------------------

                                              if (EllePtInRegion (bbox, 3, &xy))
                                                {
                                                  // --------------------------------------------------
                                                  // there are 16
                                                  // possible places
                                                  // for nodes in a
                                                  // particle. check
                                                  // if one is free
                                                  // and put node in
                                                  // do this for all
                                                  // three bounding
                                                  // particles
                                                  // --------------------------------------------------

                                                  for (jjj = 0; jjj < 16; jjj++)
                                                    {
                                                      // main
                                                      // particle

                                                      if (runParticle->elle_Node[jjj] == -1)
                                                        {
                                                          runParticle->
                                                          elle_Node
                                                          [jjj]
                                                          =
                                                            jj;
                                                          break;
                                                        }
                                                    }
                                                  for (jjj = 0; jjj < 16; jjj++)
                                                    {
                                                      // first
                                                      // neighbour

                                                      if (runParticle->neigP[neig1]->elle_Node[jjj] == -1)
                                                        {
                                                          runParticle->
                                                          neigP
                                                          [neig1]->
                                                          elle_Node
                                                          [jjj]
                                                          =
                                                            jj;
                                                          break;
                                                        }
                                                    }
                                                  for (jjj = 0; jjj < 16; jjj++)
                                                    {
                                                      // second
                                                      // neighbour
                                                      if (runParticle->neigP[j]->elle_Node[jjj] == -1)
                                                        {
                                                          runParticle->
                                                          neigP
                                                          [j]->
                                                          elle_Node
                                                          [jjj]
                                                          =
                                                            jj;
                                                          break;
                                                        }
                                                    }
                                                  // -----------------------------------------------
                                                  // now define flag
                                                  // that tells
                                                  // particles that
                                                  // this node has
                                                  // been connected
                                                  // otherwise we
                                                  // connect it at
                                                  // least three
                                                  // times
                                                  // to the same
                                                  // three particles
                                                  // ------------------------------------------------

                                                  nodes_Check
                                                  [jj]
                                                  =
                                                    true;
                                                }

                                            }
                                          // ---------------------------------
                                          // this is for the while
                                          // loop
                                          // go to next tail node in
                                          // box
                                          // ---------------------------------

                                          jj = next_inBox[jj];
                                        }
                                    }
                                }
                            }
                        }
                      // ---------------------------------------------
                      // reset neig1 to next neighbour
                      // ---------------------------------------------

                      neig1 = j;

                    }

                }
              }

          }
      }
      // ------------------------------------------------------------------------------------
      // now check the particles at the boundary of the lattice. Some
      // nodes might not be
      // within a triangle of particles. Therefore draw a square around
      // each boundary
      // particle and check if there is an unconnected node in this box.
      // -------------------------------------------------------------------------------------

      if (runParticle->is_lattice_boundary)	// if lattice boundary
        {
          // -------------------------------------
          // define a new bbox with 4 dimensions
          // particle is in the centre
          // -------------------------------------

          boundbox[0].x = runParticle->xpos - (runParticle->radius * 4.0);	// lower
          // left
          // corner
          boundbox[0].y =
            runParticle->ypos +
            (runParticle->radius * 4.0);

          boundbox[1].x = runParticle->xpos + (runParticle->radius * 4.0);	// upper
          // left
          // corner
          boundbox[1].y =
            runParticle->ypos +
            (runParticle->radius * 4.0);

          boundbox[2].x = runParticle->xpos + (runParticle->radius * 4.0);	// upper
          // right
          // corner
          boundbox[2].y =
            runParticle->ypos -
            (runParticle->radius * 4.0);

          boundbox[3].x = runParticle->xpos - (runParticle->radius * 4.0);	// lower
          // right
          // corner
          boundbox[3].y =
            runParticle->ypos -
            (runParticle->radius * 4.0);


          for (jj = 0; jj < max; jj++)	// loop through all nodes
            // again
            {
              // ----------------------------------------------------------
              // if node is active and has not been connected
              // ----------------------------------------------------------

              if (ElleNodeIsActive (jj) && !nodes_Check[jj])
                {
                  p = ElleNode (jj);	// get node struct and convert
                  // coordinates
                  xy.x = p->x;
                  xy.y = p->y;

                  if (EllePtInRegion (boundbox, 4, &xy))	// if in
                    // the box
                    {
                      for (jjj = 0; jjj < 16; jjj++)	// loop through
                        // possible nodes
                        // in particle
                        {
                          if (runParticle->elle_Node[jjj] == -1)	// if
                            // free
                            {
                              runParticle->elle_Node[jjj] = jj;	// fill
                              nodes_Check[jj] = true;	// flag up
                              break;
                            }
                          else if (runParticle->elle_Node[jjj] == jj)	// in
                            // case
                            // node
                            // is
                            // in
                            // there
                            {
                              break;	// we can exit
                            }
                        }
                    }
                }
            }
        }

      runParticle = runParticle->nextP;	// loop in particles
    }

  cout << " done with grain definitions " << endl;

  // ----------------------------------------------------------------------
  // just a test to see if all Elle nodes are connected now
  // ----------------------------------------------------------------------

  i = 0;

  for (jj = 0; jj < max; jj++)	// loop through Elle nodes
    {
      if (ElleNodeIsActive (jj) && !nodes_Check[jj])	// if node is
        // active and has
        // not been
        // checked
        {
          p = ElleNode (jj);	// define xy coordinates and pass out to
          // interface

          cout << "node not connected to lattice" << jj << " "
          << p->x << " " << p->y << endl;

          i++;
        }
    }
  if (i != 0)
    cout << i << endl;	// how many nodes were not connected

  UpdateElle();
}






// MAKE_REPULSION_BOX
/********************************************************************************
 * function is called from make mike function in the lattice class. 
 * It creates a square lattice which lies on top of the triangular one. 
 * Uses x and y as number of box. Boxlist is a onedimensional array so that 
 * x and y are translated to that array (then x are ones and y are the tens)
 * The box is filled at first with particles. Then a second run checks if there 
 * are more than one particle in the box. In the later case the first particle 
 * in the box gets a pointer to the next one in the box and the second 
 * one a pointer to the third one etc.. (pointer defined in particle.h as 
 * next_inBox). Each particle itself has a pointer to the eight boxes around
 * it and its own box (also defined in particle.h as my_Neig_Box).
 *
 * daniel spring and summer 2002
 *****************************************************************************/

// -------------------------------------------------------------------------
// function MakeRepulsionBox in lattice class
//
// called from Lattice::MakeLattice()
// -------------------------------------------------------------------------

void
Lattice::MakeRepulsionBox ()
{
  int i, ii;		// counter
  double fx, fy;
  int x, y;		// position in box
  int size;		// size of box in 1d

  // ---------------------------------------
  // fill main box
  // ---------------------------------------

  size = (particlex * (particlex * 2)) + particlex;	// size of box is
  // cubic !

  // ------------------------------------------------------
  // first we run through all particles and the whole box
  // and just fill the box with particles
  // ------------------------------------------------------

  runParticle = &refParticle;

  for (i = 0; i < numParticles; i++)
    {
      runParticle->next_inBox = NULL;
      runParticle = runParticle->nextP;
    }

  cout << "fill box" << endl;


  for (ii = 0; ii < numParticles; ii++)	// run through
    // particlelist
    {

      // -----------------------------------------------------
      // first get integers from position
      // this may be problematic depending on scaling
      // -----------------------------------------------------

      fx = ((runParticle->xpos) * particlex);
      fy = ((runParticle->ypos) * particlex);
      x = int (fx);
      y = int (fy);

      // ----------------------------------------------
      // convert x and y positions to a onedimensional
      // array, x are ones, y are tens
      // -----------------------------------------------

      x = (y * particlex * 2) + x;

      repBox[x] = runParticle;	// pointer to box
      runParticle->box_pos = x;	// position in box for particle

      runParticle = runParticle->nextP;	// run in particle list
    }


  // ------------------------------------------------------------------------
  // now we have to check for secondary and tertiary particles in
  // the boxes. For that we just run through all the particles
  // and check whether they are in the box. If not then another
  // particle is in that box and we add the particle to the local
  // box lists. So in each box there may be one particle and this
  // particle can have endless tails of other particles that are also
  // in the box.
  // ------------------------------------------------------------------------

  cout << "tailparticles" << endl;

  for (i = 0; i < numParticles; i++)	// run through list
    {

      // -------------------------------------------------------------
      // again integer x and y
      // -------------------------------------------------------------

      fx = ((runParticle->xpos) * particlex);
      fy = ((runParticle->ypos) * particlex);
      x = int (fx);
      y = int (fy);

      // cout << x << "," << y << endl;

      // ---------------------------------------------------------
      // and make one-dimensional
      // ---------------------------------------------------------

      x = (y * particlex * 2) + x;

      // cout << x << endl;
      // ----------------------------------------------------
      // if the particle is not in the box
      // check the tail of the existing particle
      // ----------------------------------------------------

      if (runParticle->nb != repBox[x]->nb)	// if not in box
        {
          // cout << repBox[x]->next_inBox << endl;

          if (repBox[x]->next_inBox != NULL)	// does a tailparticle
            // exist ?
            {
              // cout << repBox[x]->next_inBox << endl;
              if (repBox[x]->next_inBox->next_inBox != NULL)	// does a
                // second
                // tailparticle
                // exist ?
                {

                  // ---------------------------------------------------------------------
                  // at the moment only four tailparticles can exist
                  // a priori. Works so far but might not be scale
                  // independent
                  // there comes a warning at this point
                  // can make this an endless loop then this problem
                  // is gone.
                  // ---------------------------------------------------------------------

                  if (repBox[x]->next_inBox->next_inBox->next_inBox != NULL)	// third
                    {
                      if (repBox[x]->next_inBox->next_inBox->next_inBox->next_inBox != NULL)	// four
                        {
                          cout << "more than four ?" << runParticle->box_pos << endl;
                        }
                      else
                        {
                          repBox[x]->next_inBox->next_inBox->next_inBox->next_inBox = runParticle;	// tailpointer
                          runParticle->box_pos = x;	// and position
                        }
                    }
                  else
                    {
                      repBox[x]->next_inBox->next_inBox->next_inBox = runParticle;	// tailpointer
                      runParticle->box_pos = x;	// and position
                    }
                }
              else
                {

                  // ---------------------------------------------------------------
                  // in here means we had one tailparticle so that we
                  // add our
                  // particle to the tail of the first tailparticle
                  // ---------------------------------------------------------------

                  repBox[x]->next_inBox->next_inBox = runParticle;	// tailpointer
                  runParticle->box_pos = x;	// and position
                }
            }
          else
            {
              // ------------------------------------------------------------
              // in here we have a particle in the box but no tail so we
              // add our particle to the tail of the one in the box.
              // ------------------------------------------------------------

              repBox[x]->next_inBox = runParticle;	// tailpointer
              runParticle->box_pos = x;	// and position
            }
        }
      runParticle = runParticle->nextP;	// and next in list
    }
}





// FIND_UNODE_ELLE
/****************************************************************************
 * Function that finds pointer to an Elle Unode. 
 * each particle has a pointer to a Unode. Defined in the particle.h class.
 * the function ElleGetParticleUnode takes an int, the Unode number or id and
 * returns the Unode pointer itself to that Unode. 
 * function defined in unodes.cc (because need the *Unodes vector)
 *
 * daniel spring 2002
 ***************************************************************************/

// --------------------------------------------------------------------------------
// function FindUnodeElle() of Lattice class
//
// function is called from Lattice::MakeLattice()
// --------------------------------------------------------------------------------

void
Lattice::FindUnodeElle ()
{

  // -----------------------------------------------------------------------------------
  // variables
  // -----------------------------------------------------------------------------------

  int i;			// counter variables

  // -----------------------------------------------------------------------------------
  // define starting point
  // -----------------------------------------------------------------------------------

  firstParticle = &refParticle;	// get the refParticle, start for
  // particle list

  runParticle = firstParticle;	// set pointer for run through
  // list

  // -------------------------------------------------------------------------------------------
  // this loop through all particles calls "Unode
  // *ElleGetParticleUnode(int i)" in unodes.cc of
  // basic Elle functions. The function receives the number of the Unode
  // and returns directly
  // a pointer of type Unode.
  // -------------------------------------------------------------------------------------------

  for (i = 0; i < numParticles; i++)	// run through all Unodes and
    // particles and connect
    {
      runParticle->p_Unode = ElleGetParticleUnode (i);	// get my Unode
      // pointer and
      // define in
      // particle
      runParticle = runParticle->nextP;	// next in list
    }

  // ----------------------------------------------------------
  // this is just a test; dont want null pointers
  // ----------------------------------------------------------

  runParticle = firstParticle;

  for (i = 0; i < numParticles; i++)
    {
      if (!runParticle->p_Unode)	// if no Unode in particle
        {
          cout << " no Unode " << runParticle->nb << endl;
        }
      runParticle = runParticle->nextP;
    }
}

/********************************************************************
* Function for the viscoelastic Deformation, updates the length 
* of springs 
*
********************************************************************/



void
Lattice::UpdateSpringLength ()
{
  int j, i, k, rad_nb_buffer;
  float angle, buffer, buffer2, spring_angle_b;

  for (j = 0; j < numParticles; j++)
    {
      if (!runParticle->is_lattice_boundary)
        {
          for (i = 0; i < 6; i++)
            {
              if (runParticle->neigP[i])
                {
                  buffer = 2 * Pi;	// Startwert

                  for (k = 0; k < 9; k++)
                    {
                      if (k != 4)
                        {
                          switch (k)
                            {
                            case 0:
                              angle = 3.926990817;
                              break;
                            case 1:
                              angle = 4.71238898;
                              break;
                            case 2:
                              angle = 5.497787144;
                              break;
                            case 3:
                              angle = 3.141592654;
                              break;
                            case 5:
                              angle = 0;
                              break;
                            case 6:
                              angle = 2.35619449;
                              break;
                            case 7:
                              angle = 1.570796327;
                              break;
                            case 8:
                              angle = 0.7853981634;
                              break;
                            }

                          spring_angle_b =
                            runParticle->
                            spring_angle
                            [i] +
                            runParticle->
                            rot_angle;

                          if (spring_angle_b <
                              0.0)
                            spring_angle_b
                            =
                              2.0 *
                              3.14159265
                              +
                              spring_angle_b;
                          else if (spring_angle_b > 2.0 * 3.14159265)
                            spring_angle_b
                            =
                              spring_angle_b
                              -
                              2.0 *
                              3.14159265;

                          // hier suche die spring, die am dichtesten am
                          // referenz-winkel liegt, unbeachtet von vz,
                          // spruengen,...
                          buffer2 =
                            fabs (angle -
                                  spring_angle_b);

                          if (buffer2 < buffer)
                            {
                              buffer = buffer2;
                              rad_nb_buffer
                              = k;
                            }

                          if (k == 5)	// dh angle == 0.0
                            {
                              angle = 2.0 *
                                      Pi;

                              buffer2 =
                                fabs
                                (angle
                                 -
                                 spring_angle_b);

                              if (buffer2 <
                                  buffer)
                                {
                                  buffer = buffer2;
                                  rad_nb_buffer
                                  =
                                    k;
                                }
                            }
                        }
                    }

                  runParticle->rad[i] =
                    runParticle->
                    rep_rad[rad_nb_buffer];
                }
            }
        }

      runParticle = runParticle->nextP;
    }
}

/***********************************************************
*
* for the viscoelastic code. Neighbour list is imporant 
* if springs are fractured and the particle is still 
* deforming. 
*
************************************************************/

void
Lattice::UpdateNeighList ()
{

  long i, j, k, l;
  int box, pos, neig_nb;
  Particle *neig;

  runParticle->done = true;

  // no connected neighbours!
  for (i = 0; i < 8; i++)
    {
      if (runParticle->neigP[i])
        runParticle->neigP[i]->done = true;
    }

  for (i = 0; i < runParticle->merk; i++)
    {
      runParticle->neigh_list[i]->done = true;
    }

  // update the list of neighbors in the particle itself!
  if (!repBox[runParticle->box_pos])
    {
      cout << " problem replist " << endl;
    }
  else
    {

      // ---------------------------------------------------
      // nine positions in 2d box 3x3
      // ---------------------------------------------------

      for (i = 0; i < 3; i++)
        {
          for (j = 0; j < 3; j++)
            {

              // ------------------------------------------
              // case upper or lower row (+-particlex)
              // ------------------------------------------

              if (j == 0)
                box = 2*particlex * (-1);
              if (j == 1)
                box = 0;
              if (j == 2)
                box = 2*particlex;
              // -----------------------------------------
              // now check this position
              // all particles in this position !!
              // -----------------------------------------

              pos = runParticle->box_pos + (i - 1) + box;

              // -----------------------------------------
              // repBox should always be positive
              // -----------------------------------------

              if (pos > 0)
                {
                  neig = repBox[pos];	// helppointer for while loop

                  while (neig)	// as long as there is somebody
                    {
                      if (!neig->done)
                        {
                          if (neig->xpos)	// just in case
                            {
                              runParticle->neigh_list[runParticle->merk]=neig;
                              runParticle->merk++;
                              neig->done = true;
                            }
                        }

                      if (neig->next_inBox)	// if there is anotherone
                        // in box
                        {
                          neig = neig->next_inBox;	// shift the help
                          // pointer
                        }
                      else
                        {
                          break;	// if not go out to next box
                        }
                    }
                }
            }
        }
    }

  // remove the "done"-flags
  for (i = 0; i < 8; i++)
    {
      if (runParticle->neigP[i])
        runParticle->neigP[i]->done = false;
    }

  for (i = 0; i < runParticle->merk; i++)
    {
      runParticle->neigh_list[i]->done = false;
    }

  runParticle->done = false;

  //if (runParticle->merk > 88)
  //cout << "break" << endl;

}

/**************************************************
* Clean the neighour list 
*
***************************************************/

void
Lattice::EraseNeighList ()
{
  int i, z;

  for (i = 0; i < numParticles; i++)
    {
      for (z = 0; z < 88; z++)
        runParticle->neigh_list[z] = NULL;

      runParticle->merk = 0;

      runParticle = runParticle->nextP;
    }
}



// FULL_RELAX
/********************************************************************
 * This function does a full relaxation. it runs through the particle
 * list and calls the function Relax within each particle (defined in 
 * particle.cc. This function will return a true if the particle is moved
 * i.e. is relaxing. Function runs till all particles are relaxed that is
 * they are not moving anymore.
 * Function returns a flag for broken bonds 
 *
 * daniel spring 2002
 *********************************************************************/

// -----------------------------------------------------------------------------
// function FullRelax() of Lattice class
// returns flag for broken bonds
//
// called from Lattice::Relaxation()
// -----------------------------------------------------------------------------

bool
Lattice::FullRelax ()
{
  // ---------------------------------------------------
  // local variables
  // ---------------------------------------------------

  bool box_move = false;	// did particle move out of box ?
  bool relaxflag = true;	// flag for relaxation loop
  bool breakBond = false;	// second flag for broken bonds
  int i, j = 0;		// loop counters, j counts relaxation
  // steps
  int zaehler = 0;

  int zaehler2 = 0;

  float debug;
  // -----------------------------------------------------------------
  // the relaxation loop, will run until all particles are
  // relaxed. Loops through all particles and calls Relax function
  // in particle objects. Gets a flag from each object indicating
  // whether or not it had to be relaxed.
  // -----------------------------------------------------------------

  // cout << "relax" << endl;

  runParticle = &refParticle;	// go to beginning of list

  debug = 0.0;

  Tilt ();

  // UpdateSpringLength();

  do
    {

      relaxflag = false;	// new relaxation loop if flag stays false
      // jumps out of loop
      j++;		// count relaxation steps

      // cout << j << endl; // use this if you want to check how fast
      // relaxation goes

      // -------------------------------------------------------------------------------
      // loop through all particles and call Relax in the particle class
      // and set flags
      // note that we do not have to redefine the beginning of the list
      // since it is
      // circular, at the end the pointer points again to the first
      // particle
      // -------------------------------------------------------------------------------

      for (i = 0; i < numParticles; i++)	// loop through all
        // particles
        {
          // ------------------------------------------------------------------------------
          // this does a number of things. It calls the function Relax
          // in runParticle
          // and gives that function the relaxationthreshold defined in
          // the lattice class
          // (see the constructor). It receives a flag that is false if
          // the particle
          // was moved i.e. relaxed. If the flag is false the relaxflag
          // is turned true
          // which means that after all particles are done a new
          // relaxation loop will start
          // -------------------------------------------------------------------------------
          if (!runParticle->isHole)
            {
              if (!runParticle->
                  Relax (relaxthres, repBox, particlex,
                         walls, right_wall_pos,
                         left_wall_pos, lower_wall_pos,
                         upper_wall_pos, wall_rep, false,
                         debug_nb,visc_rel))
                relaxflag = true;

              // nur ne lausige abfrage. maximaler versatz.
              if (sqrt
                  (runParticle->xx * runParticle->xx +
                   runParticle->yy * runParticle->yy) >
                  debug)
                {
                  debug = sqrt (runParticle->xx *
                                runParticle->xx +
                                runParticle->yy *
                                runParticle->yy);

                  if (zaehler2 > 3)
                    debug_nb = runParticle->nb;
                }
              // ----------------------------------------------------------------------------
              // check if particle moved out of box, if it did call
              // function
              // to go to new position in box
              // ----------------------------------------------------------------------------

              runParticle->ChangeBox (repBox, particlex);

              UpdateNeighList ();
            }

          runParticle = runParticle->nextP;	// continue in particle
          // list

        }



    }
  while (relaxflag);	// loop until relaxflag is staying false
  // through a whole relaxation

  //cout << "maximum verschiebung: " << debug << endl;

  // this loop through all particles calculates the stress tensor for
  // each particle and
  // also determines which bonds will break
  // took that out of the relaxroutine so that that is really simple and
  // as fast
  // as it can be.


  for (i = 0; i < numParticles; i++)
    {
      // call routine in particles that calculates stress and returns a
      // true if
      // a bond is broken
      if (!runParticle->isHole)
        {
          if (runParticle->Relax (repBox, particlex,visc_rel,walls, right_wall_pos,
                                  left_wall_pos, lower_wall_pos,
                                  upper_wall_pos, wall_rep))
            breakBond = true;
        }
      runParticle = runParticle->nextP;
    }

  // Now erase the neigh-list, that has been updated every run-through
  // above
  // (in order to not overwhelm it)
  EraseNeighList ();

  return breakBond;	// and return whether or not a bond was
  // broken
}



// SET_BOUNDARIES
/*************************************************************************************
 * This function sets whatever boundary conditions we want. At the moment it just 
 * fixes the y position of the upper and lower row so that these are not 
 * moved during the relaxation routine
 *
 * daniel spring 2002
 *
 * bug fixed for even and uneven lattices Daniel december 2004
 ************************************************************************************/

// -----------------------------------------------------------------------------------
// function SetBoundaries() of Lattice class
//
// called from Lattice::MakeLattice()
// -----------------------------------------------------------------------------------

void
Lattice::SetBoundaries ()
{
  // ------------------------------------
  // local variables
  // ------------------------------------

  int i, j, jj;		// loop counter

  runParticle = firstParticle = &refParticle;	// set pointers to
  // beginning of list

  // ---------------------------------------------------------------------------------
  // set flag in particle object telling it not to move its y position
  // set this flag first for the lower row == particles form 0 to
  // particlex-1
  // ---------------------------------------------------------------------------------

  for (i = 0; i < particlex; i++)
    {
      runParticle->fix_y = true;	// set y
      runParticle->is_lattice_boundary = true;	// define as part
      // of boundary


      // this is a messy bit that I have to change... at the moment
      // boundary particles
      // cannot have broken bonds... otherwise it crashes.
      // have to define walls some day

      for (j = 0; j < 8; j++)
        {
          if (runParticle->neigP[j])
            {
              runParticle->no_break[j] = true;
              // runParticle->break_Str[j] = 1000000.0;

              for (jj = 0; jj < 8; jj++)
                {
                  if (runParticle->neigP[j]->neigP[jj])
                    {
                      if (runParticle->neigP[j]->
                          neigP[jj]->nb ==
                          runParticle->nb)
                        {

                          // make it unbreakable

                          runParticle->
                          neigP[j]->
                          no_break[jj] =
                            true;

                        }
                    }
                }

            }
        }
      runParticle = runParticle->nextP;
    }

  // --------------------------------------------------------------------------------
  // then go to the last particle and run backwards particlex times to
  // fix
  // the upper row of particles
  // --------------------------------------------------------------------------------

  runParticle = firstParticle->prevP;

  for (i = 0; i < particlex; i++)
    {
      runParticle->fix_y = true;
      runParticle->is_lattice_boundary = true;
      for (j = 0; j < 8; j++)
        {
          if (runParticle->neigP[j])
            {
              runParticle->no_break[j] = true;


              for (jj = 0; jj < 8; jj++)
                {
                  if (runParticle->neigP[j]->neigP[jj])
                    {
                      if (runParticle->neigP[j]->
                          neigP[jj]->nb ==
                          runParticle->nb)
                        {
                          runParticle->
                          neigP[j]->
                          no_break[jj] =
                            true;

                        }
                    }
                }
            }
        }
      runParticle = runParticle->prevP;
    }

  // ------------------------------------------------------------------
  // now do the same for the side boundaries (set x not to move)
  // ------------------------------------------------------------------

  runParticle = firstParticle;	// first do first particle
  runParticle->fix_x = true;	// fix x
  runParticle->is_lattice_boundary = true;	// define as part of
  // boundary

  do
    {
      for (i = 0; i < particlex; i++)	// loop two rows
        {
          runParticle = runParticle->nextP;
        }

      // fix left boundary

      runParticle->fix_x = true;
      runParticle->is_lattice_boundary = true;
      for (j = 0; j < 8; j++)
        {
          if (runParticle->neigP[j])
            {
              runParticle->no_break[j] = true;


              for (jj = 0; jj < 8; jj++)
                {
                  if (runParticle->neigP[j]->neigP[jj])
                    {
                      if (runParticle->neigP[j]->
                          neigP[jj]->nb ==
                          runParticle->nb)
                        {
                          runParticle->
                          neigP[j]->
                          no_break[jj] =
                            true;

                        }
                    }
                }
            }
        }

      // fix right boundary

      runParticle->prevP->fix_x = true;
      runParticle->prevP->is_lattice_boundary = true;
      for (j = 0; j < 8; j++)
        {
          if (runParticle->prevP->neigP[j])
            {
              runParticle->prevP->no_break[j] = true;


              for (jj = 0; jj < 8; jj++)
                {
                  if (runParticle->prevP->neigP[j]->
                      neigP[jj])
                    {
                      if (runParticle->prevP->
                          neigP[j]->neigP[jj]->nb ==
                          runParticle->prevP->nb)
                        {
                          runParticle->prevP->
                          neigP[j]->
                          no_break[jj] =
                            true;

                        }
                    }
                }
            }
        }

    }
  while (runParticle != firstParticle);	// loop once through all
  // lattice
}

/************************************************************
* Function that sets "wall boundaries" around the particle
* box. Walls are straight elastic lines at the moment
*
* Walls are only repulsive at the moment, therefore 
* they cannot build up tensile strain !
*
* Daniel 2005/6
************************************************************/

void
Lattice::SetWallBoundaries (int both, float constant)
{
  // ------------------------------------
  // local variables
  // ------------------------------------

  int i, j;		// loop counter

  if (both == 0)
    {
      for (i = 0; i < numParticles; i++)
        {
          runParticle->fix_y = false;	// set y
          runParticle->fix_x = false;
          runParticle->is_lattice_boundary = false;	// define as part
          // of boundary

          for (j = 0; j < 8; j++)
            {
              if (runParticle->neigP[j])
                runParticle->no_break[j] = false;
            }

          runParticle = runParticle->nextP;
        }
    }

  walls = true;

  // positions of walls

  lower_wall_pos = firstParticle->nextP->ypos - firstParticle->radius;
  left_wall_pos = firstParticle->xpos - firstParticle->radius;

  runParticle = firstParticle->prevP;

  upper_wall_pos = runParticle->ypos + runParticle->radius;
  right_wall_pos = runParticle->xpos + runParticle->radius;

  for (i = 0; i < particlex; i++)
    {
      runParticle = runParticle->prevP;
    }
  if ((runParticle->xpos + runParticle->radius) > right_wall_pos)
    right_wall_pos = runParticle->xpos + runParticle->radius;

  cout << "wallpos   " << lower_wall_pos << " lower " << upper_wall_pos
  << "upper " << endl;
  cout << left_wall_pos << " left " << right_wall_pos << " right " <<
  endl;

  wall_rep = constant; //elastic constant of wall
}



// RELAXATION
/*********************************************************************************
 * function that does a full relaxation 
 * calls FullRelax() in Lattice class. That function does a full relaxatiion
 * until the lattice is relaxed and returns a flag wether or not a new bond is
 * broken. If that is the case the function searches for the broken bond which
 * is most probable to break, breaks it and starts the relaxation again until the
 * lattice is fully relaxed and all possible bonds are broken
 *
 * daniel spring 2002
 **********************************************************************************/

// ----------------------------------------------------------------------------------
// function Relaxation() of Lattice class
//
// called in Lattice:MakeLattice() and Lattice:DeformLattice()
// ----------------------------------------------------------------------------------

void
Lattice::Relaxation ()
{
  // -------------------------------------------------------
  // some local variables
  // -------------------------------------------------------

  float maxbreak;		// probability of bond to break
  int i, z;		// loop counter,z for debugging

  //cout << "in BoxRad" << endl;
  BoxRad ();		// function writes splined radius-values
  // in particle->rep_rad[9]
  //cout << "out BoxRad" << endl;

  if (internal_break == 1)
    local_nbBreak = 0;

  // -------------------------------------------------------------------------------------
  // loop through the full relaxation routine until no more bonds break
  // -------------------------------------------------------------------------------------

  // preliminary settings
  EraseNeighList ();

  while (FullRelax ())
    {
      // -----------------------------------------------------------------
      // if in here a bond wants to break
      // -----------------------------------------------------------------

      nbBreak++;	// one more broken bond
      local_nbBreak++;	// counter for fracture plot

      // --------------------------------------------------------------------
      // This plots a picture in between timesteps if the user defines
      // a number larger than zero. Each time the code fractures
      // internally
      // that is within a time step n times it plots a picture.
      // --------------------------------------------------------------------

      if (fract_Plot_Numb != 0)	// in this case (default) no
        // internal picts taken
        {
          if (local_nbBreak == fract_Plot_Numb)	// otherwise if
            // more fractures
            // than n
            {
              UpdateElle ();
              local_nbBreak = 0;	// and reset pointer
            }
        }

      cout << "Broken Bonds " << nbBreak << endl;	// tell the world

      // -----------------------------------------------------------------------
      // now run through particles and find particle that contains the
      // bond
      // with the highest probability to break. Only break that bond.
      // What the other do that would break also will be seen after
      // the next relaxation routine. Want to relax lattice after we
      // break
      // a bond
      // ------------------------------------------------------------------------

      maxbreak = 0.0;	// set to zero

      runParticle = &refParticle;	// beginning of list

      for (i = 0; i < numParticles;)	// run through list
        {
          if (runParticle->mbreak > maxbreak)	// if the particle max
            // breaking probablity is
            // higher
            {
              maxbreak = runParticle->mbreak;	// thats the new number
              preRunParticle = runParticle;	// and catch the particle
            }
          runParticle = runParticle->nextP;
          i++;
        }
      //cout << preRunParticle->nb << endl;

      preRunParticle->BreakBond ();	// call BreakBond() in the object
      // and break
      if (visc_flag == true)
        preRunParticle->visc_flag = true;	// debugging
      // UpdateElle();
    }
}



// UPDATE_ELLE
/**************************************************************************************
 * this function writes the new values to the Unode objects for elle plotting
 * and also adjusts positions of Unodes
 * Calls ElleUpdate to talk to the Elle interface
 *
 * daniel summer and spring 2002
 * Rianne, Daniel and Jochen December 2002, Jan 2003
 *
 * add security stop 
 *
 * Daniel March 2003
 **************************************************************************************/

// --------------------------------------------------------------------------------
// function UpdateElle() of Lattice class
//
// called from Lattice::MakeLattice() and Lattice::DeformLattice()
// --------------------------------------------------------------------------------

void
Lattice::UpdateElle ()
{
  // ----------------------------------------------------------
  // local variables
  // ----------------------------------------------------------

  double pressure;	// variable for the sum of the trace of
  // the strain tensor sxx+syy
  Coords xy;		// coordinate structure defined in
  // attrib.h
  int i, ii, max;		// counter
  NodeAttrib *p;		// for nodes (x and y)
  float smax, smin, sxx, syy, sxy, energy;	// some local variables for stresses etc.
  float uxy,uxx,uyy,lame1,lame2,pois;
  double frac;

	pois = 0.33333;
	
  runParticle = &refParticle;	// jump to beginning of list

  max = ElleMaxNodes();

  for (ii = 0; ii < max; ii++)
    {
      nodes_Check[ii] = false;
    }


  // --------------------------------------------------------------------------------------
  // loop through the list of particles and change attributes and
  // positions in the Undoes
  // has to be called everytime something is changed and we talk to elle
  // again
  // this should include all kinds of plotting things, stresses,
  // energies, fractures etc.
  // --------------------------------------------------------------------------------------

  for (i = 0; i < numParticles; i++)	// and loop again
    {

      // otherwise stuff gets too long

      sxx = runParticle->sxx;
      sxy = runParticle->sxy;
      syy = runParticle->syy;

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
						
						runParticle->p_Unode->setAttribute(U_ENERGY,runParticle->eEl);
		
      // -----------------------------------------------------------------------
      // stress tensor is set in the particle object themselves at
      // relaxation
      // therefore call particles, get stresses and calculate whatever
      // you want, then pass to Unodes.
      // here now pressure = trace of tensor sxx + syy
      // -----------------------------------------------------------------------

      pressure = (sxx + syy) / 2.0;	// calc


      // ----------------------------------------------------------------------
      // this function setAttribute is defined in unodeP.h which
      // includes the
      // definition for the class plus all the functions. There is no
      // unodesP.cc in Elle
      // this function wants a reference int for which value we want to
      // change that is CONC_A at the moment (normally used for
      // concentration)
      // and also wants the value here as a double.
      //
      // this call just calls the function setAttribute in the Unode
      // connected to the particle
      // -----------------------------------------------------------------------
      runParticle->p_Unode->setAttribute (U_MEAN_STRESS,pressure);

      runParticle->p_Unode->setAttribute(U_DENSITY,runParticle->fluid_particles);

      //runParticle->p_Unode->setAttribute(U_ATTRIB_B,runParticle->grain);

      runParticle->p_Unode->setAttribute (U_YOUNGSMODULUS,runParticle->young);

      // ------------------------------------------------------------------------
      // And pass some other values like the full stress tensor
      // in N_Attrib_A N_Attrib_B and N_Attrib_C
      // ------------------------------------------------------------------------

      // runParticle->p_Unode->setAttribute(CAXIS,runParticle->sxx);

      // runParticle->p_Unode->setAttribute(EULER_3,runParticle->syy);

      // runParticle->p_Unode->setAttribute(VISCOSITY,runParticle->sxy);

      // -------------------------------------------------------------------------
      // determine smax - smin (Eigenvalues)
      // -------------------------------------------------------------------------

      smax = ((sxx + syy) / 2.0) +
             sqrt (((sxx - syy) / 2.0) * ((sxx - syy) / 2.0) +
                   sxy * sxy);

      smin = ((sxx + syy) / 2.0) -
             sqrt (((sxx - syy) / 2.0) * ((sxx - syy) / 2.0) +
                   sxy * sxy);

      pressure = smax - smin;

      runParticle->p_Unode->setAttribute (U_DIF_STRESS, pressure);

      frac = 0.0;
      if (runParticle->draw_break)
        {
          if (runParticle->no_spring_flag)
            frac = -0.3;
          else if (runParticle->visc_flag)
            frac = -0.6;
          else
            frac = -1.0;

          runParticle->p_Unode->setAttribute (U_FRACTURES, frac);	// something
          // to
          // visualize
          // fractures
        }
      else
        runParticle->p_Unode->setAttribute (U_FRACTURES, frac);

      pressure = 0.0;

      if (!transition)
        {

          if (runParticle->isHoleBoundary)
            pressure = 1.0;

          if (runParticle->isHole)
            // pressure = runParticle->conc;
            pressure = -1;

          runParticle->p_Unode->setAttribute (U_PHASE, pressure);
        }
      else
        {
          pressure = (runParticle->mineral - 1);
          runParticle->p_Unode->setAttribute(U_PHASE, pressure);
        }

      pressure = runParticle->temperature;

      runParticle->p_Unode->setAttribute (U_TEMPERATURE, pressure);

      if (runParticle->draw_break)
        {
          smin = smin * 10.0;	// something to visualize fractures
        }

      pressure = smax;

      // runParticle->p_Unode->setAttribute(ENERGY,pressure);

      // pressure = arctan ((smax-sxx)/sxx);

      // runParticle->p_Unode->setAttribute(DISLOCATION_DENSITY,pressure);

      // -------------------------------------------------------------------------
      // and now also change the Unode position because the particles
      // have moved
      // I am not exactly shure what the consequences of that for Elle
      // are, maybe
      // something crashes at some point.
      // since the function setPosition in the Unode class wants a Coord
      // structure defined in attrib.h in Elle we create one and give it
      // the
      // x and y positions in our particle and then pass it to the Unode
      // function
      // -------------------------------------------------------------------------

      xy.x = runParticle->xpos;	// pass x
      xy.y = runParticle->ypos;	// pass y

      runParticle->p_Unode->setPosition (&xy);	// and pass that to the
      // Unode

      // ------------------------------------------------------------------------------
      // This routine moves Elle nodes according to movements in the
      // particle lattice
      // At the moment a bit rough, node just moved on top of particle
      // depends a bit on resolution, make triangulation of unodes crash
      // !
      // ------------------------------------------------------------------------------





      for (ii = 0; ii < 32; ii++)
        {
          if (runParticle->elle_Node[ii] != -1)
            {
              if (!nodes_Check[runParticle->elle_Node[ii]])
                {
                  ElleNodePosition(runParticle->elle_Node[ii],&xy);

                  nodes_Check[runParticle->elle_Node[ii]] = true;

                  xy.y = xy.y + (runParticle->ypos - runParticle->oldy);
                  xy.x = xy.x + (runParticle->xpos - runParticle->oldx);
                  ElleSetPosition (runParticle->elle_Node[ii],&xy);
                }
            }
        }
      runParticle->oldx = runParticle->xpos;
      runParticle->oldy = runParticle->ypos;

      runParticle = runParticle->nextP;	// and the next particle
    }

  // ---------------------------------------------------------------------------
  // call ElleUpdateDisplay() an Elle function that updates the interface
  // then the new values will be plotted
  // if security stop is set and max number of picts is reached dont
  // plot
  // pict anymore.
  // ---------------------------------------------------------------------------

/* LE output filenames will reflect number of picts rather than stages
      Stages updated in experiment loop to indicate time
*/
  if (!set_max_pict)
    {
      // cout << "interface" << endl;
      if (ElleDisplay()) ElleUpdateDisplay() ;
    }
  else if (num_pict < max_pict)
    {
      // cout << "interface" << endl;
      if (ElleDisplay()) ElleUpdateDisplay() ;
      ElleAutoWriteFile(num_pict);
      num_pict = num_pict + 1;
//LE
      /*max_pict = max_pict + 1;*/
    }
}

void Lattice::UpdateElle (bool movenode)
{
  // ----------------------------------------------------------
  // local variables
  // ----------------------------------------------------------

  double pressure;	// variable for the sum of the trace of
  // the strain tensor sxx+syy
  Coords xy;		// coordinate structure defined in
  // attrib.h
  int i, ii, max;		// counter
  NodeAttrib *p;		// for nodes (x and y)
  float smax, smin, sxx, syy, sxy, energy;	// some local variables for stresses etc.
 float uxy,uxx,uyy,lame1,lame2,pois;
  double frac;

	pois = 0.33333;

  runParticle = &refParticle;	// jump to beginning of list

  max = ElleMaxNodes();

  for (ii = 0; ii < max; ii++)
    {
      nodes_Check[ii] = false;
    }


  // --------------------------------------------------------------------------------------
  // loop through the list of particles and change attributes and
  // positions in the Undoes
  // has to be called everytime something is changed and we talk to elle
  // again
  // this should include all kinds of plotting things, stresses,
  // energies, fractures etc.
  // --------------------------------------------------------------------------------------

  for (i = 0; i < numParticles; i++)	// and loop again
    {

      // otherwise stuff gets too long

      sxx = runParticle->sxx;
      sxy = runParticle->sxy;
      syy = runParticle->syy;

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
						
		runParticle->p_Unode->setAttribute(U_ENERGY,runParticle->eEl);
		
      // -----------------------------------------------------------------------
      // stress tensor is set in the particle object themselves at
      // relaxation
      // therefore call particles, get stresses and calculate whatever
      // you want, then pass to Unodes.
      // here now pressure = trace of tensor sxx + syy
      // -----------------------------------------------------------------------

      pressure = (sxx + syy) / 2.0;	// calc


      // ----------------------------------------------------------------------
      // this function setAttribute is defined in unodeP.h which
      // includes the
      // definition for the class plus all the functions. There is no
      // unodesP.cc in Elle
      // this function wants a reference int for which value we want to
      // change that is CONC_A at the moment (normally used for
      // concentration)
      // and also wants the value here as a double.
      //
      // this call just calls the function setAttribute in the Unode
      // connected to the particle
      // -----------------------------------------------------------------------
      runParticle->p_Unode->setAttribute (U_MEAN_STRESS,pressure);

      runParticle->p_Unode->setAttribute(U_DENSITY,runParticle->fluid_particles);

      //runParticle->p_Unode->setAttribute(U_ATTRIB_B,runParticle->grain);

      runParticle->p_Unode->setAttribute (U_YOUNGSMODULUS,runParticle->young);

      // ------------------------------------------------------------------------
      // And pass some other values like the full stress tensor
      // in N_Attrib_A N_Attrib_B and N_Attrib_C
      // ------------------------------------------------------------------------

      // runParticle->p_Unode->setAttribute(CAXIS,runParticle->sxx);

      // runParticle->p_Unode->setAttribute(EULER_3,runParticle->syy);

      // runParticle->p_Unode->setAttribute(VISCOSITY,runParticle->sxy);

      // -------------------------------------------------------------------------
      // determine smax - smin (Eigenvalues)
      // -------------------------------------------------------------------------

      smax = ((sxx + syy) / 2.0) +
             sqrt (((sxx - syy) / 2.0) * ((sxx - syy) / 2.0) +
                   sxy * sxy);

      smin = ((sxx + syy) / 2.0) -
             sqrt (((sxx - syy) / 2.0) * ((sxx - syy) / 2.0) +
                   sxy * sxy);

      pressure = smax - smin;

      runParticle->p_Unode->setAttribute (U_DIF_STRESS, pressure);

      frac = 0.0;
      if (runParticle->draw_break)
        {
          if (runParticle->no_spring_flag)
            frac = -0.3;
          else if (runParticle->visc_flag)
            frac = -0.6;
          else
            frac = -1.0;

          runParticle->p_Unode->setAttribute (U_FRACTURES, frac);	// something
          // to
          // visualize
          // fractures
        }
      else
        runParticle->p_Unode->setAttribute (U_FRACTURES, frac);


      pressure = 0.0;

      if (runParticle->isHoleBoundary)
        pressure = 1.0;

      if (runParticle->isHole)
        // pressure = runParticle->conc;
        pressure = -1;

      runParticle->p_Unode->setAttribute (U_PHASE, pressure);

      pressure = runParticle->temperature;


      runParticle->p_Unode->setAttribute (U_TEMPERATURE, pressure);

      if (runParticle->draw_break)
        {
          smin = smin * 10.0;	// something to visualize fractures
        }

      pressure = smax;

      // runParticle->p_Unode->setAttribute(ENERGY,pressure);

      // pressure = arctan ((smax-sxx)/sxx);

      // runParticle->p_Unode->setAttribute(DISLOCATION_DENSITY,pressure);

      // -------------------------------------------------------------------------
      // and now also change the Unode position because the particles
      // have moved
      // I am not exactly shure what the consequences of that for Elle
      // are, maybe
      // something crashes at some point.
      // since the function setPosition in the Unode class wants a Coord
      // structure defined in attrib.h in Elle we create one and give it
      // the
      // x and y positions in our particle and then pass it to the Unode
      // function
      // -------------------------------------------------------------------------

      xy.x = runParticle->xpos;	// pass x
      xy.y = runParticle->ypos;	// pass y

      runParticle->p_Unode->setPosition (&xy);	// and pass that to the
      // Unode

      // ------------------------------------------------------------------------------
      // This routine moves Elle nodes according to movements in the
      // particle lattice
      // At the moment a bit rough, node just moved on top of particle
      // depends a bit on resolution, make triangulation of unodes crash
      // !
      // ------------------------------------------------------------------------------


      if(movenode || !runParticle->isHole)
        {
          for (ii = 0; ii < 32; ii++)
            {
              if (runParticle->elle_Node[ii] != -1)
                {
                  if (!nodes_Check[runParticle->elle_Node[ii]])
                    {
                      ElleNodePosition(runParticle->elle_Node[ii],&xy);

                      nodes_Check[runParticle->elle_Node[ii]] = true;

                      xy.y = xy.y + (runParticle->ypos - runParticle->oldy);
                      xy.x = xy.x + (runParticle->xpos - runParticle->oldx);
                      ElleSetPosition (runParticle->elle_Node[ii],&xy);
                    }
                }
            }
          runParticle->oldx = runParticle->xpos;
          runParticle->oldy = runParticle->ypos;
        }

      runParticle = runParticle->nextP;	// and the next particle
    }

  // ---------------------------------------------------------------------------
  // call ElleUpdate() an Elle function that updates the interface
  // then the new values will be plotted
  // if security stop is set and max number of picts is reached dont
  // plot
  // pict anymore.
  // ---------------------------------------------------------------------------

/* LE output filenames will reflect number of picts rather than stages
      Stages updated in experiment loop to indicate time
*/
  if (!set_max_pict)
    {
      // cout << "interface" << endl;
      if (ElleDisplay()) ElleUpdateDisplay();
    }
  else if (num_pict < max_pict)
    {
      // cout << "interface" << endl;
      if (ElleDisplay()) ElleUpdateDisplay();
      ElleAutoWriteFile(num_pict);
      num_pict = num_pict + 1;
      /*max_pict = max_pict + 1;*/
    }
}

// DEFORM_LATTICE
/*******************************************************************************
 * This function does whatever deformation we want to perform with the lattice
 * It is called from the outside by the User and does all that is needed for 
 * a timestep including all the relaxation etc. 
 * at the moment it receives a float for how much the walls are moved
 * as a deformation it moves upper and lower row of particles inwards if 
 * positive value
 *
 * Here we can include all kinds of stuff in the future
 *
 * daniel spring and summer 2002
 ********************************************************************************/

// ---------------------------------------------------------------------------------
// function DeformLattice(float move) in Lattice class
// receives a number for deformation
// dont know why I return something ?
//
// called from the outside ! i.e. mike_elle whatever routine
// ---------------------------------------------------------------------------------

float
Lattice::DeformLattice (float move, int plot)
{
  // ------------------------------------------------------
  // and some local variables
  // ------------------------------------------------------

  int i;			// counter for loop
  float wall_pos;		// wall position

  cout << " Uniaxial Compression" << endl;

  def_time_u = def_time_u + 1;	// set deformation time

  runParticle = &refParticle;	// start of list

  runParticle = runParticle->prevP;	// go back one

  wall_pos = runParticle->ypos+ runParticle->radius;	// define wall position

  // ------------------------------------------------------------------------
  // run through particles and move their y position starting from upper
  // row
  // Upper row is pushed inwards. Remember that the yposition of upper
  // row
  // dont change in the relaxation
  // routine, they are fixed.
  // Other particles are moved as if all deformation happens in y
  // direction
  // This is a first approximation and makes the relaxation a lot faster
  // and
  // more accurate.
  // ------------------------------------------------------------------------

  for (i = 0; i < numParticles; i++)	// go through all particles
    // starting from back
    {
      runParticle->ypos =
        runParticle->ypos -
        (runParticle->ypos * move / wall_pos);

      runParticle->ChangeBox (repBox, particlex);

      runParticle = runParticle->prevP;
    }

  upper_wall_pos = upper_wall_pos - move;

  Relaxation ();		// and do a full relaxation

  if (plot == 1)
    UpdateElle ();	// and update the interface of Elle

}



// DEFORM_LATTICE_PURESHEAR
/*******************************************************************************
 * This function does whatever deformation we want to perform with the lattice
 * It is called from the outside by the User and does all that is needed for 
 * a timestep including all the relaxation etc. 
 * at the moment it receives a float for how much the walls are moved
 * as a deformation it moves upper and lower row of particles inwards if 
 * positive value. then it uses area conservation to calculate movement in 
 * x direction of the right wall. Note that at the moment elle still scales
 * everything to the window so that it looks as if the x dimension does not 
 * change and the volume shrinks... 
 *
 * Here we can include all kinds of stuff in the future
 *
 * Daniel and Rianne December 2002, 
 * area bug fixed Daniel and Jochen Feb. 2003
 ********************************************************************************/

// ---------------------------------------------------------------------------------
// function DeformLatticePUREShear in Lattice class
// receives a number for deformation
// dont know why I return something ?
//
// called from the outside ! i.e. mike_elle whatever routine
// ---------------------------------------------------------------------------------

float
Lattice::DeformLatticePureShear (float move, int plot)
{
  // ------------------------------------------------------
  // and some local variables
  // ------------------------------------------------------

  int i;			// counter for loop
  float wall_pos;		// wall position
  float movex, position_Xwall, wall_Xpos;	// stuff for x wall

  cout << " Pure Shear Deformation " << endl;

  Coords newbbox[4];	// have to change Elle box size in file

  def_time_p = def_time_p + 1;	// set deformation time

  runParticle = &refParticle;	// start of list

  runParticle = runParticle->prevP;	// go back one

  wall_pos = runParticle->ypos;	// get wall position y

  wall_Xpos = runParticle->xpos;	// get wall position x

  // have to check if particle on edge is furthest in x or particle
  // below it
  // therefore we run one x row below the highest particle and check
  // that p as well

  for (i = 0; i < particlex; i++)
    {
      runParticle = runParticle->prevP;
    }

  if (runParticle->xpos > wall_Xpos)
    {
      wall_Xpos = runParticle->xpos;
    }
  // now we can calculate the initial area of the box at time 1

  if (def_time_p == 1)
    {

      area_box = wall_pos * wall_Xpos;
      cout << area_box << endl;

    }
  // ------------------------------------------------------------------------
  // run through particles and move their y position starting from upper
  // row
  // Upper row is pushed inwards. Remember that the yposition of upper
  // row
  // dont change in the relaxation
  // routine, they are fixed.
  // Other particles are moved as if all deformation happens in y
  // direction
  // This is a first approximation and makes the relaxation a lot faster
  // and
  // more accurate.
  // ------------------------------------------------------------------------

  for (i = 0; i < numParticles; i++)	// go through all particles
    // starting from back
    {
      runParticle->ypos =
        runParticle->ypos -
        (runParticle->ypos * move / wall_pos);

      runParticle->ChangeBox (repBox, particlex);	// always call
      // Repbox if
      // positions
      // changes

      runParticle = runParticle->prevP;
    }

  if (walls)
    upper_wall_pos =
      upper_wall_pos - (upper_wall_pos * move / wall_pos);

  position_Xwall = area_box / (wall_pos - move);

  movex = wall_Xpos - position_Xwall;	// now calculate x position
  // asuming area remains constant


  for (i = 0; i < numParticles; i++)	// go through all particles
    // starting from back
    {
      // average the deformation in x..

      runParticle->xpos =
        runParticle->xpos -
        (runParticle->xpos * movex / wall_Xpos);

      runParticle->ChangeBox (repBox, particlex);

      runParticle = runParticle->prevP;
    }

  if (walls)
    right_wall_pos =
      right_wall_pos - (right_wall_pos * movex / wall_Xpos);


  // now define the new boxsize for elle

  newbbox[0].x = 0.0;
  newbbox[0].y = 0.0;

  newbbox[1].x = position_Xwall;
  newbbox[1].y = 0.0;

  newbbox[2].x = position_Xwall;
  newbbox[2].y = wall_pos - move;

  newbbox[3].x = 0.0;
  newbbox[3].y = wall_pos - move;

  // and set the new boxsize

  ElleSetCellBBox (&newbbox[0], &newbbox[1], &newbbox[2], &newbbox[3]);

  for (i = 0; i < numParticles; i++)
    {
      runParticle->ChangeBox (repBox, particlex);

      runParticle = runParticle->nextP;
    }

  Relaxation ();		// and do a full relaxation

  if (plot == 1)
    UpdateElle ();	// and update the interface of Elle if it
  // is wanted

}

// DEFORM_LATTICE_PURESHEAR_AREACHANGE
/*******************************************************************************
 * This function does a coaxial deformation including area change
 * It is called from the outside by the User and does all that is needed for 
 * a timestep including all the relaxation etc. 
 * at the moment it receives a float for how much the walls are moved
 * as a deformation it moves upper and lower row of particles inwards if 
 * positive value. then it uses area change given to calculate movement in 
 * x direction of the right wall. Note that at the moment elle still scales
 * everything to the window so that it looks as if the x dimension does not 
 * change and the volume shrinks... 
 *
 * Here we can include all kinds of stuff in the future
 *
 * Daniel July 2003
 ********************************************************************************/

// ---------------------------------------------------------------------------------
// function DeformLatticePUREShearAreaChange in Lattice class
// receives a number for deformation
// dont know why I return something ?
//
// called from the outside ! i.e. mike_elle whatever routine
// ---------------------------------------------------------------------------------

float
Lattice::DeformLatticePureShearAreaChange (float move,
    float area_change, int plot)
{
  // ------------------------------------------------------
  // and some local variables
  // ------------------------------------------------------

  int i;			// counter for loop
  float wall_pos;		// wall position
  float movex, position_Xwall, wall_Xpos;	// stuff for x wall

  Coords newbbox[4];	// have to change Elle box size in file

  def_time_p = def_time_p + 1;	// set deformation time

  runParticle = &refParticle;	// start of list

  runParticle = runParticle->prevP;	// go back one

  wall_pos = runParticle->ypos;	// get wall position y

  wall_Xpos = runParticle->xpos;	// get wall position x

  // have to check if particle on edge is furthest in x or particle
  // below it
  // therefore we run one x row below the highest particle and check
  // that p as well

  for (i = 0; i < particlex; i++)
    {
      runParticle = runParticle->prevP;
    }

  if (runParticle->xpos > wall_Xpos)
    {
      wall_Xpos = runParticle->xpos;
    }
  // now we can calculate the initial area of the box at time 1

  if (def_time_p == 1)
    {
      area_box = wall_pos * wall_Xpos;
      cout << area_box << endl;
    }

  area_box = area_box + (area_box * area_change);

  // ------------------------------------------------------------------------
  // run through particles and move their y position starting from upper
  // row
  // Upper row is pushed inwards. Remember that the yposition of upper
  // row
  // dont change in the relaxation
  // routine, they are fixed.
  // Other particles are moved as if all deformation happens in y
  // direction
  // This is a first approximation and makes the relaxation a lot faster
  // and
  // more accurate.
  // ------------------------------------------------------------------------

  for (i = 0; i < numParticles; i++)	// go through all particles
    // starting from back
    {
      runParticle->ypos =
        runParticle->ypos -
        (runParticle->ypos * move / wall_pos);

      runParticle->ChangeBox (repBox, particlex);	// always call
      // Repbox if
      // positions
      // changes

      runParticle = runParticle->prevP;
    }

  // area is not exactly 1.0, but I dont know how this changes when
  // amount of particles changes
  // is now for 400 times something particles.

  position_Xwall = area_box / (wall_pos - move);

  movex = wall_Xpos - position_Xwall;	// now calculate x position
  // asuming area remains constant


  for (i = 0; i < numParticles; i++)	// go through all particles
    // starting from back
    {
      // average the deformation in x..

      runParticle->xpos =
        runParticle->xpos -
        (runParticle->xpos * movex / wall_Xpos);

      runParticle->ChangeBox (repBox, particlex);

      runParticle = runParticle->prevP;
    }

  // now define the new boxsize for elle

  newbbox[0].x = 0.0;
  newbbox[0].y = 0.0;

  newbbox[1].x = position_Xwall;
  newbbox[1].y = 0.0;

  newbbox[2].x = position_Xwall;
  newbbox[2].y = wall_pos - move;

  newbbox[3].x = 0.0;
  newbbox[3].y = wall_pos - move;

  // and set the new boxsize

  ElleSetCellBBox (&newbbox[0], &newbbox[1], &newbbox[2], &newbbox[3]);

  Relaxation ();		// and do a full relaxation

  if (plot == 1)
    UpdateElle ();	// and update the interface of Elle if it
  // is wanted

}

/***********************************************************************
 * Function that deform uniaxially by pushing down upper wall
 * no average is taken so that only the upper wall particles
 * are moved. This will result in serious gradients in deformation
 * if the relaxation threshold is not seriously reduced. 
 * Very slow in relaxation
 ***********************************************************************/


float
Lattice::DeformLatticeNoAverage (float move, int plot)
{
  // ------------------------------------------------------
  // and some local variables
  // ------------------------------------------------------

  int i;			// counter for loop
  float wall_pos;		// wall position

  def_time_u = def_time_u + 1;	// set deformation time

  runParticle = &refParticle;	// start of list

  runParticle = runParticle->prevP;	// go back one

  wall_pos = runParticle->ypos;	// define wall position

  // ------------------------------------------------------------------------
  // run through particles and move their y position starting from upper
  // row
  // Upper row is pushed inwards. Remember that the yposition of upper
  // row
  // dont change in the relaxation
  // routine, they are fixed.
  // Other particles are moved as if all deformation happens in y
  // direction
  // This is a first approximation and makes the relaxation a lot faster
  // and
  // more accurate.
  // ------------------------------------------------------------------------

  for (i = 0; i < particlex; i++)	// go through all particles
    // starting from back
    {
      runParticle->ypos =
        runParticle->ypos -
        (runParticle->ypos * move / wall_pos);

      runParticle->ChangeBox (repBox, particlex);

      runParticle = runParticle->prevP;
    }



  Relaxation ();		// and do a full relaxation

  if (plot == 1)
    UpdateElle ();	// and update the interface of Elle

}

/***********************************************************************
 *	function that deforms uniaxially but now moves the upper and 
 * lower wall particles inwards (or outwards if move is negative)
 * no average is taken of the deformation within the box. This 
 * will result in gradients in deformation if the relaxation threshold
 * is not lowered. 
 * 
 * variable move indicates vertical strain per deformation step
 * variable plot indicates whether or not a picture is taken
 * after the deformation step (0 = no picture, 1 = picture)
 ***********************************************************************/


float
Lattice::DeformLatticeNoAverage2side (float move, int plot)
{
  // ------------------------------------------------------
  // and some local variables
  // ------------------------------------------------------

  int i;			// counter for loop
  float wall_pos;		// wall position

  def_time_u = def_time_u + 1;	// set deformation time

  runParticle = &refParticle;	// start of list

  runParticle = runParticle->prevP;	// go back one



  // ------------------------------------------------------------------------
  // run through particles and move their y position starting from upper
  // row
  // Upper row is pushed inwards. Remember that the yposition of upper
  // row
  // dont change in the relaxation
  // routine, they are fixed.
  // Other particles are moved as if all deformation happens in y
  // direction
  // This is a first approximation and makes the relaxation a lot faster
  // and
  // more accurate.
  // ------------------------------------------------------------------------

  for (i = 0; i < particlex; i++)	// go through all particles
    // starting from back, only upper
    // row
    {
      runParticle->ypos = runParticle->ypos - move;

      runParticle->ChangeBox (repBox, particlex);

      runParticle = runParticle->prevP;
    }

  runParticle = &refParticle;	// start of list





  // ------------------------------------------------------------------------
  // run through particles and move their y position starting from upper
  // row
  // Upper row is pushed inwards. Remember that the yposition of upper
  // row
  // dont change in the relaxation
  // routine, they are fixed.
  // Other particles are moved as if all deformation happens in y
  // direction
  // This is a first approximation and makes the relaxation a lot faster
  // and
  // more accurate.
  // ------------------------------------------------------------------------

  for (i = 0; i < particlex; i++)	// go through all particles
    // starting from back lower row
    {
      runParticle->ypos = runParticle->ypos + move;

      runParticle->ChangeBox (repBox, particlex);

      runParticle = runParticle->nextP;
    }


  Relaxation ();		// and do a full relaxation

  if (plot == 1)
    UpdateElle ();	// and update the interface of Elle

}

/*************************************************************************************
 * function that deforms uniaxially. Upper and lower row are pushed a factor "move"
 * inwards. Average in this case means that all particles are just moved by 
 * the factor move. Only works if there is an open surface in the middle of the model
 * and two materials are pressed together. Function dissolve x row gives the input for 
 * which part of the crystal is upper and which lower. 
 * 
 * if plot is 0 no picture is taken, if it is 1 a picture is taken
 *************************************************************************************/


float
Lattice::DeformLatticeNewAverage2side (float move, int plot)
{
  // ------------------------------------------------------
  // and some local variables
  // ------------------------------------------------------

  int i, j, count;			// counter for loop
  float wall_pos;		// wall position

  def_time_u = def_time_u + 1;	// set deformation time

  runParticle = &refParticle;	// start of list





  // ------------------------------------------------------------------------
  // run through particles and move their y position starting from upper
  // row
  // Upper row is pushed inwards. Remember that the yposition of upper
  // row
  // dont change in the relaxation
  // routine, they are fixed.
  // Other particles are moved as if all deformation happens in y
  // direction
  // This is a first approximation and makes the relaxation a lot faster
  // and
  // more accurate.
  // ------------------------------------------------------------------------

  for (i = 0; i < numParticles; i++)	// go through all particles
    // starting from back
    {
      if (!runParticle->isHole)
        {
          if(!runParticle->isHoleBoundary)
            {
              if (runParticle->isUpperBox)	// upper part move downwards

                runParticle->ypos = runParticle->ypos - move;

              else		// lower part move upwards

                runParticle->ypos = runParticle->ypos + move;

              runParticle->ChangeBox (repBox, particlex);
            }
          else
            {
              count = 0;
              for (j=0;j<8;j++)
                {
                  if (runParticle->neigP[j])
                    {
                      if (!runParticle->neigP[j]->isHole)
                        count++;
                    }
                }
              if (count>1)
                {
                  if (runParticle->isUpperBox)	// upper part move downwards

                    runParticle->ypos = runParticle->ypos - move;

                  else		// lower part move upwards

                    runParticle->ypos = runParticle->ypos + move;

                  runParticle->ChangeBox (repBox, particlex);
                }
              else
                runParticle->rate_factor = runParticle->rate_factor * 20.0;
            }
        }

      runParticle = runParticle->nextP;
    }

  runParticle = &refParticle;	// start of list



  Relaxation ();		// and do a full relaxation

  if (plot == 1)
    UpdateElle (false);	// and update the interface of Elle

}












/*****************************************************************************
 * just a dummy function that looks for the highest grain number
 *****************************************************************************/

int
Lattice::HighestGrain ()
{
  int i;
  int highest;

  highest = 0;

  runParticle = &refParticle;

  for (i = 0; i < numParticles; i++)
    {
      if (runParticle->grain > highest)
        {
          highest = runParticle->grain;
        }
      runParticle = runParticle->nextP;
    }
  return (highest);
}


// CHANGEPARTICLE (int number, float shrink)
/*********************************************************************************
 * simple function to shrink particle of number x by a certain given amount
 *
 * daniel spring 2002
 ********************************************************************************/

// ----------------------------------------------------------------------------
// function ChangeParticle in lattice class
//
// receives number of particle and amount to change radius of particle
// shrink in percent 0.9 is 90%
//
// called from outside, i.e. mike.elle by user
// ----------------------------------------------------------------------------

void
Lattice::ChangeParticle (int number, float shrink, int plot)
{
  int i,j;			// just a counter

  runParticle = &refParticle;

  for (i = 0; i < numParticles; i++)
    {
      // ------------------------------------------------------------------
      // run through all particles, if I find my particle I shrink it
      // ------------------------------------------------------------------

      if (runParticle->nb == number)
        {
          runParticle->radius = runParticle->radius * shrink;

          runParticle->a = runParticle->a - runParticle->a * shrink;
          runParticle->b = runParticle->b - runParticle->b * shrink;

          for (j = 0; j < 8; j++)
            {
              if (runParticle->neigP[j])
                {
                  runParticle->rad[j] = runParticle->radius;
                }
            }
        }

      runParticle = runParticle->nextP;
    }


  Relaxation ();		// and a full relax
  if (plot == 1)
    UpdateElle ();	// and some elle update for plotting and
  // stress

}


// CHANGEPARTICLESTRESS(float shrink)
/*********************************************************************
 * function that shrinks the particle that has the highest pressure
 * where pressure is sigma one plus sigma two
 *
 * daniel spring 2002
 ********************************************************************/

// ----------------------------------------------------------------------
// function ChangeParticleStress in Lattice class
//
// receives amount to shrink particle
// shrink in percent 0.9 is 90%
//
// called from the outside by the user in a deformation loop for example
// in mike.elle
// -----------------------------------------------------------------------

void
Lattice::ChangeParticleStress (float shrink, int plot)
{
  int i,j;			// just a counter
  float pressure, max_pressure;	// for pressure

  runParticle = preRunParticle = &refParticle;

  // -------------------------------------------------------------------
  // first define pressure (sxx plus syy) memorize first particle
  // pressure and then loop through all particles and find particle
  // with highest pressure
  // --------------------------------------------------------------------

  pressure = max_pressure = runParticle->sxx + runParticle->syy;

  for (i = 0; i < numParticles; i++)	// loop loop
    {
      pressure = runParticle->sxx + runParticle->syy;	// find pressure

      if (pressure < max_pressure)	// if higher
        {
          max_pressure = pressure;	// define new
          preRunParticle = runParticle;	// and set pointer
        }
      runParticle = runParticle->nextP;
    }
  // -------------------------------------------------------------
  // shrink the particle with highest pressure
  // -------------------------------------------------------------

  preRunParticle->radius = preRunParticle->radius * shrink;

  preRunParticle->a = preRunParticle->a - preRunParticle->a * shrink;
  preRunParticle->b = preRunParticle->b - preRunParticle->b * shrink;

  for (j = 0; j < 8; j++)
    {
      if (preRunParticle->neigP[j])
        {
          preRunParticle->rad[j] = preRunParticle->radius;
        }
    }

  Relaxation ();		// and a full relax
  if (plot == 1)
    UpdateElle ();	// and some elle update

}




/**********************************************************
 * routine shrinks the grain nb by an amount shrink. 
 * shrink is given in percent (0.0 is no change, 
 * smaller 1.0 is growth and larger shrinking
 * routine called from the outside, normally experiment.cc
 *
 * Daniel summer 2002
 **********************************************************/

void
Lattice::ShrinkGrain (int nb, float shrink, int plot)
{
  int i, j;		// run variable

  runParticle = &refParticle;	// start

  for (i = 0; i < numParticles; i++)	// loop through particles
    {
      if (runParticle->grain == nb)	// if particle is in the right
        // grain
        {
          runParticle->radius = runParticle->radius-runParticle->radius * shrink;	// shrink
          runParticle->a = runParticle->a - runParticle->a * shrink;
          runParticle->b = runParticle->b - runParticle->b * shrink;

          for (j = 0; j < 8; j++)
            {
              if (runParticle->neigP[j])
                {
                  runParticle->rad[j] = runParticle->radius;
                }
            }
        }

      runParticle = runParticle->nextP;	// go to next
    }

  Relaxation ();		// and relax
  if (plot == 1)
    UpdateElle ();	// and update elle (plot picture on
  // interface)
}

/*************************************************************
 * function used to shrink the whole box 
 * all particles shrink or expand by the same amount
 * good for mudcrack structures ! 
 * 
 * Daniel 2004/5
 *************************************************************/

void
Lattice::ShrinkBox (float shrink, int plot, int yes)
{
  int i, j;

  for (i = 0; i < numParticles; i++)
    {
      if (!runParticle->is_lattice_boundary)	// dont want to shrink
        // boundaries
        {
          if (!yes)
            {
              runParticle->radius = runParticle->radius - runParticle->radius * shrink;	// shrink
              runParticle->a = runParticle->a - runParticle->a * shrink;
              runParticle->b = runParticle->b - runParticle->b * shrink;
            }
          else
            {
              runParticle->radius = runParticle->radius - runParticle->radius * shrink/runParticle->young;	// shrink
              runParticle->a = runParticle->a - runParticle->a * shrink/runParticle->young;
              runParticle->b = runParticle->b - runParticle->b * shrink/runParticle->young;
            }

          // !!

          for (j = 0; j < 8; j++)
            {
              if (runParticle->neigP[j])
                {
                  runParticle->rad[j] = runParticle->radius;	// change
                  // individual
                  // radius
                }
            }
        }
      runParticle = runParticle->nextP;	// and go to next !
    }
  Relaxation ();		// and a full relax
  if (plot == 1)
    UpdateElle ();	// and some elle update for plotting and
  // stress
}




/*********************************************************************
 * routine to define the properties of the grain boundaries
 * first property is changing the spring constant ( in percent, 1.0 is
 * no change, larger than 1.0 is stronger, smaller is weaker). 
 * second number changes the tensile strength of the grainboundary 
 * springs (in percent, larger 1.0 is stronger, smaller is break 
 * easier. 
 * function is called from mike.elle. normally after lattice is build
 *
 * daniel autumn 2002
 *********************************************************************/

void
Lattice::MakeGrainBoundaries (float constant, float break_strength)
{
  int i, j;		// counters

  boundary_strength = break_strength;
  boundary_constant = constant;

  runParticle = &refParticle;	// start

  for (i = 0; i < numParticles; i++)	// run along
    {
      if (runParticle->is_boundary)	// if boundaryparticle
        {
          for (j = 0; j < 8; j++)	// loop through neighbours
            {
              if (runParticle->neigP[j])	// if neighbour
                {
                  if (runParticle->neigP[j]->grain != runParticle->grain)	// if
                    // neighbour
                    // is
                    // in
                    // different
                    // grain
                    {
                      runParticle->spring_boundary[j] = true;

                      runParticle->springf[j] = runParticle->springf[j] * constant;	// change
                      // spring
                      runParticle->break_Str[j] = runParticle->break_Str[j] * break_strength;	// change
                      // breakstrength
                    }
                }
            }
        }
      runParticle = runParticle->nextP;	// and loop
    }

  //Relaxation ();		// and relax

}

/************************************************************************
 * just a clean function that takes an average of springs that have 
 * different constants on both sides. Uses the youngs modulus for that
 *
 * Daniel 2004/5
 ************************************************************************/

void
Lattice::AdjustParticleConstants ()
{
  int i, j;

  for (i = 0; i < numParticles; i++)	// run along
    {
      for (j = 0; j < 8; j++)
        {
          if (runParticle->neigP[j])
            {
              runParticle->springf[j] =
                ((runParticle->young +
                  runParticle->neigP[j]->young) *
                 sqrt (3.0) / 2.0) / 2.0;
              runParticle->springv[j] =
                ((runParticle->viscosity +
                  runParticle->neigP[j]->viscosity) *
                 sqrt (3.0) / 2.0) / 2.0;
            }
        }
      runParticle = runParticle->nextP;
    }
}

/******************************************************************************
 * this function adjusts the spring constant of grain boundary springs if 
 * the two particles have different spring constants. weakening grains does
 * not affect the spring constant of boundary springs ! 
 * Function should be called after constants of all grains are defined
 *
 * if you want to have different strength for boundary springs than the grains
 * you should call this function first and then afterwards change the spring
 * strength with the MakeGrainBoundaries function
 *
 * daniel december 2002
 ******************************************************************************/

void
Lattice::AdjustConstantGrainBoundaries ()
{
  int i, j, jj, jjj;	// counters

  float grain_one_strength, grain_two_strength;	// strength of the two grains

  runParticle = &refParticle;	// start

  for (i = 0; i < numParticles; i++)	// run along
    {
      if (runParticle->is_boundary)	// if boundaryparticle
        {
          // first find your own spring constant
          // so find a spring within the grain

          for (jjj = 0; jjj < 8; jjj++)
            {
              if (runParticle->neigP[jjj])
                {
                  if (runParticle->neigP[jjj]->grain ==
                      runParticle->grain)
                    {
                      grain_one_strength =
                        runParticle->
                        springf[jjj];
                    }
                }
            }

          // then find the neighbour spring and strength of neighbours
          // may be more than one....

          for (j = 0; j < 8; j++)	// loop through neighbours
            {
              if (runParticle->neigP[j])	// if neighbour
                {
                  // have to find springs that connect boundaries and
                  // then calculate
                  // mean of spring constant of both grains.

                  if (runParticle->neigP[j]->grain != runParticle->grain)	// if
                    // neighbour
                    // is
                    // in
                    // different
                    // grain
                    {
                      for (jj = 0; jj < 8; jj++)
                        {
                          // have to find constant of particle(grain) so
                          // have to find
                          // spring that is not attached to particle of
                          // other grain

                          if (runParticle->neigP[j]->neigP[jj])	// find
                            // spring
                            // with
                            // right
                            // constant
                            {
                              // if the spring is connected to right
                              // particle

                              if (runParticle->neigP[j]->neigP[jj]->grain == runParticle->neigP[j]->grain)
                                {
                                  // should be strength of neighbour
                                  // grain

                                  grain_two_strength
                                  =
                                    runParticle->
                                    neigP
                                    [j]->
                                    springf
                                    [jj];
                                  // apply average
                                  // if the strength is zero (hole) the
                                  // boundary springs
                                  // should also be zero

                                  if (grain_two_strength == 0.0)
                                    {
                                      runParticle->
                                      springf
                                      [j]
                                      =
                                        0.0;
                                    }
                                  else if (grain_one_strength == 0.0)
                                    {
                                      runParticle->
                                      springf
                                      [j]
                                      =
                                        0.0;
                                    }
                                  else
                                    {
                                      runParticle->
                                      springf
                                      [j]
                                      =
                                        (grain_one_strength
                                         +
                                         grain_two_strength)
                                        /
                                        2.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
      runParticle = runParticle->nextP;	// and loop
    }

  Relaxation ();		// and relax

}

/***********************************************************************************
 * this routine weakens or strenghtens a grain. gets the number of the grain 
 * we are talking about and the change of the spring constant in percent (1.0 is zero
 * change, smaller is weaker and more is harder) and then a number for changing
 * the breaking strength in percent (again 1.0 is nothing and more is harder or 
 * breaks less easy and less is break more easy). 
 * normally called after lattice is constructed by mike.elle but can also be 
 * called from within a routine to have some time dependent weakening maybe ?
 *
 * daniel summer 2002
 *
 * add youngs modulus, 
 *
 * Daniel, march 2003
 ***********************************************************************************/

void
Lattice::WeakenGrain (int nb, float constant, float visc, float break_strength)
{
  int i, j;		// counters

  runParticle = &refParticle;	// start

  for (i = 0; i < numParticles; i++)	// loop through particles
    {
      if (runParticle->grain == nb)	// if the right grain
        {
          // --------------------------------------------------------
          // adjust youngs modulus of particle, default is for
          // spring constant of 1.0
          // --------------------------------------------------------

          runParticle->young = runParticle->young * constant;
          runParticle->viscosity = runParticle->viscosity * visc;

          for (j = 0; j < 8; j++)	// loop through neighbours
            {

              if (runParticle->neigP[j])	// if neighbour
                {
                  runParticle->break_Str[j] = runParticle->break_Str[j] * break_strength;	// change
                }
            }
        }
      runParticle = runParticle->nextP;	// go on looping
    }

  AdjustParticleConstants();

}

/*******************************************************************************************
 * Routine to make a horizontal layer weak or hard. gets as input y minimum and maximum 
 * position of layer (taken for whole grains) and a constant for the elastic modulus and 
 * a constant for the spring. Changes in percent, 1.0 is no change, > 1.0 stronger and 
 * < 1.0 softer. 
 *
 * routine can be called several times. 
 *
 *
 * daniel, December 2002
 *
 * adjust youngs modulus, 
 *
 * daniel, March 2003
 **********************************************************************************************/

void
Lattice::WeakenHorizontalLayer (double y_min, double y_max, float constant,
                                float break_strength)
{
  int i, j, jj;		// counters
  bool isgrain;		// is grain in list ?

  // first set the vector layer to -2 (have zero grain number)

  for (j = 0; j < 10000; j++)
    {
      layer[j] = -2;
    }

  runParticle = &refParticle;	// start

  for (i = 0; i < numParticles; i++)	// run through particles
    {
      if (runParticle->ypos > y_min)	// larger than y min
        {
          if (runParticle->ypos < y_max)	// smaller than y max ->
            // in layer
            {
              isgrain = false;	// counter is false

              for (j = 0; j < 10000; j++)	// run through the list
                {
                  if (layer[j] == runParticle->grain)	// if already
                    // defined
                    {
                      isgrain = true;	// grain is already in list
                      break;
                    }
                  else if (layer[j] == -2)	// end of list break,
                    // grain not in list
                    {
                      break;
                    }
                }
              if (!isgrain)	// grain in layer but not yet in the list
                {
                  layer[j] = runParticle->grain;	// put it in the
                  // list
                }
            }
        }
      runParticle = runParticle->nextP;
    }

  for (i = 0; i < numParticles; i++)	// loop through particles
    {
      for (jj = 0; jj < 10000; jj++)	// loop through list
        {
          if (layer[jj] == -2)
            break;	// end of list exit

          else if (runParticle->grain == layer[jj])	// if the right
            // grain
            {
              // -----------------------------------------------------
              // adjust youngs modulus of particle
              // -----------------------------------------------------

              runParticle->young =
                runParticle->young * constant;

              for (j = 0; j < 8; j++)	// loop through neighbours
                {
                  // -------------------------------------------------------------------------
                  // we do not want to change the grain boundaries as
                  // well, only springs
                  // within the grain, therefore check which springs are
                  // between particles
                  // of the same grain.
                  //
                  // call function adjustgrainboundaries later to get
                  // mean constants for boundaries
                  // --------------------------------------------------------------------------

                  if (runParticle->neigP[j])	// if neighbour
                    {
                      if (runParticle->neigP[j]->grain == runParticle->grain)	// if
                        // the
                        // same
                        // grain
                        {
                          runParticle->springf[j] = runParticle->springf[j] * constant;	// change
                          // constant
                          runParticle->break_Str[j] = runParticle->break_Str[j] * break_strength;	// change
                          // tensile
                        }
                    }
                }
            }
        }
      runParticle = runParticle->nextP;	// go on looping
    }


}


/*******************************************************************************************
 * Routine to make a horizontal layer weak or hard (particles). gets as input y minimum and maximum
 * position of layer  and a constant for the elastic modulus and
 * a constant for the spring. Changes in percent, 1.0 is no change, > 1.0 stronger and
 * < 1.0 softer.
 *
 * routine can be called several times.
 *
 *
 * daniel, January 2004
 *
 **********************************************************************************************/

void
Lattice::WeakenHorizontalParticleLayer (double y_min, double y_max,
                                        float constant, float vis,
                                        float break_strength)
{
  int i, j;		// counters



  for (i = 0; i < numParticles; i++)	// loop through particles
    {
      if (runParticle->ypos > y_min)
        {
          if (runParticle->ypos < y_max)
            {
              // -----------------------------------------------------
              // adjust youngs modulus of particle
              // -----------------------------------------------------

              runParticle->young =
                runParticle->young * constant;
              runParticle->viscosity *= vis;

              for (j = 0; j < 8; j++)	// loop through neighbours
                {
                  // -------------------------------------------------------------------------
                  // we do not want to change the grain boundaries as
                  // well, only springs
                  // within the grain, therefore check which springs are
                  // between particles
                  // of the same grain.
                  //
                  // call function adjustgrainboundaries later to get
                  // mean constants for boundaries
                  // --------------------------------------------------------------------------

                  if (runParticle->neigP[j])	// if neighbour
                    {
                      if (runParticle->neigP[j]->
                          ypos > y_min)
                        {
                          if (runParticle->
                              neigP[j]->ypos <
                              y_max)
                            {
                              runParticle->springf[j] = runParticle->springf[j] * constant;	// change
                              // constant
                              runParticle->
                              springv
                              [j] =
                                runParticle->
                                springv
                                [j] *
                                vis;
                              runParticle->break_Str[j] = runParticle->break_Str[j] * break_strength;	// change
                              // tensile
                            }
                        }
                    }
                }
            }
        }
      runParticle = runParticle->nextP;	// go on looping
    }

  AdjustParticleConstants ();

}


/*******************************************************************************************
 * Routine to make a horizontal layer weak or hard (particles). gets as input y minimum and maximum
 * position of layer  and a constant for the elastic modulus and
 * a constant for the spring. Changes in percent, 1.0 is no change, > 1.0 stronger and
 * < 1.0 softer.
 *
 * routine can be called several times.
 *
 *
 * daniel, October 2005
 *
 **********************************************************************************************/

void
Lattice::WeakenTiltedParticleLayer (double y_min, double y_max, double shift,
                                    float constant, float vis,
                                    float break_strength)
{
  int i, j;		// counters



  for (i = 0; i < numParticles; i++)	// loop through particles
    {
      if (runParticle->ypos > (y_min+(shift*runParticle->xpos)))
        {
          if (runParticle->ypos < (y_max+(shift*runParticle->xpos)))
            {
              // -----------------------------------------------------
              // adjust youngs modulus of particle
              // -----------------------------------------------------

              runParticle->young =
                runParticle->young * constant;
              runParticle->viscosity *= vis;

              for (j = 0; j < 8; j++)	// loop through neighbours
                {
                  // -------------------------------------------------------------------------
                  // we do not want to change the grain boundaries as
                  // well, only springs
                  // within the grain, therefore check which springs are
                  // between particles
                  // of the same grain.
                  //
                  // call function adjustgrainboundaries later to get
                  // mean constants for boundaries
                  // --------------------------------------------------------------------------

                  if (runParticle->neigP[j])	// if neighbour
                    {
                      if (runParticle->neigP[j]->
                          ypos > y_min)
                        {
                          if (runParticle->
                              neigP[j]->ypos <
                              y_max)
                            {
                              runParticle->springf[j] = runParticle->springf[j] * constant;	// change
                              // constant
                              runParticle->
                              springv
                              [j] =
                                runParticle->
                                springv
                                [j] *
                                vis;
                              runParticle->break_Str[j] = runParticle->break_Str[j] * break_strength;	// change
                              // tensile
                            }
                        }
                    }
                }
            }
        }
      runParticle = runParticle->nextP;	// go on looping
    }

  AdjustParticleConstants ();

}


void
Lattice::WeakenTiltedParticleLayerX (double x_min, double x_max, double shift,
                                     float constant, float vis)
{
  int i, j;		// counters



  for (i = 0; i < numParticles; i++)	// loop through particles
    {
      if (runParticle->xpos > (x_min+(shift*runParticle->ypos)))
        {
          if (runParticle->xpos < (x_max+(shift*runParticle->ypos)))
            {
              // -----------------------------------------------------
              // adjust youngs modulus of particle
              // -----------------------------------------------------

              runParticle->young =
                runParticle->young * constant;
              runParticle->viscosity *= vis;


            }
        }
      runParticle = runParticle->nextP;	// go on looping
    }

  AdjustParticleConstants ();

}


/*********************************************************************
 * function in lattice class used for multi layer boudins
 * function creates layers of two sorts. Input is the spring constant
 * of layer a and b, the viscosity of layer a and b, the breaking strength of layer 
 * a and b and the thickness of layer a and b.
 * layers are arranged parallel to the x direction (horizontal)
 * starting from the bottom. Elle grains are ignored, only particles are taken
 * 
 * arranged like : 	b
 *                       a
 *                       b
 *                       a
 *                       ....
 * last input (integer av) determines whether or not an average 
 * of the two layer constants is taken (0 = no average, 1 = average)
 * Becomes important when layers are only as thick as a particle
 * 
 * Daniel and Arzu 2004
 **********************************************************************/

void
Lattice::MakeHorizontalLayers (float young_a, float viscous_a,
                               float break_a, float thick_a, float young_b,
                               float viscous_b, float break_b,
                               float thick_b, int av)
{
  int layernumber;
  int i, j, jj, k;
  float average;

  layernumber = (int)(1.0 / (thick_a + thick_b));	// determine how many
  // layers fit in the model

  cout << "layernumber " << layernumber << endl;	// give out layer
  // number


  for (i = 0; i < numParticles; i++)	// loop through all particles
    {
      // algorithm runs through the layers in sets of two, always a
      // sequence,
      // then checks first the sequence for the particle
      // and afterwards determines if its layer a or b in that sequence.

      for (j = 1; j < (layernumber + 1); j++)	// loop through the layers
        // to find layer of
        // particle
        {
          if (runParticle->ypos < ((thick_a + thick_b) * j))	// in this
            // layer
            // sequence
            // (defined
            // by j)
            {
              if (runParticle->ypos < ((thick_a + thick_b) * (j - 1)) + thick_a)	// is
                // it
                // a
                // or
                // b
                // ?
                {
                  for (k = 0; k < 8; k++)	// its layer a, give it a
                    // constants !
                    {
                      runParticle->springf[k] = young_a;	// spring
                      // constant
                      // (absolute)
                      runParticle->springv[k] = viscous_a;	// viscosity
                      // of
                      // spring
                      // (absolute)
                      runParticle->young = young_a * 2.0 / sqrt (3.0);	// young
                      // modulus
                      // (absolute)
                      runParticle->break_Str[k] = runParticle->break_Str[k] * break_a;	// breaking
                      // strength
                      // (factor)
                    }
                  break;	// and exist, we have found the layer
                }
              else	// oh, this is layer b -> give it the b
                // constants !
                {
                  for (k = 0; k < 8; k++)	// loop through springs
                    {
                      runParticle->springf[k] = young_b;	// spring
                      // constant
                      // (absolute)
                      runParticle->springv[k] = viscous_b;	// viscosity
                      // (absolute)
                      runParticle->young = young_b * 2.0 / sqrt (3.0);	// young
                      // modulus
                      // (absolute)
                      runParticle->break_Str[k] = runParticle->break_Str[k] * break_b;	// breaking
                      // strength
                      // (factor)
                    }
                  break;	// get out !
                }
            }
          if (runParticle->ypos > ((thick_a + thick_b) * layernumber))	// last
            // layer...
            {
              for (k = 0; k < 8; k++)
                {
                  runParticle->springf[k] = young_a;
                  runParticle->springv[k] = viscous_a;
                  runParticle->young = young_a;
                  runParticle->break_Str[k] =
                    runParticle->break_Str[k] *
                    break_a;
                }
            }
        }
      runParticle = runParticle->nextP;	// next particle please...
    }

  // now we have to adjust the spring constants, spring constants on
  // layer boundaries are now different
  // event though springs are the same. Now a spring has a constant b if
  // looking from layer b and one of
  // a if we look from layer a... this is unstable and physically
  // wrong...

  for (i = 0; i < numParticles; i++)	// run along
    {

      // first find your own spring constant
      // so find a spring within the grain

      for (j = 0; j < 8; j++)	// loop through springs
        {
          if (runParticle->neigP[j])	// if there is a spring
            {
              for (jj = 0; jj < 8; jj++)	// loop second time
                {
                  if (runParticle->neigP[j]->neigP[jj])	// find my
                    // neighbour
                    // (spring
                    // from
                    // both
                    // sides..)
                    {
                      if (runParticle->neigP[j]->
                          neigP[jj]->nb ==
                          runParticle->nb)
                        {
                          if (av == 1)	// take the average of the
                            // two values
                            {
                              average =
                                (runParticle->
                                 springf
                                 [j] +
                                 runParticle->
                                 neigP
                                 [j]->
                                 springf
                                 [jj])
                                / 2.0;
                              runParticle->
                              springf
                              [j] =
                                average;
                              runParticle->
                              neigP
                              [j]->
                              springf
                              [jj] =
                                average;
                              average =
                                (runParticle->
                                 springv
                                 [j] +
                                 runParticle->
                                 neigP
                                 [j]->
                                 springv
                                 [jj])
                                / 2.0;
                              runParticle->
                              springv
                              [j] =
                                average;
                              runParticle->
                              neigP
                              [j]->
                              springv
                              [jj] =
                                average;
                            }
                          else	// no average, take always the
                            // lower constant
                            {
                              if (runParticle->springf[j] > runParticle->neigP[j]->springf[jj])
                                {
                                  runParticle->
                                  springf
                                  [j]
                                  =
                                    runParticle->
                                    neigP
                                    [j]->
                                    springf
                                    [jj];
                                }
                              else
                                {
                                  runParticle->
                                  neigP
                                  [j]->
                                  springf
                                  [jj]
                                  =
                                    runParticle->
                                    springf
                                    [j];
                                }
                              if (runParticle->springv[j] > runParticle->neigP[j]->springv[jj])
                                {
                                  runParticle->
                                  springv
                                  [j]
                                  =
                                    runParticle->
                                    neigP
                                    [j]->
                                    springv
                                    [jj];
                                }
                              else
                                {
                                  runParticle->
                                  neigP
                                  [j]->
                                  springv
                                  [jj]
                                  =
                                    runParticle->
                                    springv
                                    [j];
                                }

                            }
                        }
                    }
                }
            }
        }

      runParticle = runParticle->nextP;	// and loop
    }

}

void Lattice::ChangeYoung(float change)
{
	int i; 
	
	for (i=0;i<numParticles;i++)
	{
		runParticle->young = runParticle->young * change;
		
		runParticle = runParticle->nextP;
	}
}

/***********************************************************************
 * function changes elastic constant, viscosity and breaking strength of all
 * Particles.
 * all variables are factors for the change -> so that initial set 
 * distributions are not changed, just shifted
 *
 * Daniel January 2004
 *************************************************************************/

void
Lattice::WeakenAll (float constant, float viscosity, float break_strength)
{
  int i, j;		// loop counter

  for (i = 0; i < numParticles; i++)
    {
      runParticle->young = runParticle->young * constant;
      runParticle->viscosity *= viscosity;

      for (j = 0; j < 8; j++)
        {
          if (runParticle->neigP[j])
            {
              runParticle->springf[j] =
                runParticle->springf[j] * constant;
              runParticle->springv[j] =
                runParticle->springv[j] * viscosity;
              runParticle->break_Str[j] =
                runParticle->break_Str[j] *
                break_strength;
            }
        }
      runParticle = runParticle->nextP;
    }
}

void Lattice::SetSinAnisotropy(float nb, float ampl)
{
  int i;

  for (i=0;i<numParticles;i++)
    {
      runParticle->young = runParticle->young * ((sin(runParticle->xpos*nb) + 1.1)*ampl);

      runParticle = runParticle->nextP;
    }

  AdjustParticleConstants();
}


/******************************************************************************************
 *  function that sets an initial anisotropy by setting a pseudorandom distribution of 
 *  grains that consist of one to three particles aligned parallel to x. 
 *  breaking strength, youngs modulus and viscosity of these particles can be changed. 
 *  length determines how long they are in x , at the moment only 1-3 particles 
 *  ratio determines how many of these particles there are (percent from 0.0 to 1.0)
 *  these new particles will have no direct neighbours, there is always supposed to be 
 *  at least one other particle in between. 
 *******************************************************************************************/

void
Lattice::SetAnisotropy (float break_strength, float youngs_mod,
                        float viscosity, float ratio, float length)
{
  int i, j, jj, nb;	// counter
  float ran_nb;		// pseudorandom number
  float str_nb, br_nb;	// number for spring constant, number for
  // breaking strength
  float str_m, br_m;	// helpers for the mean
  bool no_neighbour;	// check if there is a matrix
  // particle in between

  runParticle = &refParticle;	// and go


  ran_nb = rand () / (float) RAND_MAX;	// pic pseudorandom from 0 to 1.0
  // for grains


  for (i = 0; i < numParticles; i++)	// loop through particles
    {
      if (runParticle->xpos > 0.02)
        {
          if (runParticle->xpos < 0.98)
            {
              if (runParticle->ypos > 0.02)
                {
                  if (runParticle->ypos < 0.98)	// leave a matrix
                    // boundary at the
                    // rims of the
                    // box.
                    {
                      ran_nb = rand () / (float) RAND_MAX;	// pic
                      // pseudorandom
                      // from 0
                      // to 1.0
                      // for
                      // grains
                      no_neighbour = true;	// assume I have only
                      // matrix neighbours

                      if (ran_nb < ratio / length)	// rannumber used
                        // to determine
                        // propability for
                        // particles to be
                        // there
                        {
                          if (!runParticle->done)	// if I have not
                            // been converted
                            // before
                            {
                              if (!runParticle->is_lattice_boundary)	// if
                                // I
                                // am
                                // not
                                // at
                                // the
                                // lattice
                                // boundary
                                // (just
                                // in
                                // case)
                                {
                                  for (j = 0; j < 8; j++)	// loop
                                    // through
                                    // springs
                                    {
                                      if (runParticle->neigP[j])	// if
                                        // I
                                        // have
                                        // a
                                        // neighbour
                                        {
                                          if (runParticle->neigP[j]->done)
                                            no_neighbour = false;	// if
                                          // this
                                          // neighbour
                                          // is
                                          // already
                                          // transformed
                                          // get
                                          // out
                                          // !
                                        }
                                    }
                                  if (no_neighbour)	// if I have no
                                    // neighbour that
                                    // is transformed
                                    {
                                      no_neighbour = true;	// ???
                                      runParticle->young = runParticle->young * youngs_mod;	// now
                                      // apply
                                      // the
                                      // factors
                                      runParticle->
                                      viscosity
                                      *=
                                        viscosity;

                                      for (j = 0; j < 8; j++)	// loop
                                        // through
                                        // neighbours
                                        {
                                          if (runParticle->neigP[j])	// if
                                            // neighbour
                                            {
                                              runParticle->
                                              springf
                                              [j]
                                              =
                                                runParticle->
                                                springf
                                                [j]
                                                *
                                                youngs_mod;
                                              runParticle->
                                              break_Str
                                              [j]
                                              =
                                                runParticle->
                                                break_Str
                                                [j]
                                                *
                                                break_strength;
                                              runParticle->
                                              springv
                                              [j]
                                              =
                                                runParticle->
                                                springv
                                                [j]
                                                *
                                                viscosity;
                                            }
                                        }
                                      runParticle->done = true;	// ok
                                      // this
                                      // one
                                      // is
                                      // done
                                      if (length > 1)	// if length is
                                        // larger than one
                                        // than try to
                                        // transform
                                        // neighbours in x
                                        // direction as
                                        // well
                                        {
                                          for (j = 0; j < 8; j++)	// loop
                                            // through
                                            // neighbours
                                            {
                                              if (runParticle->neigP[j])	// if
                                                // neighbour
                                                {
                                                  if (runParticle->neigP[j]->ypos == runParticle->ypos)	// if
                                                    // neighbour
                                                    // in
                                                    // same
                                                    // row
                                                    {
                                                      if (!runParticle->neigP[j]->done)	// if
                                                        // not
                                                        // done
                                                        {
                                                          if (!runParticle->neigP[j]->is_lattice_boundary)	// if
                                                            // not
                                                            // lattice
                                                            {
                                                              for (jj = 0; jj < 8; jj++)	// loop
                                                                // through
                                                                // neighbours
                                                                {
                                                                  if (runParticle->neigP[j]->neigP[jj])	// if
                                                                    // neighbour
                                                                    {
                                                                      if (runParticle->neigP[j]->neigP[jj]->done)	// if
                                                                        // that
                                                                        // one
                                                                        // is
                                                                        // transformed
                                                                        {
                                                                          if (runParticle->nb != runParticle->neigP[j]->neigP[jj]->nb)	// if
                                                                            // its
                                                                            // not
                                                                            // myself
                                                                            no_neighbour = false;	// dont
                                                                          // transform
                                                                        }
                                                                    }
                                                                }
                                                              if (no_neighbour)	// if
                                                                // I
                                                                // can
                                                                // transform
                                                                {
                                                                  runParticle->
                                                                  neigP
                                                                  [j]->
                                                                  young
                                                                  =
                                                                    runParticle->
                                                                    neigP
                                                                    [j]->
                                                                    young
                                                                    *
                                                                    youngs_mod;
                                                                  runParticle->
                                                                  neigP
                                                                  [j]->
                                                                  done
                                                                  =
                                                                    true;
                                                                  for (jj = 0; jj < 8; jj++)	// loop
                                                                    // through
                                                                    // neighbours
                                                                    {
                                                                      if (runParticle->neigP[j]->neigP[jj])	// if
                                                                        // neighbour
                                                                        {
                                                                          runParticle->
                                                                          neigP
                                                                          [j]->
                                                                          springf
                                                                          [jj]
                                                                          =
                                                                            runParticle->
                                                                            neigP
                                                                            [j]->
                                                                            springf
                                                                            [jj]
                                                                            *
                                                                            youngs_mod;
                                                                          runParticle->
                                                                          neigP
                                                                          [j]->
                                                                          break_Str
                                                                          [jj]
                                                                          =
                                                                            runParticle->
                                                                            neigP
                                                                            [j]->
                                                                            break_Str
                                                                            [jj]
                                                                            *
                                                                            break_strength;
                                                                          runParticle->
                                                                          neigP
                                                                          [j]->
                                                                          springv
                                                                          [jj]
                                                                          =
                                                                            runParticle->
                                                                            neigP
                                                                            [j]->
                                                                            springv
                                                                            [jj]
                                                                            *
                                                                            viscosity;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                              if (length == 2)	// if
                                                // length
                                                // is
                                                // two
                                                // stop
                                                // otherwise
                                                // try
                                                // to
                                                // do
                                                // the
                                                // second
                                                // neibhour
                                                // in
                                                // x
                                                // as
                                                // well
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
      runParticle = runParticle->nextP;

    }
  for (i = 0; i < numParticles; i++)	// loop through particles to do
    // some averaging.
    {
      runParticle->done = false;
      for (j = 0; j < 8; j++)	// loop through neighbours
        {
          if (runParticle->neigP[j])	// if neighbour
            {
              runParticle->springf[j] =
                (runParticle->neigP[j]->young +
                 runParticle->young) / 2.0;


            }
        }

      runParticle = runParticle->nextP;
    }



}

/*******************************************************************+
* Function that set a horizontal anisotropy. Litte horizontally 
* aligned mica flakes are generated with varying length (3-7) and
* a randomly chosen Youngs modulus between 1 and max. 
*
* Daniel 2005/6
*********************************************************************/

void
Lattice::SetAnisotropyRandom (float max, float boundary)
{
  int size, i, j, ii;
  float ran_nb;

  runParticle = firstParticle;

  srand (std::time (0)); // pick random seed


  for (ii = 0; ii < numParticles;)
    {

      ran_nb = rand () / (float) RAND_MAX; // draw a number

      // distribute length of micas randomly bewteen 3 and 7

      if (ran_nb < 0.2)
        size = 3;
      else if (ran_nb < 0.4)
        size = 4;
      else if (ran_nb < 0.6)
        size = 5;
      else if (ran_nb < 0.8)
        size = 6;
      else
        size = 7;

      ran_nb = rand () / (float) RAND_MAX; // draw second number

      ran_nb = (ran_nb * (max - 1)) + 1; // base for youngs modulus variation



      for (i = 0; i < size; i++)
        {
          if (runParticle->is_lattice_boundary)
            {
              runParticle->young =
                runParticle->young * boundary;
              runParticle->viscosity *= boundary;

              for (j = 0; j < 8; j++)
                {
                  if (runParticle->neigP[j])
                    {
                      runParticle->springf[j] =
                        runParticle->
                        springf[j] * boundary;
                      runParticle->springv[j] =
                        runParticle->
                        springv[j] * boundary;
                    }
                }
            }
          else
            {
              runParticle->young =
                runParticle->young * ran_nb;
              runParticle->viscosity *= ran_nb;

              for (j = 0; j < 8; j++)
                {
                  if (runParticle->neigP[j])
                    {
                      runParticle->springf[j] =
                        runParticle->
                        springf[j] * ran_nb;
                      runParticle->springv[j] =
                        runParticle->
                        springv[j] * ran_nb;
                    }
                }
            }

          ii++;
          if (ii >= numParticles)
            break;

          runParticle = runParticle->nextP;

        }


    }

  AdjustParticleConstants ();
}

/***********************************************************
 * function to set a max for pict dumps so that the 
 * harddisk is not filled up completely
 * 
 * Daniel March 2003
 ************************************************************/

void
Lattice::Set_Max_Pict (int max)
{
  set_max_pict = true;
  max_pict = max;
}


/***************************************************************************
 * This little thing can be called at the beginning after the lattice 
 * was made (in mike.elle). You give it a number. It makes a plot of the 
 * stuff each time that n fractures developed within a time step. Is 
 * useful sometimes when a lot of fractures develop after a deformation
 * step and you want to see whats happening. Or you want to watch the 
 * fracture develop. Default is 0 so if you dont call this function the 
 * program will only make plots after time steps. 
 *
 * daniel summer 2002
 **************************************************************************/

void
Lattice::SetFracturePlot (int numbfrac, int local)
{
  internal_break = local;	// default 0
  fract_Plot_Numb = numbfrac;	// define the number (default is 0)
}



/********************************************************************************
 * Function that is used to dump a statistic file. This file gets the strain as 
 * input from the main function (the strain that you put in the deformation file)
 * and an int for the grain that you want the statistic for. Dumps mean values for 
 * stress tensor and mean and differential stress for this grain. Called normally 
 * by the user from the mike.elle.cc file within the Run section after deformation. 
 *
 * file is called statisticgrain.txt
 *
 * dumps strain, mean stress smax + smin/2, sxx, syy, sxy and differential stress
 * smax - smin. Note that all compressive stresses are by definition negative ! 
 *
 * can be directly importet by all kinds of programs including Matlab and Excel
 *
 * Daniel and Jochen Feb. 2003
 * thanks for some help from Jens
 **********************************************************************************/

void
Lattice::DumpStatisticStressGrain (double strain, int nb)
{
  FILE *stat;		// file pointer
  int i, p_counter;	// counters
  float mean, differential;	// mean stress, differential
  // stress, pressure
  float m_sxx, m_syy, m_sxy;	// mean values of tensor
  float finite_strain;	// finite strain on the box
  float sxx, syy, sxy, smax, smin;	// stress tensor and eigenvalues

  // zero out some values

  p_counter = 0;

  mean = 0.0;
  differential = 0.0;
  m_sxx = 0.0;
  m_syy = 0.0;
  m_sxy = 0.0;

  finite_strain = strain * (def_time_p + def_time_u);	// calculate
  // finite strain

  runParticle = &refParticle;	// start to loop through particles

  for (i = 0; i < numParticles; i++)	// loop
    {
      if (runParticle->grain == nb)	// if its the grain
        {
          p_counter = p_counter + 1;	// counts number of particles

          // get the tensor from particles, just used to save space

          sxx = runParticle->sxx;
          syy = runParticle->syy;
          sxy = runParticle->sxy;

          // calculate eigenvalues

          smax = ((sxx + syy) / 2.0) +
                 sqrt (((sxx - syy) / 2.0) * ((sxx -
                                               syy) / 2.0) +
                       sxy * sxy);

          smin = ((sxx + syy) / 2.0) -
                 sqrt (((sxx - syy) / 2.0) * ((sxx -
                                               syy) / 2.0) +
                       sxy * sxy);

          // calculate mean stress

          mean = mean + (smax + smin) / 2.0;

          // calculate tensor means

          m_sxx = m_sxx + sxx;

          m_syy = m_syy + syy;

          m_sxy = m_sxy + sxy;

          // calculate differential stress

          differential = differential + (smax - smin);
        }
      runParticle = runParticle->nextP;
    }

  // divide by number of particles

  mean = mean / p_counter;

  m_sxx = m_sxx / p_counter;

  m_syy = m_syy / p_counter;

  m_sxy = m_sxy / p_counter;

  differential = differential / p_counter;

  // and dump the data for this step

  // append "a" means he is just adding values directly to the file
  // if the program crashes the file is still there with data in it
  // but if you start the program twice he will add data to the file
  // that is originally there so you have several steps of loading !!!!
  // just delete or move or rename old file and he makes a new one.

  stat = fopen ("statisticgrain.txt", "a");	// open statistic output
  // append file

  // and write the data in one line with a semicolon ; between the
  // numbers
  // and jump to next line for the next deformation step

  fprintf (stat, "%d", nb);
  fprintf (stat, ";%f", finite_strain);
  fprintf (stat, ";%f", mean);
  fprintf (stat, ";%f", m_sxx);
  fprintf (stat, ";%f", m_syy);
  fprintf (stat, ";%f", m_sxy);
  fprintf (stat, ";%f\n", differential);

  fclose (stat);		// close file
}


/*************************************************************************
 * ok, this is almost the same as the above code (to dump statistic for just
 * one grain). This is now for two grains. So the function gets the strain 
 * from the deformation you set and two grain numbers. This is useful to
 * compare stresses in a hard layer and a soft layer. The function is an
 * append file again, so the program writes the data directly to the file 
 * each time and the file is there even if the code crashes (thanks, Jens)
 * dumps number of grain, 
 * strain on box, mean stress, differential stress and sxx, syy and sxy
 *
 * Jochen and Daniel Feb 2003
 *******************************************************************************/


void
Lattice::DumpStatisticStressTwoGrains (double strain, int nb1, int nb2)
{
  FILE *stat;		// file pointer
  int i, j, p_counter;	// counters
  float mean, differential;	// mean stress, differential
  // stress, pressure
  float m_sxx, m_syy, m_sxy;	// mean stresses for tensor
  float finite_strain;	// finite strain on the box
  float sxx, syy, sxy, smax, smin;	// stress tensor and eigenvalues

  // and zero values

  p_counter = 0;

  mean = 0.0;
  differential = 0.0;
  m_sxx = 0.0;
  m_syy = 0.0;
  m_sxy = 0.0;

  finite_strain = strain * (def_time_p + def_time_u);	// calculate
  // finite strain

  runParticle = &refParticle;

  for (i = 0; i < numParticles; i++)	// and loop
    {
      if (runParticle->grain == nb1)	// if grain one
        {
          p_counter = p_counter + 1;	// counter for particles

          // get tensor

          sxx = runParticle->sxx;
          syy = runParticle->syy;
          sxy = runParticle->sxy;

          // eigenvalues

          smax = ((sxx + syy) / 2.0) +
                 sqrt (((sxx - syy) / 2.0) * ((sxx -
                                               syy) / 2.0) +
                       sxy * sxy);

          smin = ((sxx + syy) / 2.0) -
                 sqrt (((sxx - syy) / 2.0) * ((sxx -
                                               syy) / 2.0) +
                       sxy * sxy);

          // mean stress

          mean = mean + (smax + smin) / 2.0;

          // mean of tensor

          m_sxx = m_sxx + sxx;

          m_syy = m_syy + syy;

          m_sxy = m_sxy + sxy;

          // differential stress

          differential = differential + (smax - smin);
        }
      runParticle = runParticle->nextP;
    }

  // and divide by number of particles

  mean = mean / p_counter;

  m_sxx = m_sxx / p_counter;

  m_syy = m_syy / p_counter;

  m_sxy = m_sxy / p_counter;

  differential = differential / p_counter;

  stat = fopen ("statistictwograins.txt", "a");	// open statistic
  // output append
  // file

  // and dump data for first grain

  fprintf (stat, "%d", nb1);
  fprintf (stat, ";%f", finite_strain);
  fprintf (stat, ";%f", mean);
  fprintf (stat, ";%f", m_sxx);
  fprintf (stat, ";%f", m_syy);
  fprintf (stat, ";%f", m_sxy);
  fprintf (stat, ";%f", differential);

  // and do second grain
  // reset counters

  p_counter = 0;

  mean = 0.0;
  differential = 0.0;
  m_sxx = 0.0;
  m_syy = 0.0;
  m_sxy = 0.0;


  for (j = 0; j < numParticles; j++)	// and loop again
    {
      if (runParticle->grain == nb2)	// if second grain
        {
          p_counter = p_counter + 1;	// count particles

          // stress tensor

          sxx = runParticle->sxx;
          syy = runParticle->syy;
          sxy = runParticle->sxy;

          // eigenvalues

          smax = ((sxx + syy) / 2.0) +
                 sqrt (((sxx - syy) / 2.0) * ((sxx -
                                               syy) / 2.0) +
                       sxy * sxy);

          smin = ((sxx + syy) / 2.0) -
                 sqrt (((sxx - syy) / 2.0) * ((sxx -
                                               syy) / 2.0) +
                       sxy * sxy);

          // mean stress

          mean = mean + (smax + smin) / 2.0;

          // mean stress tensor

          m_sxx = m_sxx + sxx;

          m_syy = m_syy + syy;

          m_sxy = m_sxy + sxy;

          // differential stress

          differential = differential + (smax - smin);
        }
      runParticle = runParticle->nextP;
    }

  // and divide by number of particles

  mean = mean / p_counter;

  m_sxx = m_sxx / p_counter;

  m_syy = m_syy / p_counter;

  m_sxy = m_sxy / p_counter;

  differential = differential / p_counter;

  // and put in the file

  fprintf (stat, "%d", nb2);
  fprintf (stat, ";%f", mean);
  fprintf (stat, ";%f", m_sxx);
  fprintf (stat, ";%f", m_syy);
  fprintf (stat, ";%f", m_sxy);
  fprintf (stat, ";%f\n", differential);

  fclose (stat);		// and close

}

/*************************************************************************
 * function that plots the x and y coordinates of surface particles
 * and the strain. First row is strain than x and y coordinates 
 * of one particle. Note that this is a scan along the particles so that 
 * an additional function is needed to sort the data so that a 
 * continuous function of the surface is attained. 
 ************************************************************************/


void
Lattice::DumpStatisticSurface (double strain)
{
  FILE *stat;		// file pointer
  int i, j, jj, p_counter;	// counters
  float mean, differential;	// mean stress, differential
  // stress, pressure
  float m_sxx, m_syy, m_sxy;	// stress tensor
  float finite_strain;	// finite strain within the box of
  // interest
  float sxx, syy, sxy, smax, smin;	// stress tensor and eigenvalues

  // set counters to zero

  finite_strain = strain * (def_time_u + def_time_p);	// calculate
  // finite strain

  stat = fopen ("statisticsurface.txt", "a");	// open statistic output
  // append file


  runParticle = &refParticle;	// start

  for (i = 0; i < numParticles; i++)	// look which particles are in the
    // box
    {
      if (runParticle->isHoleBoundary)	// larger ymin
        {

          // and dump the data
          fprintf (stat, "%f", finite_strain);
          fprintf (stat, " %f", runParticle->xpos);
          fprintf (stat, " %f\n", runParticle->ypos);

        }
      runParticle = runParticle->nextP;
    }

  fclose (stat);		// close file
}

/**************************************************************************************
 * function that dumps the y coordinate of surface particles within a region 
 * xmin < x < xmax. end of line is the strain 
 **************************************************************************************/

void
Lattice::DumpTimeStatisticSurface (double strain, double xmin, double xmax)
{
  FILE *stat;		// file pointer
  int i, j, jj, p_counter;	// counters
  float mean, differential;	// mean stress, differential
  // stress, pressure
  float m_sxx, m_syy, m_sxy;	// stress tensor
  float finite_strain;	// finite strain within the box of
  // interest
  float sxx, syy, sxy, smax, smin;	// stress tensor and eigenvalues

  // set counters to zero

  finite_strain = strain * (def_time_u + def_time_p);	// calculate
  // finite strain

  stat = fopen ("timestatsurf.txt", "a");	// open statistic output
  // append file


  runParticle = &refParticle;	// start

  for (i = 0; i < numParticles; i++)	// look which particles are in the
    // box
    {
      if (runParticle->isHoleBoundary)	// larger ymin
        {
          if (runParticle->xpos < xmax)
            {
              if (runParticle->xpos > xmin)
                {
                  // and dump the data


                  fprintf (stat, " %f", runParticle->ypos);	// one row
                  // of y
                  // coordinates
                }
            }

        }
      runParticle = runParticle->nextP;
    }

  fprintf (stat, " %f\n", finite_strain);	// give out strain

  fclose (stat);		// close file
}

// SetGaussianStrengthDistribution
/*******************************************************************
 * puts a quenched noise on the breaking strength of the springs
 * in the box. This is a gaussian distribution of breaking strength
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
 * Function writes also a text file with the set spring strengths distribution
 * calld gauss.txt 
 *******************************************************************/

void
Lattice::SetGaussianStrengthDistribution (double g_mean, double g_sigma)
{
  FILE *stat;		// for the outfile

  float prob;		// probability from gauss
  float k_spring;		// pict spring
  int i, j;		// counters
  float ran_nb;		// rand number

  stat = fopen ("gauss.txt", "a");	// open statistic output append
  // file

  // grain_counter counter how many grain there were intially
  // times two is for security, not all grains may be there
  // this can certainly be made nicer

  runParticle = &refParticle;

  srand (std::time (0));

  for (i = 0; i < numParticles; i++)	// loop through the particles now
    {
      for (j = 0; j < 8; j++)
        {
          if (runParticle->neigP[j])
            {
              do
                {
                  k_spring = rand () / (float) RAND_MAX;	// pic a
                  // pseudorandom
                  // float
                  // between
                  // 0 and 1

                  // ----------------------------------------------------------------
                  // now convert this distribution to a distribution
                  // from
                  // zero minus 8 times sigma to plus 8 times sigma
                  // -----------------------------------------------------------------

                  k_spring =
                    (k_spring -
                     0.5) * 8.0 * (8.0 * g_sigma);

                  // --------------------------------------------------------------
                  // and shift it to mean plus minus 2 times sigma
                  // --------------------------------------------------------------

                  k_spring = k_spring + g_mean;

                  // ---------------------------------------------------------------
                  // now apply gauss function to determine a probability
                  // for this
                  // reaction constant to occur
                  // ---------------------------------------------------------------

                  prob = (1 / (g_sigma * sqrt (2.0 * 3.1415927)));	// part
                  // one
                  // (right
                  // part)
                  // of
                  // function

                  prob = prob * (exp (-0.5 * (((k_spring) - g_mean) / g_sigma) * (((k_spring) - g_mean) / g_sigma)));	// rest

                  // --------------------------------------------------------------
                  // now adjust probablity to run from 0 to 1.0
                  // --------------------------------------------------------------

                  prob = prob * sqrt (2.0 * 3.1415927) *
                         g_sigma;

                  // ---------------------------------------------------------------
                  // pic the second pseudorandom number
                  // ---------------------------------------------------------------

                  ran_nb = rand () / (float) RAND_MAX;

                  // --------------------------------------------------------------
                  // if the number picted is smaller or the same as the
                  // probability
                  // from the gauss function accept the rate constant,
                  // if not go on
                  // looping.
                  // -------------------------------------------------------------------

                  if (ran_nb <= prob)	// if smaller
                    {
                      if (k_spring <= 0.0)	// if not 0.0 or smaller
                        // (dont want that)
                        ;
                      else
                        {

                          // scale to spring default strength

                          k_spring =
                            k_spring *
                            0.0125;

                          // set the constant in the list !
                          runParticle->
                          break_Str[j] =
                            k_spring;

                          // and dump some data

                          fprintf (stat, "%f",
                                   k_spring);
                          fprintf (stat, "\n");

                          // and break the while loop !

                          break;
                        }
                    }
                }
              while (1);	// loop until break

            }
        }

      runParticle = runParticle->nextP;
    }


}

// SetGaussianYoungDistribution
/*******************************************************************
 * puts a quenched noise on the breaking strength of the springs
 * in the box. This is a gaussian distribution of breaking strength
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
 * Function writes also a text file with the set spring strengths distribution
 * calld gauss.txt 
 *******************************************************************/

void Lattice::SetGaussianYoungDistribution (double g_mean, double g_sigma)
{
  FILE *stat;		// for the outfile

  float prob;		// probability from gauss
  float k_spring;		// pict spring
  int i, j;		// counters
  float ran_nb;		// rand number

  stat = fopen ("gauss.txt", "a");	// open statistic output append
  // file

  // grain_counter counter how many grain there were intially
  // times two is for security, not all grains may be there
  // this can certainly be made nicer

  runParticle = &refParticle;

  srand (std::time (0));

  for (i = 0; i < numParticles; i++)	// loop through the particles now
    {

      do
        {
          k_spring = rand () / (float) RAND_MAX;	// pic a
          // pseudorandom
          // float
          // between
          // 0 and 1

          // ----------------------------------------------------------------
          // now convert this distribution to a distribution
          // from
          // zero minus 8 times sigma to plus 8 times sigma
          // -----------------------------------------------------------------

          k_spring =
            (k_spring -
             0.5) * 8.0 * (8.0 * g_sigma);

          // --------------------------------------------------------------
          // and shift it to mean plus minus 2 times sigma
          // --------------------------------------------------------------

          k_spring = k_spring + g_mean;

          // ---------------------------------------------------------------
          // now apply gauss function to determine a probability
          // for this
          // reaction constant to occur
          // ---------------------------------------------------------------

          prob = (1 / (g_sigma * sqrt (2.0 * 3.1415927)));	// part
          // one
          // (right
          // part)
          // of
          // function

          prob = prob * (exp (-0.5 * (((k_spring) - g_mean) / g_sigma) * (((k_spring) - g_mean) / g_sigma)));	// rest

          // --------------------------------------------------------------
          // now adjust probablity to run from 0 to 1.0
          // --------------------------------------------------------------

          prob = prob * sqrt (2.0 * 3.1415927) *
                 g_sigma;

          // ---------------------------------------------------------------
          // pic the second pseudorandom number
          // ---------------------------------------------------------------

          ran_nb = rand () / (float) RAND_MAX;

          // --------------------------------------------------------------
          // if the number picted is smaller or the same as the
          // probability
          // from the gauss function accept the rate constant,
          // if not go on
          // looping.
          // -------------------------------------------------------------------

          if (ran_nb <= prob)	// if smaller
            {
              if (k_spring <= 0.0)	// if not 0.0 or smaller
                // (dont want that)
                ;
              else
                {

                  // scale to spring default strength

                  k_spring =
                    k_spring *
                    1;

                  // set the constant in the list !
                  runParticle->
                  young =
                    k_spring;


                  // and dump some data

                  fprintf (stat, "%f",
                           k_spring);
                  fprintf (stat, "\n");

                  // and break the while loop !

                  break;
                }
            }
        }
      while (1);	// loop until break



      runParticle = runParticle->nextP;
    }
  AdjustParticleConstants();

}


/***************************************************************************
 * This function generates a gaussian distribution of springconstants k_gauss for a 
 * a given grain population. This means that the number of grains with a certain
 * function sprinconstant f(k) within a desired intervall of values follows a  
 * density function of the form f(k)=(sqrt(2*pi)*g_sigma)**(-1) exp((k_gauss-g_mean)/g_sigma)**2 
 * where g_sigma is the standard variation, g_mean is the mean value.
 * Remember that g_mean determines the centre of the distribution and 
 * sigma the sharpness and height of the distribution. 
 * 
 * the function receives the mean and the sigma for the distribution. 
 * AS mentioned above the mean will be the mean of the spring constants of the 
 * grains so that a mean statistic of the whole box should give this constant as 
 * long as the system behaves linear elastic. The distibution is set for grains
 * so that all particles in a grain will have the same spring constants (i.e. their
 * springs will). mean will give absolute spring constants so that this function
 * should be called first before anything else is set, otherwise it will be wiped out. 
 * No problem however in setting a gauss distribution and afterwards making a layer
 * hard or soft. Default spring constant in the model is 1.0 so that a mean of 1.0 
 * will produce values clustering around 1.0 and 2.0 values with a mean spring constant 
 * of 2.0 etc.. 
 * We determine distribution within the interval mean plus minus two times sigma ! 
 *
 * The distribution is determined as follows: Pic a pseudorandom number (from mean 
 * plus/minus 2 times sigma)  to give a grain
 * a spring constant. Then determine the probability for that constant to appear from
 * the Gauss function. Then pic a second pseudo random number (from 0 to sigma) to 
 * determine whether or not the grain will actually get this spring constant. If 
 * the random number is below the probability the spring contant is accepted, 
 * if it is higher it is rejected and a new one is drawn. 
 *
 * Function writes also a text file with the set spring constant distribution
 * calld gauss.txt
 *
 * Daniel and Jochen Feb. 2003
 *
 * add adjustment to youngs modulus of particles, 
 * 
 * Daniel, March 2003
 **************************************************************************/

void
Lattice::SetGaussianSpringDistribution (double g_mean, double g_sigma)
{
  FILE *stat;		// for the outfile
  float spring_list[10000];	// distribution list for grains
  float prob;		// probability from gauss
  float k_spring;		// pict spring
  int i, j;		// counters
  float ran_nb;		// rand number

  stat = fopen ("gauss.txt", "a");	// open statistic output append
  // file

  // grain_counter counter how many grain there were intially
  // times two is for security, not all grains may be there
  // this can certainly be made nicer

  for (i = 0; i < (grain_counter * 2); i++)	// loop through grains
    {
      do
        {
          k_spring = rand () / (float) RAND_MAX;	// pic a
          // pseudorandom
          // float between 0
          // and 1

          // ----------------------------------------------------------------
          // now convert this distribution to a distribution from
          // zero minus 2 times sigma to plus 2 times sigma
          // -----------------------------------------------------------------

          k_spring = (k_spring - 0.5) * 2.0 * (2.0 * g_sigma);

          // --------------------------------------------------------------
          // and shift it to mean plus minus 2 times sigma
          // --------------------------------------------------------------

          k_spring = k_spring + g_mean;

          // ---------------------------------------------------------------
          // now apply gauss function to determine a probability for
          // this
          // spring constant to occur
          // ---------------------------------------------------------------

          prob = (1 / (g_sigma * sqrt (2.0 * 3.1415927)));	// part
          // one
          // (right
          // part)
          // of
          // function

          prob = prob * (exp (-0.5 * (((k_spring) - g_mean) / g_sigma) * (((k_spring) - g_mean) / g_sigma)));	// rest

          // --------------------------------------------------------------
          // now adjust probablity to run from 0 to 1.0
          // --------------------------------------------------------------

          prob = prob * sqrt (2.0 * 3.1415927) * g_sigma;

          // ---------------------------------------------------------------
          // pic the second pseudorandom number
          // ---------------------------------------------------------------

          ran_nb = rand () / (float) RAND_MAX;

          // --------------------------------------------------------------
          // if the number picted is smaller or the same as the
          // probability
          // from the gauss function accept the spring constant, if not
          // go on
          // looping.
          // -------------------------------------------------------------------

          if (ran_nb <= prob)	// if smaller
            {
              if (k_spring <= 0.0)	// if not 0.0 or smaller (dont
                // want that)
                ;
              else
                {
                  // set the constant in the list !

                  spring_list[i] = k_spring;
                  grain_young[i] = k_spring;

                  // and dump some data

                  fprintf (stat, "%f", k_spring);
                  fprintf (stat, "\n");

                  // and break the while loop !

                  break;
                }
            }
        }
      while (1);	// loop until break
    }

  runParticle = &refParticle;

  for (i = 0; i < numParticles; i++)	// loop through the particles now
    {
      for (j = 0; j < 8; j++)	// loop through neighbours
        {

          if (runParticle->neigP[j])	// if neighbour
            {

              // ------------------------------------------------------------------------
              // we use the place in the spring_list as number of the
              // grain
              // then we set all particles in the grain to that spring
              // constant
              // including all springs of particles.
              // ------------------------------------------------------------------------

              runParticle->springf[j] =
                spring_list[runParticle->grain];

              // ---------------------------------------------------
              // and set the new youngs modulus
              // ---------------------------------------------------

              runParticle->young =
                runParticle->springf[j] * 2.0 /
                sqrt (3.0);
            }
        }
      runParticle = runParticle->nextP;
    }

  fclose (stat);		// and close file

  AdjustConstantGrainBoundaries ();	// clean up (make mean of grain
  // boundary springs)
}


/************************************************************************************************
 * Another file to dump statistics in a text file. This is for a whole box. It gets the input values
 * box position with is box min in y box max in y, box min in x and box max in x and also the strain
 * applied on the whole box (the elle box now) from your deformation file. 
 * The function takes a mean of all particles in the box. It calculates means for the mean stress, 
 * the differential stress and the stress tensor sxx, syy, sxy. 
 *
 * output is finite strain on box, mean stress, sxx,syy,sxy and differential stress. Function
 * can be called from whereever you want it. 
 *
 * daniel and jochen Feb. 2003
 *****************************************************************************************************/


void
Lattice::DumpStatisticStressBox (double y_box_min, double y_box_max,
                                 double x_box_min, double x_box_max,
                                 double strain)
{
  FILE *stat;		// file pointer
  int i, j, jj, p_counter;	// counters
  float mean, differential;	// mean stress, differential
  // stress, pressure
  float m_sxx, m_syy, m_sxy;	// stress tensor
  float finite_strain;	// finite strain within the box of
  // interest
  float sxx, syy, sxy, smax, smin;	// stress tensor and eigenvalues

  // set counters to zero

  p_counter = 0;

  mean = 0.0;
  differential = 0.0;
  m_sxx = 0.0;
  m_syy = 0.0;
  m_sxy = 0.0;

  finite_strain = strain * (def_time_u + def_time_p);	// calculate
  // finite strain


  runParticle = &refParticle;	// start

  for (i = 0; i < numParticles; i++)	// look which particles are in the
    // box
    {
      if (runParticle->ypos > y_box_min)	// larger ymin
        {
          if (runParticle->ypos < y_box_max)	// smaller ymax
            {
              if (runParticle->xpos > x_box_min)	// larger xmin
                {
                  if (runParticle->xpos < x_box_max)	// smaller xmax
                    {
                      p_counter = p_counter + 1;	// count particles

                      // set stress tensor

                      sxx = runParticle->sxx;
                      syy = runParticle->syy;
                      sxy = runParticle->sxy;

                      // eigenvalues

                      smax = ((sxx + syy) / 2.0) +
                             sqrt (((sxx -
                                     syy) / 2.0) *
                                   ((sxx -
                                     syy) / 2.0) +
                                   sxy * sxy);

                      smin = ((sxx + syy) / 2.0) -
                             sqrt (((sxx -
                                     syy) / 2.0) *
                                   ((sxx -
                                     syy) / 2.0) +
                                   sxy * sxy);

                      // values added for particles

                      m_sxx = m_sxx + sxx;

                      m_syy = m_syy + syy;

                      m_sxy = m_sxy + sxy;

                      mean = mean + (smax +
                                     smin) / 2.0;

                      differential =
                        differential + (smax -
                                        smin);
                    }
                }
            }
        }
      runParticle = runParticle->nextP;
    }

  // and divide by number of particles

  mean = mean / p_counter;

  differential = differential / p_counter;

  m_sxx = m_sxx / p_counter;

  m_syy = m_syy / p_counter;

  m_sxy = m_sxy / p_counter;

  stat = fopen ("statisticbox.txt", "a");	// open statistic output
  // append file

  // and dump the data

  fprintf (stat, "%f", finite_strain);
  fprintf (stat, ";%f", mean);
  fprintf (stat, ";%f", m_sxx);
  fprintf (stat, ";%f", m_syy);
  fprintf (stat, ";%f", m_sxy);
  fprintf (stat, ";%f\n", differential);

  fclose (stat);		// close file
}


/*************************************************************************************************
 * Function that sets a pseudorandom distribution on spring constants and/or on breaking-
 * strength of springs. Spring constants are set for grains, breaking strength for the 
 * whole model. First number is for the mean spring constant. This number is multiplied by 
 * the set values (default 1.0) and the result substracted or added so that 0.0 does not change the mean. 
 * Therefore anisotropies or distributions in the model will not be distroyed but redistributed
 * depending on what they are. The second number gives the distribution size. The distribution 
 * is original mean times str_mean plus/minus half of str_size. 
 * The same applies for the breaking strength of springs. 
 * if second or fourth numbers are 0.0 no distributions are set for these values
 *
 * Daniel and Jochen, Feb. 2003
 *
 * adjust youngs modulus of particle, bug in mean, add helpers
 *
 * Daniel, March 2003
 *************************************************************************************************/

void
Lattice::SetPhase (float str_mean, float str_size, float br_mean,
                   float br_size)
{
  int i, j, nb;		// counter
  float ran_nb;		// pseudorandom number
  float str_nb, br_nb;	// number for spring constant, number for
  // breaking strength
  float str_m, br_m;	// helpers for the mean

  srand (std::time (0));

  runParticle = &refParticle;	// and go

  for (nb = 0; nb < (grain_counter * 2); nb++)	// loop through
    // grains
    {
      if (str_size != 0.0)	// in case a distribution is wanted
        {
          ran_nb = rand () / (float) RAND_MAX;	// pic pseudorandom from 0
          // to 1.0 for grains
          str_nb = ran_nb * str_size;	// now from 0 to distribution size
        }

      for (i = 0; i < numParticles; i++)	// loop through particles
        {
          if (runParticle->grain == nb)	// if particle of grain
            {
              if (str_size != 0.0)	// if distribution
                {
                  if (!runParticle->is_lattice_boundary)
                    {
                      for (j = 0; j < 8; j++)	// loop through neighbours
                        {

                          if (runParticle->neigP[j])	// if neighbour
                            {
                              // ------------------------------------------------
                              // multiply mean by springconstant
                              // ------------------------------------------------

                              str_m = runParticle->springf[j] * str_mean;

                              // --------------------------------------------------
                              // apply random distribution to spring
                              // --------------------------------------------------

                              runParticle->
                              springf
                              [j] =
                                runParticle->
                                springf
                                [j] *
                                (1.0 +
                                 str_nb
                                 -
                                 (str_size
                                  /
                                  2.0));

                              // ---------------------------------------------------
                              // and shift by the str_mean
                              // ---------------------------------------------------

                              runParticle->
                              springf
                              [j] =
                                runParticle->
                                springf
                                [j] +
                                str_m;
                            }
                        }
                      // ------------------------------------------
                      // and set youngs modulus of particle
                      // ------------------------------------------

                      str_m = runParticle->young *
                              str_mean;
                      runParticle->young =
                        runParticle->young *
                        (1.0 + str_nb -
                         (str_size / 2.0));
                      runParticle->young =
                        runParticle->young +
                        str_m;
                    }
                }
              // --------------------------------------------------------
              // now do the same for all springs
              // --------------------------------------------------------

              if (br_size != 0.0)	// if distribution is wanted
                {
                  for (j = 0; j < 8; j++)	// loop through neighbours
                    {

                      if (runParticle->neigP[j])	// if neighbour
                        {

                          // ------------------------------------------------------
                          // we pic a different pseudorandom number for
                          // each spring
                          // --------------------------------------------------------

                          ran_nb = rand () / (float) RAND_MAX;	// pic
                          // number
                          // between
                          // 0.0 and
                          // 1.0

                          br_nb = ran_nb * br_size;	// now from 0.0 to
                          // br_size

                          //br_m = runParticle->break_Str[j] * br_mean;	// set
                          // br_mean

                          // --------------------------------------------------------
                          // and apply distribution to the spring
                          // breaking strength
                          // -------------------------------------------------------

                          runParticle->
                          break_Str[j] = runParticle->break_Str[j] +
                                         (runParticle->
                                          break_Str[j] *
                                          (br_nb -
                                           (br_size /
                                            2.0)));

                          // ----------------------------------------------------------
                          // and shift the distribution
                          // ---------------------------------------------------------

                          runParticle->
                          break_Str[j] =
                            runParticle->
                            break_Str[j] *
                            br_mean;
                        }
                    }
                }
            }
          runParticle = runParticle->nextP;
        }
    }
  AdjustConstantGrainBoundaries ();
}



/**********************************************************************************
 * viscous retardation after the elastic deformation step. 
 * the equilibrium lengths of springs is adjusted following an exponential 
 * stress retardation depending on the time step and the youngs modulus to 
 * viscosity ratio. 
 * input is an integer that determines whether or not a picture is taken 
 * (0 is no picture, 1 is take a picture). 
 * the second variable is the time step for the viscous relaxation. 
 * It does determine how fast the viscous step relaxes stresses before the next
 * elastic deformation step. 
 ***********************************************************************************/

/*
 * kleine fehlerliste/problem-list/komments: in
 * 
 * bool Particle::Relax(Particle **list,int size)
 * 
 * wrd noch immer der radius ( statt rad[j] fuer die sxx/syy-berrechnung
 * benutzt, in 2. Relax-funktion sicher auch.
 * 
 * * runParticle->springv ==> viscosity of spring reales verhaeltnis der
 * springs statt 2.0? formel f. ldl: haeh?
 * 
 */

void
Lattice::ViscousRelax (int dump, float timestep)
{
  int i, j, ii, jj, k, n, l;
  int lcounter;
  float ldx, ldy, ldd, lalen, ldl, lchange, ltest;

  float dx, dy;		// x and y differences
  float dd, alen;		// distances between particles
  float rep_constant;	// repulsion constant
  float fn, fx, fy;	// forces
  float uxi, uyi;		// unitdistance
  float xx, yy, anc;	// move xx and yy
  float area;		// area of particle
  float ten_break;	// help for breaking strenght
  int box, pos;		// help for Repulsion
  float sigma;		// the stress in direction of the
  // neighbour

  float angle, n_force;	// repulsion: relevant angle with regard
  // to str.tensor of neig, n_force normal
  // force in that dir.
  float rad_factor;
  float xdiff, ydiff;
  float add
    ;
  float debug;
  float buffer, buffer2, angle_buffer;
  int int_buffer, rad_nb_buffer;
  float phi,
  sigma_angle, sigman, tau, mean_stress, merker1, merker2, r,
  sigma_new, G;
  bool tens[9];
  float sxx,
  young,
  syy,
  sxy,
  sigma1,
  sigma2,
  devsxx,
  devsxy,
  devsyy,
  strain,
  shear,
  x,
  y,
  m,
  part1,
  part2,
  a, b;

  long viscosity;
  float buffer3i, rad_buf;
  int z;

  bool flag;

  visc_rel = 1;

  DumpStatisticStressBox (0.1, 0.9, 0.1, 0.9, 0.001);

  Tilt ();

  SetSpringAngle ();

  cout << "in viscous relax" << endl;

  debug = 0.0;

  for (z = 0; z < 10; z++)
    {
      for (i = 0; i < numParticles; i++)	// loop through particles
        {
          if (!runParticle->is_lattice_boundary)
          {
            // -------------------------------------------------------------------------------------------
            // preliminary settings
            // -------------------------------------------------------------------------------------------

            mean_stress = (-runParticle->sxx - runParticle->syy) / 2.0;

            // deviatoric stresses!
            sxx = -runParticle->sxx - mean_stress;
            syy = -runParticle->syy - mean_stress;
            sxy = -runParticle->sxy - mean_stress;

            // define main axes of ellipse, if not yet done
            if (runParticle->maxis1length == runParticle->maxis2length)
              {
                sigma_angle = 0.5 * atan ((2.0 * (sxy) / ((sxx)) - (sxy)));

                sigma1 = 0.5 * (sxx + syy) + 0.5 * (sxx - syy) * cos (2.0 * sigma_angle) + sxy * sin (2.0 * sigma_angle);
                sigma2 = 0.5 * (sxx + syy) + 0.5 * (sxx - syy) * cos (2.0 * (sigma_angle + Pi / 2.0)) + sxy * sin (2.0 * (sigma_angle + Pi / 2.0));

                if (sigma2 > sigma1)
                  sigma_angle += Pi / 2.0;

                runParticle->maxis1angle = sigma_angle;
                runParticle->maxis2angle = sigma_angle + Pi / 2.0;
              }

            // correct angles
            if (runParticle->maxis2angle >= 2.0 * Pi)
              runParticle->maxis2angle -= 2.0 * Pi;
            if (runParticle->maxis1angle >= 2.0 * Pi)
              runParticle->maxis1angle -= 2.0 * Pi;

            if (runParticle->maxis2angle < 0.0)
              runParticle->maxis2angle += 2.0 * Pi;
            else if (runParticle->maxis2angle > 2.0 * Pi)
              runParticle->maxis2angle -= 2.0 * Pi;

            // use angles of upper sphere
            if (runParticle->maxis1angle > Pi)
              runParticle->maxis1angle -= Pi;
            if (runParticle->maxis2angle > Pi)
              runParticle->maxis2angle -= Pi;

            // shear-constant:
            young = runParticle->young * sqrt (3.0) / 2.0;
            G = young / (2.0 * (1.0 + 0.333333333));

            // calculations for axis 1
            sigma = 0.5 * (sxx + syy) + 0.5 * (sxx - syy) * cos (2.0 * runParticle->maxis1angle) + sxy * sin (2.0 * runParticle->maxis1angle);
            tau = 0.5 * (syy - sxx) * sin (2.0 * runParticle->maxis1angle) + sxy * cos (2.0 * runParticle->maxis1angle);
            strain = (sigma * timestep / 10.0) / (2.0 * runParticle->viscosity);
            shear = atan ((tau * timestep / 10.0) / (G * 2.0 * runParticle->viscosity));

            runParticle->maxis1length *= (1.0 - strain);
            if (runParticle->maxis1angle <= 0.5 * Pi)
              {
                runParticle->maxis1angle += shear;
                runParticle->maxis2angle = runParticle->maxis1angle + (Pi / 2.0);
              }

            // calculations for axis 2
            sigma = 0.5 * (sxx + syy) + 0.5 * (sxx - syy) * cos (2.0 * runParticle->maxis2angle) + sxy * sin (2.0 * runParticle->maxis2angle);
            tau = 0.5 * (syy - sxx) * sin (2.0 * runParticle->maxis2angle) + sxy * cos (2.0 * runParticle->maxis2angle);
            strain = (sigma * timestep / 10.0) / (2.0 * runParticle->viscosity);
            shear = atan ((tau * timestep / 10.0) / (G * 2.0 * runParticle->viscosity));

            runParticle->maxis2length *= (1.0 - strain);
            if (runParticle->maxis2angle <= 0.5 * Pi)
              {
                runParticle->maxis1angle = runParticle->maxis2angle + (Pi / 2.0);
              }

            if (runParticle->maxis2angle >= 2.0 * Pi)
              runParticle->maxis2angle -= 2.0 * Pi;
            if (runParticle->maxis1angle >= 2.0 * Pi)
              runParticle->maxis1angle -= 2.0 * Pi;

            if (runParticle->maxis2angle < 0.0)
              runParticle->maxis2angle += 2.0 * Pi;
            if (runParticle->maxis1angle < 0.0)
              runParticle->maxis2angle += 2.0 * Pi;

            if (runParticle->maxis2angle >= Pi)
              runParticle->maxis2angle -= Pi;
            if (runParticle->maxis1angle >= Pi)
              runParticle->maxis1angle -= Pi;

            // for solution in particle::alen():
            // separate larger and smaller axis, put into a and b
            if (runParticle->maxis1length > runParticle->maxis2length)
              {
                runParticle->a = runParticle->maxis1length;
                runParticle->b = runParticle->maxis2length;
                runParticle->fixangle = runParticle->maxis1angle;
              }
            else
              {
                runParticle->a = runParticle->maxis2length;
                runParticle->b = runParticle->maxis1length;
                runParticle->fixangle = runParticle->maxis2angle;
              }
          }
          runParticle = runParticle->nextP;
        }

      for (i = 0; i < numParticles; i++)
        {
          runParticle->Relax (repBox, particlex,visc_rel,walls, right_wall_pos,
                              left_wall_pos, lower_wall_pos,
                              upper_wall_pos, wall_rep);

          runParticle = runParticle->nextP;
        }
    }

  UpdateSpringLength ();

  Tilt ();

  Relaxation ();

  Tilt ();

  DumpStatisticStressBox (0.1, 0.9, 0.1, 0.9, 0.001);
}



/*******************************************
 ** vorweg:
 ** irgendwo musste ein standard-wert f�r die polygon-flï¿�he
 ** definiert werden (polygon-flï¿�he != kreisflï¿�he).
 ** habe sie momentan in die funktion Def_Polyg_Area() verlegt, aufruf im konstruktor,
 ** name der globalen variablen polyg_area, lattice-klasse.
 *******************************************/
void
Lattice::Adjust_Radii ()
{
  int i, k, j, l, m, n;

  double a,		// flï¿�he (des polygons)
  ratio_a,		// flï¿�he wie sie sein soll durch
  // flï¿�he neu
  alpha,		// winkel zur x-achse einer spring bzw.
  // rad[]
  polyg_feld[9][2],	// speichert eck-koordinaten des neuen
  // polygons (x-wert, y-wert)
  xdiff, ydiff, xpos, ypos, factor, angle, buffer500;

  bool springs_present;

  SetSpringRatio ();

  // polyg_area8 =
  // sin(Pi/8)*refParticle.radius*cos(Pi/8)*refParticle.radius*8;

  polyg_area8 =
    2.0 * refParticle.radius * refParticle.radius * sqrt (2.0);


  cout << "polyg_area8: " << polyg_area8 << endl;

  // the following part guarantees area constance if wished

  for (j = 0; j < numParticles; j++)
    {
      if (!runParticle->is_lattice_boundary)
        {
          // koordinaten ins feld
          // benutzt (klar) standardkoord. + aktuellen boxrad
          for (i = 0; i < 8; i++)
            {
              if (i == 3)
                {
                  factor = 0;
                  k = 5;
                }
              else if (i == 7)
                {
                  factor = 4;
                  k = 3;
                }
              else if (i < 3)
                {
                  factor = (i + 5);
                  k = i;
                }
              else
                {
                  factor = (i - 3);
                  k = 12 - i;
                }

              polyg_feld[i][0] =
                cos (factor * (Pi / 4)) *
                runParticle->rep_rad[k];
              polyg_feld[i][1] =
                sin (factor * (Pi / 4)) *
                runParticle->rep_rad[k];
            }

          // letzter punkt im feld = 1. punkt
          polyg_feld[8][0] = polyg_feld[0][0];
          polyg_feld[8][1] = polyg_feld[0][1];

          // aktuelle flaeche berechnen (s.
          // www.iti.fh-flensburg.de/lang/algorithmen/geo/polygon.htm#section3)
          a = 0.0;
          for (i = 0; i < 8; i++)
            {
              a += ((polyg_feld[i + 1][0] -
                     polyg_feld[i][0]) * (polyg_feld[i +
                                                     1][1] +
                                          polyg_feld[i][1]))
                   / 2;
            }

          a = fabs (a);

          ratio_a = sqrt (polyg_area8 / a);

          if (ratio_a > buffer500)
            {
              buffer500 = ratio_a;
            }

          runParticle->ratio_a = ratio_a;

          for (i = 0; i < 9; i++)
            {
              if (i != 4)
                {
                  runParticle->rep_rad[i] *= ratio_a;
                  if (runParticle->rep_rad[i] <
                      (runParticle->radius / 10.0))
                    cout << "halt" << endl;
                }
            }
        }
      runParticle = runParticle->nextP;
    }

  cout << "ratio_a max:" << buffer500 << endl;
}



// if no neighbour exists in the Adjust_Radii-function (or in the
// SetSpringRatio-function), then go and get x-/y-coordinates HERE
// returns x-value, if xy==0;;;y-value, if xy==1

double
Lattice::VirtualNeighbour (int spring, int xy)
{
  double coord,
  alpha,
  beta,
  total_angle,
  numerator,
  denominator,
  ratio,
  upper_angle,
  lower_angle,
  upper_xdiff,
  lower_xdiff, upper_ydiff, lower_ydiff, buffer, buffer2,
  buffer3, buffer4;
  int i, j, k, l, n, upper_empty_springs, lower_empty_springs,
  total_empty_springs, lower_real_spring, upper_real_spring,
  sign;
  bool upper_atan_exists, lower_atan_exists;

  // first find out, if there exist more virtual neighbours, and if so,
  // count them
  // while, da 5/0 grenze ueberschritten werden kann
  upper_empty_springs = lower_empty_springs = 0;
  // nach oben
  i = (spring < 5) ? spring + 1 : 0;
  while (!runParticle->neigP[i])
    {
      upper_empty_springs++;
      i++;
      if (i == 6)
        {
          i = 0;
        }
    }
  // nach unten
  i = (spring > 0) ? spring - 1 : 5;
  while (!runParticle->neigP[i])
    {
      lower_empty_springs++;
      i--;
      if (i == -1)
        {
          i = 5;
        }
    }

  total_empty_springs = upper_empty_springs + lower_empty_springs + 1;

  if (total_empty_springs != 6)	// achtung: s.u.
    {
      lower_real_spring = spring - lower_empty_springs - 1;
      lower_real_spring =
        (lower_real_spring <
         0) ? 6 + lower_real_spring : lower_real_spring;
      upper_real_spring = spring + upper_empty_springs + 1;
      upper_real_spring =
        (upper_real_spring >
         5) ? upper_real_spring - 6 : upper_real_spring;
    }
  else			// dh wenn nur noch eine spring exitiert
    // (kreis->keine 5 total_empty_springs!!!)
    {			// dieser workaround sollte bei
      // bestimmungsgemaessem ablauf NICHT zur
      // anwendung kommen!
      cout << "fucked up in VirtualNeighbour()" << endl;
      i = 0;
      while (!runParticle->neigP[i])
        {
          i++;
        }
      lower_real_spring = i;
      upper_real_spring = i;
    }

  // differenzen festlegen, um richtung zu indizieren
  upper_xdiff =
    runParticle->xpos -
    runParticle->neigP[upper_real_spring]->xpos;
  upper_ydiff =
    runParticle->ypos -
    runParticle->neigP[upper_real_spring]->ypos;
  lower_xdiff =
    runParticle->xpos -
    runParticle->neigP[lower_real_spring]->xpos;
  lower_ydiff =
    runParticle->ypos -
    runParticle->neigP[lower_real_spring]->ypos;

  // falls atan nicht loesbar...
  if (lower_xdiff == 0)
    {
      lower_atan_exists = false;
    }
  else
    {
      lower_angle =
        atan ((runParticle->neigP[lower_real_spring]->ypos -
               runParticle->ypos) /
              (runParticle->neigP[lower_real_spring]->xpos -
               runParticle->xpos));
      lower_atan_exists = true;
    }

  if (upper_xdiff == 0)
    {
      upper_atan_exists = false;
    }
  else
    {
      upper_angle =
        atan ((runParticle->neigP[upper_real_spring]->ypos -
               runParticle->ypos) /
              (runParticle->neigP[upper_real_spring]->xpos -
               runParticle->xpos));
      upper_atan_exists = true;
    }

  if (lower_xdiff > 0)
    {
      lower_angle = lower_angle + Pi;
    }
  else if (lower_xdiff < 0 && lower_ydiff > 0)
    {
      lower_angle = lower_angle + 2 * Pi;
    }

  if (upper_xdiff > 0)
    {
      upper_angle = upper_angle + Pi;
    }
  else if (upper_xdiff < 0 && upper_ydiff > 0)
    {
      upper_angle = upper_angle + 2 * Pi;
    }

  if (lower_real_spring < upper_real_spring)
    {
      if (lower_angle < upper_angle)
        total_angle = upper_angle - lower_angle;
      else
        total_angle = 2 * Pi - lower_angle + upper_angle;
    }
  else
    {
      if (lower_angle > upper_angle)
        total_angle = 2 * Pi - lower_angle + upper_angle;
      else
        total_angle = upper_angle - lower_angle;
    }

  if (total_empty_springs == 5)
    total_angle = 2 * Pi;

  // hier erstmal ratio ueber gesamtheit der kaputten (nachbar-)springs
  // kalkulieren
  // variablen: ratio, buffer, i, j u.v.m.
  // ursprngliche version

  // dritte version

  ratio = 0;

  i = lower_real_spring + 1;
  if (i > 5)
    i = 0;

  n = spring + 1;
  if (n > 5)
    n = 0;

  while (i != n)
    {
      j = i;
      buffer = 1.0;

      while (j != n)
        {
          buffer /= runParticle->ratio[j];

          j++;
          if (j > 5)
            j = 0;
        }

      buffer2 = buffer;

      j = spring + 1;
      if (j > 5)
        j = 0;

      while (j != upper_real_spring)
        {
          buffer2 += buffer2 / runParticle->ratio[j];

          j++;
          if (j > 5)
            j = 0;
        }

      ratio += 1.0 / buffer2;

      i++;
      if (i > 5)
        i = 0;
    }

  // winkel berechnen
  alpha = total_angle / ((1.0 / ratio) + 1);	// angle OVER missing
  // spring (in terms of
  // indices)
  beta = total_angle / (ratio + 1);	// abgle UNDER missing spring (in
  // terms of indices)

  if (xy == 0)		// x-wert berechnen und zurueckgeben
    {
      // hier vermutlich noch jede MENGE SPEZIALFAELLE (vielleicht auch
      // nicht...)
      coord = cos (alpha + lower_angle) * runParticle->rad[spring] +
              runParticle->xpos;
    }
  else			// y-wert berechnen und zurueckgeben
    {
      // hier vermutlich noch jede MENGE SPEZIALFAELLE (hier wohl auch
      // nicht...)
      coord = sin (alpha + lower_angle) * runParticle->rad[spring] +
              runParticle->ypos;
    }

  return (coord);
}



void
Lattice::SetSpringAngle ()
{
  int i, j;
  float spring_angle[6];
  float ref_xpos, ref_ypos, xdiff, ydiff;
  bool around_flag;

  for (i = 0; i < numParticles; i++)
    {
      around_flag = false;
      // einmal durch alle springs des partikels loopen und jeder spring
      // einen absolut-winkelwert zuweisen
      for (j = 0; j < 6; j++)
        {
          if (runParticle->neigP[j])
            {
              ref_xpos = runParticle->neigP[j]->xpos;
              ref_ypos = runParticle->neigP[j]->ypos;

              spring_angle[j] =
                atan ((runParticle->ypos -
                       ref_ypos) /
                      (runParticle->xpos - ref_xpos));

              xdiff = runParticle->xpos - ref_xpos;
              ydiff = runParticle->ypos - ref_ypos;

              if (xdiff > 0)
                {
                  spring_angle[j] =
                    spring_angle[j] + Pi;
                }
              else if (xdiff < 0 && ydiff > 0)
                {
                  spring_angle[j] =
                    spring_angle[j] + 2 * Pi;
                }

              if (xdiff == 0.0)
                {
                  if (ydiff < 0.0)
                    {
                      spring_angle[j] =
                        3.14159265 / 2.0;
                    }
                  else
                    {
                      spring_angle[j] =
                        3.14159265 * 1.5;
                    }
                }

              runParticle->spring_angle[j] =
                spring_angle[j];
            }
        }
      runParticle = runParticle->nextP;
    }
}



// setze nach jeder deformation neue ratio fuer noch AKTIVE springs
// aufruf am ende von AdjustRadii + konstruktor
void
Lattice::SetSpringRatio ()
{
  int i, j, k, l, m, o;
  double ratio, angle[3], xpos, ypos, ref_xpos, ref_ypos, ref_angle,
  xdiff, ydiff;
  bool springs_present, atan_exists, ref_atan_exists, zero_jump_up,
  zero_jump_down;

  runParticle = &refParticle;

  for (j = 0; j < numParticles; j++)
    {
      // find out, if there exist ANY springs (otherwise seg-fault in
      // VirtualNeighbour)
      o = 0;
      springs_present = true;
      for (m = 0; m < 6; m++)
        {
          if (!runParticle->neigP[m])
            {
              o++;
            }
        }
      if (o > 5)
        springs_present = false;

      // if there are springs, continue
      if (!runParticle->is_lattice_boundary)
        {
          if (springs_present)
            {

              runParticle->no_spring_flag = false;

              for (i = 0; i < 6; i++)
                {

                  if (runParticle->neigP[i])
                    {

                      // for neigP i, angle = angle from positive x-axis
                      ref_xpos =
                        runParticle->
                        neigP[i]->xpos;
                      ref_ypos =
                        runParticle->
                        neigP[i]->ypos;
                      ref_angle =
                        atan ((ref_ypos -
                               runParticle->
                               ypos) /
                              (ref_xpos -
                               runParticle->
                               xpos));
                      ref_atan_exists = true;

                      xdiff = runParticle->xpos -
                              ref_xpos;
                      ydiff = runParticle->ypos -
                              ref_ypos;

                      if (xdiff > 0)
                        {
                          ref_angle += Pi;
                        }
                      else if (xdiff < 0
                               && ydiff > 0)
                        {
                          ref_angle += 2 * Pi;
                        }

                      for (k = i - 1; k <= i + 1;
                           k += 2)
                        {

                          l = k;

                          zero_jump_up = false;
                          zero_jump_down =
                            false;

                          if (l == -1)
                            {
                              l = 5;
                              zero_jump_down
                              =
                                true;
                            }
                          if (l == 6)
                            {
                              l = 0;
                              zero_jump_up =
                                true;
                            }
                          if (l > 6)
                            cout << "autsch" << endl;


                          if (runParticle->
                              neigP[l])
                            {
                              xpos = runParticle->neigP[l]->xpos;
                              ypos = runParticle->neigP[l]->ypos;
                            }
                          else
                            {
                              xpos = VirtualNeighbour (l, 0);
                              ypos = VirtualNeighbour (l, 1);
                            }

                          xdiff = runParticle->
                                  xpos - xpos;
                          ydiff = runParticle->
                                  ypos - ypos;

                          angle[k - i + 1] =
                            atan (ydiff /
                                  xdiff);

                          if (xdiff > 0)
                            {
                              angle[k - i +
                                    1] +=
                                      Pi;
                            }
                          else if (xdiff < 0
                                   && ydiff > 0)
                            {
                              angle[k - i +
                                    1] +=
                                      2 * Pi;
                            }

                          if (zero_jump_down)
                            {
                              if (ref_angle
                                  <
                                  angle[k -
                                        i +
                                        1])
                                {
                                  angle[k - i + 1] = 2 * Pi - angle[k - i + 1] + ref_angle;
                                }
                              else
                                {
                                  angle[k - i + 1] = ref_angle - angle[k - i + 1];
                                }
                            }
                          else if (zero_jump_up)
                            {
                              if (ref_angle
                                  <
                                  angle[k -
                                        i +
                                        1])
                                {
                                  angle[k - i + 1] = angle[k - i + 1] - ref_angle;
                                }
                              else
                                {
                                  angle[k - i + 1] = 2 * Pi + angle[k - i + 1] - ref_angle;
                                }
                            }
                          else
                            {
                              if (k < i)
                                {
                                  if (ref_angle > angle[k - i + 1])
                                    {
                                      angle[k - i + 1] = ref_angle - angle[k - i + 1];
                                    }
                                  else
                                    {
                                      angle[k - i + 1] = 2 * Pi + ref_angle - angle[k - i + 1];
                                    }
                                }
                              else
                                {
                                  if (angle[k - i + 1] > ref_angle)
                                    {
                                      angle[k - i + 1] = angle[k - i + 1] - ref_angle;
                                    }
                                  else
                                    {
                                      angle[k - i + 1] = 2 * Pi - ref_angle + angle[k - i + 1];
                                    }
                                }
                            }
                        }
                      // achtung: so rum oder anders rum???
                      runParticle->ratio[i] =
                        fabs (angle[2] /
                              angle[0]);
                    }
                  else
                    {
                      // dummy else for debugging
                    }
                }
            }
          else	// no springs present
            {
              runParticle->no_spring_flag = true;
            }
        }
      runParticle = runParticle->nextP;
    }
}



// function, that writes the averaged radius into a field, depending on
// the direction
// directions associated to values 1..9 depend on box in particle-class
// (for the repulsion)

// result will be written in particle->rep_rad[9]

// new stuff: now calculates length of the rep_rad using quadratic
// interpolation, based on the slopes on the edges and the x/y-positions
// of the edges themselves
// variable/name definition: y = a + bx + cx^2
void
Lattice::BoxRad ()
{}



/*************************************************************
 * function that changes the relaxation threshold for 
 * the relaxation routine. number < 1.0 is better relaxation 
 * but relaxation takes longer. 
 ************************************************************/

void
Lattice::ChangeRelaxThreshold (float change)
{
  relaxthres = relaxthres * change;
}

/***************************************************
* function tilts elliptical particles 
*****************************************************/

void
Lattice::Tilt ()
{
  int i;
  float buffer;

  buffer = 0.0;

  EraseNeighList ();

  SetSpringAngle ();

  for (i = 0; i < numParticles; i++)
    {
      // runParticle->Tilt();
      UpdateNeighList ();

      runParticle = runParticle->nextP;
    }

  SetSpringAngle ();

  UpdateSpringLength ();
}

