/******************************************************
 * Spring Code Latte 2.0
 *
 * Functions for particle class in particle.cc
 *
 * Class lists defined in Lattice header
 *
 * Daniel Koehn and Jochen Arnold
 * Feb. 2002 to Feb. 2003
 *
 * Daniel Dec. 2003
 *
 * We thank Anders Malthe-Srenssen for his enormous help
 * and introduction to these codes
 *
 * new version daniel/Till 2004/5
 * koehn_uni-mainz.de
 ******************************************************/
#include <iostream>

#include "particle.h"

using std::cout;
using std::endl;

// CONSTRUCTOR
/****************************************************
 * Does not do much , defines radius
 * sets flags and some variables to 0
 *
 * Daniel spring 2002
 **************************************************/

Particle::Particle():

    //-------------------------------------------
    // variable definitions
    //-------------------------------------------

    radius (0.005)  // set radius default only for x = 100 ! now scaled automatically

{
  fix_y = false;  // particle can move in y direction
  fix_x = false;  // particle can move in x direction

  done = false;   // flag for done Relaxation of particle

  mineral = 1;	// default number for mineral
  conc = 44000.0;   // default concentration
  fluid_P = 0.0;   // default fluid pressure

  merk = 0;
	
	fluid_particles = 0; // for lattice gas 

  isUpperBox = true; // for stylolites 

  rate_factor = 1.0; // for distribution in dissolution 

  isHole = false; // flag for dissolution etc.
  isHoleBoundary = false;  // flag for hole boundaries
  isHoleInternalBoundary = false;   // particle of fluid along interface

  box_pos = -1;   // no boxposition is -1

  draw_break = false;  // particle has no broken bond
  neigh = 0;           // no neighbour breaks

  next_inBox = NULL;     // pointer  to  next  in  box

  young = 2.0/sqrt(3.0);  // default youngs modulus

  grain = -1;        // no grain is -1
  
  mV = 1;	// just in case for Energy

  is_boundary = false;   // default not a grainboundary particle
  is_lattice_boundary = false;  // boundary of box for boundary conditions

  movement = 0.0;
  temperature = 0.0;
  
  nofluid = false;  // for lattice gas fracture walls
  
  //---------------------------------------------
  // if 1.0 is 10 GPa then viscosity 1e11 means
  // viscosity of 1e21 Pas
  //---------------------------------------------

  viscosity = 1e11;  // set viscosity default
  
  

  //---------------------------------------------
  // set all neighbours to zero
  //---------------------------------------------

  for (i=0;i<8;i++)
    {
      neigP[i] = 0;                   		// pointer  to  neighbour
      break_Str[i] = 0.0;             		// tensile normal stress for spring
      break_Sp[i] = false;            		// does spring break ?
      no_break[i] = false;				  	// springs can break
		fluid_particle[i] = 0;				// lattice gas fluid particles 
		fluid_particle_new[i] = 0;			// lattice gas fluid particles 
    }

  // set list for elle nodes to -1 = empty slot

  for (i=0;i<32;i++)          // maximum of 32 Elle_nodes per particle at moment
    {
      elle_Node[i] = -1;
    }

  visc_flag = false; //for debugging
}

/******************************************************************
*
* function is used to tilt ellipses, only in viscoelastic code 
*
******************************************************************/

void
Particle::Tilt()
{
  int i, counter;
  float spring_angle, angle, xdiff, ydiff;

  counter = 0;

  rot_angle = 0.0;

  for (i=0; i<6; i++)
    {
      if (neigP[i])
        {
          switch (i)
            {
              //neu: jetzt mit ursprungswinkel
            case 0:
              spring_angle = 0.0;
              break;
            case 1:
              spring_angle = 1.047197551;
              break;
            case 2:
              spring_angle = 2.094395102;
              break;
            case 3:
              spring_angle = 3.141592654;
              break;
            case 4:
              spring_angle = 4.188790205;
              break;
            case 5:
              spring_angle = 5.235987756;
              break;
            }

          counter++;

          xdiff = xpos - neigP[i]->xpos;
          ydiff = ypos - neigP[i]->ypos;

          angle = atan(ydiff/xdiff);

          if (xdiff > 0.0)
            {
              angle = angle + 3.14159265;
            }
          else if (xdiff < 0.0 && ydiff > 0.0)
            {
              angle = angle + 2.0*3.14159265;
            }

          if (xdiff == 0.0)
            {
              if (ydiff < 0.0)
                {
                  angle = 3.14159265/2.0;
                }
              else
                {
                  angle = 3.14159265*1.5;
                }
            }


          if (fabs(spring_angle - angle) > 3.14159265)
            {
              if ((spring_angle - angle) < 0.0)
                {
                  angle = spring_angle + (3.14159265*2.0-angle);
                }
              else
                {
                  angle = -(3.14159265*2.0 - spring_angle + angle);
                }
            }
          else
            {
              angle = spring_angle - angle;
            }

          rot_angle -= angle;
        }
    }

  if (counter != 0)
    rot_angle /= counter;
}



// RELAX (float,Particle**,int,bool)
/*****************************************************
 * relaxation routine
 * returns true if the particle is relaxed, i.e. not 
 * moving anymore.  
 * gets the relaxationthreshold form the lattice
 * and the boxlist and x dimension
 * the routine calculates stresses from neighbours
 * and repulsive forces from nonconnected neighbours
 * and calculates how much particle will move to find
 * equilibrium. Then the particle is overrelaxed.                                            
 *
 * Daniel spring, summer 2002
 *
 * added interactive repulsion constant, caluclated
 * from the youngs moduli of particles
 *
 * daniel March, 2003
 ****************************************************/

//----------------------------------------------------------------------------
// function Relax in particle class
//
// receives relaxationthreshold, box list from lattice, size of x direction
//  positions of the walls and wall constant, whether or not code is viscous
//
// called form Lattice::FullRelax()
//-----------------------------------------------------------------------------


bool
Particle::Relax(float relaxthresh,Particle **list,int size,bool wall,float rightwall,float leftwall,float lowerwall,float upperwall, float wallconstant, bool length, int debug_nb, int visc_rel)
{
  //------------------------------------------------
  // local variables
  //------------------------------------------------

  int i,j,k,s;               // counter
  float dx,dy;           // x and y differences
  float dd,alen;         // distances between particles
  float fn,fx,fy;        // forces
  float uxi,uyi;         // unitdistance
  float anc;       // move xx and yy
  float fover = 0.9;     // overrelaxation factor
  float area;            // area of particle
  float rep_constant;    // repulsion constant
  int x,y,z;         	   // for list
  int box,pos;           // box  position helper
  float rad_factor1, rad_factor2, rad0, rad1; // rad0 & rad1 enthalten richtungsbezogene radii
  float refrad, angle;
  float xxx, yyy;
  bool while_flag;
  int zaehl;

  Particle *buffer_pointer;
  float buffer, fac;
  int number;
  int merker2;

  //-----------------------------------------------------------
  //  set  box_move  indicator  to  not  moved  out  of  box
  //-----------------------------------------------------------

  //box_move = false;      // does particle move out of box ?

  //----------------------------------------------------
  // set some to Zero
  //----------------------------------------------------

  //    do
  //    {
  xx = 0.0;
  yy = 0.0;
  anc = 0.0;

  //------------------------------------------------------
  // dont relax particle itself in Rep list
  //------------------------------------------------------

  done = true;

  //-----------------------------------------------------------
  // loop through all connected neighbours of particle
  //-----------------------------------------------------------

  z = 0;
  //    zaehl = 0;
  //    if (debug_nb == nb)
  //	cout << "debug this particle!" << endl;

  for(i = 0; i < 8; i++)    // try all
    {
      if (neigP[i])      // if neighbour is connected
        {

          zaehl++;

          //--------------------------------------------------------------
          // get distance to neighbour
          //--------------------------------------------------------------

          dx = neigP[i]->xpos - xpos;
          dy = neigP[i]->ypos - ypos;

          dd = sqrt((dx*dx)+(dy*dy));

          //--------------------------------------------------------------
          // get equilibrium length
          //--------------------------------------------------------------

		  
		  if (visc_rel==1)  // is viscous 
		  {
			  neig = neigP[i];
			  rad[i] = Alen(0,dx,dy);
			  alen = rad[i] + Alen(1,dx,dy);
		  }
		  else // not viscous
		  {
			   alen = radius + neigP[i]->radius;
		  }

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
              //cout << " zero divide in Relax " << endl;
              uxi = 0.0;
              uyi = 0.0;
            }

          //---------------------------------------------------------------
          // determine the force on particle = constant times strain
          //---------------------------------------------------------------

          fn = springf[i] * (dd - alen);    // normal force
          fx = fn * uxi;                    // force x
          fy = fn * uyi;                    // force y

          //----------------------------------------------------------------
          // add forces in xx and yy and springconstants
          //----------------------------------------------------------------

          xx = xx + fx;
          yy = yy + fy;
          anc = anc + springf[i];

          neigP[i]->done = true;   // this one was done
        }
    }

  //-----------------------------------------------------------------------
  // now do the Repulsion box, try all next neighbours,
  // if they are already done (i.e. had connection) and if they apply
  // a repulsion on particle
  //-----------------------------------------------------------------------

  if (merk > 0)
    {
      for(i=0; i<(merk); i++)
        {

          //----------------------------------------------
          // get distance to neighbour
          //---------------------------------------------

          neig = neigh_list[i];

          if (neig)
            {
              dx = neig->xpos - xpos;
              dy = neig->ypos - ypos;

              dd = sqrt((dx*dx)+(dy*dy));

              //------------------------------------------------
              // get equilibrium length
              //------------------------------------------------

				if (visc_rel == 1)
				{
				  alen = Alen(1,dx,dy) + Alen(0,dx,dy);
				}
				else
					alen = radius + neig->radius;

              //----------------------------------------
              // determine the unitlengths in x and y
              //----------------------------------------

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

              //-----------------------------------------------------------
              // determine the force on particle = constant times strain
              //-----------------------------------------------------------

              rep_constant = (young + neig->young)/2.0;

              if (young == 0.0)
                rep_constant = 0.0;
              if (neig->young == 0.0)
                rep_constant = 0.0;

              // and rescale to spring constant

              rep_constant = rep_constant * sqrt(3.0)/2.0;

              fn = rep_constant * (dd - alen);    // normal force

              if (fn < 0.0)              // only compressive forces
                {
                  fx = fn * uxi;                    // force x
                  fy = fn * uyi;                    // force y

                  //------------------------------------------------------------
                  // add forces in xx and yy and springconstants
                  // repulsion constant 1.0 at moment
                  //------------------------------------------------------------

                  xx = xx + fx;
                  yy = yy + fy;
                  anc = anc + rep_constant;
                }
            }
        }
    }

	// now handle possible forces from sidewalls
	
  if (wall)
    {
      fy = - wallconstant * ((ypos + radius) - upperwall);

      if (fy < 0.0) 
        {
          yy = yy + fy;
          anc = anc + wallconstant;
        }

      fy = - wallconstant * (lowerwall - (ypos - radius));

      if (fy < 0.0)
        {
          yy = yy - fy;
          anc = anc + wallconstant;
        }

      fx = - wallconstant * (leftwall - (xpos - radius));

      if (fx < 0.0)
        {
          xx = xx - fx;
          anc = anc + wallconstant;
        }

      fx = wallconstant * (rightwall- xpos + radius);

      if (fx < 0.0)
        {
          xx = xx + fx;
          anc = anc + wallconstant;
        }
    }

  //-------------------------------------------------------
  //		Add Fluid Pressure if it is there
  //-------------------------------------------------------

  if (fluid_P < 0.0 )
    {
	  i = 0;
		for (j = 0; j < 8; j++)
		{
			if (neigP[j])
			{
				if(!neigP[j]->isHole)
					i = i+1;
			 }
		  }
		  if (i > 1 && rightNeighbour && leftNeighbour)
			{
				  dd = ((leftNeighbour->xpos-rightNeighbour->xpos)*(leftNeighbour->xpos-rightNeighbour->xpos));
				  dd= sqrt(dd+((leftNeighbour->ypos-rightNeighbour->ypos)*(leftNeighbour->ypos-rightNeighbour->ypos)));
				  xx = xx + fluid_P * radius * 2.0 * ((leftNeighbour->ypos - rightNeighbour->ypos)/dd);
				  yy = yy + fluid_P * radius * 2.0 * - ((leftNeighbour->xpos - rightNeighbour->xpos)/dd);
  
				anc = anc + 1.0;
			}
	   }

  //---------------------------------------------------------
  // set all particle flags back to not done
  //---------------------------------------------------------

  done = false;   // myself

  for (i=0;i<8;i++)    // and all my neighbours
    {
      if (neigP[i])
        {
          neigP[i]->done = false;
        }
    }

  //-----------------------------------------------------------------
  // now devide forces again by springconstants (anc)
  // to get movement of particle
  //-----------------------------------------------------------------

  if (anc != 0.0)
    {
      anc = 1.0/anc;
      xx = xx * anc;
      yy = yy * anc;
    }
  else
    {
      xx = 0.0;
      yy = 0.0;
    }

  if (fix_x)
    xx = 0.0;
  if (fix_y)
    yy = 0.0;

  movement = sqrt(xx*xx + yy*yy);

  // a WORKING slow-down mechanism)
  if (movement > 0.5 * radius)
  {
	  fac = sqrt(0.25*radius*radius)/movement;

	  xx *= fac;
	  yy *= fac;

	  movement = sqrt(xx*xx + yy*yy);
  }
	  
  //---------------------------------------------------------
  // now return flag if movement larger than relaxthreshold
  //---------------------------------------------------------

  //new boundary code
  if (relaxthresh < ((xx*xx)+(yy*yy)))
    {
     
        xpos = xpos + xx * fover;    // overrelax
        ypos = ypos + yy * fover;    // overrelax

        return false;
    }
  else
    {
      return true;
    }

}


// RELAX (particle **, int)
/*****************************************************
 * Routine that calculates stresses and which 
 * bonds will break
 * gets Repulsion box from lattice plus x size
 * 
 *
 * Daniel spring and summer 2002
 *
 * Daniel and Jochen scale bug fixed
 * Feb. 2003
 *
 * added active repulsion constant calculated from 
 * the youngs modulus of particles
 *
 * daniel March 2003
 ****************************************************/

//-----------------------------------------------------------
// function Relax (bool) in particle class
//
//
//  gets RepulsionBox and x size
//
//
// called form Lattice::FullRelax()
//------------------------------------------------------------


bool
Particle::Relax(Particle **list,int size, int visc_rel,bool wall,float rightwall,float leftwall,float lowerwall,float upperwall, float wallconstant)
{
  //------------------------------------------------
  // local variables
  //------------------------------------------------

  int   i,j,k;           			// counter
  float dx,dy;           			// x and y differences
  float dd,alen;         			// distances between particles
  float rep_constant;    			// repulsion constant
  float fn,fx,fy;        			// forces
  float uxi,uyi;         			// unitdistance
  float anc;       					// move xx and yy
  float area;            			// area of particle
  float ten_break;       			// help for breaking strenght
  int   box,pos;         			// help for Repulsion
  float rad_factor1, rad_factor2, rad0, rad1; // rad0 for radius in given direction for THIS particle, rad1 for NEIG particle
  float xdiff, ydiff, phi;
  float angle;
  int   add;


  bool  bbreak = false;

  //----------------------------------------------------
  // set some to Zero
  //----------------------------------------------------

  xx = 0.0;
  yy = 0.0;
  anc = 0.0;
  sxx = 0.0;
  syy = 0.0;
  sxy = 0.0;

  mbreak = 0.0;

  done = true;

  //-----------------------------------------------------------
  // loop through all connected neighbours of particle
  //-----------------------------------------------------------

  for(i = 0; i < 8; i++)    // try all
    {
      if (neigP[i])      // if neighbour is connected
        {
          //--------------------------------------------------------------
          // get distance to neighbour
          //--------------------------------------------------------------

          dx = neigP[i]->xpos - xpos;
          dy = neigP[i]->ypos - ypos;

          dd = sqrt((dx*dx)+(dy*dy));

          //--------------------------------------------------------------
          // get equilibrium length
          //--------------------------------------------------------------

 
		  
		    if (visc_rel==1)  // with viscosity 
		  {
			  neig = neigP[i];
			  rad[i] = Alen(0,dx,dy);
			  alen = rad[i] + Alen(1,dx,dy);
		  }
		  else  // just elastic 
		  {
			 
			   alen = radius + neigP[i]->radius;
		  }


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

          fn = springf[i] * (dd - alen);    // normal force
          fx = fn * uxi;                    // force x
          fy = fn * uyi;                    // force y

          //---------------------------------------------------------------
          // check breaking strength
          //---------------------------------------------------------------


          fn = fn * alen;    // scaling to particle size

          if (fn < 0.0)
            fn = 0.0;     // only want tensile force

          ten_break = break_Str[i]*alen*alen;  // a force

          if (ten_break > 0.0)
            {
              ten_break = fn / ten_break; // ratio of forces
            }
          else
            {
              ten_break = 0.0;
            }

          if (ten_break > 1.0) // spring breaks
            {
              if (ten_break > mbreak)  // not used at moment....
                {
                  if (!no_break[i] /*|| !neigP[i]->no_break[neig_spring[i]]*/)  // dont break particle walls
                    {
                      mbreak = ten_break;      // define max breaking prob
                      neigh = i;        // and this is the neighbour
                      bbreak = true;
                      if (is_lattice_boundary || neigP[i]->is_lattice_boundary)
                        cout << "debug" << endl;
                    }
                }
            }

          //----------------------------------------------------------------
          // add forces in xx and yy and springconstants
          //----------------------------------------------------------------

          xx = xx + fx;
          yy = yy + fy;
          anc = anc + springf[i];

          //----------------------------------------------------------------
          // add stresses in stress tensor * uxi in order to get no
          // direction (forces cancel out, not stress)
          //----------------------------------------------------------------

          sxx = sxx + rad[i]*uxi*fx;
          syy = syy + rad[i]*uyi*fy;
          sxy = sxy + rad[i]*uyi*fx;

          neigP[i]->done = true;
        }
    }

  if (merk > 0)
    {
      for(i=0; i<merk; i++)
        {

          //----------------------------------------------
          // get distance to neighbour
          //---------------------------------------------

          neig = neigh_list[i];

          if (neig)
            {
              dx = neig->xpos - xpos;
              dy = neig->ypos - ypos;

              dd = sqrt((dx*dx)+(dy*dy));

              //------------------------------------------------
              // get equilibrium length
              //------------------------------------------------

				
				if (visc_rel == 1)
				{
				  alen = Alen(1,dx,dy) + Alen(0,dx,dy);
				}
				else
					alen = radius + neig->radius;

              //----------------------------------------
              // determine the unitlengths in x and y
              //----------------------------------------

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

              //-----------------------------------------------------------
              // determine the force on particle = constant times strain
              //-----------------------------------------------------------

              rep_constant = (young + neig->young)/2.0;

              if (young == 0.0)
                rep_constant = 0.0;
              if (neig->young == 0.0)
                rep_constant = 0.0;

              // and rescale to spring constant

              rep_constant = rep_constant * sqrt(3.0)/2.0;

              fn = rep_constant * (dd - alen);    // normal force

              if (fn < 0.0)              // only compressive forces
                {
                  fx = fn * uxi;                    // force x
                  fy = fn * uyi;                    // force y

                  //------------------------------------------------------------
                  // add forces in xx and yy and springconstants
                  // repulsion constant 1.0 at moment
                  //------------------------------------------------------------

                  xx = xx + fx;
                  yy = yy + fy;
					sxx = sxx + fx*radius*uxi;
					syy = syy + fy*radius*uyi;
					sxy = sxy + fx * radius * uyi;
                  anc = anc + rep_constant;
                }
            }
        }
    }
	// now handle possible forces from sidewalls
	
  if (wall)
    {
      fy = - wallconstant * ((ypos + radius) - upperwall);

      if (fy < 0.0) 
        {
          yy = yy + fy;
			syy = syy + fy*radius;
          anc = anc + wallconstant;
        }

      fy = - wallconstant * (lowerwall - (ypos - radius));

      if (fy < 0.0)
        {
          yy = yy - fy;
			syy = syy + fy*radius;
          anc = anc + wallconstant;
        }

      fx = - wallconstant * (leftwall - (xpos - radius));

      if (fx < 0.0)
        {
          xx = xx - fx;
			sxx = sxx + fx*radius;
          anc = anc + wallconstant;
        }

      fx = wallconstant * (rightwall- xpos + radius);

      if (fx < 0.0)
        {
          xx = xx + fx;
			sxx = sxx + fx*radius;
          anc = anc + wallconstant;
        }
    }
  //-------------------------------------------------------
  //		Add Fluid Pressure if it is there
  //-------------------------------------------------------

  /*****************************************
   * this hasnt been adapted yet for viscous stuff!!!
   * *****************************************/
     if (fluid_P < 0.0)
      {
		i = 0;
		for (j = 0; j < 8; j++)
		{
			if (neigP[j])
			{
				if(!neigP[j]->isHole)
				i = i+1;
			}
		}
		if (i > 1 && rightNeighbour && leftNeighbour)
		{
  
			dd = ((leftNeighbour->xpos-rightNeighbour->xpos)*(leftNeighbour->xpos-rightNeighbour->xpos));
			dd= sqrt(dd+((leftNeighbour->ypos-rightNeighbour->ypos)*(leftNeighbour->ypos-rightNeighbour->ypos))); 
			fx = (fluid_P * radius*radius*2.0*sqrt((leftNeighbour->ypos-rightNeighbour->ypos)*(leftNeighbour->ypos-rightNeighbour->ypos))/dd);
			fy = (fluid_P * radius*radius*2.0*sqrt((leftNeighbour->xpos-rightNeighbour->xpos)*(leftNeighbour->xpos-rightNeighbour->xpos))/dd); 
			sxx = sxx + fx;
			syy = syy + fy;
		}
      }

  //------------------------------------------
  // reset all the flags
  //------------------------------------------

  done = false;

  for (i=0;i<8;i++)
    {
      if (neigP[i])
        {
          neigP[i]->done = false;
        }
    }

  //----------------------------------------------------------------
  // rescale stress by area of particle
  //----------------------------------------------------------------

  //area wird in viscous routine constant gehalten, daher kann sie weiterhin
  //so benutzt werden.
  area = radius*radius*3.1415926;
  sxx = sxx/area;
  syy = syy/area;
  sxy = sxy/area;


  return bbreak;

}

// CHANGEBOX (particle **, int)
/**********************************************************************
 * routine that changes the box for a particle if it moved out of 
 * its box during the relaxation routine or a deformation 
 * First have to remove particle from box depending on where it is 
 * in the box list and reconnect the box list of remaining particles 
 * and then we have to add the particle to the end of the list 
 * in the new box. 
 *
 * Daniel spring 2002
 **********************************************************************/

//--------------------------------------------------------------------------
// function ChangeBox in particle class
//
// receives box list from Lattice and the x size.
//
// called from relaxation routines in Lattice and from deformation routines
//--------------------------------------------------------------------------


void Particle::ChangeBox(Particle **list,int size)
{
  int x,y;   // some variables

  //------------------------------------------------
  // again find current position of particle in box
  // is a double check
  //------------------------------------------------

  x = int(xpos*size);
  y = int(ypos*size);
  x = (y*size*2) + x;

  if (x!=box_pos)  // if current position not the same as saved position
    {

      //--------------------------------------------------------------
      //  first remove particle from old position
      //--------------------------------------------------------------


      if (list[box_pos])
        {
          neig = neig2 = list[box_pos];

          if(neig->nb == nb)  // if we are the first
            {
              list[box_pos] = neig->next_inBox; // the next is the first
            }
          else   // if not loop
            {
              while(neig!=this)  // loop until particle is found
                {
                  if(neig->next_inBox)  // if not the particle is not there in box
                    {
                      neig2 = neig;   // have a pointer to the last particle
                      neig = neig->next_inBox;  // set pointer to the next
                    }
                  else
                    {
                      cout << "Problem in RepBox, changeBox  " <<  nb  <<  endl;
                      break;
                    }
                }
              //-------------------------------------------------------
              // in here means we found the particle. Now the previous
              // particle is made to point to the next one that is
              // we just take our particle out of the list
              //-------------------------------------------------------

              neig2->next_inBox = neig->next_inBox;
            }
          //--------------------------------------------------------------------
          // next inBox is empty because we put particle to the end of the list
          //--------------------------------------------------------------------

          next_inBox = NULL;
        }
      else
        {
          cout << "problem in repBox, ChangeBox  "<<  nb  <<  endl;
        }

      //----------------------------------------------------------------
      //  put  in  new  position  at  end  of  list
      //  neig  now  points  at  me =  at this particle
      //----------------------------------------------------------------

      // *** 8.1.04: Line 843: Verbesserung eines bugs aus alter mike-version: ***

      box_pos = x;

      if (!list[box_pos])
        {
          list[box_pos] = neig; // in case the box is empty
        }
      else
        {
          neig2 = list[box_pos];  // for the last particle

          while(neig2->next_inBox)  // go to end of list
            {
              neig2 = neig2->next_inBox;
            }

          neig2->next_inBox = neig;  // and add particle
        }
    }
}


// SET_SPRINGS
/***********************************************************
 * function that gives the springs a constant
 * and a breaking strength
 * checks which neighbours are there. 
 * gives just one constant at the moment, 
 * should be overwritten later to make grain boundaries
 *
 * Daniel spring 2002
 * fixed breaking strength bug
 * Daniel and Jochen Feb. 2003
 *********************************************************/

void Particle::SetSprings()
{
  int i;

  for (i = 0;i < 8; i++)
    {

      if (neigP[i])      // if the neighbour is there
        {
          springf[i] = 1.0;    // set spring to 1.0
          springv[i] = 1.0;	// viscosity
          break_Str[i] = 0.0125;  // breaking strength

          rad[i] = radius;	// equilibrium length
			
			spring_boundary[i] = false;

        }
    }

  for (i=0; i<9; i++)
    {
      rep_rad[i] = radius;
    }
}


// FIND_NEIGHBOUR_F
/*****************************************************
 * Simple Routine to catch a pointer to another 
 * particle object in the list of particles
 * that is neigDist from the current particle
 * in the list, returns the pointer
 *
 * Daniel spring 2002
 *****************************************************/

Particle *Particle::FindNeighbourF(int neigDist)
{
  neig = nextP;
  for (i=0;i<(neigDist-1);i++)
    {
      neig = neig->nextP;
    }
  return neig;
}



// FIND_NEIGHBOUR_B
/******************************************************
 * same as above but runs backwards in list
 * returns pointer
 *
 * Daniel spring 2002
 *****************************************************/

Particle *Particle::FindNeighbourB(int neigDist)
{
  neig = prevP;
  for (i=0;i<(neigDist-1);i++)
    {
      neig = neig->prevP;
    }
  return neig;
}




// CONNECT
/******************************************************
 * This Routine connects all the particles with their
 * Neighbours. Each particle receives pointers to the
 * neighbours. 
 * function receives type of lattice and number of 
 * particles in x and y direction from lattice 
 * function is called from lattice class
 * from the MakeLattice functiion of the lattice object
 * which builds the complete lattice
 *
 * Daniel spring 2002
 *****************************************************/

void Particle::Connect(int lType,int lParticlex,int lParticley)
{
  if (lType == 1) // if triagonal

    /***************************************************
     * now we connect all particles to each other
     * even and uneven rows have different neighbours
     * and all the boundary particles also !!
     ************************************************/

    {
      // left lower corner
      if (nb == 0)
        {
          neigP[0] = nextP;
          neigP[1] = FindNeighbourF(lParticlex);
        }
      // right lower corner
      else if ((nb+1) == lParticlex)
        {
          neigP[1] = FindNeighbourF(lParticlex);
          neigP[2] = FindNeighbourF(lParticlex-1);
          neigP[3] = prevP;
        }
      // left upper corner
      else if (nb == ((lParticlex*lParticley)-lParticlex))
        {
          // check if last row is even or not
          if (lParticley == (2*(lParticley/2)))
            {
              neigP[4] = FindNeighbourB(lParticlex);
              neigP[5] = FindNeighbourB(lParticlex-1);
              neigP[0] = nextP;
              cout << "even" << endl;
            }
          else //uneven
            {
              neigP[5] = FindNeighbourB(lParticlex);
              neigP[0] = nextP;
              cout << "uneven" << endl;
            }
        }
      // right upper corner
      else if ((nb+1) == (lParticlex*lParticley))
        {
          // check if last row is even or not
          if (lParticley == (2*(lParticley/2)))
            {
              neigP[3] = prevP;
              neigP[4] = FindNeighbourB(lParticlex);
            }
          else // uneven
            {
              neigP[3] = prevP;
              neigP[5] = FindNeighbourB(lParticlex);
              neigP[4] = FindNeighbourB(lParticlex+1);
            }
        }
      // rest of lower row
      else if ((nb+1) < lParticlex)
        {
          neigP[0] = nextP;
          neigP[1] = FindNeighbourF(lParticlex);
          neigP[2] = FindNeighbourF(lParticlex-1);
          neigP[3] = prevP;
        }
      // rest of upper row
      else if ((nb+1) > ((lParticlex*lParticley)-lParticlex))
        {
          if (lParticley == (2*(lParticley/2)))
            {
              neigP[3] = prevP;
              neigP[4] = FindNeighbourB(lParticlex);
              neigP[5] = FindNeighbourB(lParticlex-1);
              neigP[0] = nextP;
            }
          else
            {
              neigP[3] = prevP;
              neigP[4] = FindNeighbourB(lParticlex+1);
              neigP[5] = FindNeighbourB(lParticlex);
              neigP[0] = nextP;
            }
        }
      // left boundary row
      else if ((nb+1) == (((nb/lParticlex)*lParticlex)+1))
        {
          if (xpos == 0.0 ) // rows 0,2,etc.
            {
              neigP[5] = FindNeighbourB(lParticlex);
              neigP[0] = nextP;
              neigP[1] = FindNeighbourF(lParticlex);
            }
          else // rows 3,5,etc.
            {
              neigP[4] = FindNeighbourB(lParticlex);
              neigP[5] = FindNeighbourB(lParticlex-1);
              neigP[0] = nextP;
              neigP[1] = FindNeighbourF(lParticlex+1);
              neigP[2] = FindNeighbourF(lParticlex);
            }
        }
      // right boundary row
      else if ((nb+1) == (((nb+1)/lParticlex)*lParticlex))
        {
          // rows 0,2 etc.
          if (((nb+1)/lParticlex) == ((((nb+1)/lParticlex)/2)*2))
            {
              neigP[2] = FindNeighbourF(lParticlex);
              neigP[3] = prevP;
              neigP[4] = FindNeighbourB(lParticlex);
            }
          else // rows 1,3 etc.
            {
              neigP[1] = FindNeighbourF(lParticlex);
              neigP[2] = FindNeighbourF(lParticlex-1);
              neigP[3] = prevP;
              neigP[4] = FindNeighbourB(lParticlex+1);
              neigP[5] = FindNeighbourB(lParticlex);
            }
        }
      // the rest
      else
        {
          // first even numbers (row 1,3,5 etc)
          // since now even numbers are uneven rows !
          if (((nb+1)/lParticlex) == (2*(((nb+1)/lParticlex)/2)))
            {
              neigP[0] = nextP;
              neigP[1] = FindNeighbourF(lParticlex);
              neigP[2] = FindNeighbourF(lParticlex-1);
              neigP[3] = prevP;
              neigP[4] = FindNeighbourB(lParticlex+1);
              neigP[5] = FindNeighbourB(lParticlex);
            }
          else // rows 2,4,etc.
            {
              neigP[0] = nextP;
              neigP[1] = FindNeighbourF(lParticlex+1);
              neigP[2] = FindNeighbourF(lParticlex);
              neigP[3] = prevP;
              neigP[4] = FindNeighbourB(lParticlex);
              neigP[5] = FindNeighbourB(lParticlex-1);
            }
        }
    }
}

// BREAKBOND()
/****************************************************************
 * routine that breaks bonds between particle
 * the routine breaks the bond of particle i and also the bond
 * of its connected neighbour (same bond just specified in both 
 * particles) 
 * 
 *
 * Daniel spring 2002
 *
 * Particle with broken bonds is now part of grain boundary, 
 * is_boundary flag is set true for both particles
 *
 * Daniel March 2003
 ***************************************************************/

//-----------------------------------------------------------------
// function BreakBond in particle class
//
//-----------------------------------------------------------------

float Particle::BreakBond()
{
  int i;  // counter
  int j;  // Reference for neighbour

  //-----------------------------------------------------------------
  // first break the neighbour
  // find neighbour and break
  //-----------------------------------------------------------------

  //cout << neigP[neigh]->nb << endl;

  for (i = 0; i < 8; i++)
    {
      if (neigP[neigh]->neigP[i])
        {
          if (neigP[neigh]->neigP[i]->nb == nb)
            {
              j = i; // thats the neighbour
            }
        }
    }

  neigP[neigh]->neigP[j] = 0;           // neighbour gone
  neigP[neigh]->draw_break = true;      // particle has a broken bond
  neigP[neigh]->springf[j] = 0.0;       // springconstant is zero
  neigP[neigh]->break_Str[j] = 0.0;     // breakingstrength is zero
  neigP[neigh]->break_Sp[j] = false;    // spring is broken
  neigP[neigh]->is_boundary = true;     // particle is now part of grain boundary

  //-----------------------------------------------------
  // now the particle itself
  //-----------------------------------------------------

  neigP[neigh] = 0;
  draw_break = true;
  springf[neigh] = 0.0;
  break_Str[neigh] = 0.0;
  break_Sp[neigh] = false;
  is_boundary = true;
  mbreak = 0.0;

  return 0.0;
}

 /*************************************************** 
 *
 * 	Visco elastic routine changes lengths of springs 
 *
 ****************************************************/

float Particle::Alen(int par, float dx, float dy)
{
  float m, angle, x, y, part1, part2;

  switch (par)
    {

    case 0:

      dx *= -1.0;
      dy *= -1.0;

      angle = atan(dy/dx);

      if (dx > 0)
        {
          angle = angle + 3.14159265;
        }
      else if (dx < 0 && dy > 0)
        {
          angle = angle + 2*3.14159265;
        }

      if (dx == 0.0)
        {
          if (dy < 0.0)
            {
              angle = 3.14159265/2.0;
            }
          else
            {
              angle = 3.14159265*1.5;
            }
        }

      angle -= fixangle;

      if (angle < 0.0)
        angle += 2.0*3.14159265;
      else if (angle > 2.0*3.14159265)
        angle -= 2.0*3.14159265;

      if (angle > 3.14159265)
        angle -= 3.14159265;

      m = tan(angle);

      part1 = (a*b)/(b*b+a*a*m*m);
      part2 = sqrt(a*a*m*m+b*b);
      x = fabs(part1*part2);

      part1 = (a*b*m)/(b*b+a*a*m*m);
      part2 = sqrt(a*a*m*m + b*b);
      y = fabs(part1*part2);

      break;

    case 1:

      angle = atan(dy/dx);

      if (dx > 0)
        {
          angle = angle + 3.14159265;
        }
      else if (dx < 0 && dy > 0)
        {
          angle = angle + 2*3.14159265;
        }

      if (dx == 0.0)
        {
          if (dy < 0.0)
            {
              angle = 3.14159265/2.0;
            }
          else
            {
              angle = 3.14159265*1.5;
            }
        }

      angle -= neig->fixangle;

      if (angle < 0.0)
        angle += 2.0*3.14159265;
      else if (angle > 2.0*3.14159265)
        angle -= 2.0*3.14159265;

      if (angle > 3.14159265)
        angle -= 3.14159265;

      m = tan(angle);

      part1 = (neig->a*neig->b)/(neig->b*neig->b+neig->a*neig->a*m*m);
      part2 = sqrt(neig->a*neig->a*m*m+neig->b*neig->b);
      x = fabs(part1*part2);

      part1 = (neig->a*neig->b*m)/(neig->b*neig->b+neig->a*neig->a*m*m);
      part2 = sqrt(neig->a*neig->a*m*m + neig->b*neig->b);
      y = fabs(part1*part2);

      break;
    }

  return(sqrt(x*x + y*y));
}

