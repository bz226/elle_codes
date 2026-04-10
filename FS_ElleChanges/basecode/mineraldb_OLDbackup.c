
/*
 *----------------------------------------------------
 * Copyright: (c) L. A. Evans, M. W. Jessell
 * File:      $RCSfile: mineraldb.c,v $
 * Revision:  $Revision: 1.6 $
 * Date:      $Date: 2014/06/17 05:29:50 $
 * Author:    $Author: levans $
 *
 *----------------------------------------------------
 */

static const char rcsid[] =
       "$Id: mineraldb.c,v 1.6 2014/06/17 05:29:50 levans Exp $";
/*
 *----------------------------------------------------
 */
#include <stdio.h>
#include <math.h>

#include "error.h"
#include "file.h"
#include "mineraldb.h"

/*void main()
{
    double val;

    val=GetMineralAttribute(QUARTZ,GB_MOBILITY);
    printf("quartz gb mobility is %e\n\n",val);

    val=GetMineralAttribute(MICA,RECOVERY_RATE);
    printf("mica recovery rate is %e\n\n",val);

 
    
}*/
/*
 * CRITICAL_MISORIENT is in degrees
 * garnet & mica have value for quartz - may need changing
 */
double GetMineralAttribute(int mineral, int attribute)
{
    double val = 0.0;

    switch(mineral) {
    case QUARTZ    :
        switch(attribute) {
        case GB_MOBILITY       :
                    val=1e-11;
                                     break;
        case RECOVERY_RATE     :
                    val=0.95;
                                     break;
        case CRITICAL_SG_ENERGY:
                    val=0.5*7e5;
                                     break;
        case CRITICAL_RX_ENERGY:
                    val=1*7e5;
                                     break;
        case VISCOSITY_BASE    :
                    val=1.0;
                                     break;
        case SURFACE_ENERGY    :
                    val=7e-2;
                                     break;
        case DD_ENERGY         :
                    val=7e-9;
                                     break;
        case GB_ACTIVATION_ENERGY:
                    val=53.24e3; //  J mol-1
                                     break;
        case MIN_FLYNN_AREA    :
                    val=1.25e-3*0.25;
                                     break;
        case MAX_FLYNN_AREA    :
                    val=1.25e-3;
                                     break;
        case CRITICAL_MISORIENT:
                    val=10;
                                     break;
        case BURGERS_VECTOR:
                    val=4.5e-10;
                                     break;
        case SYMMETRY:
                      val = (double)(HEXAGONAL);
                                     break;
        default                : OnError("GetMineralAttribute",0);
                                     break;
        }
        break;
    case FELDSPAR:
        switch(attribute) {
        case GB_MOBILITY       :
                    val=1e-14;
                                     break;
        case RECOVERY_RATE     :
                    val=0.98;
                                     break;
        case CRITICAL_SG_ENERGY:
                    val=8*7e5;
                                     break;
        case CRITICAL_RX_ENERGY:
                    val=10*7e5;
                                     break;
        case VISCOSITY_BASE    :
                    val=1.0;
                                     break;
        case SURFACE_ENERGY    :
                    val=7e-2;
                                     break;
        case DD_ENERGY         :
                    val=7e-9;
                                     break;
        case MIN_FLYNN_AREA    :
                    val=6.25e-3*0.25;
                                     break;
        case MAX_FLYNN_AREA    :
                    val=6.25e-3;
                                     break;
        case CRITICAL_MISORIENT:
                    val=5;
                                     break;
        case BURGERS_VECTOR:
                    val=4.5e-10;
                                     break;
        case SYMMETRY:
                      val = (double)(TRIGONAL);
                                     break;
        default                : OnError("GetMineralAttribute",0);
                                     break;
        }
        break;
    case GARNET    :
        switch(attribute) {
        case GB_MOBILITY       :
                    val=1e-14;
                                     break;
        case RECOVERY_RATE     :
                    val=0.999999;
                                     break;
        case CRITICAL_SG_ENERGY:
                    val=8*7e5;
                                     break;
        case CRITICAL_RX_ENERGY:
                    val=10*7e5;
                                     break;
        case VISCOSITY_BASE    :
                    val=1.0;
                                     break;
        case SURFACE_ENERGY    :
                    val=7e-2;
                                     break;
        case DD_ENERGY         :
                    val=7e-9;
                                     break;
        case MIN_FLYNN_AREA    :
                    val=6.25e-3*0.25;
                                     break;
        case MAX_FLYNN_AREA    :
                    val=6.25e-3;
                                     break;
        case CRITICAL_MISORIENT:
                    val=10;
                                     break;
        case BURGERS_VECTOR:
                    val=4.5e-10;
                                     break;
        case SYMMETRY:
                      val = (double)(CUBIC);
                                     break;
        default                : OnError("GetMineralAttribute",0);
                                     break;
        }
        break;
    case MICA    :
        switch(attribute) {
        case GB_MOBILITY       :
                    val=1e-14;
                                     break;
        case RECOVERY_RATE     :
                    val=0.9999;
                                     break;
        case CRITICAL_SG_ENERGY:
                    val=8*7e5;
                                     break;
        case CRITICAL_RX_ENERGY:
                    val=10*7e5;
                                     break;
        case VISCOSITY_BASE    :
                    val=1.0;
                                     break;
        case SURFACE_ENERGY    :
                    val=7e-2;
                                     break;
        case DD_ENERGY         :
                    val=7e-9;
                                     break;
        case MIN_FLYNN_AREA    :
                    val=6.25e-3*0.25;
                                     break;
        case MAX_FLYNN_AREA    :
                    val=6.25e-3;
                                     break;
        case CRITICAL_MISORIENT:
                    val=10;
                                     break;
        case BURGERS_VECTOR:
                    val=4.5e-10;
                                     break;
        case SYMMETRY:
                      val = (double)(MONOCLINIC);
                                     break;
        default                : OnError("GetMineralAttribute",ATTRIBID_ERR);
                                     break;
        }
        break;
    case ICE    :
        switch(attribute) {
        case GB_MOBILITY:
                    val=0.023;// m4/s·J or m2/Kg·s Nasello et al. 2005 Intrinsic Mobility Mo; 1st stage of groove movement-low velocity
                                     break;
        case GB_MOBILITY_2:
                    val=2.3; // m^4/(Js)// m4/s·J or m2/Kg·s Nasello et al. 2005 Intrinsic Mobility Mo; 2nd stage of groove movement- fast
                                     break;

        case DD_ENERGY:
                    //val=3.5e-10;// Core energy dislocations Gb**2/2, 3500 (MPa) *4.52e-10 (m)**2 /2; Jm-1
                    // FS: I like to use 3.6e-10, which is justified by Scholson & Duval book
                    val=3.6e-10;
                                     break;

        case SURFACE_ENERGY:
                    val=0.065; // Jm-2
                                     break;

        case GB_ACTIVATION_ENERGY:
                    val=51.1e3; //  J mol-1
                                     break;
        case CRITICAL_MISORIENT:
                    val=4;
                                     break;
        case BURGERS_VECTOR:
                    val=4.5e-10;
                                     break;
        case SYMMETRY:
                      val = (double)(HEXAGONAL);
                                     break;
        default                : OnError("GetMineralAttribute",ATTRIBID_ERR);
                                     break;
        }
        break;
    case MAGNESIUM    :
        switch(attribute) {
// !!! Has values for ICE except for burgers vector!!!!!
        case GB_MOBILITY:
                    val=0.023;
                                     break;
        case GB_MOBILITY_2:
                    val=2.3;
                                     break;

        case DD_ENERGY:
                    val=3.5e-10;
                                     break;

        case SURFACE_ENERGY:
                    val=0.065; // Jm-2
                                     break;

        case GB_ACTIVATION_ENERGY:
                    val=51.1e3; //  J mol-1
                                     break;
        case CRITICAL_MISORIENT:
                    val=10;
                                     break;
        case BURGERS_VECTOR:
                    val=2.79e-10;  //Wheeler,J.Microscopy,2009
                                     break;
        case SYMMETRY:
                      val = (double)(HEXAGONAL);
                                     break;
        default                : OnError("GetMineralAttribute",ATTRIBID_ERR);
                                     break;
        }
        break;
    case OLIVINE:
        switch(attribute) {
        case GB_MOBILITY:
                     val=3.98e-8;
/*Karato 1989, with grain growth exponent p=2; m0 == kp0=3.98e-8 m**2·s-1
--dry,olivine synthetic? fast (Karato 1989:Grain Growth Kinetics in Olivine
Aggregates, Tectonophysics, 168, 255-273) from Evans et al., 2001
Kametama et al. 1997 (source Braun et al. 1999) obtained 10-8, , p=2
wet peridotite synthetic values between 1.6e-8 and 5e-8 (Karato(1989) & Hirt & Kohlstedt, 1995), p=2*/
                                     break;
        case GB_MOBILITY_2:
                      val=1e-9;
 // 3e-9 to 1e-14, p=3 to 4; Faul and Scott (2006), (2007)
                                     break;
        case DD_ENERGY:
                    val=1e-9;
/* Core energy dislocations Gb**2/2, 6000 (MPa)[Kohlstedt & Weathers] *5.0e-10 (m)**2 /2; Jm-1
Twiss (1977), Karato and Wu (1993) shear modulus 8000 MPa */
                                     break;
        case SURFACE_ENERGY:
                     val=1.0; // Jm-2
/*  Duyster and Stockhert, 2001 -All people use the same value, Karato (); Platt & Behr used 1.4 from data of Duyster and Stockhert (2001) */
                                     break;

        case GB_ACTIVATION_ENERGY:
                      val=200e3; //  J mol-1 
/* Karato (1989) Olivine
Faul and Scott (2006) 390e3, Faul and Scott (2007) 400e3,
Farver and Yund (2000) 375e3
wet conditions synthetic peridotite, 160e3 Karato(1989), H&K (1995)
dry conditions synthetic peridotite, 600e3-520e3 Karato (1989)*/
                                     break;
        case CRITICAL_MISORIENT:
                    val=10;
                                     break;
        case BURGERS_VECTOR:
                    val=6.0e-10;  
                                     break;
        case SYMMETRY:
                      val = (double)(ORTHORHOMBIC);
                                     break;
            
        default                : OnError("GetMineralAttribute",ATTRIBID_ERR);
                                     break;
        }
        break;
    default     : OnError("GetMineralAttribute",0);
                  break;
    }
    return(val);

}

/*
 * Function ElleLoadMineralSymmetry
 * Utility function which loads the array "symm" with the symmetry
 * symmetry operators for the mineral and returns the number of
 * operators(symm_ops)
 */
/*!
        \brief          Load the symmetry array for the mineral

        \param          symm - symmetry matrix 24x3x3
                        mineral - value of MINERAL
                        symm_ops - number of symmetry entries (use symm_opsx3x3)

        \return         \a symm_ops - number of symmetry entries (0->24)

        \par            Description:
                        Calls GetMineralAttribute() to get the symmetry
                        Calls symmetry_set to load symm and determin symm_ops

        \exception
                        

        \par            Example:

\verbatim
                        int symm_ops, mineral, symmetry;
                        double symm[24][3][3];

                        if (ElleFlynnIsActive(i)) {
                            ElleGetFlynnMineral(i,&mineral);
                            symm_ops = ElleLoadMineralSymmetry(symm,mineral);
                        }
\endverbatim
*/
int ElleLoadMineralSymmetry(double symm[24][3][3], int mineral )
{
    int symm_ops=0, symmetry=0;

    symmetry = (int)GetMineralAttribute(mineral,SYMMETRY);
    symm_ops = symmetry_set(symm, (const int)symmetry);
    return(symm_ops);
}

const int Cubic_size = 24;
static double cubic_symm[][3][3] = {
{ {1,0,0 },
  {0,1,0 },
  {0,0,1 } },
{ {0,0,1 },
  {1,0,0 },
  {0,1,0 } },
{ {0,1,0 },
  {0,0,1 },
  {1,0,0 } },
{ {0,-1,0 },
  {0,0,1 },
  {-1,0,0 } },
{ {0,-1,0 },
  {0,0,-1 },
  {1,0,0 } },
{ {0,1,0 },
  {0,0,-1 },
  {-1,0,0 } },
{ {0,0,-1 },
  {1,0,0 },
  {0,-1,0 } },
{ {0,0,-1 },
  {-1,0,0 },
  {0,1,0 } },
{ {0,0,1 },
  {-1,0,0 },
  {0,-1,0 } },
{ {-1,0,0 },
  {0,1,0 },
  {0,0,-1 } },
{ {-1,0,0 },
  {0,-1,0 },
  {0,0,1 } },
{ {1,0,0 },
  {0,-1,0 },
  {0,0,-1 } },
{ {0,0,-1 },
  {0,-1,0 },
  {-1,0,0 } },
{ {0,0,1 },
  {0,-1,0 },
  {1,0,0 } },
{ {0,0,1 },
  {0,1,0 },
  {-1,0,0 } },
{ {0,0,-1 },
  {0,1,0 },
  {1,0,0 } },
{ {-1,0,0 },
  {0,0,-1 },
  {0,-1,0 } },
{ {1,0,0 },
  {0,0,-1 },
  {0,1,0 } },
{ {1,0,0 },
  {0,0,1 },
  {0,-1,0 } },
{ {-1,0,0 },
  {0,0,1 },
  {0,1,0 } },
{ {0,-1,0 },
  {-1,0,0 },
  {0,0,-1 } },
{ {0,1,0 },
  {-1,0,0 },
  {0,0,1 } },
{ {0,1,0 },
  {1,0,0 },
  {0,0,-1 } },
{ {0,-1,0 },
  {1,0,0 },
  {0,0,1 } }
};

const int Hexag_size=6;
// sqrt(3)/2=0.8660254038
double hexag_symm[][3][3] = {
{ {1,0,0},
  {0,1,0},
  {0,0,1} },
{ {-0.5,0.8660254038,0},
  {-0.8660254038,-0.5,0},
  {0,0,1} },
{ {-0.5,-0.8660254038,0},
  {0.8660254038,-0.5,0},
  {0,0,1} },
{ {0.5,0.8660254038,0},
  {-0.8660254038,0.5,0},
  {0,0,1} },
{ {-1,0,0},
  {0,-1,0},
  {0,0,1} },
{ {0.5,-0.8660254038,0},
  {0.8660254038,0.5,0},
  {0,0,1} }
};

const int Ortho_size = 4;
static double ortho_symm[][3][3] = {
{ {1,0,0 },
  {0,1,0 },
  {0,0,1 } },
{ {-1,0,0 },
  {0,1,0 },
  {0,0,-1 } },
{ {1,0,0 },
  {0,-1,0 },
  {0,0,-1 } },
{ {-1,0,0 },
  {0,-1,0 },
  {0,0,1 } }
};

const int Trigonal_size = 1;
static double trig_symm[][3][3] = {
{ {1,0,0 },
  {0,1,0 },
  {0,0,1 } }
};

int symmetry_set(double symm[24][3][3], const int symmetry)
{
    int n, i, j, symm_ops=1;

    // reset all values
    for(n=0;n<24;n++) {
        for(i=0;i<3;i++) {
            for (j=0;j<3;j++) {
                symm[n][i][j]=0.0;
            }
        }
    }

    switch(symmetry) {
        // store symmetry operators
    case CUBIC:
        symm_ops = Cubic_size;

        for(n=0;n<symm_ops;n++) {
          for (i=0;i<3;i++) {
            symm[n][i][0]=cubic_symm[n][i][0];
            symm[n][i][1]=cubic_symm[n][i][1];
            symm[n][i][2]=cubic_symm[n][i][2];
          }
        }
        break;
    case HEXAGONAL:
        symm_ops = Hexag_size;

        for(n=0;n<symm_ops;n++) {
          for (i=0;i<3;i++) {
            symm[n][i][0]=hexag_symm[n][i][0];
            symm[n][i][1]=hexag_symm[n][i][1];
            symm[n][i][2]=hexag_symm[n][i][2];
          }
        }
        break;
    case TRIGONAL:
        symm_ops = Trigonal_size;

        for(n=0;n<symm_ops;n++) {
          for (i=0;i<3;i++) {
            symm[n][i][0]=trig_symm[n][i][0];
            symm[n][i][1]=trig_symm[n][i][1];
            symm[n][i][2]=trig_symm[n][i][2];
          }
        }
        break;
#if XY
    case MONOCLINIC:
        symm_ops = Monoclinic_size;

        for(n=0;n<symm_ops;n++) {
          for (i=0;i<3;i++) {
            symm[n][i][0]=mono_symm[n][i][0];
            symm[n][i][1]=mono_symm[n][i][1];
            symm[n][i][2]=mono_symm[n][i][2];
          }
        }
        break;
#endif
    case ORTHORHOMBIC:
        symm_ops = Ortho_size;

        for(n=0;n<symm_ops;n++) {
          for (i=0;i<3;i++) {
            symm[n][i][0]=ortho_symm[n][i][0];
            symm[n][i][1]=ortho_symm[n][i][1];
            symm[n][i][2]=ortho_symm[n][i][2];
          }
        }
        break;
    default:
        break;
    }

    return(symm_ops);
}


int symmetry_load(double symm[24][3][3])
{

        int i,j,n,symm_op;
        FILE *f=fopen("symmetry.symm","r");
        double dum1,dum2,dum3;

    // reset all values
        for(n=0;n<24;n++) {
                for(i=0;i<3;i++) {
                        for (j=0;j<3;j++) {
                                symm[n][i][j]=0.0;
                        }
                }
        }

        // first line of with the information of number of symmetry operation
        fscanf(f,"%d",&symm_op);

        // store symmetry operators
     for(n=0;n<symm_op;n++) {

                for (i=0;i<3;i++) {
                        fscanf(f,"%lf %lf %lf",&dum1,&dum2,&dum3);
                        symm[n][i][0]=dum1;
                        symm[n][i][1]=dum2;
                        symm[n][i][2]=dum3;

                }
        }

        fclose(f);

return(symm_op);
}

#if XY
const int Mono_size = 1;
double monoclinic_symm[Mono_size][3][3] =
{ 1 0 0
  0 1 0
  0 0 1
}
ortho.symm

albite - monoclinic
horneblende - triclinic

#endif

