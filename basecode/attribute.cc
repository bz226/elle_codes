 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <iostream>
#include <algorithm>
#include "attribute.h"
#include "file.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

const int NUM_IFA=8;
static int IntFlynnAttrib[]={ COLOUR,EXPAND,MINERAL,GRAIN,SPLIT,
                              F_ATTRIB_I, F_ATTRIB_J, F_ATTRIB_K };
const int NUM_RFA=30;
static int RealFlynnAttrib[]={ ENERGY,VISCOSITY,S_EXPONENT,
                       INCR_S, BULK_S, E_XX, E_XY, E_YX, E_YY, E_ZZ,
                       F_INCR_S, F_BULK_S, TAU_XX, TAU_YY, TAU_ZZ, TAU_XY,
                       TAU_1, PRESSURE, E3_ALPHA,E3_BETA,E3_GAMMA,
                     AGE,CYCLE,DISLOCDEN,F_ATTRIB_A,F_ATTRIB_B,F_ATTRIB_C,
                     F_ATTRIB_D,F_ATTRIB_E,F_ATTRIB_F };

bool Attribute::isIntAttribute()
{
    return(std::find(IntFlynnAttrib,IntFlynnAttrib+NUM_IFA,getType()) !=
                  IntFlynnAttrib+NUM_IFA);
}

bool Attribute::isRealAttribute()
{
    return(std::find(RealFlynnAttrib,RealFlynnAttrib+NUM_RFA,getType()) !=
                  RealFlynnAttrib+NUM_RFA);
}

bool Attribute::isFlynnAttribute()
{
    return(isIntAttribute() || isRealAttribute());
}

//Write object to an ostream reference
std::ostream & operator<< (std::ostream &os, const Attribute &t)
{
		std::cout << t.id << ", "<< t.value << '\n';
    return os;
}

