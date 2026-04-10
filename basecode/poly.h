 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_poly_h
#define _E_poly_h

#define ELLE_KEY "ELLE"
#define AREA_KEY "AREA"
#define QUALITY_KEY "QUALITY"
#define VISCOSITY_KEY "VISCOSITY"
#define SE_KEY "S_EXPONENT"
#define FLYNN_REAL_ATTRIB_KEY "F_ATTRIB_A"
#define UNODES_KEY "UNODES"
#define U_VISCOSITY_KEY "U_VISCOSITY"
#define DEFAULT "Default"

/*
 * arbitrary minimum area for triangulation
 * smaller values can cause basil to fail array allocation
 */
/* min area for flynn triangulation eg splitting */
#define AREA_MIN_F 0.0001
/* min area for basil triangulation eg elle2poly */
#define AREA_MIN 0.0002

#endif
