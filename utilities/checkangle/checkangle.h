 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
/*!
	\file		checkangle.h
	\brief		header for checkangle utility
	\par		Description:
                Defines user data indices to control the checks
                done on an elle file by checkangle
*/
#ifndef _E_checkangle_h
#define _E_checkangle_h
/*************************************************************
 *	INCLUDE FILES
 */
/*************************************************************
 *	CONSTANT DEFINITIONS
 */
const int INDEXMINANGLE=0; // index into user data
const int INDEXMINAREAFACTOR=1;
const int INDEXMAXRATIO=2;
const double MINANGLE=0.16; // rad (9.167 deg)
const double MINAREAFACTOR=1; // factor to vary min flynn area
const double MAXRATIO=0;  // flynn ellipsoidal axes ratio 
                         // default ratio 0 turns off check
/*************************************************************
 *	MACRO DEFINITIONS
 */
/************************************************************
 *	ENUMERATED DATA TYPES
 */
/*************************************************************
 *	STRUCTURE DEFINITIONS
 */
/*************************************************************
 *	IN-LINE FUNCTION DEFINITIONS
 */
/*************************************************************
 *	CLASS DECLARATIONS
 */
/*************************************************************
 *	EXTERNAL DATA DECLARATIONS
 */
/*************************************************************
 *	EXTERNAL FUNCTION PROTOTYPES
 */
#endif	// _E_checkangle_h
