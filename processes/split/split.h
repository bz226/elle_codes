
 /*****************************************************
 * Copyright: (c) 2006 L. A. Evans, M.Jessell
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA
 ******************************************************/
/*!
	\file		split.h
	\brief		header for split process
	\par		Description:
                indices for User data values for this process
                udata[SMode] controls which split calculation to use 
						0=Piazolo et al 
						1=Jessell et al
*/
#if !defined(E_split_h)
#define E_split_h
/*************************************************************
 *	INCLUDE FILES
 */
/*************************************************************
 *	CONSTANT DEFINITIONS
 */
const int SMode=0;   
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
#endif	// E_split_h
