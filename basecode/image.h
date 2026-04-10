 /*****************************************************
 * Copyright: (c) 2010 L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 * Elle Project Software
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
	\file		image.h
	\brief		header for image reading utility (ppm/pnm)
	\par		Description:
                Defines valid strings for the comment section of ppm
                file
*/
#if !defined(E_image_h)
#define E_image_h
/*************************************************************
 *	INCLUDE FILES
 */
#include <vector>
#include "attrib.h"
#include "string_utils.h"
/*************************************************************
 *	CONSTANT DEFINITIONS
 */
/*************************************************************
 *	MACRO DEFINITIONS
 */
#define PPM_CAXIS 10
#define PPM_DIM 11
/************************************************************
 *	ENUMERATED DATA TYPES
 */
static valid_terms ppm_option_terms[] = {
                         { "caxis", PPM_CAXIS },
                         { "dimension", PPM_DIM },
                           NULL
                          };
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
int ElleReadImage(char *fname, int ***image, int *rows, int *cols,
                  int decimate,
                  std::vector<int> &rgbcols, std::vector<Coords_3D> &orient,
                  double *dim);
#endif	// E_image_h
