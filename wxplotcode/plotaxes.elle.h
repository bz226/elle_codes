/*****************************************************
 * Copyright: (c) 2009 L. A. Evans
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
    \file       plotaxes.elle.h
    \brief      header for stereonet option, plotaxes
    \par        Description: Creates Postscript file of
                stereonet projection of Euler angles
                and c-axis data for unodes
*/
#ifndef PLOTAXES_ELLE_H_
#define PLOTAXES_ELLE_H_
/*************************************************************
 *  INCLUDE FILES
 */
#include <stdio.h>
#include <math.h>
#include "error.h"
#include "file.h"
#include "init.h"
#include "runopts.h"
#include "interface.h"
#include "unodes.h"
#include "convert.h"
#include "mat.h"
#include "wx/wx.h"
/*************************************************************
 *  CONSTANT DEFINITIONS
 */
const int REVERSE_ORDER=0; //index in user data
const int SAMPLE_STEP=1; //index in user data
const int CAXIS_OUT=2; //index in user data

/*************************************************************
 *  FUNCTION PROTOTYPES
 */
int PlotAxes(wxString filename);
void change(double *axis, double axis2[3], double rmap[3][3]);
void firo(double *a, double *phi, double *rho);
void pstartps(FILE *);
void pendps(FILE *);
void plotonept(double *axis, double rmap[3][3], double *center, double radius,
                FILE *psout,FILE *polarout);
void splotps(double *center, double radius, double phi, double rho,
             FILE *psout, FILE *polarout);
void startsteronet(double *center, double radius, FILE *psout, char *title, int ngns);
int FindRowsCols(int *rows, int *numperrow);
void old_main(); // no longer used

#endif /*PLOTAXES_ELLE_H_*/
