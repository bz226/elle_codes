 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
/*!
	\file		voronoi.h
	\brief		header for voronoi points
	\par		Description:
                Class for voronoi point data
*/
#ifndef E_voronoi_h
#define E_voronoi_h
/*************************************************************
 *	INCLUDE FILES
 */
#include "attrib.h"
/*************************************************************
 *	CONSTANT DEFINITIONS
 */
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
class VoronoiPt {
private:
	Coords _position;
	bool _isBnode;
public:
    VoronoiPt(): _isBnode(false) { _position.x=0.0; _position.y=0.0; }
    VoronoiPt(Coords *xy): _isBnode(false) { _position.x=xy->x;
											 _position.y=xy->y;  }
    VoronoiPt(Coords *xy,bool bnodeval): _isBnode(bnodeval) { _position.x=xy->x;
											 _position.y=xy->y; } 
	void getPosition(Coords *xy) {
								xy->x = _position.x;
								xy->y = _position.y;
								}
	void setPosition(Coords *xy) {
								_position.x = xy->x;
								_position.y = xy->y;
								}
	bool isBnode() { return(_isBnode); }
	void markAsBnode() { _isBnode=true; }
	void markNotBnode() { _isBnode=false; }
};
/*************************************************************
 *	EXTERNAL DATA DECLARATIONS
 */
/*************************************************************
 *	EXTERNAL FUNCTION PROTOTYPES
 */
#endif	// E_voronoi_h
