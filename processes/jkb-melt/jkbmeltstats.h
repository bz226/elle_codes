/***************************************************************************
                          meltstats.h  -  description
                             -------------------
    copyright            : (C) 2003 by J.K. Becker
    email                : becker@jkbecker.de
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
#ifndef _E_meltsats_h
#define _E_meltstats_h


class stats{
public:
/*!Initializes the melt statistics (writes out the initial file)*/
	void initmeltstats(void);
    /*!Saves out statistics in file */
	void SaveMeltStats(int mineral,int Count);
private:
	FILE *statsout;
	void CalcAreaMineral(int mineral,int iflynn);
	void getFlynnCircumference(int mineral,int Flynn);
	void calculateAngle(int mineral, int Flynn);
};
#endif
