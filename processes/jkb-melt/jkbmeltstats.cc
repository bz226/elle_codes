/***************************************************************************
                          meltstats.cpp
This file is (an inofficial) part of the elle_melt package. It calculates
the area, circumference, angle at triple points and the curvature of
boundaires of melt pockets (not yet). There is a problem calculating the angles:
If the switch distance is very small, the angles are not correct, the distances
between the points is so small that rounding errors occur.
Also, up to now all the calculated values are in Elle-coordinates, NOT
real-world coordinates!

Make sure you have the updated version of melt.main.cc and elle.melt.cc
including all the header files!


    copyright            : (C) 2003 by J.K. Becker
    email                : becker@jkbecker.de


This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
***************************************************************************/
#include "interface.h"
#include "jkbmeltstats.h"
#include "nodes.h"
#include <stdio.h>
#include <stdlib.h>
#include <file.h>
#include <math.h>


void stats::initmeltstats()
{
	statsout = fopen( "melt-statistics.csv", "w" );
	fprintf(statsout,"Run,Flynn,Area,Circumference,angles at triple points");
	fclose(statsout);
}
void stats::SaveMeltStats(int mineral,int Count)
{
	div_t rest=div(Count,100);
	if(rest.rem==0) //only writes stuff every tenth stage,change to what you like
	{
		statsout = fopen( "melt-statistics.csv", "a+" );
		//printf("\n%d",Count);       //make linefeed before printing current number of runs
		fprintf(statsout,"\n%d,",Count); //make linefeed and stuff in file too!
		CalcAreaMineral(mineral,8);
		fclose(statsout);
	}
}
/*!calculate melt area */
void stats::CalcAreaMineral(int mineral,int iflynn)
{
	double area;
	area=ElleFlynnArea(iflynn);   //get melt-area
	//printf("Flynn:%d-Area:%lf-",i,area);  //print area of melt pocket
	fprintf(statsout,"%d,%lf,",iflynn,area);  //print area of melt pockets in file
	getFlynnCircumference(mineral,iflynn);
	calculateAngle(mineral,iflynn);
}
/*!needs interface.h and nodes.h, iflynn is the flynn to use*/
void stats::getFlynnCircumference(int mineral, int iflynn)
{
	double dist,circumference=0.0;
	int n,num;
	int *farr;
	circumference=0.0;
	dist=0,0;
	ElleFlynnNodes(iflynn,&farr,&num); //see Elle manual, page 27
	//printf(" Nodes:%d-",num);
	for(n=0;n<num-1;n++)//cycle through all nodes of Flynn iflynn
	{
		if(n==num-1) //if n=last Flynn in array, calculate distance between last and first Node
		{
			dist=ElleNodeSeparation(farr[n],farr[0]);
		}
		else //if not calculate distance between this and next node
		{
			dist=ElleNodeSeparation(farr[n],farr[n+1]);
		}
		circumference=circumference+dist; //add distances
	}
	//printf("-Circumference: %lf\n",circumference);
	fprintf(statsout,"%lf,",circumference);
	circumference=0.0; //reset circumference for next Flynn
	free(farr);
}

//needs math.h, interface.h and nodes.h, iflynn is the flynn to use
void stats::calculateAngle(int mineral, int iflynn)
{
	int z,zn,j[3],tpnode[20],nbnode[3],check=0,memoryzn,counttpnodes=0;
	int *zarr;
	double a,b,c,cosgamma,angle;
	ElleFlynnNodes(iflynn,&zarr,&zn);
	memoryzn=zn;//I dunno why but somewhere along the line in the for-cycle zn always gets set to the value of j[2] ?!?!?!?!?!?!?
	//so this memoryzn is the workaround for that! Stupid computer!
	for(z=0;z<zn;z++) //cycle through all nodes
	{
		if(ElleNodeIsActive(zarr[z])) //only do stuff if node exists
		{
			if(ElleNodeIsTriple(zarr[z])) //is it a triple-point? 1=yes
			{
				//tpnode[counttpnodes]=zarr[z];//put all triplenodes in this array - ATTENTION: IF THERE ARE MORE THEN 21 TRIPLENODES THE PROGRAM WILL CRASH!
				//counttpnodes++; //that is for the calculation of the curvature between triplenodes
				nbnode[check]=zarr[z];
				check++;
				ElleNeighbourNodes(zarr[z],j);
				if(ElleNodeOnRgnBnd(j[0],iflynn))
				{
					nbnode[check]=j[0];
					check++;
				}
				if(ElleNodeOnRgnBnd(j[1],iflynn))
				{
					nbnode[check]=j[1];
					check++;
				}
				if(ElleNodeOnRgnBnd(j[2],iflynn))
				{
					nbnode[check]=j[2];
				}
				check=0;//reset checking variable for next run
				zn=memoryzn;
				c=ElleNodeSeparation(nbnode[1],nbnode[2]);//and these should be the lengths of the lines a, b and c
				b=ElleNodeSeparation(nbnode[0],nbnode[1]);
				a=ElleNodeSeparation(nbnode[0],nbnode[2]);
				//calculate the angle between points and print them out
				//cos(alpha)=(b^2+c^2-a^2)/2bc
				//cos(beta)=(a^2+c^2-b^2)/2ac
				//cos(gamma)=(a^2+b^2-c^2)/2ab
				//triple point should always be C in the triangle...
				cosgamma=((a*a)+(b*b)-(c*c))/(2*a*b);
				//printf("Cos(gamma)-%lf",cosgamma);
				angle=acos(cosgamma)*(180/3.1415926535);//now of course acos() is in radiants! 1degree=phi/180radiant--1radiant=180/phi degrees
				//printf("Node %d is triple, neighbouring nodes are %d and %d,",nbnode[0], nbnode[1],nbnode[2]);
				//printf("a=%f, b=%f, c=%f, Angle:%lf\n",a,b,c,angle);
				fprintf(statsout,"%d,%lf,",zarr[z],angle);
			}
		}
	}
	if (zarr) free(zarr);
	fprintf(statsout,",");
}
