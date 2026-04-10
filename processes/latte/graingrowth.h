
#ifndef _E_graingrowth_h
#define _E_graingrowth_h

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "attrib.h"
#include "nodes.h"
#include "file.h"
#include "display.h"
#include "check.h"
#include "error.h"
#include "runopts.h"
#include "init.h"
#include "general.h"
#include "stats.h"
#include "update.h"


class GrainGrowth
	{
	public:
		
		double TotalTime;
		
	GrainGrowth();                           // Constructor
	~GrainGrowth()
	{}
	;                        // Destructor

	int DoGrowth(int step);
	int MoveDoubleJ(int node1);
	int MoveTripleJ(int node1);
	void GetRay(int node1,int node2,int node3,double *ray,Coords *movedist);
	void CheckAngles();
		int IncreaseAngle(Coords *xy,Coords *xy1,Coords *xy2,Coords *diff);
		
	};

#endif
