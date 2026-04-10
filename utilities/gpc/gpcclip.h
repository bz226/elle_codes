#ifndef _E_gpcclip_h
#define _E_gpcclip_h
#include "gpc.h"

int gpcclip(std::vector<Coords> &pv2, std::vector<Coords> &pv1,
				std::vector<std::vector<Coords> > &pclip , gpc_op op);

#endif
