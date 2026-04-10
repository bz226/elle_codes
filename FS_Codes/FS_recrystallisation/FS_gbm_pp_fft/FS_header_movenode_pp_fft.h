#ifndef movenode_pp_fft_h
#define movenode_pp_fft_h

#include <ctime> // only for log-ellefile output if nan error occurs
#include <vector>
#include "convert.h"
#include "log.h"
#include "triattrib.h"

#include "FS_header_gbm_pp_fft.elle.h"

//extern runtime_opts Settings_run;

using namespace std;

/*
 * Function declaration, indentation indicates dependencies and which function 
 * calls which etc.
 */
int GetMoveDir( int node, Coords pvec, Coords * movedir,double trialdist, int *iNodeTooFast2, double *dTimeStepHighest2  );
    
    int MoveTNode( int node, Coords pvec, Coords * movedir, int *iNodeTooFast2, double *dTimeStepHighest2 );
    int MoveDNode( int node, Coords pvec, Coords * movedir, int *iNodeTooFast2, double *dTimeStepHighest2 );
        double GetMisorientation(int rgn1, int rgn2, int id);
            int ElleGetFlynnEulerCAxis(double alpha, double beta, double gamma, Coords_3D *dircos);
        double GetBoundaryMobility( int node, int nb );
        double Get2BoundaryMobility(double misorientation); // Calculates boundary mobility as a function of misorientation (see Holm et al., 2003 etc.)
        double GetVectorLength( Coords vec );
        double DEGCos( Coords vec1, Coords vec2 );
        int SearchUnode (int rgn, Coords *xy);
            // double FS_GetRoi(int iFact);
            
// FS to colour bnodes for velocity
void FS_SetBnodeVelocityAttrib(int node,Coords cVelocity);

// Other, unused at the moment
int FS_DEBUG_FUNCTION_BoundaryType(int iNode);

#endif
