#ifndef gbm_pp_fft_elle_h
#define gbm_pp_fft_elle_h
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <list>
#include <iostream>
#include <fstream>
#include <sstream>
#include "attrib.h"
#include "nodes.h"
#include "unodes.h"
#include "file.h"
#include "display.h"
#include "check.h"
#include "error.h"
#include "runopts.h"
#include "init.h"
#include "general.h"
#include "stats.h"
#include "update.h"
#include "interface.h"
#include "crossings.h"
#include "mineraldb.h"
#include "polygon.h"
#include "gpcclip.h"
#include "lut.h"
#include "FS_header_movenode_pp_fft.h"

/*
#define PI 3.1415926
#define DTOR PI/180
#define RTOD 180/PI
*/

#define MAX_PHASES 2

using namespace std;

/*
 * Declaration of structures
 */
struct UnodeDist
{
	int iNode;
	double dDist;
	
	bool operator< (const UnodeDist& a) const
    {
        return dDist < a.dDist;
    }
};
 
typedef struct
{
	double mobility;
	double b_energy;
	double dGbActEn;
} PhaseBoundaryProps;

typedef struct
{
	int cluster_diff;
	int diffusion_times;
	double kappa;
	double stored;
	int disscale;
	double disbondscale;
} PhaseProps;

typedef struct
{
	int no_phases; // means Numner Of phases
	int first_phase;
	int p_en;
	int p_track;
	PhaseProps phasep[15];
	PhaseBoundaryProps pairs[120][120];
} AllPhases;

/*
 * Declaration of cluster tracking class
 */
class clusters
{
public:
	clusters( std::vector<int>, double );
	~clusters ();
	std::vector<int> ReturnClusterFlynns ( void );
	double  ReturnClusterArea ( void );
private:
	std::vector<int> vFlynns;
	double dArea;
};

class clusterTracking
{
public:
	clusterTracking();
	~clusterTracking();
	bool writeInitialData( const char* );
	bool writeData( const char*, int );
	void setClusterAreas( void );
	void findSplit( void );
	void findMerge( void );
	void findClusters( void );
	void updateClusters( void );
	void checkDoubleClusterAreaLoop( void );
	double returnClusterAreaEnergy ( int , Coords * );

private:
	int iMaxFlynns; // max Flynns. Has to by initialized with this value
	double dAreaShift, dMultiplierA, dMultiplierB, dMultiplierC, dMultiplierD;
	//std::vector<clusters> vClusters;
	std::vector<int> vFlynns, vFlynnPhase;
	std::vector<double> vPhaseAreas; // Areas of the phases
	std::vector<std::vector<double> > vPhasesClusterAreas; // Areas of the Clusters for all Clusterphases

	// vClusterPhases -> stores Phases which are set to clusterdiffusion
	// |---------|
	// | Phase 1 | --> Phase 1 is set to cluster diffusion
	// |---------|     ------------------------------
	// |   ...   |     | Cluster 1 | Cluster 2 | ...
	//                 ------------------------------
	//                   --> First Cluster of Flynns of Phase 1
	//                       |---------|
	//                       | Flynn 1 | --> First Flynn of that Cluster
	//                       |---------|
	//                       | Flynn 2 |
	//                       |---------|
	//                       |   ...   |
	std::vector<std::vector<std::vector<int> > > vPhasesClusters; // Flynns of all Clusters of all Clusterphases

	std::vector<int> vClusterPhases; // a copy of the ClusterDiffPhases (copied in the Constructor)
	

	// FUNCTIONS RELATED TO CLUSTER TRACKING

	std::vector<double> returnMultiplier ( std::vector<double > );
	double returnFlynnAreaChange ( int, int, Coords * );
	std::vector<double> returnClusterAreaChange ( std::vector<std::vector<int > > , int, Coords * );
	bool checkDoubleClusterArea( int, int, int );
	void getPhaseAreas( void );
	bool getClusters( void );
	void getClusterAreas( void );
	void resolveSplit( std::vector<std::vector<int > > );
	void resolveMerge( int, int, std::list<double> );
	bool resolveDoubleClusterArea( int, int, int, double );
};
/*
 * DECLARATION OF REMAINING FUNCTIONS:
 * ATTENTION: Indentation indicates dependencies and which function calls / de-
 * pends on which other functions etc.
 */
int InitGrowth(void);
void FS_check_unodes_in_flynn();

    int Read2PhaseDb(char *dbfile, AllPhases *phases);
        bool fileExists(const char *filename);
        
    int GBMGrowth(void);
    
        /* uses several functions of cluster tracking class FOR INSTANCE:
         * clusters.writeInitialData();
         * clusters.updateClusters(); // in case anything in microstructure changed
         * etc.
         */
        double ReturnArea(ERegion poly, int node, Coords *pos);
        void FS_InitialTopoCheck(int iNode);
        
        int GGMoveNode_all(int node, Coords *xy, clusterTracking *clusterData);
            // uses returnClusterAreaEnergy
            double GetNodeEnergy(int node, Coords *);
                // GetVectorLength, which is defnied and declared in movenode.cc / .h code
                double CheckPair(int node1, int node2, int type);
            double GetNodeStoredEnergy( int node, Coords * xy );
                double density_unodes_3( int iFlynn, Coords * xy, int node );
                double FS_density_unodes(int iFlynn,Coords * cTrialXY,int iBnode);
                    double FS_ScaleDDForBasalActivity(int iUnode);
                double FS_GetROI(int iFact);
                double area_swept_gpclip( int node, Coords * xy );
                    double area_triangle (Coords n1, Coords n2, Coords n3);
            // GetMoveDir, which is defnied and declared in movenode.cc / .h code
        
        void CheckAngle(double min_ang);
            int IncreaseAngle(Coords *xy,Coords *xy1,Coords *xy2,Coords *diff);
        int DeleteSmallFlynns( void );
            int MergeSmallFlynn( int iFlynn1, int iFlynn2 );
        int FS_update_dislocden(void);
        int FS_update_dislocden2(void);
            // double FS_GetROI(int iFact);
        
// FS additional functions
double ZZ_FS_GetNodeEnergy( int node, Coords * xy );
int FS_NodeOnExcludeBoundary(int iNode);
double FS_GetTotalBoxArea(); // For my changes in cluster tracking
            
#endif

