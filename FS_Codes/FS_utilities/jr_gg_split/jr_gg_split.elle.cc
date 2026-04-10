#include <vector>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
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
#include "interface.h"
#include "polygon.h"
#include "split2.elle.cc"

#define PI 3.141592653589793
#define DtoR PI/180
#define RtoD 180/PI

using std::vector;

int InitGG_Split(void);
int Init_GG_Split(void);
int GG_Split(int splitmode, double chance, double min_area, double max_area, double mcs, int x);
int MoveDoubleJ(int node1);
int MoveTripleJ(int node1);
extern void GetRay(int node1,int node2,int node3,double *ray,Coords *movedist);
void MoveFlynnNodes(int **nodes, int num, int moves);
void TimeWrite(FILE **where);
double ListBNodes(int **nodes, int n, double *dir_x, double *dir_y);

extern runtime_opts Settings_run;

/* mid_area: the average of all grain areas
 * TotalTime: not really used
 * max_split_age: age a daughter grain has to be before it can split again
 * chance: chance for a grain to split
 * min_child_area: minimum fraktion of the area of the parent flynn for the 2 daughter flynns
 * max_area: area when the chance comes to 100% for splitting
 */

double TotalTime, gb_energy;
FILE *fp;  //this is where the log is written.

int InitGG_Split(void)
{
    int err=0;
    char *infile;

    printf("Usage:\ncommand line parameter -u x1 x2 x3 (x4) (x5) (x6) -- (x)=optional, *=standard (used if nothing else is supplied\nx1: splitmode\n\t1* = every grain same chance (x2)\n\t2 every grain starting from min_area (x5) same chance (x2)\n\t3 increasing chance from min_area (x5) with chance (x2) to max_area (x6) with 100%% chance\n");
	printf("x2: split chance from 0 to 1*\nx3: randomshuffle&randomD forward - supplied int-1, 0* is default\nx4: restart step -- supplied in case of crash to restart at the given step number (e.g. for numeration), 0* is default\nx5: min_area - double (should be supplied in split mode 2&3)\nx6: max_area - double (should be supplied in split mode 3)\n");


    ElleReinit();
    ElleSetRunFunction(Init_GG_Split);

    infile = ElleFile();

	
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);

        ElleAddDoubles();
    }
	
	if (!ElleFlynnAttributeActive(SPLIT))
	    ElleInitFlynnAttribute(SPLIT);
}

int Init_GG_Split(void)
{
	int split_mode, start_stage, x, i;
	double chance, min_area, max_area, min_child_area;

	UserData userdata;

	ElleUserData(userdata);
    // 1=every grain has same chance2split,
    // 2=every grain>MinArea same chance2split,
    // 3= grains<MinArea 0%chance till grains>MaxArea 100%chance2split,
	split_mode = (int)userdata[0];
    if (split_mode<1 || split_mode>3)
    	split_mode=1;
    chance = (double)userdata[1];
    if (chance<0)
    	chance=0;
    else if (chance>1)
    	chance=1;
    x = (int)userdata[2];
    x = fabs(x);
    x -= 1;
    for (i=0; i<x; i++)
		ElleRandomD();
    min_area = (double)userdata[4];
    max_area = (double)userdata[5]; // area where chance to split becomes 100% used for split mode 3
    if (min_area>max_area) {
    	printf("Error: min_area > max_area: correction of max_area\n");
    	max_area=min_area*.5;
    }
	// last stage if simulations has crashed.
    if ((start_stage=(int)userdata[3])!=1)
    	Settings_run.Count = start_stage;

    min_child_area = .4;
    // command line can be started with e.g. -u 0.01 0.001 0.35

    //printf("How many steps should the randomD generator be forwarded? - Enter an int.\n");
	//scanf("%d", &x);
	//x = fabs(x);
    

    fp=fopen("log.txt", "a");
    //filename for the log

    fprintf(fp,"\n***********************************\n************** RUN ****************\n***********************************\n\n");
    TimeWrite(&fp);
    fprintf(fp, "%d\tforward of randomD\n", x);
    fprintf(fp, "%d\t\tsplit mode\n", split_mode);
    fprintf(fp, "%E\tmin grain size\n", min_area);
    fprintf(fp, "%E\tmax grain size\n", max_area);
	fprintf(fp, "%E\tchance for a grain to split\n", chance);
	fprintf(fp, "%E\tminimum area for child grains as fraction of the parent grain\n", min_child_area);
	fflush(fp);

	fclose(fp);

	TotalTime=0.0;

	GG_Split(split_mode, chance, min_area, max_area, min_child_area, x);
}

void TimeWrite(FILE **where)
{
	char timestore[80];

	time_t rawtime;
	struct tm * timeinfo;

	time ( &rawtime );
	timeinfo = localtime ( &rawtime );
	// for the time...

	strftime(timestore, 80, "Date: %d/%m/%Y, %H:%M:%S\n", timeinfo);
	fprintf(*where,"%s", timestore);
}

int GG_Split(int splitmode, double chance, double min_area, double max_area, double mcs, int x)
{

	int i=0, j=0, n=0, c1, c2, k, savestep;
    int maxn, maxf;
    int errors=0, splits=0, small_errors=0, nd=0, *nodes, num;
    vector<int> seq;
    double a, test, try_x, try_y;

    savestep = Settings_run.save.frequency;

    fp=fopen("log.txt", "a");

    fprintf(fp,"**************START****************\nStages: %d\n***********************************\n", EllemaxStages());
    fprintf(fp,"STEP\tSplits+ndSplits\tsGrains\tErrors\n");

    if (ElleCount()==0) ElleAddDoubles();
    if (ElleDisplay()) EllePlotRegions(ElleCount());
	ElleCheckFiles();

	gb_energy = (1e-7)*ElleTimestep();

	for (i=0;i<EllemaxStages();i++) {
        printf("Stage: %d\n", Settings_run.Count);
        maxn = ElleMaxNodes();
        seq.clear();
        for (j=0;j<maxn;j++)
        	if (ElleNodeIsActive(j))
        		seq.push_back(j);

        random_shuffle(seq.begin(),seq.end());
        maxn = seq.size();
        for (n=0;n<maxn;n++) {
            j=seq[n];
            if (ElleNodeIsActive(j)) {
                if (ElleNodeIsDouble(j)) {
                    MoveDoubleJ(j);
                    ElleCheckDoubleJ(j);
                }
                else if (ElleNodeIsTriple(j)) {
                    MoveTripleJ(j);
                    ElleCheckTripleJ(j);
                }
            }
        }
		seq.clear();
		maxf = ElleMaxFlynns();

		for (j=0;j<maxf;j++)
			if (ElleFlynnIsActive(j))
				seq.push_back(j);
		for (j=0;j<x;j++)
			random_shuffle(seq.begin(),seq.end());
		maxf = seq.size();

		for (n=0;n<maxf;n++) {
			j=seq[n];

			//area of the flynn
			a = fabs(ElleRegionArea(j));
			test = ElleRandomD();
			if (splitmode == 3)
				chance += ((a-min_area)/(max_area-min_area));
			//printf("Chance: %f\n", chance);
			if (splitmode == 1) {
				if (test<chance) {
					ElleFlynnNodes(j, &nodes, &num);
					ListBNodes(&nodes, num, &try_x, &try_y);
					k = directionsplit2(j, try_x, try_y, mcs, &c1, &c2);
					if (k==1) {
						splits++;
						EllePromoteFlynn(c1);
						EllePromoteFlynn(c2);
						ElleRemoveFlynn(j);
					}
					else if (k==2)
						small_errors++;
					else if (k==3) {
						nd++;
						EllePromoteFlynn(c1);
						EllePromoteFlynn(c2);
						ElleRemoveFlynn(j);
					}
					else
						errors++;
				}
			}
			else if (splitmode == 2 || splitmode ==3) {
				if (a >= min_area) {
					if (test<chance) {
						ElleFlynnNodes(j, &nodes, &num);
						ListBNodes(&nodes, num, &try_x, &try_y);
						k = directionsplit2(j, try_x, try_y, mcs, &c1, &c2);
						if (k==1) {
							splits++;
							EllePromoteFlynn(c1);
							EllePromoteFlynn(c2);
							ElleRemoveFlynn(j);
							ElleSetFlynnIntAttribute(c1,1,SPLIT);
							ElleSetFlynnIntAttribute(c2,1,SPLIT);
						}
						else if (k==2)
							small_errors++;
						else if (k==3) {
							nd++;
							EllePromoteFlynn(c1);
							EllePromoteFlynn(c2);
							ElleRemoveFlynn(j);
							ElleSetFlynnIntAttribute(c1,1,SPLIT);
							ElleSetFlynnIntAttribute(c2,1,SPLIT);
						}
						else
							errors++;
					}
				}

			}
		}

		if (savestep == 0) {
			if (i%5000==0) //Write the time in the log every 10k steps
				TimeWrite(&fp);
			if (i%200==0)
				fprintf(fp,"%d\t%d+%d\t%d\t%d\n", Settings_run.Count, splits, nd, small_errors, errors);
			fflush(fp);
		} else {
			if (i%(savestep*5)==0) //Write the time in the log every 10k steps
				TimeWrite(&fp);
			if (i%savestep==0)
				fprintf(fp,"%d\t%d+%d\t%d\t%d\n", Settings_run.Count, splits, nd, small_errors, errors);
			fflush(fp);
		}

        ElleUpdate();
    }
	fprintf(fp, "%d\t%d+%d\t%d\t%d\nEND: STEP\tSplits+ndSplits\tsGrains\tErrors\n", Settings_run.Count, splits, nd, small_errors, errors);
    fclose(fp);
}

double ListBNodes(int **nodes, int n, double *dir_x, double *dir_y)
{
	int i, j;
	double l=0, cl=0;
	Coords node, dist;

	// Checks distance from every node to the other nodes and keeps the direction perpendicular to the longest direction

	for (i=0; i<n; i++) {
		ElleNodePosition(*(*nodes+i), &node);
		for (j=0; j<n; j++) {
			ElleNodePlotXY(*(*nodes+j),&dist,&node);
			dist.x-=node.x;
			dist.y-=node.y;
			cl = sqrt((double)((dist.x*dist.x)+(dist.y*dist.y)));
			if (l<cl) {
				l=cl;
				*dir_x = -dist.y/cl;
				*dir_y = dist.x/cl;
			}
		}


	}
	return l;
}

void MoveFlynnNodes(int **nodes, int num, int moves)
{
	int i, n, j, maxn;
	vector<int> seq;
/*
	printf("%d:", flynn);
	for (j=0;j<num;j++)
		printf(" %d", *(*nodes+j));
	printf("\n");
*/
	for (j=0;j<num;j++)
		if (ElleNodeIsActive(*(*nodes+j)))
			seq.push_back(*(*nodes+j));
	random_shuffle(seq.begin(),seq.end());
	maxn = seq.size();
	//printf("%d\n", maxn);
	for (i=0;i<moves;i++) {
		for (n=0;n<maxn;n++) {
			j=seq[n];
			if (ElleNodeIsActive(j)) {
				if (ElleNodeIsDouble(j)) {
					MoveDoubleJ(j);
					ElleCheckDoubleJ(j);
				}
				else if (ElleNodeIsTriple(j)) {
					MoveTripleJ(j);
					ElleCheckTripleJ(j);
				}
			}
		}
	}
}

int MoveDoubleJ(int node1)
{
    int i, nghbr[2], nbnodes[3], err;
    double maxV,ray,deltaT,vlen;
    double switchDist, speedUp;
    Coords xy1, movedist;

    switchDist = ElleSwitchdistance();
    speedUp = ElleSpeedup() * switchDist * switchDist * 0.02;
    maxV = ElleSwitchdistance()/5.0;
    /*
     * allows speedUp to be 1 in input file
     */
    //gb_energy = speedUp;
    deltaT = 0.0;
    /*
     * find the node numbers of the neighbours
     */
    if (err=ElleNeighbourNodes(node1,nbnodes))
        OnError("MoveDoubleJ",err);
    i=0;
    while (i<3 && nbnodes[i]==NO_NB) i++;
    nghbr[0] = nbnodes[i]; i++;
    while (i<3 && nbnodes[i]==NO_NB) i++;
    nghbr[1] = nbnodes[i];

    GetRay(node1,nghbr[0],nghbr[1],&ray,&movedist);
    if (ray > 0.0) {
    /*if (ray > ElleSwitchdistance()/100.0) {*/
        vlen = gb_energy/ray;
        if (vlen > maxV) {
            vlen = maxV;
            deltaT = 1.0;
        }
        if (vlen>0.0) {
            movedist.x *= vlen;
            movedist.y *= vlen;
        }
        else {
            movedist.x = 0.0;
            movedist.y = 0.0;
        }
        TotalTime += deltaT;
        ElleUpdatePosition(node1,&movedist);
    }
    else {
        vlen = 0.0;
    }
}

int MoveTripleJ(int node1)
{
    int i, nghbr[3], finished=0, err=0;
    double maxV,/*gb_energy[3],*/ray[3],deltaT,vlen[3],vlenTriple;
    double switchDist, speedUp;
    Coords xy1, movedist[3], movedistTriple;

    switchDist = ElleSwitchdistance();
    /*
     * allows speedUp to be 1 in input file
     */
    speedUp = ElleSpeedup() * switchDist * switchDist * 0.02;
    maxV = switchDist/5.0;
    //for (i=0;i<3;i++) gb_energy[i] = speedUp;
    deltaT = 0.0;
    /*
     * find the node numbers of the neighbours
     */
    if (err=ElleNeighbourNodes(node1,nghbr))
        OnError("MoveTripleJ",err);

    GetRay(node1,nghbr[0],nghbr[1],&ray[0],&movedist[0]);
    GetRay(node1,nghbr[1],nghbr[2],&ray[1],&movedist[1]);
    GetRay(node1,nghbr[2],nghbr[0],&ray[2],&movedist[2]);
    for(i=0;i<3;i++) {
        if (ray[i] > 0.0) {
        /*if (ray[i] > switchDist/100.0) {*/
            vlen[i] = gb_energy/ray[i];
            /*if (vlen[i] > maxV) vlen[i] = maxV;*/
        }
        else {
            vlen[i] = 0.0;
            finished = 1;
        }
    }
    if (!finished) {
        for(i=0;i<3;i++) {
            if (vlen[i] < maxV) {
                movedist[i].x *= vlen[i];
                movedist[i].y *= vlen[i];
            }
            else {
                movedist[i].x *= maxV;
                movedist[i].y *= maxV;
            }
        }
        movedistTriple.x = movedist[0].x+movedist[1].x+movedist[2].x;
        movedistTriple.y = movedist[0].y+movedist[1].y+movedist[2].y;
        vlenTriple = sqrt(movedistTriple.x*movedistTriple.x +
                          movedistTriple.y*movedistTriple.y);
        if (vlenTriple > maxV) {
            vlenTriple = maxV/vlenTriple;
            movedistTriple.x *= vlenTriple;
            movedistTriple.y *= vlenTriple;
            deltaT = 1.0;
        }
        if (vlenTriple <= 0.0) movedistTriple.x = movedistTriple.y = 0.0;

        TotalTime += deltaT;
    }
    else {
        ElleNodePosition(node1,&xy1);
        ElleNodePlotXY(nghbr[0],&movedist[0],&xy1);
        ElleNodePlotXY(nghbr[1],&movedist[1],&xy1);
        ElleNodePlotXY(nghbr[2],&movedist[2],&xy1);
        for(i=0;i<3;i++) {
            movedist[i].x = movedist[i].x - xy1.x;
            movedist[i].y = movedist[i].y - xy1.y;
        }
        movedistTriple.x = (movedist[0].x+movedist[1].x+movedist[2].x)/2.0;
        movedistTriple.y = (movedist[0].y+movedist[1].y+movedist[2].y)/2.0;
#if XY
        vlenTriple = sqrt(movedistTriple.x*movedistTriple.x +
                          movedistTriple.y*movedistTriple.y);
        if (vlenTriple > maxV) {
            vlenTriple = maxV/vlenTriple;
            movedistTriple.x *= vlenTriple;
            movedistTriple.y *= vlenTriple;
            deltaT = 1.0;
        }
#endif
    }
    ElleUpdatePosition(node1,&movedistTriple);
}
