#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "display.h"
#include "nodes.h"
#include "unodes.h"
#include "error.h"
#include "string_utils.h"
#include "interface.h"
#include "check.h"
#include "crossings.h"
#include "runopts.h"
#include "polygon.h"
#include "file.h"
#include "init.h"
#include "convert.h"
#include "general.h"
#include "update.h"
#include "expand.h"

using std::cout;
using std::endl;
using std::ios;
using std::vector;

int MovingNode(int node, ERegion *rgn);
int MoveNodeOut(int node1,int expflynn,bool use_area,int calc_type);
int CalcMove(int node, int *nbs, Coords *incr,bool use_area);
int CalcMoveGG(int node, int *nbs, Coords *incr,bool use_area);
double NodeArea(int node);
int ExpandGrain(), InitExpand();
extern void GetRay(int node1,int node2,int node3,double *ray,Coords *movedist);

bool delete_attrib=0;

int InitExpand()
{
    char *infile;
    int err=0;
    int max;
    extern char InFile[];

    ElleReinit();
    ElleSetRunFunction(ExpandGrain);

    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError("",err);
        /*
         * check for any necessary attributes
         */
        if (!ElleFlynnAttributeActive(EXPAND)) 
            ElleAttributeNotInFile(infile,EXPAND);
        ElleAddDoubles();
		cout.setf(ios::scientific,ios::floatfield);
    }

}

#ifdef DEBUG
extern void display_row(int k);
#endif

double MaxVExp = 1.0e-4;
double VExp = 0.0;
int ExpandGrain()
{
    int i, j, k, n, exp, expflynn;
    int interval=0,err=0,max,maxf, update;
    int start, attrib_id;
    int calc_type=BisectAngle;
    int nbs[3];
	vector<int> seq;
    bool use_area=0, reset=0;
    double speedup;
    UserData udata;
    double maxarea=1e6,currarea,eps=1e-10;
    double mass = 0.0, val=0;
    double nmass = 0.0, narea = 0.0;
    double totmass = 0.0;
    double tmp = 0.0;

/*FILE *fp = fopen("area","w");*/
    ElleUserData(udata);
    speedup = udata[Velocity];  
    start = (int)udata[NodeId];
    attrib_id = (int)udata[AttribId];
    use_area = (udata[AreaFlag]>eps);
    calc_type = (int)udata[VectorCalc];
    reset = (start!=NO_NB);
    if (reset) {
        if (!ElleNodeAttributeActive(N_ATTRIB_A))
            ElleInitNodeAttribute(N_ATTRIB_A);
        ElleNeighbourNodes(start,nbs);
        for (j=0;j<3;j++)
            if (nbs[j]!=NO_NB) narea += ElleNodeSeparation(start,nbs[j]);
        narea *= ElleBndWidth();
        if (ElleResetNodeConc((int)udata[NodeId],(int)udata[ResetNum],
                           udata[AttribValue],attrib_id)==0)
            cout << "Resetting " << (int)udata[ResetNum]*2+1 <<
                 " nodes to " << udata[AttribValue] << endl; 
    }

    max = ElleMaxNodes();
    for (j=0;j<max;j++) if ( ElleNodeIsActive(j)) seq.push_back(j);
	if (ElleDisplay()) ElleUpdateDisplay();
    for (i=0;i<EllemaxStages();i++) {
        max = ElleMaxNodes();
        seq.clear();
        for (j=0;j<max;j++) if ( ElleNodeIsActive(j)) seq.push_back(j);
        if (reset && ElleResetNodeConc((int)udata[NodeId],(int)udata[ResetNum],
                           udata[AttribValue],attrib_id)==0)
#if XY
            cout << "Resetting " << (int)udata[ResetNum]*2+1 <<
                 " nodes to " << udata[AttribValue] << endl; 
        for (j=0;j<max;j++)
            if (reset && ElleNodeIsActive(j) &&
                    (val=ElleNodeAttribute(j,N_ATTRIB_A))!=0.0 /*&&
                        ElleNodeAttribute(j,attrib_id)<1*/) {
                            tmp = NodeArea(j);
                            /*ElleSetNodeAttribute(j,val*tmp/narea,attrib_id);*/
                            ElleSetNodeAttribute(j,val,attrib_id);
            }
            if (reset && ElleNodeIsActive(j))
              ElleResetNodeConc((int)udata[NodeId],(int)udata[ResetNum],
                           udata[AttribValue],attrib_id);
#endif
        if (ElleNodeAttributeActive(attrib_id) &&
						ElleUnodeAttributeActive(attrib_id) && !(i%10)) {
        nmass = ElleTotalNodeMass(attrib_id);
        mass = ElleTotalUnodeMass(attrib_id);
        totmass = mass+nmass;
		cout << i << "umass " << mass << "  nmass " << nmass << 
             "  tot " << totmass << endl;
        }
        maxf = ElleMaxFlynns();
        for (j=0,currarea=0.0;j<maxf;j++) {
            if (ElleFlynnIsActive(j) && !isParent(j)) {
                ElleGetFlynnIntAttribute(j,&exp,EXPAND);
                if (exp!=0) currarea += ElleRegionArea(j);
            }
        }
        /*
         * rate of increase in flynn area will decrease as area
         * approaches maxarea (0 when area of expanding grains is maxarea,
         *                     1 when area of expanding grains is 0)
         */
        /*MaxVExp = (double)ElleminNodeSep()*0.01*speedup**/
                     /*(pow(2.0,(1.0-currarea/maxarea)) - 1.0);*/
        VExp = MaxVExp*speedup*(pow(2.0,(1.0-currarea/maxarea)) - 1.0);

        random_shuffle(seq.begin(),seq.end());
        for (n=0;n<seq.size();n++) {
            j=seq[n];
            if (ElleNodeIsActive(j)) {
                if (MovingNode(j,&expflynn))
                    MoveNodeOut(j,expflynn,use_area,calc_type);
					  /*
        nmass = ElleTotalNodeMass(attrib_id);
        mass = ElleTotalUnodeMass(attrib_id);
        totmass = mass+nmass;
		cout << j << "umass " << mass << "  nmass " << nmass << 
             "  tot " << totmass << endl;
			 */
            }
        }
		max = ElleMaxNodes();
        for (n=0;n<max;n++) {
                if (ElleNodeIsActive(n))
                  if (ElleNodeIsDouble(n)) ElleCheckDoubleJ(n);
                  else if (ElleNodeIsTriple(n)) ElleCheckTripleJ(n);
        }
#ifdef DEBUG
display_row(50);
#endif
        ElleUpdate();
        if (ElleNodeAttributeActive(attrib_id) &&
						ElleUnodeAttributeActive(attrib_id) && !(i%10)) {
        nmass = ElleTotalNodeMass(attrib_id);
        mass = ElleTotalUnodeMass(attrib_id);
        totmass = mass+nmass;
		cout << i << "umass " << mass << "  nmass " << nmass << 
             "  tot " << totmass << endl;
        }
    }
}


/*
 * Allow node to move if it is on the boundary of
 * an expanding grain but not on the boundary
 * between two expanding grains
 */
int MovingNode(int node, ERegion *rgn)
{
    int i,nbnodes[3],move=0,cnt=0,val=0;
    ERegion tmp;

    ElleNeighbourNodes(node,nbnodes);
    for (i=0;i<3;i++) {
        if (nbnodes[i]!=NO_NB) {
            ElleNeighbourRegion(node,nbnodes[i],&tmp);
            ElleGetFlynnIntAttribute(tmp,&val,EXPAND);
            if (val) {
                if (!move) *rgn = tmp;
                move++;
            }
            cnt++;
        }
    }
    if (move==cnt) move=0;
    return(move);
}

int MoveNodeOut(int node1,int expflynn, bool use_area, int calc_type)
{
    int i, nghbr[2], nbnodes[3], rgns[3], err;
    int  cnt=0, shrinking=NO_NB, common_nb,  val;
    Coords movedist;

    /*
     * find the node numbers of the neighbours
     * and put them in the order, walking around expanding flynn,
     * nghbr[0], node1, nghbr[1]
     */
    if (err=ElleNeighbourNodes(node1,nbnodes))
        OnError("MoveNodeOut",err);
    ElleRegions(node1,rgns);
    /*
     * if it is a triple with 2 expanding regions, calculate
     * vector for each region and use average (allows for
     * expanding grains with different areas)
     */
    for (i=0,cnt=0;i<3;i++) {
        if (rgns[i]!=NO_NB) {
            ElleGetFlynnIntAttribute(rgns[i],&val,EXPAND);
            if (val) cnt++;
            else shrinking=rgns[i];
        }
    }
    if (cnt>1) {
        if ((nghbr[0] = ElleFindBndNb(node1,shrinking))==NO_NB)
            OnError("MoveNodeOut",NONB_ERR);
        i=0;
        while (i<3 && (nbnodes[i]==NO_NB || nbnodes[i]==nghbr[0])) i++;
       if (ElleNodeOnRgnBnd(nbnodes[i],shrinking))  {
            nghbr[1] = nbnodes[i];
        }
        else {
            i++;
            while (i<3 && (nbnodes[i]==NO_NB || nbnodes[i]==nghbr[0])) i++;
            if (ElleNodeOnRgnBnd(nbnodes[i],shrinking) && i<3)
                nghbr[1] = nbnodes[i];
        }
        if (i==3) OnError("MoveNodeOut - bnd nb not found",0);
    }
    else {
        if ((nghbr[1] = ElleFindBndNb(node1,expflynn))==NO_NB)
            OnError("MoveNodeOut",NONB_ERR);
        i=0;
        while (i<3 && (nbnodes[i]==NO_NB || nbnodes[i]==nghbr[1])) i++;
        if (ElleNodeOnRgnBnd(nbnodes[i],expflynn))  {
            nghbr[0] = nbnodes[i];
        }
        else {
            i++;
            while (i<3 && (nbnodes[i]==NO_NB || nbnodes[i]==nghbr[1])) i++;
            if (ElleNodeOnRgnBnd(nbnodes[i],expflynn) && i<3)
                nghbr[0] = nbnodes[i];
        }
        if (i==3) OnError("MoveNodeOut - bnd nb not found",0);
    }

    if (calc_type==BisectAngle)
        CalcMove(node1,nghbr,&movedist,use_area);
    else if (calc_type==GrainGrowth)
        CalcMoveGG(node1,nghbr,&movedist,use_area);
    ElleCrossingsCheck(node1,&movedist);
}

double NodeArea(int node)
{
    int nbs[3], j;
    double area=0.0;

    ElleNeighbourNodes(node,nbs);
    for (j=0;j<3;j++)
        if (nbs[j]!=NO_NB) area += ElleNodeSeparation(node,nbs[j]);
    return(area*ElleBndWidth());
}

int CalcMoveGG(int node, int *nbs, Coords *incr, bool use_area)
{
    int err=0, i, j, rgns[2];
    double a, factor, factor2, step;
    double ang, eps=1e-6;
    Coords xy[3];
    double dir_x,dir_y,newxy[2];
    double xpts[3], ypts[3], ray, area_old, area_new;
    float dist, dist2;

    ElleNodePosition(node,&xy[1]);
    ElleNodePlotXY(nbs[0],&xy[0],&xy[1]);
    ElleNodePlotXY(nbs[1],&xy[2],&xy[1]);

    for (i=0;i<3;i++) {
        xpts[i] = xy[i].x;
        ypts[i] = xy[i].y;
    }
    area_old = polyArea(xpts,ypts,3);

    GetRay(node,nbs[0],nbs[1],&ray,&xy[2]);
    /*
     * move a straight boundary
     */
    if (ray==0) {
        xy[2].x = sin(PI_2);
        xy[2].y = cos(PI_2);
    }
        
    /*
     * step is proportional to sqrt flynn area
     */
    ElleNeighbourRegion(node,nbs[1],&rgns[1]);
    step = VExp;
    if (use_area) {
        factor = sqrt(fabs(ElleRegionArea(rgns[1])));
        ElleNeighbourRegion(nbs[0],node,&rgns[0]);
        if (rgns[0]!=rgns[1]) {
            /*  
             *  triple - with 2 expanding grains so
             *    use average from nb regions
             */
            factor2 = sqrt(fabs(ElleRegionArea(rgns[0])));
            factor = (factor+factor2)*0.5;
        }
        step = VExp*(1.0-factor);
    }
    newxy[0] = xy[1].x+eps*xy[2].x;
    newxy[1] = xy[1].y+eps*xy[2].y;
    xpts[1] = newxy[0]; ypts[1] = newxy[1];
    area_new = polyArea(xpts,ypts,3);

    /*
     * local area change should increase flynn area
     */
    if (area_old > area_new) {
        incr->x = (-step*xy[2].x);
        incr->y = (-step*xy[2].y);
    }
    else {
        incr->x = (step*xy[2].x);
        incr->y = (step*xy[2].y);
    }

    return(err);
}

int CalcMove(int node, int *nbs, Coords *incr, bool use_area)
{
    int err=0, i, j, rgns[2];
    double a, factor, factor2, step;
    double ang, eps=1e-6;
    Coords xy[3];
    double dir_x,dir_y,newxy[2];
    double xpts[3], ypts[3], area_old, area_new;
    float dist, dist2;

    ElleNodePosition(node,&xy[1]);
    ElleNodePlotXY(nbs[0],&xy[0],&xy[1]);
    ElleNodePlotXY(nbs[1],&xy[2],&xy[1]);

    for (i=0;i<3;i++) {
        xpts[i] = xy[i].x;
        ypts[i] = xy[i].y;
    }
    area_old = polyArea(xpts,ypts,3);

    angle(xpts[1],ypts[1],
          xpts[2],ypts[2],
          xpts[0],ypts[0],&ang);

    /*
     * step is proportional to sqrt flynn area
     */
    ElleNeighbourRegion(node,nbs[1],&rgns[1]);
    step = VExp;
    if (use_area) {
        factor = sqrt(fabs(ElleRegionArea(rgns[1])));
        ElleNeighbourRegion(nbs[0],node,&rgns[0]);
        if (rgns[0]!=rgns[1]) {
            /*  
             *  triple - with 2 expanding grains so
             *    use average from nb regions
             */
            factor2 = sqrt(fabs(ElleRegionArea(rgns[0])));
            factor = (factor+factor2)*0.5;
        }
        step = VExp*(1.0-factor);
    }
    /*
     * bisect the angle
     */
    a = ang*0.5;
    if (ang<0) a += PI;
    rotate_coords(xpts[2],ypts[2],
                  xpts[1],ypts[1],
                  &dir_x, &dir_y, a);
    if (dir_y==ypts[1] && dir_x==xpts[1]) a=0;
    else 
        a = atan2(dir_y-ypts[1],dir_x-xpts[1]);
    newxy[0] = eps*cos(a) + xpts[1];
    newxy[1] = eps*sin(a) + ypts[1];
    xpts[1] = newxy[0]; ypts[1] = newxy[1];
    area_new = polyArea(xpts,ypts,3);

    /*
     * local area change should increase flynn area
     */
    if (area_old > area_new) {
        incr->x = (-step*cos(a));
        incr->y = (-step*sin(a));
    }
    else {
        incr->x = (step*cos(a));
        incr->y = (step*sin(a));
    }

    return(err);
}

#if XY
int WriteCircleNodeFile(float centrex,float centrey,float rad,
                      char *filename)
{
    char label[20];
    float max,incr,ang;
    int j, err=0, max_node, nn, num[360];
    int nn1,nn2,indx1,indx2;
    Coords current;
    FILE *fp;

    max = M_PI * 2.0;
    if ((fp=fopen(filename,"w"))==NULL) return(OPEN_ERR);
    incr = M_PI/90.0*0.45/rad;  /* 2 deg if rad=0.45 */
    max_node = (int)(max/incr);
    for (j=0;j<360;j++) num[j] = -1;
    RandomiseSequence(num,max_node);
    
    ang = 0.0;
    indx1 = 0;
    nn = num[indx1++];
    if (!id_match(FileKeys,LOCATION,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    do {
        current.x = centrex + rad*cos(ang);
        current.y = centrey + rad*sin(ang);
        fprintf(fp,"%d %f %f\n",nn,current.x,current.y);
        ang += incr;
        nn = num[indx1++];
    } while( nn != -1 );

    fclose(fp);
    return( err );
}

int WriteCircleFile(char *nodefname, char *grainfname)
{
    char label[20];
    int nn, num, grain=2, nnarray[300], i=0, j;
    float x,y;
    FILE *fin, *fout;

    fin = fopen(nodefname,"r");
    fout = fopen(grainfname,"w");
    if ((num = fscanf(fin,"%19s\n", label))!=1) return(READ_ERR);
    if (!id_match(FileKeys,FLYNNS,label)) return(KEY_ERR);
    fprintf(fout,"%s\n",label);
    while (!feof(fin)) {
        if ((num = fscanf(fin,"%d %f %f\n",
                            &nn, &x, &y)) != 3) {
            printf("Read error for node file\n");
            exit(1);;
        }
        nnarray[i++] = nn;
    }
    fprintf(fout,"%d",grain);
    fprintf(fout," %d",i);
    for (j=0;j<i;j++) fprintf(fout," %d",nnarray[j]);
    fprintf(fout,"\n");
    grain = 1;
    fprintf(fout,"%d",grain);
    fprintf(fout," %d",i);
    for (j=i-1;j>=0;j--) fprintf(fout," %d",nnarray[j]);
    fprintf(fout,"\n");

    rewind(fin);
    while ((i=fgetc(fin))!=EOF) fputc(i,fout);

    fclose(fin);
    fclose(fout);
}
#endif
