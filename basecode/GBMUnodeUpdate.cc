#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <list>
#include "nodes.h"
#include "check.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "polygon.h"
#include "stats.h"
#include "crossings.h"
#include "convert.h"
#include "timefn.h"
#include "mineraldb.h"
#include "unodes.h"
#include "attrib.h"
#include "string_utils.h"
#include "tripoly.h"
#include "general.h"
#include "log.h"

using std::cout;
using std::endl;
using std::vector;

typedef struct {
double area;
double gb_mass_i;
double nb_mass;
double node_mass;
double swept_mass_i;
double enrich_mass_i;
Coords mid_pt,end_pt[2];
} Incr_Data;
void InitIncrData(Incr_Data *p);

void CalculateInitialAreas(double xval, double u_rad, vector<int> &unodelist);
void Partition(double prev[2],Incr_Data *incr_enrich,Incr_Data *incr_sweep,
                 int increments, double *total_gb_mass,
                 double *nb_mass_f,vector <double> &delta_mass,
                 double gb_area_f,double node_mass_i,
                 double node_sweep_frac,
                 int attr_id,set_int *enrichlist,set_int *sweeplist,
                 set_int *unodelist, double roi);
void UpdateUnodes(set_int *unodelist, vector<double> &delta_mass,
                  int attr_id);
double SumTriangleWeights(Coords *current,double *incr_areas,
                          Coords *mid_pts, int increments, double ROI);
double PointWeight(Coords *line_pos,Coords *current,double ROI);
double EstimateConcFromUnodeValues(Coords *pt_pos,
                                   vector<int> *unodelist);
void ReassignUnodes(double tri_pos[3][2],double roi,
               int enrich_id,int sweep_id,int attr_id,
               int *enrich_indx,int *sweep_indx,
               set_int **unodeset,set_int *flynnset);
double SumLineWeights(double tri_pos[3][2],Coords current,double ROI);
void set_up_mid_line(double tri_pos[3][2],double pivot[2],double *inc_width,double *line_height_inc,double *length,double *height, double del[2][2]);
void set_up_tri_values(double tri_pos[3][2],double pivot[2],
                       double gb_area,double gb_area_f,
                       double conc_node,double conc_nb,
                       double *nb_sweep_frac,double sweep_area,
                       Incr_Data *incr,int increments);
double point_to_line(Coords pt, double line_pt1[2], double line_pt2[2]);
void display_row(int k);

void calc_mass();
int umax=0;
const int Increments=20;
double mass;
double *Weights[Increments];
double Total_Enrich_Weights[Increments];
double Total_Sweep_Weights[Increments];
double partition_coeff=1.0;
double Overlap_frac=0.0;
double gb_width=DFLT_BNDWIDTH, gbwidth_on_2;
double Unode_area;
static double massincr;
static int print;

/* modify unode values for all triangular areas swept by boundary */
/* all calculations done in Elle units */
void UpdateGBMUnodeValues(int n,Coords new_incr,
                          set_int **unodeset, set_int *flynnset,
                          double roi)
{
    Coords old_node,pivot_node,new_node;
    int    j,i,k,nb[3],nbnbs[3];        
    int sweep_id=NO_NB, sweep_indx, enrich_id=NO_NB, enrich_indx;
    int num_unodes;
    double gb_area_i[3],gb_area_f[3];
    double gb_mass[3],gb_mass_f[3];
    double xpts[4],ypts[4];
    double length, gb_width_on_2;
    ERegion  rgnl,rgnr;
    double tri_pos[3][2],tri_pos_s[3][2],tri_pos_e[3][2];
	double pivot[2],del[2][2],triarea;
    double total_gb_mass_f=0.0, total_gb_area_f=0.0, total_gb_mass_i=0.0;
    double total_sweep_area=0.0,sweep_area[3];
    double nb_mass_f[3],nb_mass[3],nb_area[3],gb_nb_mass[3],nb_sweep_frac[3];
    double total_nb_mass_i=0.0, total_nb_mass_f=0.0;
    double delta_unode_mass=0.0, delta_node_mass=0.0;
    double delta_nb=0.0,nb_tot=0.0;
    double conc_n, conc_nb;
Incr_Data incr_sweep[Increments], incr_enrich[Increments];
    vector<double> delta_mass;
    static int first=1;
    int e1,e2,rgn_exp[2][2];
    
        /*
         * this should calculate all conc attributes
         */
    int attr_id = CONC_A;
    ElleNodePosition(n,&old_node);
    if(first == 1)
    {
        first=0;
        umax=ElleMaxUnodes();
        Unode_area=1.0/umax;
        /* hexagon area */
        /*double s=roi/4.0/cos(PI/6.0);*/
        /*Unode_area=3.0*sin(PI/3.0)*s*s;*/
        sprintf( logbuf, "max_nodes:%d\n", umax );
        Log( 2, logbuf );
        massincr = ElleMassIncr();
        gb_width = ElleBndWidth();
		gbwidth_on_2 = gb_width/ElleUnitLength()*0.5;
        sprintf( logbuf, "ROI= %lf\n", ElleUnodeROI() );
        Log( 2, logbuf );
        print=0;
    }

    num_unodes = unodeset[COMBINED_LIST]->size();
    for (i=0;i<num_unodes;i++) delta_mass.push_back(0.0);
    for (j=0;j<Increments;j++)
        if (!(Weights[j]=(double *)malloc(num_unodes*sizeof(double))))
            OnError("UpdateGBMUnodeValues",MALLOC_ERR);
    for (i=0;i<3;i++) 
        gb_area_f[i] = gb_area_i[i] = gb_mass[i] = gb_mass_f[i] = 
        nb_area[i] = nb_mass[i] = nb_mass_f[i] = nb_sweep_frac[i] =
        sweep_area[i] = gb_nb_mass[i] = 0.0;

    tri_pos[0][0]= old_node.x;
    tri_pos[0][1]= old_node.y;
    xpts[0] = old_node.x;
    ypts[0] = old_node.y;
	tri_pos_e[0][0]=tri_pos[0][0]-gbwidth_on_2;
    tri_pos_e[0][1]= old_node.y;
    tri_pos_s[0][0]=tri_pos[0][0]+gbwidth_on_2;
    tri_pos_s[0][1]= old_node.y;
    xpts[0] = tri_pos_s[0][0];
    ypts[0] = tri_pos_s[0][1];

    tri_pos[1][0] = (double) (old_node.x + new_incr.x);
    tri_pos[1][1] = (double) (old_node.y + new_incr.y);
    xpts[2] = new_node.x=old_node.x + new_incr.x;
    ypts[2] = new_node.y=old_node.y + new_incr.y;
    tri_pos_e[1][0]=tri_pos[1][0]-gbwidth_on_2;
    tri_pos_e[1][1] = tri_pos[1][1];
    xpts[2] = tri_pos_s[1][0]=tri_pos[1][0]+gbwidth_on_2;
    ypts[2] = tri_pos_s[1][1]= tri_pos[1][1];

    ElleNeighbourNodes(n,nb);
    conc_n =  ElleNodeAttribute(n,attr_id);

    /* for each neighbour node*/

    for (i=0;i<3;i++) 
    {
        
        if (nb[i]!=NO_NB) //for each gb segment adjacent to the moving node
        {
			RegionExpanding(n,nb[i],rgn_exp,&new_node);
            ElleNodePlotXY(nb[i],&pivot_node,&old_node);
            tri_pos[2][0]= xpts[1]= pivot_node.x;
            tri_pos[2][1]= ypts[1]= pivot_node.y;
			tri_pos_e[2][0]=tri_pos[2][0]-gbwidth_on_2;
            tri_pos_e[2][1] = tri_pos[2][1];
			xpts[1] = tri_pos_s[2][0]=tri_pos[2][0]+gbwidth_on_2;
            ypts[1] = tri_pos_s[2][1]= tri_pos[2][1];
            triarea = polyArea(xpts,ypts,3);
            sweep_area[i] = fabs(triarea);

			e1=rgn_exp[0][1];
			e2=rgn_exp[1][1];
			rgnl=rgn_exp[0][0];
			rgnr=rgn_exp[1][0];
			/*
            ElleGetFlynnIntAttribute(rgnl, &e1, EXPAND);
            ElleGetFlynnIntAttribute(rgnr, &e2, EXPAND); // prevents trip junctions getting too much mass
            if (triarea>0) {
                sweep_id = rgnl;
                enrich_id = rgnr;
            }
            else {
                sweep_id = rgnr;
                enrich_id = rgnl;
            }
			*/

            gb_area_i[i] = pointSeparation(&pivot_node,&old_node)*
                                               gb_width/ElleUnitLength();
            length = pointSeparation(&pivot_node,&new_node);
            gb_area_f[i] = length* gb_width/ElleUnitLength();
			if (sweep_area[i]>gb_area_f[i])
				//swept area that does not become grain boundary
  			    Overlap_frac = (sweep_area[i]-gb_area_f[i])/sweep_area[i];

            gb_mass[i] = conc_n * gb_area_i[i] * 0.5;

            if(e1 != e2) // only if segment sweeps lattice
            {
				//expanding grain is enriched
				if (e1==1) {
                  sweep_id = rgn_exp[1][0];
                  enrich_id = rgn_exp[0][0];
                }
                else {
                  sweep_id = rgn_exp[0][0];
                  enrich_id = rgn_exp[1][0];
                }
                ReassignUnodes(tri_pos,roi,enrich_id,sweep_id,attr_id,
                           &enrich_indx,&sweep_indx,unodeset,flynnset);
                // nb_area and nb_mass are the values for nb[i]'s
                // segments not attached to node n (ie not changed in
                // this calculation
                conc_nb =  ElleNodeAttribute(nb[i],attr_id);
                ElleNeighbourNodes(nb[i],nbnbs);
                for (j=0;j<3;j++)
                    if (nbnbs[j]!=NO_NB && nbnbs[j]!=n)
                        nb_area[i] += ElleNodeSeparation(nb[i],nbnbs[j])/2.0;
                nb_area[i] *= gb_width/ElleUnitLength();
                nb_mass[i] = conc_nb * nb_area[i];
                gb_nb_mass[i] = conc_nb * gb_area_i[i] * 0.5;
                for (j=0;j<Increments;j++) InitIncrData(&incr_sweep[j]);
                for (j=0;j<Increments;j++) InitIncrData(&incr_enrich[j]);
                for (k=0;k<Increments;k++) 
                    for (j=0;j<num_unodes;j++) Weights[k][j]=0.0;

                set_up_tri_values(tri_pos_s,pivot,gb_area_i[i],gb_area_f[i],
                                  conc_n,conc_nb,&nb_sweep_frac[i],
                                  sweep_area[i],
                                  incr_sweep,Increments);
                set_up_tri_values(tri_pos_e,pivot,gb_area_i[i],gb_area_f[i],
                                  conc_n,conc_nb,&nb_sweep_frac[i],
                                  sweep_area[i],
                                  incr_enrich,Increments);

                Partition(tri_pos[0],incr_sweep,incr_enrich,Increments,
						  &gb_mass_f[i], &nb_mass_f[i],
                          delta_mass,gb_area_f[i],gb_mass[i],
                          (1.0-nb_sweep_frac[i]),
                          attr_id,unodeset[enrich_indx],
                          unodeset[sweep_indx],unodeset[COMBINED_LIST],roi);
            }
        }
    } 
    for (i=0;i<3;i++) {
        total_gb_mass_f += gb_mass_f[i];
        total_gb_area_f += gb_area_f[i];
        total_gb_mass_i += gb_mass[i];
        total_sweep_area += sweep_area[i];
        total_nb_mass_i += gb_nb_mass[i];
        total_nb_mass_f += nb_mass_f[i];
    }
    delta_unode_mass = 
                accumulate(delta_mass.begin(),delta_mass.end(),0.0);
    delta_node_mass = total_gb_mass_f-total_gb_mass_i +
                      total_nb_mass_f-total_nb_mass_i;
    if (delta_unode_mass>0 && (delta_unode_mass+delta_node_mass)>0)
        delta_unode_mass = -delta_node_mass;
    UpdateUnodes(unodeset[COMBINED_LIST],delta_mass,attr_id);
    if (delta_unode_mass = 
                accumulate(delta_mass.begin(),delta_mass.end(),0.0)) {
        for (i=0;i<3;i++) {
          if (nb[i]!=NO_NB) {
            delta_nb = 
            delta_unode_mass*(sweep_area[i]/total_sweep_area)*nb_sweep_frac[i];
            if (delta_nb < -nb_mass_f[i]) delta_nb = -nb_mass_f[i];
            nb_mass_f[i] += delta_nb;
            nb_tot += delta_nb;
          }
        }
        total_gb_mass_f += (delta_unode_mass-nb_tot);
    }
    if (total_gb_mass_f<0.0) total_gb_mass_f = 0.0;
    for (i=0;i<3;i++)
        if (nb[i]!=NO_NB)
            ElleSetNodeAttribute(nb[i],(nb_mass[i]+nb_mass_f[i])/
                                       (nb_area[i]+gb_area_f[i]/2),attr_id);
    ElleSetNodeAttribute(n,total_gb_mass_f/(total_gb_area_f/2.0),attr_id);

    for (j=0;j<Increments;j++)
        if (Weights[j]) free(Weights[j]);
}

void UpdateUnodes(set_int *unodelist, vector<double> &delta_mass,
                  int attr_id)
{
    double conc, mass_chge, u_mass, delta_unode_mass, u_area=0.0;
    int cnt=0, i=0, j=0, max, err=0;

    if(unodelist!=0) {
        for (set_int :: iterator it = unodelist->begin();
                    it != unodelist->end() && !err; it++,cnt++) 
        {
            i=(*it);
            ElleGetUnodeAttribute((*it),&conc,attr_id);
                
            if (delta_mass[cnt]!=0.0) {
              ElleGetUnodeArea(*it,&u_area );
              u_mass = conc*u_area;
              mass_chge=delta_mass[cnt];
              if (u_mass+mass_chge<0.0) 
                  mass_chge=-u_mass;
              ElleSetUnodeAttribute(i,(u_mass+mass_chge)/u_area,
                                     attr_id);
              delta_mass[cnt]-= mass_chge;
            }
        }
        if (delta_unode_mass = 
                accumulate(delta_mass.begin(),delta_mass.end(),0.0)) {
				max=delta_mass.size();
        for (set_int :: iterator it = unodelist->begin();
                    it != unodelist->end() && !err; it++,cnt++) 
        {
            i=(*it);
            ElleGetUnodeAttribute((*it),&conc,attr_id);
            u_mass = conc*u_area;
			if (u_mass>0) {
				j=0;
				while (j<max&&
						(delta_mass[j]>=0.0||(delta_mass[j]+u_mass)<0.0)) j++;
				if (j<max) {
                  mass_chge=delta_mass[j];
	              ElleSetUnodeAttribute(i,(u_mass+mass_chge)/u_area,attr_id);
                  delta_mass[j]-= mass_chge;
				}
			}
        }
        }
    }
}

void ReassignUnodes(double tri_pos[3][2],double roi,
               int enrich_id,int sweep_id,int attr_id,
               int *enrich_indx,int *sweep_indx,
               set_int **unodeset,set_int *flynnset)
{
    int found=0, i, cnt=0;
    Coords xy, node_xy, current;
    double pnt[2], conc, conc0, conc_est=0;
    double weight, weight_n, tot_weight=0, del_conc;
    set_int :: iterator f_it;
    set_int :: iterator it, it2;
    vector<double> weights;

    *enrich_indx=NO_NB, *sweep_indx=NO_NB;
    for (f_it = flynnset->begin(),cnt=0; f_it != flynnset->end();
                                                        f_it++,cnt++) {
        if (*f_it == enrich_id) *enrich_indx=cnt;
        if (*f_it == sweep_id) *sweep_indx=cnt;
    }
    if (*enrich_indx==NO_NB||*sweep_indx==NO_NB)
        OnError("flynn not found",0);
    for (cnt=0;cnt<unodeset[*enrich_indx]->size();cnt++)
        weights.push_back(0.0);
    /* 
     * unodes on boundary or in region to be swept
     * should be moved to enrich list
     */
    for (it = unodeset[*sweep_indx]->begin();
                  it != unodeset[*sweep_indx]->end(); it++) {
        ElleGetUnodePosition(*it,&xy);
        pnt[0] = xy.x; pnt[1] = xy.y;
        if (find(unodeset[*enrich_indx]->begin(),
                 unodeset[*enrich_indx]->end(),*it)==
                                         unodeset[*enrich_indx]->end() &&
                CrossingsTest(&tri_pos[0][0],3,pnt))
            unodeset[*enrich_indx]->insert(*it);
    }
}

double EstimateConcFromUnodeValues(Coords *pt_pos, int attr_id,
                                   vector<int> *unodelist, double roi)
{
    int i, err=0;
    double conc=0, conc_estimate=0.0;
    double total_pointweight=0, pointweight=0.0;
    Coords current;

    if(unodelist!=0) {
        for (vector<int> :: iterator it = unodelist->begin();
                    it != unodelist->end() && !err; it++,i++)
        {
            i=(*it);
            ElleGetUnodePolyInfo((*it),attr_id,&current,&conc);
            ElleCoordsPlotXY(&current,pt_pos);

            pointweight=PointWeight(pt_pos,&current,roi);

            total_pointweight+=pointweight;

            conc_estimate+=pointweight*conc;
        }
    }
    if (total_pointweight>0.0)
        conc_estimate=conc_estimate/total_pointweight;
    else {
        sprintf( logbuf, "EstimateConcFromUnodeValues: no contributing unodes\n" );
        Log( 1, logbuf );
    }
    return(conc_estimate);
}

void Partition(double prev_pt[2], Incr_Data *incr_sweep,Incr_Data *incr_enrich,
				 int increments, 
                 double *total_gb_mass,double *nb_mass_f,
                 vector <double> &delta_mass,
                 double gb_area_f, double node_mass_i,
                 double node_sweep_frac,
                 int attr_id,set_int *enrichlist,set_int *sweeplist,
                 set_int *unodelist, double roi)
{
    int i=0,k=0,cnt=0;
    int err=0;
    double increments2=increments*2.0;
    double gb_incr_area, overlap;
    Coords current,prev;
    double conc, conc_e1, conc_s1, weight, weight1, weight2;
    double mass_chge, mass_chge_e, mass_chge_b, mass_chge_s;
    double swept_area_frac, enrich_area_frac;
    double conc_s, conc_e, conc_b;
    double conc_s_f, conc_e_f, conc_b_f;
    double final_gb_mass, mass_tmp;
    double u_area=0;
    double pc1, pc2, pc;
    double tmp, frac, mfrac=1;
    set_int :: iterator it;
    double tot_sarea=0.0, tot_gbmass=0.0;
    double tot_gbmass_f=0.0, tot_weight=0.0;
    double tot_mass_e=0.0, node_chge_e=0.0;
    
    prev.x = prev_pt[0];
    prev.y = prev_pt[1];

    gb_incr_area = gb_area_f/increments;

    pc1 = pc2 = pc=1.0;
    vector<int> reassigned;
    for (it = sweeplist->begin(); it != sweeplist->end(); it++) 
        if (find(enrichlist->begin(),enrichlist->end(),*it)!=
                                         enrichlist->end())
            reassigned.push_back(*it);

    for (k=0,frac=(increments2-1)/increments2;k<increments;
                                    k++,frac-=2.0/increments2) {
      conc_s = conc_e = conc_s1 = conc_e1 = 0.0;
      mass_chge_b = mass_chge_e = mass_chge_s = final_gb_mass = 0.0;
      tmp= tot_weight = mass_tmp = 0;
      Total_Sweep_Weights[0] = Total_Enrich_Weights[0] = 0.0;
      Total_Sweep_Weights[1] = Total_Enrich_Weights[1] = 0.0;
      Total_Sweep_Weights[2] = Total_Enrich_Weights[2] = 0.0;
      for (cnt=0,it = unodelist->begin();
                   it != unodelist->end() && !err; it++,cnt++) 
      {
            i=(*it);
            ElleGetUnodePolyInfo((*it),attr_id,&current,&conc);
            ElleCoordsPlotXY(&current,&prev);
            ElleGetUnodeArea(*it,&u_area );
            //u_area = ElleUnodeArea(*it);
                
            /*weight = PointWeight(&current,&incr[k].mid_pt,roi);*/
            weight = PointWeight(&current,&incr_sweep[k].end_pt[0],roi);
            weight2 = PointWeight(&current,&incr_sweep[k].end_pt[1],roi);
            if (weight!=0.0||weight2!=0) {
                weight1 = (weight+weight2)/2.0;
                weight = weight1;
                tot_weight += weight;
            // incr_areas are in Elle units, scale area to bell units
                weight1 *= (PI/roi)*(PI/roi)*incr_sweep[k].area;
            // weights is fraction of bell volume
                weight1 /= ((0.5*PI)*(PI*PI-4.0));
                if (find(reassigned.begin(),reassigned.end(),i)!=
                                         reassigned.end()) {
                    weight1 /= 2;
                    weight /= 2;
                }
                Weights[k][cnt] = weight1;
                tmp = conc*u_area*weight1;
                if (find(sweeplist->begin(),sweeplist->end(),i)!=
                                         sweeplist->end()) {
                    /*if (find(reassigned.begin(),reassigned.end(),i)==*/
                                         /*reassigned.end()) {*/
                        Total_Sweep_Weights[1] += weight1;
                        conc_s1 += weight1*conc;
                    /*}*/
                    conc_s += weight1*conc;
                    Total_Sweep_Weights[0] += weight1;
                    Total_Sweep_Weights[2] += weight;
                    /*delta_mass[cnt] -= tmp;*/
                    incr_sweep[k].swept_mass_i += tmp;
                }
                if ( find(enrichlist->begin(),enrichlist->end(),i)!=
                                         enrichlist->end()) {
                    conc_e1 += weight1*conc;
                    Total_Enrich_Weights[1] += weight1;
                    /*if (find(reassigned.begin(),reassigned.end(),i)==*/
                                         /*reassigned.end()) {*/
                      conc_e += weight1*conc;
                      Total_Enrich_Weights[0] += weight1;
                      Total_Enrich_Weights[2] += weight;
                      incr_enrich[k].enrich_mass_i += tmp;
                      /*delta_mass[cnt] += tmp;*/
                    /*}*/
                }
            }
        }
        if (Total_Enrich_Weights[0]>0) conc_e /= Total_Enrich_Weights[0];
        if (Total_Enrich_Weights[1]>0) {
            conc_e1 /= Total_Enrich_Weights[1];
        }
        if (Total_Sweep_Weights[0]>0) {
            conc_s /= Total_Sweep_Weights[0];
        }
        if (Total_Sweep_Weights[1]>0) conc_s1 /= Total_Sweep_Weights[1];
        incr_sweep[k].gb_mass_i = incr_sweep[k].node_mass+incr_sweep[k].nb_mass;
        tot_gbmass +=incr_sweep[k].gb_mass_i;
        final_gb_mass = incr_sweep[k].gb_mass_i;
        conc_e_f = conc_s_f = conc_b = incr_sweep[k].gb_mass_i/gb_incr_area;
        tmp = 0.0;
        swept_area_frac = Total_Sweep_Weights[2]/tot_weight;
        enrich_area_frac = Total_Enrich_Weights[2]/tot_weight;
        if (incr_sweep[k].gb_mass_i>0)
            mfrac = incr_sweep[k].node_mass/incr_sweep[k].gb_mass_i;

        // boundary and unodes
        // include boundary flux here if known
        if (conc_s!=conc_b || conc_e!=conc_b) {
          conc_b_f = (incr_sweep[k].gb_mass_i+incr_sweep[k].swept_mass_i+
                        incr_enrich[k].enrich_mass_i)/
                        (incr_sweep[k].area + gb_incr_area);
          conc_s_f = conc_b_f*pc1;
          conc_e_f = conc_b_f*pc2;
          mass_chge = conc_s_f*incr_sweep[k].area*Overlap_frac*swept_area_frac;
          mass_chge_s = -incr_sweep[k].swept_mass_i+mass_chge;
          mass_chge = conc_e_f*(1.0-Overlap_frac)*incr_sweep[k].area+
				  conc_e_f*incr_enrich[k].area*enrich_area_frac*Overlap_frac;
          mass_chge_e = -incr_enrich[k].enrich_mass_i+mass_chge;
		  /*
          mass_chge_e = conc_e_f*incr_enrich[k].area*enrich_area_frac-
                                                incr_enrich[k].enrich_mass_i;
          mass_chge = conc_s_f*incr_sweep[k].area*(1.0-Overlap_frac)*swept_area_frac;
          mass_chge_s = mass_chge-mass_chge_s;
          mass_chge_e += mass_chge;
		  */
          mass_chge_b = gb_incr_area*conc_b_f-incr_sweep[k].gb_mass_i;
        }
        /*mass_chge_b = -(mass_chge_s + mass_chge_e);*/
		mass_chge_e = -(mass_chge_b + mass_chge_s);
        final_gb_mass = incr_sweep[k].gb_mass_i+mass_chge_b;
        if (mass_chge_b>0)
            node_chge_e += mass_chge_b*frac;
        else if (mass_chge_b<0)
            node_chge_e += mass_chge_b*mfrac;
            
        for (cnt=0,it = unodelist->begin();
                it != unodelist->end() && !err; it++,cnt++) {
            i=(*it);
            if (Weights[k][cnt] &&
              find(sweeplist->begin(),sweeplist->end(),i)!=
                                 sweeplist->end())
              delta_mass[cnt] += (/*incr[k].swept_mass_i+*/mass_chge_s)*
                             Weights[k][cnt]/Total_Sweep_Weights[0];
            if (Weights[k][cnt] &&
              find(enrichlist->begin(),enrichlist->end(),i)!=
                                 enrichlist->end()){
              /*if (find(reassigned.begin(),reassigned.end(),i)==*/
                                 /*reassigned.end())*/
              delta_mass[cnt] += (/*incr[k].enrich_mass_i+*/ mass_chge_e)*
                             Weights[k][cnt]/Total_Enrich_Weights[1];
			tmp+=Weights[k][cnt]/Total_Enrich_Weights[1];
			}
        }
mass_tmp= accumulate(delta_mass.begin(),delta_mass.end(),0.0);
        tot_sarea += incr_sweep[k].area;
        tot_gbmass_f += final_gb_mass;
    }
    *total_gb_mass = node_mass_i+node_chge_e;
    if (*total_gb_mass>tot_gbmass_f) *total_gb_mass = tot_gbmass_f;
    *nb_mass_f = tot_gbmass_f - *total_gb_mass;
tmp= accumulate(delta_mass.begin(),delta_mass.end(),0.0);
}       

#if XY
void Partition(double tri_pos[3][2], Incr_Data *incr,int increments, 
                 double *total_gb_mass,double *nb_mass_f,
                 vector <double> &delta_mass,
                 double gb_area_f, double node_mass_i,
                 double node_sweep_frac,
                 int attr_id,set_int *enrichlist,set_int *sweeplist,
                 set_int *unodelist, double roi)
{
    int i=0,k=0,cnt=0;
    int err=0;
    double increments2=increments*2.0;
    double gb_incr_area;
    Coords current,prev;
    double conc, conc_s_f, conc_e1, conc_s1, conc_e_f, weight, weight1;
    double mass_chge_e, mass_chge_s, conc_s, conc_e, conc_s2, conc_tmp;
    double mass_chge_b = 0;
    double conc_e2;
    double final_gb_mass;
    double pc=1.0;
    double tmp, frac, mfrac;
    set_int :: iterator it;
    double tot_sarea=0.0, tot_gbmass=0.0;
    double tot_gbmass_f=0.0, tot_weight=0.0;
    double node_chge_s=0.0, node_chge_e=0.0;
    
    prev.x = tri_pos[0][0];
    prev.y = tri_pos[0][1];

    gb_incr_area = gb_area_f/increments;

    vector<int> reassigned;
    for (it = sweeplist->begin(); it != sweeplist->end(); it++) 
        if (find(enrichlist->begin(),enrichlist->end(),*it)!=
                                         enrichlist->end())
            reassigned.push_back(*it);
    for (k=0,frac=(increments2-1)/increments2;k<increments;
                                    k++,frac-=2.0/increments2) {
      conc_s = conc_e = conc_s1 = conc_e1 = conc_s2 = conc_e2 = 0.0;
      mass_chge_s = mass_chge_e = tot_weight = 0.0;
      tmp=0;
      Total_Sweep_Weights[0] = Total_Enrich_Weights[0] = 0.0;
      Total_Sweep_Weights[1] = Total_Enrich_Weights[1] = 0.0;
      for (cnt=0,it = unodelist->begin();
                    it != unodelist->end() && !err; it++,cnt++) {
            i=(*it);
            ElleGetUnodePolyInfo((*it),attr_id,&current,&conc);
            ElleCoordsPlotXY(&current,&prev);
                
            weight = PointWeight(&current,&incr[k].end_pt[0],roi);
            weight1 = PointWeight(&current,&incr[k].end_pt[1],roi);
            if (weight!=0.0||weight1!=0.0) {
                Weights[k][cnt] = weight1;
                if (find(sweeplist->begin(),sweeplist->end(),i)!=
                                         sweeplist->end()) {
if (print && (k==0||k==19||k==39)) cout << "sweep "<< i << ' ' << conc << endl;
                    Total_Sweep_Weights[1] += weight1;
                    conc_s1 += weight1*conc;
                }
                if ( find(enrichlist->begin(),enrichlist->end(),i)!=
                                         enrichlist->end()) {
if (print && (k==0||k==19||k==39)) cout << "enrich "<< i << ' ' << conc << endl;
                    Total_Enrich_Weights[1] += weight1;
                    conc_e1 += weight1*conc;
                }
tot_weight += Weights[k][cnt];
            }
        }
        if (Total_Enrich_Weights[0]>0) conc_e /= Total_Enrich_Weights[0];
        if (Total_Enrich_Weights[1]>0) conc_e1 /= Total_Enrich_Weights[1];
        if (Total_Sweep_Weights[0]>0) conc_s /= Total_Sweep_Weights[0];
        if (Total_Sweep_Weights[1]>0) conc_s1 /= Total_Sweep_Weights[1];
        incr[k].gb_mass_i = incr[k].node_mass+incr[k].nb_mass;
        tot_gbmass +=incr[k].gb_mass_i;
        tot_sarea += incr[k].area;
        conc_e_f = conc_s_f = incr[k].gb_mass_i/gb_incr_area;
        tmp = 0.0;
        if (conc_s1 != conc_s_f) {
            mfrac = incr[k].node_mass/incr[k].gb_mass_i;
        //mass taken from swept unodes
            conc_s_f = (conc_s1*Total_Sweep_Weights[1]/tot_weight*
                        incr[k].area+incr[k].gb_mass_i)*pc/
                         (incr[k].area + gb_incr_area);
            if (conc_s_f<conc_s1)
                mass_chge_s += (conc_s_f-conc_s1)*incr[k].area*
                            Total_Sweep_Weights[1]/tot_weight;
            conc_e_f = (incr[k].gb_mass_i-mass_chge_s+
                       conc_e1*incr[k].area*
                         Total_Enrich_Weights[1]/tot_weight)/
                         (incr[k].area + gb_incr_area);
            mass_chge_b = (conc_e_f-conc_e1)*
                           Total_Enrich_Weights[1]/tot_weight*incr[k].area;
            if (mass_chge_b > (incr[k].gb_mass_i-mass_chge_s))
                conc_e_f = (incr[k].gb_mass_i-mass_chge_s)/incr[k].area*
                            tot_weight/Total_Enrich_Weights[1]+conc_e1;
            /*if (conc_s1>conc_s_f || conc_e1<conc_s_f) {*/
            if (conc_e1!=conc_e_f) {
                for (cnt=0,it = unodelist->begin();
                        it != unodelist->end() && !err; it++,cnt++) {
                    i=(*it);
                    if ( Weights[k][cnt] &&
                        find(enrichlist->begin(),enrichlist->end(),i)!=
                                         enrichlist->end()) {
                        ElleGetUnodePolyInfo((*it),attr_id,&current,&conc);
                        /*tmp = (conc_e_f*Total_Enrich_Weights[1]/tot_weight-conc)* */
                        tmp = (conc_e_f-conc)* 
                               Weights[k][cnt]/Total_Enrich_Weights[1]*
                                incr[k].area;
                        delta_mass[cnt] += tmp;
                        mass_chge_e += tmp;
                    }
                }
           }
           conc_tmp = (incr[k].gb_mass_i-mass_chge_s-mass_chge_e)/gb_incr_area;
           if (mass_chge_e<0)
             node_chge_e += conc_tmp*gb_incr_area*frac-incr[k].node_mass;
           else if (mass_chge_e>0)
             node_chge_e += conc_tmp*gb_incr_area*mfrac-incr[k].node_mass;
           tot_gbmass_f += conc_tmp * gb_incr_area;
        }
    }
    *total_gb_mass = node_mass_i+node_chge_e;
    if (*total_gb_mass>tot_gbmass_f) *total_gb_mass = tot_gbmass_f;
    *nb_mass_f = tot_gbmass_f - *total_gb_mass;
tmp= accumulate(delta_mass.begin(),delta_mass.end(),0.0);
}
#endif

/* function to sum weights in bell for point location*/
double PointWeight(Coords *line_pos,Coords *current,double ROI)
{
    double gauss;
    double radial;
        
    radial=sqrt(((current->x-line_pos->x)*(current->x-line_pos->x)+
                ((current->y-line_pos->y)*(current->y-line_pos->y))));

    if(radial<=ROI)
        gauss=(1+cos(PI*radial/ROI))/2.0;
    else
        gauss=0.0;
        
    
    return(gauss);        
}

/* function to sum weights in bell wrt gb segment*/
double SumLineWeights(double tri_pos[3][2],Coords current,double ROI)
{
    double weights,gauss;
    double mid[2],end[2],dist[3];
    double a2,height,mindist,radial;
    int i,j,k;
    double line_pos[2];
    double line_x,line_y,line_height;
    int increments=100;
        

    line_pos[0]=tri_pos[1][0];
    line_pos[1]=tri_pos[1][1];
    
    radial=point_to_line(current, tri_pos[1], tri_pos[2]);
    
    if(radial<=ROI)
        gauss=PI*(2.0*((1+cos(PI*radial/ROI))/4.0))/2.0;        
    else
        gauss=0.0;
        
    
    return(gauss);        
}


/* set up mid line parameters */
void set_up_tri_values(double tri_pos[3][2],double pivot[2],
                       double gb_area, double gb_area_f,
                       double conc_n,double conc_nb,
                       double *nb_sweep_frac,double sweep_area,
                       Incr_Data *incr,int increments)
{
    int i,j,k;
    double mid[2],end[2][2],xpts[3],ypts[3];
    double del[2][2],del_mid[2];
    double gb_incr_area=gb_area/increments;
    double tot_area=0.0;
    double gb_mass_i=(conc_n+conc_nb)/2 * gb_area;
    double nb_sweep_area=0.0;
    double conc=conc_n;
    double del_conc_n = 0.0, del_conc_nb = 0.0;

    mid[0]=(tri_pos[0][0]+tri_pos[1][0])/2.0;
    mid[1]=(tri_pos[0][1]+tri_pos[1][1])/2.0;

    end[0][0]=tri_pos[0][0];
    end[0][1]=tri_pos[0][1];

    end[1][0]=tri_pos[1][0];
    end[1][1]=tri_pos[1][1];

    pivot[0]=tri_pos[2][0];
    pivot[1]=tri_pos[2][1];
    
    del[0][0]=    (end[0][0]-pivot[0])/increments;
    del[0][1]=    (end[0][1]-pivot[1])/increments;

    del[1][0]=    (end[1][0]-pivot[0])/increments;
    del[1][1]=    (end[1][1]-pivot[1])/increments;

    del_mid[0]=    (mid[0]-pivot[0])/increments;
    del_mid[1]=    (mid[1]-pivot[1])/increments;

    xpts[0] = end[0][0]; ypts[0] = end[0][1];
    xpts[1] = end[1][0]; ypts[1] = end[1][1];
    xpts[2] = pivot[0]; ypts[2] = pivot[1];
    // area of swept triangle
    incr[0].area = sweep_area;
    incr[0].mid_pt.x = mid[0]-del_mid[0]/2.0;
    incr[0].mid_pt.y = mid[1]-del_mid[1]/2.0;
    incr[0].end_pt[0].x = end[0][0]-del[0][0]/2.0;
    incr[0].end_pt[0].y = end[0][1]-del[0][1]/2.0;
    incr[0].end_pt[1].x = end[1][0]-del[1][0]/2.0;
    incr[0].end_pt[1].y = end[1][1]-del[1][1]/2.0;
    
    tot_area = sweep_area+gb_area;
    // assumes linear conc gradient along segment from node to nb
    del_conc_n = conc_n/(increments*2);
    del_conc_nb = conc_nb/(increments*2);
    conc_n -= del_conc_n;
    conc_nb = del_conc_nb;
    incr[0].nb_mass = conc_nb*gb_incr_area;
    incr[0].node_mass= conc_n*gb_incr_area;
    conc_n -= del_conc_n*2;
    conc_nb += del_conc_nb*2;

    // area of swept triangle
    // use similar triangles to create increments
    // store areas and midpts (pts on mid_line of triangle)
double tot=0.0;
    for(k=1;k<increments;k++) {
        xpts[0] -= del[0][0];
        ypts[0] -= del[0][1];
        xpts[1] -= del[1][0];
        ypts[1] -= del[1][1];
        incr[k].area = fabs(polyArea(xpts,ypts,3));
        incr[k-1].area -= incr[k].area;
        /*incr[k-1].gb_mass_i = gb_mass_i * (incr[k-1].area+gb_incr_area)/*/
                               /*(tot_area);*/
        if (k==(increments/2)) nb_sweep_area = incr[k].area;
        incr[k].mid_pt.x = incr[k-1].mid_pt.x-del_mid[0];
        incr[k].mid_pt.y = incr[k-1].mid_pt.y-del_mid[1];
        incr[k].end_pt[0].x = incr[k-1].end_pt[0].x-del[0][0];
        incr[k].end_pt[0].y = incr[k-1].end_pt[0].y-del[0][1];
        incr[k].end_pt[1].x = incr[k-1].end_pt[1].x-del[1][0];
        incr[k].end_pt[1].y = incr[k-1].end_pt[1].y-del[1][1];
        incr[k].node_mass = conc_n*gb_incr_area;
        incr[k].nb_mass = conc_nb*gb_incr_area;
        conc_n -= del_conc_n*2;
        conc_nb += del_conc_nb*2;
        tot += incr[k-1].gb_mass_i;
    }
    /*incr[increments-1].gb_mass_i = gb_mass_i **/
                           /*(incr[increments-1].area+gb_incr_area)/tot_area;*/
    *nb_sweep_frac = nb_sweep_area/sweep_area;
}

/* set up mid line parameters */
void set_up_mid_line(double tri_pos[3][2],double pivot[2],double *inc_width,double *line_height_inc,
                     double *length,double *height, double del[2][2])
{
    double weights,gauss;
    double mid[2],end[2][2],dist[3];
    double a2,mindist,radial;
    int i,j,k;
    int increments=100;
    double line_x,line_y,line_height;

    
/*            
    for(k=0;k<3;k++)
    {
        dist[k]=sqrt(((tri_pos[k][0]-tri_pos[(k+1)%3][0])*(tri_pos[k][0]-tri_pos[(k+1)%3][0]))+
                     ((tri_pos[k][1]-tri_pos[(k+1)%3][1])*(tri_pos[k][1]-tri_pos[(k+1)%3][1])));
        
    }
    
    for(k=0,mindist=1e10;k<3;k++)
    {
        if(dist[k]<mindist)
        {
            mindist=dist[k];

            mid[0]=(tri_pos[k][0]+tri_pos[(k+1)%3][0])/2.0;
            mid[1]=(tri_pos[k][1]+tri_pos[(k+1)%3][1])/2.0;

            end[0][0]=tri_pos[k][0];
            end[0][1]=tri_pos[k][1];

            end[1][0]=tri_pos[(k+1)%3][0];
            end[1][1]=tri_pos[(k+1)%3][1];

            pivot[0]=tri_pos[(k+2)%3][0];
            pivot[1]=tri_pos[(k+2)%3][1];
        }
    }
*/    
    
    mid[0]=(tri_pos[0][0]+tri_pos[1][0])/2.0;
    mid[1]=(tri_pos[0][1]+tri_pos[1][1])/2.0;

    end[0][0]=tri_pos[0][0];
    end[0][1]=tri_pos[0][1];

    end[1][0]=tri_pos[1][0];
    end[1][1]=tri_pos[1][1];

    pivot[0]=tri_pos[2][0];
    pivot[1]=tri_pos[2][1];
    
    
    
    mindist=sqrt(((tri_pos[0][0]-tri_pos[1][0])*(tri_pos[0][0]-tri_pos[1][0]))+
                     ((tri_pos[0][1]-tri_pos[1][1])*(tri_pos[0][1]-tri_pos[1][1])));
                         
    *line_height_inc=mindist/increments;    
    *length=sqrt(((pivot[0]-mid[0])*(pivot[0]-mid[0]))+((pivot[1]-mid[1])*(pivot[1]-mid[1])));
    *inc_width=*length/(increments-1);
    
    del[0][0]=    (end[0][0]-pivot[0])/increments;
    del[0][1]=    (end[0][1]-pivot[1])/increments;

    del[1][0]=    (end[1][0]-pivot[0])/increments;
    del[1][1]=    (end[1][1]-pivot[1])/increments;
}


void calc_mass()
{
    int max,i,err,nb[3],j=0,no_unodes=0;
    double lattice_mass=0.0,all_gb_mass=0.0;
    double conc;
    double node_mass,length,gb_mass;
    Coords current;
    double minconc=1e10,maxconc=-1e10;
            
    max = ElleMaxFlynns();

    for(i=0;i<max;i++)
    {
        if(ElleFlynnIsActive(i))
        {
            vector<int> unodelist;
            ElleGetFlynnUnodeList(i,unodelist);
            
            for (vector<int> :: iterator it = unodelist.begin();
                    it != unodelist.end() ; it++) 
            {
                ElleGetUnodePolyInfo((*it),CONC_A,&current,&conc);
                node_mass=conc*Unode_area;
                lattice_mass+=node_mass;
                
                if(conc<minconc)
                    minconc=conc;            
                if(conc>maxconc)
                    maxconc=conc;    
                
                no_unodes++;    
            }            
        }                
    }
    
    if(no_unodes<max) {
        sprintf( logbuf, "only %d of all %d unodes accounted for\n", no_unodes, max );
        Log( 1, logbuf );
    }
        
    max = ElleMaxNodes();

    for (i=0;i<max;i++) 
    {
         if (ElleNodeIsActive(i)) 
        {
            ElleNeighbourNodes(i,nb);
            for (j=0;j<3;j++) 
            {
                if (nb[j]!=NO_NB && nb[j]<i) 
                {
                    length=ElleNodeSeparation(i,nb[j]);
                    gb_mass=length*gb_width*
                            (ElleNodeAttribute(nb[j],CONC_A)+
                             ElleNodeAttribute(i,CONC_A))/2.0;
                    if(length<ElleSwitchdistance()*10.0);
                        all_gb_mass+=gb_mass;
                }
            }
        }
    }
    
    //printf("minconc,maxconc:%f %f\n",minconc,maxconc);
    //printf("lattice_mass,gb_mass,total_mass:%12.10le %12.10le %12.10le\n",lattice_mass,all_gb_mass,lattice_mass+all_gb_mass);
}

double point_to_line(Coords pt, double line_pt1[2], double line_pt2[2])
{
    double a2,height;
    
    a2=((pt.y-line_pt1[1])*(line_pt2[0]-line_pt1[0]))-((pt.x-line_pt1[0])*(line_pt2[1]-line_pt1[1]));
    
    height=2*sqrt((a2*a2)/(((line_pt2[0]-line_pt1[0])*(line_pt2[0]-line_pt1[0]))+((line_pt2[1]-line_pt1[1])*(line_pt2[1]-line_pt1[1]))));
    
    return(height);
}

void CalculateInitialAreas(double xval, double u_rad,
                            vector<int> &unodelist)
{
    double rad, area;
    Coords current;
    vector<int> :: iterator it;

    if (!UnodeAttributeActive(U_ATTRIB_A))
        ElleInitUnodeAttribute(U_ATTRIB_A);
    for (it = unodelist.begin(); it != unodelist.end(); it++) {
        ElleGetUnodePosition((*it),&current);
        rad = fabs(current.x-xval);
        if (fabs(rad)<u_rad) {
            area = u_rad*u_rad*(PI-acos(rad/u_rad)) +
                   rad * sqrt(u_rad*u_rad + rad*rad);
            ElleSetUnodeAttribute(*it,area/Unode_area,U_ATTRIB_A);
        }
    }
}

void display_row(int k)
{
    FILE *out;
    int i,min,max;
    double conc;
    
    min=k*100;
    max=min+100;
    
    out=fopen("row","a");
    
    for(i=min;i<max;i++)
    {
        ElleGetUnodeAttribute(i,CONC_A,&conc);
        fprintf(out,"%e\t",conc);
    }
    fprintf(out,"\n");
    fclose(out);
}

void InitIncrData(Incr_Data *p)
{
   p->area=0.0;
   p->gb_mass_i=0.0;
   p->nb_mass=0.0;
   p->node_mass=0.0;
   p->swept_mass_i=0.0;
   p->enrich_mass_i=0.0;
   p->mid_pt.x=0.0;
   p->mid_pt.y=0.0;
   p->end_pt[0].x=0.0;
   p->end_pt[0].y=0.0;
   p->end_pt[1].x=0.0;
   p->end_pt[1].y=0.0;
}

#if XY
void set_up_weights(double tri_pos[3][2], double *incr_areas,
                 Coords *mid_pts, double *mass, int increments,
                 int attr_id,set_int *unodelist,
                 double roi, int bnd);
void SweepUnodes(double tri_pos[3][2], double *incr_areas,
                 Coords *mid_pts, double *mass, int increments,
                 int attr_id,set_int *allunodes, double roi, int bnd);
void EnrichUnodes(double *p_mass,set_int *unodelist, int attr_id);
void PartitionMass( int n, int attr_id, double pc, double *swept_area,
                    double *swept_mass, double *put_mass,
                    double *gb_area_i, double *gb_area_f, double *gb_mass);
void RemoveSweptMass(set_int *unodelist, int attr_id);

void set_up_weights(double tri_pos[3][2], double *incr_areas,
                 Coords *mid_pts, double *mass, int increments,
                 int attr_id,set_int *unodelist,
                 double roi, int bnd);
                // remove swept mass
                SweepUnodes(tri_pos,tri_areas,mid_pts,
                            &swept_mass[i],Increments,attr_id,
                            unodeset[COMBINED_LIST],roi,i);
    RemoveSweptMass(unodeset[COMBINED_LIST],attr_id);

    PartitionMass(n,attr_id,partition_coeff,
                  swept_area,swept_mass,put_mass,
                  gb_area_f,gb_mass);

    EnrichUnodes(put_mass,unodeset[COMBINED_LIST],attr_id);
}

/* reduce unode values after being swept by boundary */
void SweepUnodes(double tri_pos[3][2], double *incr_areas,
                 Coords *mid_pts, double *mass, int increments,
                 int attr_id,set_int *unodelist,
                 double roi, int bnd)
{
    /* 
     * Sum mass from remaining swept part of Unodes and
     * remove mass from them
     */

    int i=0,cnt=0;
    int err=0;
    double swept_mass,unode_mass;
    double total=0.0;
    Coords current,prev;
    double conc;
    set_int :: iterator it2;
    
    prev.x = tri_pos[0][0];
    prev.y = tri_pos[0][1];

    if(unodelist!=0) {
        for (set_int :: iterator it = unodelist->begin();
                    it != unodelist->end() && !err; it++) 
        {
            i=(*it);
            if (i>=umax) {
                sprintf( logbuf, "unode index too big %d\n", i );
                Log( 2, logbuf );
            }
            ElleGetUnodePolyInfo((*it),attr_id,&current,&conc);
            ElleCoordsPlotXY(&current,&prev);
                
            if (Weights[bnd][i]>0.0) {
                unode_mass = conc*Unode_area;
                swept_mass=unode_mass*Weights[bnd][i];

                *mass += swept_mass;
                total += Weights[bnd][i];
            }
        }
    }
}           

/* reduce unode values after being swept by boundary */
void RemoveSweptMass(set_int *unodelist, int attr_id)
{
    /* 
     * Sum mass of Unodes and remove mass from them
     */

    int i=0,cnt=0,j;
    int err=0;
    double swept_mass,unode_mass;
    double total=0.0, temp_mass=0.0;
    double conc;
    
    if(unodelist!=0) {
        for (set_int :: iterator it = unodelist->begin();
                    it != unodelist->end() && !err; it++) 
        {
            i=(*it);
            if (i>=umax) {
                sprintf( logbuf, "unode index too big %d\n", i );
                Log( 2, logbuf );
            }
            ElleGetUnodeAttribute((*it),&conc,attr_id);
                
            for (total=0.0,j=0;j<3;j++) total += Weights[j][i];
            if (total>0.0) {
                unode_mass = conc*Unode_area;
                swept_mass=unode_mass*total;

                err=ElleSetUnodeAttribute((*it),
                           (unode_mass-swept_mass)/Unode_area,attr_id);
                temp_mass += swept_mass;
            }
        }
    }
}           

void PartitionMass( int n, int attr_id, double pc, double *swept_area,
                    double *swept_mass, double *put_mass,
                    double *gb_area_i,double *gb_area_f, double *gb_mass)
{
    int i, nbs[3], nb_cnt=0;
    double conc=0.0, mass=0.0, total_put_mass=0.0, gb_mass_f;
    double total_swept_mass=0.0,total_swept_area=0.0,total_gb_mass=0.0;
    double total_gb_area_i=0.0,total_gb_area_f=0.0;

    ElleNeighbourNodes(n,nbs);
    for (i=0;i<3;i++) {
        total_swept_mass+=swept_mass[i];
        total_swept_area+=swept_area[i];
        total_gb_area_i+=gb_area_i[i];
        total_gb_area_f+=gb_area_f[i];
        total_gb_mass+=gb_mass[i];
        put_mass[i] = 0.0;

        if (nbs[i]!=NO_NB) nb_cnt++;
    }
    mass = total_swept_mass+ total_gb_mass;
    gb_mass_f = mass * (total_gb_area_f/2.0) * pc /
                (total_swept_area + (total_gb_area_f/2.0));
    total_put_mass = mass - gb_mass_f;
    // ratio of swept area to total swept area
    for (i=0;i<3;i++) {
        if (nbs[i]!=NO_NB) {
          if (total_swept_area!=0.0)
              put_mass[i] = total_put_mass*(swept_area[i]/total_swept_area);
          else
              put_mass[i] = (total_put_mass/nb_cnt);
        }
    }

    Conc = gb_mass_f/(total_gb_area_f/2.0);
    ElleSetNodeAttribute(n,Conc,attr_id);
}


void EnrichUnodes(double *p_mass,set_int *unodelist, int attr_id)
{
    /* 
     * Sum mass of Unodes and remove mass from them
     */

    int i=0,cnt=0,j;
    int err=0;
    double put_mass=0.0,unode_mass;
    double u_total=0.0, total=0.0;
    double conc;
    
    
    for (j=0;j<3;j++) total += p_mass[j];
    if(unodelist!=0) {
        for (set_int :: iterator it = unodelist->begin();
                    it != unodelist->end() && !err; it++) 
        {
            i=(*it);
            if (i>=umax) {
                sprintf( logbuf, "unode index too big %d\n", i );
                Log( 2, logbuf );
            }
            ElleGetUnodeAttribute((*it),&conc,attr_id);
                
            put_mass = 0.0;
            for (j=0;j<3;j++) {
                if (Total_Weights[j]>0.0)
                    put_mass+=p_mass[j]*Weights[j][i]/Total_Weights[j];
            }

            unode_mass = conc*Unode_area;
            err=ElleSetUnodeAttribute((*it),
                           (unode_mass+put_mass)/Unode_area,attr_id);
            total -= put_mass;
        }
    if (fabs(total) > 1e-20) cout << total <<endl;
    }
}        

/* function to sum weights in bell swept by triangle increment*/
/* weighting bell is 1 unit high and has radius PI  */
double IncrementWeight(Coords *centre,double *incr_areas,
                          Coords *mid_pts, int increments, double ROI)
{
    int k;
    double weights,gauss,radial;
    
    for(k=0,weights=0.0;k<increments;k++)
    {
        radial=pointSeparation(centre,&mid_pts[k]);

        if(radial<=ROI)
            gauss=(1.0+cos(PI*radial/ROI))/2.0;
        else
            gauss=0.0;

        weights += (gauss*incr_areas[k]);
    }
    // incr_areas are in Elle units, scale area to bell units
    weights *= (PI/ROI)*(PI/ROI);
    // weights is fraction of bell volume
    weights /= ((0.5*PI)*(PI*PI-4.0));
    return(weights);        
}

#endif
