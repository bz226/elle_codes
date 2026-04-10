#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include "convert.h"
#include "attrib.h"
#include "nodes.h"
#include "update.h"
#include "interface.h"
#include "init.h"
#include "triattrib.h"
#include "unodes.h"
#include "display.h"
#include "file.h"
#include "error.h"
#include "string_utils.h"
#include "runopts.h"
#include "polygon.h"
#include "tripoly.h"
#include "timefn.h"

extern const int KAPPA_INDEX, NODE_ID, ATTRIB_VAL, DIFFUSION_ONLY, TEMP_INCR;
extern const double KAPPA_DFLT;

int InitSetUnodes();
int CalcConc();
void SwapPointers();
void CalcConcAtUnode(int n, double dt,int num_nbs);
void CalcConcAtNode(int n, double dt,int num_nbs);
void UpdateConc(int flynnid);
void UpdateBdConc(int flynnid);
void SetupConc(int flynnid,int attr_id,double *minlen,int *max_nbs);
void SetupBdConc(int flynnid,int attr_id,int *bd_max_nbs);
int CleanupConc();
double CalcUnodeSep(int unode1, int unode2,int index);
double CalcSegWidth(int num,int num_nbs,int i);
void SetVoronoi(int tmp,int unode1,int unode2,int index2,int unode3,int index3);
double CalcArea(int num,int unode1,int num_nbs);
double CalcBdSegWidth(int num);
double CalcBdTriArea(int num,int node1);
double CalcTriArea(int num,int i,int unode1,int num_nbs);
void SetBdVoronoi(int tmp,int node1,int node2,int,int node3,int);
void SetMargins();
double CalcInitMinConc(int minid); //mwj*** 23/06/2003
double GrtBtCalcMinEnergy(double init_Bt_conc,double init_Grt_conc,double T, double grt_mole, double Bt_mole, double *sgc, double *sbc);//mwj*** 23/06/2003
double FlynnBoundaryArea(int flynn); //mwj*** 25/06/2003
double TotalBoundaryArea(int minid); //mwj*** 25/06/2003

void ExchangeReaction();
void CalcExchange(int count,double min_energy,double);
void CleanupExchangeArrays();
int ReactionIsActive(int flynnid);
double CalcMinEnergy(int flynnid,double,double);
double CalcGrtMole();
double CalcBtMole();
double CalcBtMole(double);
double CalcGrtMole(double);
void CalcMinBndyArea();

Coords *Voronoi=0;
Coords *BdVoronoi=0;
int Count=0,Max_nbs;
int *Neighbours=0,*BdNeighbours=0;

int *NbNodeSize=0,*NodeIndex=0, *BdNodeIndex=0;
double *OldConc=0, *NewConc=0, *SegLength=0, *SegWidth=0;
double *BdOldConc=0, *BdNewConc=0, *BdSegLength=0, *BdSegWidth=0;
double *LatticeConc=0,*BoundaryConc=0;
double Minlen;
double Kappa;
double G_RC = 1.0;
double B_RC = 1.0;
double Grt_area =8.18352838e-2;
double Bt_area = 2.77094724e-1;
double CoolingRate=-0.25;
/*float Kappa=0.000005;*/



/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitSetUnodes()
{
    char *infile;
    int err=0;

    extern int Count;
    /*
     * clear the data structures
     */
    ElleReinit();

    ElleSetRunFunction(CalcConc);
    ElleSetExitFunction(CleanupConc);
    CleanupConc();
    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
        /*
         * check for any necessary attributes which may
         * not have been in the elle file
         */
        if (ElleNodeAttributeActive(CONC_A)&&
	       ElleUnodeAttributeActive(CONC_A)&&
	         !ElleNodeMarginAttributeActive(CONC_A))
	    ElleInitNodeMarginAttribute(CONC_A);
	        
        
    }
    
    SetMargins();    
    
    Kappa = KAPPA_DFLT;
    std::cout << "Kappa= "<< Kappa << std::endl;
    Count = 1;

 	CalcMinBndyArea();
}

void SetMargins()
{
    int i, j, k, count, numunodes;
    int max_stages, max_flynns, max_unodes, max_nbs, bd_max_nbs;
    double val, roi=0.01;
	
    
    
    std::vector<int> bdnodelist, unodelist;
    
    max_flynns = ElleMaxFlynns(); // maximum flynn number used
    max_unodes = ElleMaxUnodes(); // maximum unode number used
    roi = sqrt(1.0/(double)max_unodes/3.142)*5;
    for (i=0;i<max_flynns;i++)   // cycle through stages
	{    
        if(ElleFlynnIsActive(i)){
        ElleClearTriAttributes();  // reset triangulation
        TriangulateUnodes(i,MeshData.tri); // triangulate one flynns unodes
	
	ElleGetFlynnUnodeList(i,unodelist);
        ElleFlynnNodes(i,bdnodelist); // get the list of unodes for a flynn
        count = bdnodelist.size();           
        for(j=0;j<count;j++)
	   {
	    val=ElleNodeMarginConcEstimate(bdnodelist[j],CONC_A,&unodelist,roi);
	    ElleSetNodeMarginAttribute(bdnodelist[j],val,CONC_A,i);
             	   
	   }
        numunodes = unodelist.size();
	for(j=0;j<count;j++) bdnodelist.pop_back();
	for(j=0;j<numunodes;j++) unodelist.pop_back(); 
	 
	 }
       }
}

int CalcConc()
{
    int i, j, k;
    int max_stages, max_flynns, max_unodes, max_nbs, bd_max_nbs;
    int interval=0,err=0;
    int *seq;
    int stages, steps=1;
    int reset_node;
    FILE *fp;
    double timestep, dt=1.0, length, tmp,kappa_m, temp_incr;
    double lastconc;
    double minlen_elle=10.0,tmp_minlen=10.0;
    UserData udata;
    Coords incr;
    extern int Count;
    int mintype;
    double Matrix_area=0.0;
       
    ElleUserData(udata);
   
    length = ElleUnitLength();   
    
    ElleCheckFiles();
    
    max_stages = EllemaxStages(); // number of stages 
    max_flynns = ElleMaxFlynns(); // maximum flynn number used
    max_unodes = ElleMaxUnodes(); // maximum unode number used
 	
    CoolingRate = udata[TEMP_INCR]; 	   //LE 4/1/2006
    if(udata[DIFFUSION_ONLY] != 1) // for lattice diffusion only calculation
		ExchangeReaction();
    // LE 3/10/06 The following line was at the end of
    // ExchangeReaction(). I moved it here as it should be done even if
    // DIFFUSION_ONLY is set. 
    // Cooling Rate is rate per Stage (timestep) so half increment
    // before and half increment after matches original coding
    ElleSetTemperature(ElleTemperature()+CoolingRate/2.0); //this is for cooling rate  
    
    for (i=0;i<max_flynns;i++)   // cycle through flynns
	{ 
        Kappa =  udata[KAPPA_INDEX]*
                   exp(-239000.0/(8.3144*(ElleTemperature()+273.15)));
 	    if(ElleFlynnIsActive(i)&& ReactionIsActive(i))
	    {
    	    ElleClearTriAttributes();  // reset triangulation
            TriangulateUnodes(i,MeshData.tri); // triangulate one flynns unodes
    	    
    	    ElleGetFlynnIntAttribute(i, &mintype, MINERAL); //mwj*** 20/06/2003
    		//if(mintype==GARNET) //mwj*** 20/06/2003
    	    	//WritePoly(MeshData.tri,"Unode4.poly");
    	    
    	    SetupConc(i,CONC_A,&minlen_elle,&max_nbs);
    	    
    	    SetupBdConc(i,CONC_A,&bd_max_nbs);
    	    Minlen = minlen_elle*ElleUnitLength();
    	    	    
            kappa_m = Kappa*ElleUnitLength()*ElleUnitLength();
    	   if (EllemaxStages()==1) 
    	   {
                  timestep = ElleTimestep();
                  /* 
                  * using value from FE: dt <= len*len/Kappa
                  */
                  dt = Minlen*Minlen/kappa_m*0.25;
                  tmp = timestep/dt;
                  if (tmp > (double)INT_MAX) 
                     OnError("Timestep too large for process",0);
                  steps = (int)(tmp);
                  if (steps<100) 
                  {
                     fprintf(stderr,"Calculated %d steps\n",steps);
                     steps = 100;
                  }
                  dt = timestep/steps;
                  fprintf(stderr,"Running %d steps at dt of %le s = %le s\n",steps,dt,steps*dt);
                  ElleSetStages(1);
           }    	
    	   else if(EllemaxStages()>1) 
    	   {
    	          //check that area diffused in 1 stage is less
                  //than minimum possible area

				  if((Minlen*(Minlen/2)/kappa_m)<1.0)
                  	fprintf(stderr,"Warning:Kappa, node separation or stages need adjustment. Minlen: %le Kappa: %le\n",Minlen,Kappa); 
           }
    	   
    	   for (j=0;j<max_stages;j++) 
    	   {
             	for(k=0;k<steps;k++)
             	{
    	     		Kappa = Kappa/length/length;
    
    	     		SwapPointers();	

             		CalcConcAtUnode(i,dt,max_nbs);  
                 
    	     		CalcConcAtNode(i,dt,bd_max_nbs); 
                     
    	     		Kappa = Kappa*length*length;	    	       
    	     	}
    	    }

    	    UpdateConc(i);    
    	    UpdateBdConc(i);
    	    CleanupConc();   	     
		}     
    }

    
    if(udata[DIFFUSION_ONLY] != 1)
    	ExchangeReaction();
    ElleSetTemperature(ElleTemperature()+CoolingRate/2.0); //this is for cooling rate  
    ElleUpdate();

} 
void CalcMinBndyArea()
{
    int i, ii, j, k, n ,count;
    int  max_flynns, max_nbs, bd_max_nbs,num_nbs;
    double length,minlen_elle=10.0;
    int mintype;
    
	
	Grt_area=0.0;
	Bt_area=0.0;   
   
    length = ElleUnitLength();   
        
    max_flynns = ElleMaxFlynns(); // maximum flynn number used
	
	for (i=0;i<max_flynns;i++)   // cycle through flynns
	{    

  	    if(ElleFlynnIsActive(i)&& ReactionIsActive(i))
	    {
    	    ElleClearTriAttributes();  // reset triangulation
            TriangulateUnodes(i,MeshData.tri); // triangulate one flynns unodes
    	        		    	    
    	    SetupConc(i,CONC_A,&minlen_elle,&max_nbs);
    	    
    	    SetupBdConc(i,CONC_A,&bd_max_nbs);
    	    	       	   
            ElleGetFlynnIntAttribute(i, &mintype, MINERAL);//mwj*** 20/06/2003
                        
            std::vector<int> bdnodelist;
            ElleFlynnNodes(i,bdnodelist); // get the list of bnodes for a flynn
            count = bdnodelist.size();  

    	    //std::cout << "flynnid " << i << std::endl;
    	    //std::cout << "mintype " << mintype << std::endl;
    	    //std::cout << "count " << count << std::endl;

            for (ii=0;ii<count;ii++)
            {
                n = bdnodelist[ii];    

                for (k=0;k<bd_max_nbs-1;k++) 
                {
                    j = BdNeighbours[ii*bd_max_nbs+k];
            	    if (j!=NO_NB) 
            	    {                
	  					//std::cout << "n ii k j " << n << " "<<ii << " "<< k << " " << j << std::endl;
                	    if(mintype==MICA)
                	    	Bt_area += CalcBdTriArea(ii*bd_max_nbs+k,n);
                	    else if(mintype==GARNET)
                	    	Grt_area += CalcBdTriArea(ii*bd_max_nbs+k,n);               	    
        	        }
                }   
            }        	     	
    	    
    	    CleanupConc();
    	  }
      }
	  Bt_area*=acos(0.0)*10.0;
	  Grt_area*=acos(0.0)*10.0;
	  
	  std::cout << "Bi bndy area:" << Bt_area << std::endl;
	  std::cout << "Gt bndy area:" << Grt_area << std::endl;
}
int CleanupConc(void)
{
    if (OldConc) { free(OldConc); OldConc=0; }
    if (NewConc) {free(NewConc); NewConc=0;}
    if (Neighbours) {free(Neighbours); Neighbours=0;}
    if (SegLength) {free(SegLength); SegLength=0;}
    if (SegWidth) {free(SegWidth); SegWidth=0;}
    if (Voronoi) {free(Voronoi); Voronoi=0;}
    if (NbNodeSize) {free(NbNodeSize); NbNodeSize=0;}
    if (NodeIndex) {free(NodeIndex); NodeIndex=0;}
    
    if (BdOldConc) { free(BdOldConc); BdOldConc=0; }
    if (BdNewConc) {free(BdNewConc); BdNewConc=0;}
    if (BdNeighbours) {free(BdNeighbours); BdNeighbours=0;}
    if (BdSegLength) {free(BdSegLength); BdSegLength=0;}
    if (BdSegWidth) {free(BdSegWidth); BdSegWidth=0;}
    if (BdVoronoi) {free(BdVoronoi); BdVoronoi=0;}
    if (BdNodeIndex) {free(BdNodeIndex); BdNodeIndex=0;}
}

void SetupConc(int flynnid,int attr_id,double *minlen,int *max_nbs)
{
    int i,ii, j,jj, k, tmp=1,a,aa;
    int id, num_nbs, count, bd_size,index,found;
    double val,xx;
    std::vector<int> nbnodes;
    std::vector<int> node_index;
    *minlen = 1.0;
    std::vector<int> unodelist;  // create a vector list of unodes
    std::vector<int> bdnodelist;  // create a vector list of unodes
    
    ElleFlynnNodes(flynnid,bdnodelist);
    bd_size = bdnodelist.size();
    ElleGetFlynnUnodeList(flynnid,unodelist); // get the list of unodes for a flynn
    count = unodelist.size();
    
    for (i=0; i<count; i++) 
    {					
	    std::vector<int> nbnodes1;
        std::vector<int> node_index1;
	    ElleGetTriPtNeighbours(unodelist[i],nbnodes1,node_index1,1); //get the  list of neighbours for a unode
        num_nbs = nbnodes1.size();
        if(num_nbs>tmp)
	    { 
	        tmp = num_nbs;	    	   
	    }
    }
    
    *max_nbs = tmp;
    
    if ((OldConc=(double *)malloc(10000*sizeof(double)))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((NewConc=(double *)malloc(10000*sizeof(double)))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((SegLength=(double *)malloc(count*sizeof(double)*tmp))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((SegWidth=(double *)malloc(count*sizeof(double)*tmp))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((Neighbours=(int *)malloc(count*sizeof(int)*tmp))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((Voronoi=(Coords *)malloc(count*sizeof(Coords)*tmp*2))==0)
        OnError("SetupConc",MALLOC_ERR);		
    if ((NbNodeSize=(int *)malloc(count*sizeof(int)*tmp))==0)
        OnError("SetupConc",MALLOC_ERR);	
    if ((NodeIndex=(int *)malloc(count*sizeof(int)*tmp))==0)
        OnError("SetupConc",MALLOC_ERR);    
    
    for (i=0; i<count; i++)
    {     
	 ElleGetUnodeAttribute(unodelist[i],attr_id,&val); 
         
	 a = unodelist[i];
	 NewConc[a] = OldConc[a] = val; 
         
     ElleGetTriPtNeighbours(unodelist[i],nbnodes,node_index,1);
	 ii = nbnodes.size();
	 j=0, k=0;
	 NbNodeSize[i] = ii;
	 
	 
	 while(j<tmp)
	 {
	       Neighbours[i*tmp+j]= NO_NB;
	       NodeIndex[i*tmp+j]= NO_NB;
	       Voronoi[i*tmp+j].x = NO_NB;
	       Voronoi[i*tmp+j].y = NO_NB;
	       SegLength[i*tmp+j]= 0.0;
	       SegWidth[i*tmp+j]= 0.0;

	       if(j<ii && nbnodes[j]!=NO_NB )
	         {
	          
		  if(node_index[j]==0){
		  for(index=0,found=0;found<1&&index<count;index++){
		    if(unodelist[index]==nbnodes[j])
		      found=1;
		    
		    }
		  if(!found)
		      OnError("nbindex not found",0);
		  }   
	          if(node_index[j]==1){
		  for(index=0,found=0;found<1&&index<bd_size;index++){
		    if(bdnodelist[index]==nbnodes[j])
		      found=1;
		   
		    }
		  if(!found)
		      OnError("nbindex not found",0);
		  }   
		  
		  if(j+1==ii)
		 
SetVoronoi(i*tmp+k,unodelist[i],nbnodes[j],node_index[j],nbnodes[0],node_index[0]);
		  else

SetVoronoi(i*tmp+k,unodelist[i],nbnodes[j],node_index[j],nbnodes[j+1],node_index[j+1]);
		  
		  Neighbours[i*tmp+k]=nbnodes[j];
                  NodeIndex[i*tmp+k]=node_index[j];
		  SegLength[i*tmp+k] =
		  CalcUnodeSep(unodelist[i],nbnodes[j],node_index[j]);
                  
		  if(SegLength[i*tmp+k]<(*minlen))
		    {
		     *minlen = SegLength[i*tmp+k];
		      
		    }
		  
		   k++;
		 }
		
		  j++;
       } 
	 
	 }
}

void SetupBdConc(int flynnid,int attr_id,int *bd_max_nbs)
{ 
    int i,ii, j,jj, k, tmp=1,a,aa, unode_size, index, found;
    int id, num_nbs, count, nb_id, nb_index;
    double val,xx;
    
    
    std::vector<int> bdnodelist;  // create a vector list of unodes
    std::vector<int> nbnodes;
    std::vector<int> node_index;
    std::vector<int> unodelist;
    
    ElleFlynnNodes(flynnid,bdnodelist); // get the list of unodes for a flynn
    count = bdnodelist.size();
    ElleGetFlynnUnodeList(flynnid,unodelist); // get the list of unodes for a flynn
    unode_size = unodelist.size();
     
    for (i=0; i<count; i++) 
    {			
         std::vector<int> nbnodes1;
         std::vector<int> node_index1;
         ElleGetNodeTriPtNeighbours(bdnodelist[i],nbnodes1,node_index1,bdnodelist[i+1],1); 
         if(num_nbs>tmp)
         {
            tmp = num_nbs;	 
         }
    }
    tmp = tmp+1;     
    *bd_max_nbs = tmp;
    
    if ((BdOldConc=(double *)malloc(10000*sizeof(double)))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((BdNewConc=(double *)malloc(10000*sizeof(double)))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((BdSegLength=(double *)malloc(count*sizeof(double)*tmp))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((BdSegWidth=(double *)malloc(count*sizeof(double)*tmp))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((BdNeighbours=(int *)malloc(count*sizeof(int)*tmp))==0)
        OnError("SetupConc",MALLOC_ERR);
    if ((BdVoronoi=(Coords *)malloc(count*sizeof(Coords)*tmp*2))==0)
        OnError("SetupConc",MALLOC_ERR);		
    if ((BdNodeIndex=(int *)malloc(count*sizeof(int)*tmp))==0)
        OnError("SetupConc",MALLOC_ERR);    
    for (i=0; i<count; i++)
        {     
         
	 val = ElleNodeMarginAttribute(bdnodelist[i],CONC_A,flynnid);	          
	 a = bdnodelist[i];
	 BdNewConc[a] = BdOldConc[a] = val; 
         nb_index = (i+1)%count;
	 nb_id = bdnodelist[nb_index];
	 
	 ElleGetNodeTriPtNeighbours(bdnodelist[i],nbnodes,node_index,nb_id,1);
         
	 ii = nbnodes.size();
	 for(jj=0;jj<tmp;jj++){
	    BdVoronoi[i*tmp+jj].x = NO_NB;
	    BdVoronoi[i*tmp+jj].y = NO_NB; 
	 }
	 
	 j=0, k=0;
	 while(j<tmp-1)
	 {


	       BdNeighbours[i*tmp+j]= NO_NB;
	       BdNodeIndex[i*tmp+j]= NO_NB;
	       BdSegLength[i*tmp+j]= 0.0;
	       BdSegWidth[i*tmp+j]= 0.0;
	       if(j<ii && nbnodes[j]!=NO_NB )
	         {
		  if(node_index[j]==0){
		  for(index=0,found=0;found<1&&index<unode_size;index++){
		    if(unodelist[index]==nbnodes[j])
		      found=1;
		    }
		    if(!found)
		      OnError("nbindex not found",0);
		    
		  }   
	          if(node_index[j]==1){
		  for(index=0,found=0;found<1&&index<count;index++){
		    if(bdnodelist[index]==nbnodes[j])
		      found=1;
		    }
		    if(!found)
		      OnError("nbindex not found",0);
		    
		  }   
		          
		  BdNeighbours[i*tmp+k]=nbnodes[j];
          BdNodeIndex[i*tmp+k]=node_index[j];
		  if(j+1<ii){		  
                 
SetBdVoronoi(i*tmp+k,bdnodelist[i],nbnodes[j],node_index[j],nbnodes[j+1],node_index[j+1]);}
		  
		  
		  if(node_index[j]==1)
		      BdSegLength[i*tmp+k] = ElleNodeSeparation(bdnodelist[i],nbnodes[j]);
		  else
		      BdSegLength[i*tmp+k] =
		      CalcUnodeSep(nbnodes[j],bdnodelist[i],1);
		  
		   k++;
		 }
		
		  j++;
         } 	 
	 }	   
}


void CalcConcAtUnode(int flynnid, double dt,int num_nbs)
{
    int i,j,ii,n,count,tmp_nbs,nb_size,index;
    double tmp1, tmp2, tmp3,xx;
    double double_kappa, area, estimate_area, estimate_width; 
    double_kappa=Kappa;
    int mintype; //mwj*** 20/06/2003
    
    ElleGetFlynnIntAttribute(flynnid, &mintype, MINERAL);//mwj*** 20/06/2003
    
    if(mintype==MICA)//mwj*** 20/06/2003
    	double_kappa = Kappa*0.001;
    
    std::vector<int> unodelist;
    ElleGetFlynnUnodeList(flynnid,unodelist); // get the list of unodes for a flynn
    count = unodelist.size();  
    for (ii=0;ii<count;ii++)
    {
        n = unodelist[ii];    
        nb_size= NbNodeSize[ii];
    	tmp_nbs=0;   
    	tmp1 = tmp2 = tmp3 = area = 0.0;
    	for (i=0;i<num_nbs;i++) 
    	{
        	j = Neighbours[ii*num_nbs+i];
        	index = NodeIndex[ii*num_nbs+i];                 
			if (j!=NO_NB) 
			{
            
	    		SegWidth[ii*num_nbs+i]= CalcSegWidth(ii*num_nbs+i,nb_size,i);
	    		area += CalcTriArea(ii*num_nbs+i,i,n,nb_size);
	    
	    		if(index==1)
	    			xx= BdOldConc[j];
	    		else
	    			xx= OldConc[j];
	    
	    		tmp1 += xx*SegWidth[ii*num_nbs+i]/SegLength[ii*num_nbs+i];
            
		  		tmp2 += SegLength[ii*num_nbs+i];
            
	    		tmp3 += SegWidth[ii*num_nbs+i]/SegLength[ii*num_nbs+i];
            	
            	tmp_nbs = tmp_nbs+1;      
			}
    	}
   
    	NewConc[n] =  OldConc[n]*(1 - double_kappa*dt*(tmp3/area))+(double_kappa*dt*tmp1/area);
    }
}
void CalcConcAtNode(int flynnid, double dt,int num_nbs)
{
    int i,j,ii,n,count,tmp_nbs,index;
    double tmp1, tmp2, tmp3, tmp_area;
    double double_kappa, area,xx; 
    double_kappa=Kappa;
    int mintype; //mwj*** 20/06/2003   

    ElleGetFlynnIntAttribute(flynnid, &mintype, MINERAL);//mwj*** 20/06/2003
    
    if(mintype==MICA)//mwj*** 20/06/2003
        double_kappa = Kappa*0.001;
    
    std::vector<int> bdnodelist;
    ElleFlynnNodes(flynnid,bdnodelist); // get the list of unodes for a flynn
    count = bdnodelist.size();  
    for (ii=0;ii<count;ii++)
    {
        n = bdnodelist[ii];    
        tmp_nbs = 0;   
        tmp1 = tmp2 = tmp3 = area= 0.0;
        for (i=0;i<num_nbs-1;i++) 
        {
            j = BdNeighbours[ii*num_nbs+i];
            index = BdNodeIndex[ii*num_nbs+i];                 
    	    if (j!=NO_NB) 
    	    {                
        	    BdSegWidth[ii*num_nbs+i]= CalcBdSegWidth(ii*num_nbs+i);
        	    area += CalcBdTriArea(ii*num_nbs+i,n);
        	    
        	    if(index==1)
        	        xx = BdOldConc[j];
        	    else
        	        xx = OldConc[j];
        	    
        	    tmp1 += xx*BdSegWidth[ii*num_nbs+i]/BdSegLength[ii*num_nbs+i];
                    	    
        	    tmp2 += BdSegLength[ii*num_nbs+i];
                    
        	    tmp3 += BdSegWidth[ii*num_nbs+i]/BdSegLength[ii*num_nbs+i];
                
                tmp_nbs = tmp_nbs+1;        
	        }
        }   
        BdNewConc[n] =  BdOldConc[n]*(1 - double_kappa*dt*(tmp3/area))+(double_kappa*dt*tmp1/area);
    }        
}


void SwapPointers()
{
    double *tmp,*tmp1;

    tmp = OldConc;
    OldConc = NewConc;
    NewConc = tmp;

    tmp1 = BdOldConc;
    BdOldConc = BdNewConc;
    BdNewConc = tmp1;


}

void UpdateConc(int flynnid)
{
    int j,count;
    std::vector<int> unodelist;  // create a vector list of unodes
    ElleGetFlynnUnodeList(flynnid,unodelist); // get the list of unodes for a flynn
    count = unodelist.size();

    for (j=0;j<count;j++) 
    {       
	  ElleSetUnodeAttribute(unodelist[j],CONC_A,NewConc[unodelist[j]]);             
    }  
}

void UpdateBdConc(int flynnid)
{
    int j,count;
    std::vector<int> bdnodelist;  // create a vector list of unodes
    ElleFlynnNodes(flynnid,bdnodelist); // get the list of unodes for a flynn
    count = bdnodelist.size();

    for (j=0;j<count;j++) 
    {
      ElleSetNodeMarginAttribute(bdnodelist[j],BdNewConc[bdnodelist[j]],CONC_A,flynnid);          
    }   
}

double CalcUnodeSep(int unode1, int unode2,int index)
{
   double tmp;
   Coords xy1, xy2;
   
   
   ElleGetUnodePosition(unode1,&xy1);
   
   if (index==1)
       ElleNodePlotXY(unode2,&xy2,&xy1);
   else
       ElleGetUnodePosition(unode2,&xy2);
   
   
   ElleCoordsPlotXY(&xy2,&xy1);
   tmp = pointSeparation(&xy1,&xy2);
                        
   return tmp;
}


void SetVoronoi(int tmp,int unode1,int unode2,int index2,int unode3,int index3)
{
   Coords xy1, xy2, xy3;

   ElleGetUnodePosition(unode1,&xy1);
   
   if(index2==1)
   ElleNodePosition(unode2,&xy2);
   else
   ElleGetUnodePosition(unode2,&xy2);
   
   if(index3==1)
   ElleNodePosition(unode3,&xy3);
   else
   ElleGetUnodePosition(unode3,&xy3);

   ElleCoordsPlotXY(&xy2,&xy1);
   ElleCoordsPlotXY(&xy3,&xy1);
   
   Voronoi[tmp].x = (xy1.x+xy2.x+xy3.x)/3.0;
   Voronoi[tmp].y = (xy1.y+xy2.y+xy3.y)/3.0;
}


void SetBdVoronoi(int tmp,int node1,int node2,int index2,int node3,int index3)
{
   Coords xy1, xy2, xy3;
   ElleNodePosition(node1,&xy1);
   if(index2==1){
      ElleNodePosition(node2,&xy2); 
      ElleCoordsPlotXY(&xy2,&xy1);
      BdVoronoi[tmp].x = (xy1.x+xy2.x)/2.0;
      BdVoronoi[tmp].y = (xy1.y+xy2.y)/2.0;
   } 
   else{
      ElleGetUnodePosition(node2,&xy2);}
   
   if(index3==1){
      ElleNodePosition(node3,&xy3); 
      ElleCoordsPlotXY(&xy3,&xy1);
      BdVoronoi[tmp+2].x = (xy1.x+xy3.x)/2.0;
      BdVoronoi[tmp+2].y = (xy1.y+xy3.y)/2.0;
   } 
   else{
      ElleGetUnodePosition(node3,&xy3);}  
   
   ElleCoordsPlotXY(&xy2,&xy1);
   ElleCoordsPlotXY(&xy3,&xy1);

   BdVoronoi[tmp+1].x = (xy1.x+xy2.x+xy3.x)/3.0;
   BdVoronoi[tmp+1].y = (xy1.y+xy2.y+xy3.y)/3.0;

}

double CalcSegWidth(int num,int num_nbs,int i)
{

   double tmp; 
   Coords xy1, xy2;
   
   xy1 = Voronoi[num];
    
   if(i==0)
        xy2 = Voronoi[num+num_nbs-1];
   else
        xy2 = Voronoi[num-1];
   
   tmp = pointSeparation(&xy1, &xy2);
   return tmp;
   
}

double CalcTriArea(int num,int i,int unode1,int num_nbs)
{
   
   double a, b, c, s, area;
   Coords xy1, xy2, xy3;
  
   ElleGetUnodePosition(unode1,&xy1);
   
   xy2 = Voronoi[num];
   if(i==0)
        xy3 = Voronoi[num+num_nbs-1];
   else
        xy3 = Voronoi[num-1];
   
   a = pointSeparation(&xy1, &xy2);
   b = pointSeparation(&xy2, &xy3);
   c = pointSeparation(&xy3, &xy1);

   s = (a+b+c)/2.0;
   area = sqrt(s*(s-a)*(s-b)*(s-c));
   
   return area;
}


double CalcBdSegWidth(int num)
{

   double tmp; 
   Coords xy1, xy2;
   
   xy1 = BdVoronoi[num];
   
   xy2 = BdVoronoi[num+1];
   
   tmp = pointSeparation(&xy1, &xy2);
   return tmp;
   
}

double CalcBdTriArea(int num,int node1)
{
   
   double a, b, c, s, area;
   Coords xy1, xy2, xy3;
  
   ElleNodePosition(node1,&xy1);
   
   xy2 = BdVoronoi[num];
   
   xy3 = BdVoronoi[num+1];
   
   a = pointSeparation(&xy1, &xy2);
   b = pointSeparation(&xy2, &xy3);
   c = pointSeparation(&xy3, &xy1);

   s = (a+b+c)/2.0; //Youngdo 23/06/2003
   area = sqrt(s*(s-a)*(s-b)*(s-c));
   
   return area;
}

void ExchangeReaction()
{

   int i,ii,j,max_flynns,count;
   double l_val,b_val,min_energy,bt_min_energy,grt_min_energy,grt_mole,bt_mole;
   double init_Bt_conc,init_Grt_conc,T,T_K,k;
   std::vector<int> bdnodelist;
   int mintype; //mwj*** 20/06/2003   
   int randflynns[10000],targetflynns[10000]; // arrays for randomised flynn numbers mwj*** 21/06/2003
   static int first_ran=0; // flag to check for first time through function mwj*** 21/06/2003
   int test_ran,no_active_flynns=0; // test of random flynn, number of reactive flynns  mwj*** 21/06/2003
   
/*
 ****************
 * changed this to use random in libelle
 * but generator is initialised in ElleInit() so do we want it again??
 ****************
 */
   if(first_ran==0) //  initialise random number generator mwj*** 21/06/2003
   {
   		/*srand(currenttime());  			*/
        ElleInitRandom((unsigned long)currenttime());
   		first_ran=1;
   		
   }
   max_flynns = ElleMaxFlynns();

   for(i=0;i<max_flynns;i++)
   {
		if(ElleFlynnIsActive(i) && ReactionIsActive(i)) //  need to store flynn in random list mwj*** 21/06/2003
		{
			targetflynns[i]=-1;
			no_active_flynns++;
		}
		else
			targetflynns[i]=0;
   }
		
   for(i=0;i<no_active_flynns;i++) //  create random flynn id list mwj*** 21/06/2003
   {
		do
		{
			test_ran=(int)(ElleRandomD()*max_flynns);
		}while(targetflynns[test_ran] != -1 );
		
		randflynns[i]=test_ran;
		targetflynns[test_ran]=0;
   }
 
   init_Bt_conc = CalcInitMinConc(MICA); //calculate the average conc. of Bt margin mwj*** 21/06/2003
   init_Grt_conc = CalcInitMinConc(GARNET);//calculate the average conc. of Grt margin mwj*** 21/06/2003
   printf("init_Bt_conc=%.8e\n",init_Bt_conc);
   printf("init_Grt_conc=%.8e\n",init_Grt_conc);
   grt_mole = CalcGrtMole(init_Grt_conc);
   bt_mole = CalcBtMole(init_Bt_conc);
   printf("grt_mole=%.8e\n",grt_mole);
   printf("bt_mole=%.8e\n",bt_mole);
   T = ElleTemperature();
   printf("T = %.8e\n",T);
   T_K = T +273.15;
      
   GrtBtCalcMinEnergy(init_Bt_conc,init_Grt_conc,T_K,grt_mole,bt_mole,&grt_min_energy,&bt_min_energy); //mwj*** 21/06/2003
   printf("bt_min_energy=%.8e\n",bt_min_energy);
   printf("grt_min_energy=%.8e\n",grt_min_energy);
   
   for (ii=0;ii<no_active_flynns;ii++)   // cycle through flynns
	{    
		i=randflynns[ii]; //  get flynn from randomised list mwj*** 21/06/2003
		
        if(ElleFlynnIsActive(i)&& ReactionIsActive(i))
        {
        
	
			min_energy = CalcMinEnergy(i,bt_min_energy,grt_min_energy);

			ElleFlynnNodes(i,bdnodelist); // get the list of nodes for a flynn
        	count = bdnodelist.size();           
         	
         	if ((LatticeConc=(double *)malloc(count*sizeof(double)))==0)
         		OnError("SetupConc",MALLOC_ERR);
         	if ((BoundaryConc=(double *)malloc(count*sizeof(double)))==0)
         		OnError("SetupConc",MALLOC_ERR);
        
	
        	for(j=0;j<count;j++)
        	{ 
        	    l_val = ElleNodeMarginAttribute(bdnodelist[j],CONC_A,i);	          
        	    LatticeConc[j]= l_val; 
                b_val = ElleNodeAttribute(bdnodelist[j],CONC_A);
        	    BoundaryConc[j]= b_val;
            }
        
        
            ElleGetFlynnIntAttribute(i, &mintype, MINERAL);//mwj*** 20/06/2003
        	if(mintype==GARNET)//mwj*** 20/06/2003 
        		k = G_RC;
        	else
        		k = B_RC;
        	
        	CalcExchange(count,min_energy,k);
            for(j=0;j<count;j++)
        	{
        	    ElleSetNodeAttribute(bdnodelist[j],BoundaryConc[j],CONC_A);
                ElleSetNodeMarginAttribute(bdnodelist[j],LatticeConc[j],CONC_A,i);
            } 
                
        	for(j=0;j<count;j++) 
        		bdnodelist.pop_back();
        }       
        CleanupExchangeArrays();
       
    }
}

void CalcExchange(int count,double min_energy,double K)
{
    
   int j;
   double amount_of_Ex,tmp_val;
    
   for(j=0;j<count;j++)
   { 
        amount_of_Ex = (LatticeConc[j]-min_energy)*K;
    	if(amount_of_Ex>=0)
    	{
    	    if((0.1-BoundaryConc[j])<amount_of_Ex)
    	        amount_of_Ex = 0.1- BoundaryConc[j];
    	 
    	    LatticeConc[j]=LatticeConc[j]-amount_of_Ex;
    	    BoundaryConc[j]=BoundaryConc[j]+amount_of_Ex; 
    	}
    	if(amount_of_Ex<0)
    	{
    	    tmp_val = -amount_of_Ex;
    	    if(BoundaryConc[j]<tmp_val) 
                amount_of_Ex = BoundaryConc[j];
    	    else 
    	        amount_of_Ex = tmp_val;    	 
    	    LatticeConc[j] = LatticeConc[j]+amount_of_Ex;
    	    BoundaryConc[j] = BoundaryConc[j]-amount_of_Ex;
    	}
    }
} 
 
void CleanupExchangeArrays()      
{
   if (LatticeConc) {free(LatticeConc); LatticeConc=0;}
   if (BoundaryConc) {free(BoundaryConc); BoundaryConc=0;}

}

double CalcMinEnergy(int flynnid,double bt_min_energy, double grt_min_energy)
{
   int mintype; //mwj*** 20/06/2003   
   ElleGetFlynnIntAttribute(flynnid, &mintype, MINERAL);//mwj*** 20/06/2003
   
   if(mintype==GARNET)//mwj*** 20/06/2003 
   	    return grt_min_energy;
   if(mintype==MICA)//mwj*** 20/06/2003 
	    return bt_min_energy;

}

int ReactionIsActive(int flynnid)
{
   int mintype; //mwj*** 20/06/2003   
   ElleGetFlynnIntAttribute(flynnid, &mintype, MINERAL);//mwj*** 20/06/2003
  
   if(mintype==GARNET || mintype==MICA) //mwj*** 20/06/2003
   	    return 1;
   else
   	    return 0;
}

double GrtBtCalcMinEnergy(double init_Bt_conc,double init_Grt_conc,double T, double grt_mole, double Bt_mole, double *sgc, double *sbc)
{
   double ln_KD, KD, grt_change, bt_change, Bt_mf, Grt_mf, term_1, term_2, term_3, a, b;
   double stable_Bt_conc, stable_Bt_conc1, stable_Bt_conc2, stable_Grt_conc; 
   
   ln_KD = -2109.0/T + 0.782;
   KD = exp(ln_KD);   
   
   Bt_mf = Bt_mole/(Bt_mole+grt_mole);
   Grt_mf = grt_mole/(Bt_mole+grt_mole);   
   grt_change = 0.01;
   bt_change = -(grt_change*(1.0/Bt_mf))*Grt_mf;
   printf("bt_change=%.8e\n",bt_change);
   
   a = (init_Grt_conc-(init_Grt_conc-grt_change))/(init_Bt_conc-(init_Bt_conc-bt_change));
   b = init_Grt_conc-(a*init_Bt_conc);
   
   term_1 = a*(KD-1);
   term_2 = b*(KD-1)-a*KD+1;
   term_3 = -b*KD;
   
   
   stable_Bt_conc1 = (-term_2+sqrt(term_2*term_2-4.0*term_1*term_3))/(2.0*term_1);
   stable_Bt_conc2 = (-term_2-sqrt(term_2*term_2-4.0*term_1*term_3))/(2.0*term_1);
 
   if(stable_Bt_conc1<0)
        stable_Bt_conc = stable_Bt_conc2;
   else
        stable_Bt_conc = stable_Bt_conc1;
   
   stable_Grt_conc = a*stable_Bt_conc+b;
   
   *sgc=stable_Grt_conc;
   *sbc=stable_Bt_conc;
}

double CalcInitMinConc(int minid)
{
    int i,j,max_flynns,count,temp_count=0;
    double conc_x,conc_sum=0,conc_average; 
    std::vector<int> bdnodelist;
    int mintype; //mwj*** 20/06/2003   
    
    max_flynns = ElleMaxFlynns();
    for(i=0;i<max_flynns;i++)
    {
        if(ElleFlynnIsActive(i)) //mwj*** 20/06/2003
        {
        	ElleGetFlynnIntAttribute(i, &mintype, MINERAL); //mwj*** 20/06/2003
            if(mintype==minid) //mwj*** 20/06/2003
        	{
        	 	ElleFlynnNodes(i,bdnodelist); // get the list of nodes for a flynn
                count = bdnodelist.size();           
                if ((LatticeConc=(double *)malloc(count*sizeof(double)))==0)
                 	OnError("SetupConc",MALLOC_ERR);
                for(j=0;j<count;j++)
        	    {
        	     	if(ElleNodeIsActive(bdnodelist[j]))//mwj*** 20/06/2003
        	     	{
        	     		conc_x = ElleNodeMarginAttribute(bdnodelist[j],CONC_A,i);	          
        	     		conc_sum = conc_sum + conc_x; 
        				//printf("flynn, node, conc_x: %d %d %le\n",i, bdnodelist[j],conc_x);
        	     	}   
        	    }           
        		temp_count = temp_count+count;

        	}
    	}
    conc_average = conc_sum/temp_count;       
    }

    return conc_average; 

}

double CalcGrtMole(double init_Grt_conc)
{
    
    double volume, density, mass, molecular_weight, mole_grt;
    
    volume = Grt_area*ElleUnitLength();
    density = 3.58+(init_Grt_conc*0.74);
    mass = density*volume;
    molecular_weight = 403.13+(94.627*init_Grt_conc); 
    mole_grt = mass/ molecular_weight;

    return mole_grt; 

}

double CalcBtMole(double init_Bt_conc)
{
 
    double volume, density, mass, molecular_weight, mole_bt;
    
    volume = Bt_area*ElleUnitLength();
    density = 2.86+(init_Bt_conc*0.34);
    mass = density*volume;
    molecular_weight = 417.13+(94.618*init_Bt_conc); 
    mole_bt = mass/ molecular_weight;
    
    return mole_bt; 
}

double FlynnBoundaryArea(int flynnid) //mwj*** 25/06/2003
{
	int i,index,j,nbs[3],m,n,count,bd_size,tmp_nb=0,num_nbs;
	double area=0.0;
    std::vector<int> bdnodelist;  // create a vector list of unodes
    std::vector<int> unodelist;  // create a vector list of unodes
    
    ElleFlynnNodes(flynnid,bdnodelist);
    bd_size = bdnodelist.size();
    ElleGetFlynnUnodeList(flynnid,unodelist); // get the list of unodes for a flynn
    count = unodelist.size();
    
    for (i=0; i<count; i++) 
    {					
	    std::vector<int> nbnodes1;
        std::vector<int> node_index1;
	    ElleGetTriPtNeighbours(unodelist[i],nbnodes1,node_index1,1); //get the  list of neighbours for a unode
        tmp_nb = nbnodes1.size();
        if(tmp_nb>num_nbs)
	    { 
	        num_nbs = tmp_nb;	    	   
	    }
    }
    std::cout << "num nbs: " << num_nbs << std::endl;
    std::cout << "count: " << count << std::endl;
    std::cout << "flynnid: " << flynnid << std::endl;
    for (i=0;i<count;i++)
    {
        n = bdnodelist[i];    
        for (i=0;i<num_nbs-1;i++) 
        {
            j = BdNeighbours[i*num_nbs+i];
            index = BdNodeIndex[i*num_nbs+i];                 
    	    if (j!=NO_NB) 
    	    {                
        	    area += CalcBdTriArea(i*num_nbs+i,n);
   	        }
        }   
    } 
	return area;
}

double TotalBoundaryArea(int minid) //mwj*** 25/06/2003
{
    int i,max_flynns;
    int mintype;  
    double total=0.0;
    
    max_flynns = ElleMaxFlynns();
    for(i=0;i<max_flynns;i++)
    {
    	if(ElleFlynnIsActive(i))
    	{
    	        ElleGetFlynnIntAttribute(i, &mintype, MINERAL); 
    	        if(minid==mintype)
    	        	total+=FlynnBoundaryArea(i);
    	}     		
    }
	return total;	
}
