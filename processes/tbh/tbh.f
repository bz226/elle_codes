      subroutine tbh(etensor,ermap,ework,eerr,fncrss,fnxlinfo)
c-------------------------------------------------------               
c     tbh + drx  + gg  + anneal + dynamic hardening  (tbh v.3)   oct 1988                     
c--------------------------------------------------------               
c--------documentation of input output characteristics in setup---------
      implicit real * 8 (d),logical (q)                                 
      real*8 etensor(4),ermap(9),ework
      integer*4 eerr
      character*32 fncrss,fnxlinfo
      logical qerror,qbasis,basic                                       
      real * 8 cost,matrix,rhs,yields,pivec,slips                       
      real  defpaf,dummy,deviat                                         
      dimension stress(3,3), scrat(3,3)
      integer map(102,102)
      integer*2 age(102,102)
      real mob
* variables and constants for increasing the stack size

      integer*4 appllimit                ! address of current aplication limit 
      integer*4 newlimit                ! new application limit
      parameter (appllimit=z'00000130')


      integer ij,kl
      character*32 ifn,lfn,mfn
c      integer*4 window,toolbx
c      include  :include files:toolbx.par
      real u(97), c, cd, cm
      integer i97, j97
      logical test
      common /raset1/ u, c, cd, cm, i97, j97, test
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      common /logic/ qd,qt,qr,qdata,qbasis                              
      common /crystl/ nrep,irepre(20),itypof,rotat(103,3,3)             
c---matrices relate bond axial frame to external coordinates.           
c-----zmap describes orientation of frame after the deformation step.   
c-----rmap describes orientation of frame prior to the deformation step.
c-----ngrain=grain number                                               
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)            
      common /path/ defpaf(400,9),alltau(2,103),nconf,nchang(2)         
      common /storer/ iactiv(15),nactiv,zactiv(10),ngrain,iterat,work   
      common /degen/ nbasic,redcst(103),holds(65,103)                   
      common /param/ itype,kcycle,lcycle,lcrit,number,nom(22,4)         
      common /xtalog/ dalpha,dbeta,dgamma,da,db,dc,dlat(3,3),drecl(3,3) 
      common pi,root2,root3,pi90,pifac,jrand                            
      common /lp1/ matrix(6,103),rhs(6),yields(103),slips(103),pivec(6),
     1cost,qerror                                                       
      common /lp2/ num(103),basic(103)                                  
      common /lp3/ nrow,ncol,nlimit,nsys,nartif 
      common /tbhdrx/ xlmap(3,4,30000)
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
      common /newpar/ itdef,rrate,hrdrate
      common /agecom/ age
      common /print/nvals,linc
    
* by default, applications running on the mac ii are allocated a stack of
* 24k. a fortran program which uses a significant amount of local memory 
* (ie. declares large arrays as local variables) will exceed this limit. the
* following lines show one way of increasing the amount of stack available.

      test = .true.
      ij=1802
      kl=9373
      call rmarin(ij,kl)
      
c                                                                       
c---initialization------------------------------------------------------
c                                                                       
      ngrain=0                                                          
c-----set up crystallographic data and perform initialization etc.      
      qdata=.true.
      ndata=19                                                          
                                                          
      
c----------------------------------------------------------------

c      write(*,910)
c  910 format('what is the input file name?')
c      read(*,920) ifn
c  920 format(a32)
c      open(unit=16,file=ifn,status='old')                                          
c      read(16,911)itrans
c  911 format(i2)



c----------------------------------------------------------------

      itrans=1

c----------------------------------------------------------------
 
      jtrans=1
      qfirst=.true.
      
      call setup (qfirst,jtrans,etensor,ermap,fncrss,fnxlinfo)                                               
c      qdata=.false.
      l=1  
      number=1
      nincs=kcycle
      
      l2=0
      ic=1                                                      
      call realtbh                                                
c      call outit(1)
      
      jinc=0
      do 110 kk=1,3
      do 110 jj=1,3
         jinc=jinc+1
         ermap(jinc)=xlmap(kk,jj,1)
  110 continue

      ework=xlmap(3,4,1)
      eerr=0
      return
      end

c---------------------------------------------------------------------------

      subroutine realtbh
      implicit real * 8 (d),logical (q)                                 
      logical qerror,qbasis,basic                                       
      real * 8 cost,matrix,rhs,yields,pivec,slips                       
      real  defpaf,dummy,deviat                                         
      dimension stress(3,3), scrat(3,3)
      integer map(102,102)
      real mob
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      common /logic/ qd,qt,qr,qdata,qbasis                              
      common /crystl/ nrep,irepre(20),itypof,rotat(103,3,3)             
c---matrices relate bond axial frame to external coordinates.           
c-----zmap describes orientation of frame after the deformation step.   
c-----rmap describes orientation of frame prior to the deformation step.
c-----ngrain=grain number                                               
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)            
      common /path/ defpaf(400,9),alltau(2,103),nconf,nchang(2)         
      common /storer/ iactiv(15),nactiv,zactiv(10),ngrain,iterat,work   
      common /degen/ nbasic,redcst(103),holds(65,103)                   
      common /param/ itype,kcycle,lcycle,lcrit,number,nom(22,4)         
      common /xtalog/ dalpha,dbeta,dgamma,da,db,dc,dlat(3,3),drecl(3,3) 
      common pi,root2,root3,pi90,pifac,jrand                            
      common /lp1/ matrix(6,103),rhs(6),yields(103),slips(103),pivec(6),
     1cost,qerror                                                       
      common /lp2/ num(103),basic(103)                                  
      common /lp3/ nrow,ncol,nlimit,nsys,nartif 
      common /tbhdrx/ xlmap(3,4,30000)
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
      common /print/nvals,linc
      
      zpetbn=0.02e0                                                     
      dpetbn=0.08d0                                                     
      iconf=2                                                           
      qfirst=.false.
      linc=l

c                                                                       
c-----------------------------------------------------------------------
c            ---major loop-grain by grain deformation---                
c-----------------------------------------------------------------------
c                                                                       
      do 300 nvals=1,number                                             
      if(xlmap(1,4,nvals).eq.0) goto 280
      qbasis=.false.                                                    
      nconf=1                                                           
      call reqx(nvals)      
      iterat=l
      if (qr) write (nlist,160) ngrain                                  
160   format(5x,5hgrain,i4,2x,20(1h=))                                  

      qbasis=.false.                                                                                                                             
      if (qr) write (nlist,170) iterat                                  
170   format (1h+,30x,10(1h-),9hiteration,i4,1x,10(1h-))                
c-----critical resolved shear stresses for yield may alter.             
      qset=iterat.ne.1.and.iterat.lt.nchang(nconf)                      
      if(qset) go to 150                                                
c-----initialize yield values                                           
      if (nconf.gt.iconf)  write (nlist,175)                            
175   format(10x,'error in number of allowed configuration changes')    
c-----initialize object function for lp procedures                      
c-----initialize num at same time-flags basic variables                 
      do 40 j=1,nsys                                                    
      num(j)=j                                                          
      dummy=alltau(nconf,j)                                             
      if(dummy.gt.1.0e6) num(j)=-j                                      
      yields(j)=dummy                                                   
40    continue                                                          
      if (lcrit.eq.0) go to 150                                         
c-----perturb yield values in attempt to remove degeneracy              
      do 140 jk=1,nsys                                                  
      dz=1.0+randz(xr)*zpetbn                                            
      yields(jk)=yields(jk)*dz                                          
140   continue                                                          
      nconf=nconf+1                                                     
150   continue                                                          
c-----form the infinitesimal strain tensor and the rigid body           
c-----rotation tensor.crystal axes now define reference frame and       
c-----tensors are subjected to appropriate similarity transform.        
c-----defpaf and grad initially wrt external frame                      
c-----rmap is rotation matrix mapping external frame to bond frame.     
      jk=0                                                              
      do 180 j=1,3                                                      
      do 180 k=1,3                                                      
      jk=jk+1                                                           
      grad(j,k)=defpaf(iterat,jk)                                      
180   continue                                                          
      call trans (rmap,scrat)                                           
      call mult (grad,scrat,grad)                                       
      call mult (rmap,grad,grad)                                        
      if (qd) write (nlist,190)                                         
      do 181 k=1,3                                                      
      if (qd)  write(nlist,191) (rmap(k,j),j=1,3),                      
     1     (grad(k,j),j=1,3)                                            
181   continue                                                          
190   format (10x,'bond reference frame orientation',15x,               
     1     'deformation gradient w.r.t. bond frame')                    
191   format (10x,3f9.4,15x,3f9.4)                                      
      jk=0                                                              
      do 200 j=1,3                                                      
      grad(j,j)=grad(j,j)-1.0                                           
      do 200 k=j,3                                                      
      vort(j,k)=0.5*(grad(j,k)-grad(k,j))                               
      vort(k,j)=-vort(j,k)                                              
      jk=jk+1                                                           
      rhs(jk)=0.5*(grad(j,k)+grad(k,j))                                 
200   continue                                                          
c-----perform lp                                                        
c-----assume warm start for calculation                                 
      call lp                                                           
      if(qerror) write (nlist,610)                                      
610   format (10x,'-----error termination from lp procs in main-----')  
c-----flag activated mechanisms-calculate reduced cost.                 
      call flag                                                         
      if (.not.qd) go to 230                                            
      jk=0                                                              
c-----stress state w.r.t. bond frame.                                   
      stress (3,3) = 0.00e0                                             
      pivec(6) = 0.00                                                   
      do 210 j=1,3                                                      
      do 210 k=j,3                                                      
      jk=jk+1                                                           
      z=pivec(jk)                                                       
      stress(j,k)=+z                                                    
      stress(k,j)=+z                                                    
210   continue                                                          
      deviat=(stress(1,1)+stress(2,2)+stress(3,3))/3.0                  
      stress(1,1)=stress(1,1)-deviat                                    
      stress(2,2)=stress(2,2)-deviat                                    
      stress(3,3)=stress(3,3)-deviat                                    
      if (qr) write (nlist,220) ((stress(ko,kj),ko=1,3),kj=1,3)                 
220   format (10x,'stress causing deformation'/(10x,3f12.4))            
230   continue                                                          
c---specify degeneracy in solutions to taylor analysis                  
c       to each solution of the stress tensor                           
c       corresponds the same set of solutions for                       
c       activity on the shear mechanisms.                               
c-------commence lp with perturbation of the yield surface.             
      if (itype.ne.1.and.nactiv.eq.5) go to 250                         
      call degena                                                       
      if (itype.eq.1) go to 270                                         
c-----choose solution and bias                                          
      nsoln=randz(xr)*float(nbasic)+1.00                                 
      do 240 j=1,nsys                                                   
      slips(j)=holds(nsoln,j)                                           
      yields(j)=yields(j)*(1.0d0-slips(j)*dpetbn)                       
240   continue                                                          
250   continue                                                          
c-----work done                                                         
      work=cost
      xlmap(3,4,nvals)=work
c-----update grain orientation and output results on unit ndata if cycle
      call zipzap                                                       
      call mult (zmap,rmap,zmap)                                        
      if (qdata) call drawit                                            
      do 260 j=1,3                                                      
      do 260 k=1,3                                                      
260   rmap(j,k)=zmap(j,k)                                               
      call xeqr(nvals)
  270 continue                                                          
      if (qd) write (nlist,290)                                         
290   format (10x,40(1h=))                                              
280   continue 
300   continue                                                          
c-----------------------------------------------------------------------
      if (qr) write(nlist,777)
  777 format('9.99999')
      call daytim (idate,itime)                                         
      if (qr) write (nlist,360) ndata                                           
  360 format (10x,'data file',i6,5x,'used'/                             
     1     10x,'end of fabric simulation increment'/10x,40(1h=))                  
      return
90    if (qr) write (nlist,620 )         ngrain                                 
620   format (10x,'end of file encountered-only',i4,2x,'input sets')    
      
      return
      end
c******************************************************************************                                                                        
      subroutine setup (qfirst,jtrans,etensor,ermap,fncrss,fnxlinfo)                                         
c---documentation for setup routine.                                    
c     programs use the taylor-bishop-hill approach to simulate          
c     the development of prferred orientation in a plastically          
c     deforming polycrystalline mass.one data set represents one grain  
c     in the polycrystalline mass.                                      
c---routine sets up crystallographic information for lp calculation.    
c---input                                                               
card1                                      ------- format 7l1-------    
c    columns 1-3 contain t or f to initialize logical variables.        
c          col1   t     listing for terminal monitoring                 
c          col2   t     run time information                            
c          col3   t     full dump listing of calculation                
card2 itype,kcycle,lcycle,lcrit            -----format 4i4-------       
c          itype -method of program operation.                          
c          kcycle-number of deformation steps.                          
c          lcycle-frequency at which plot diagrams produced.            
c          lcrit -criterion to be used to allow unique specification    
c          of possible crystal rotations.                               
cols 1-4 0000       uniaxial strain is assumed and the orientation of   
c     the principal axis of strain for each data point must be input.   
c---itype=2                                                             
cols 1-4 0001 to 0999        no assumptions made about total strain.    
c      number   data points input as  rotation matrices describing      
c      orientation of the crystal.output from program can be re-input   
c      in this mode.                                                    
c---itype=3                                                             
cols 1-4 -001 to -999      no assumptions made about total strain.      
c     magnitude of number determines how many random relative           
c     orientations will be generated.                                   
c---lcrit                                                               
c   in general more than one slip activity vector can be specified      
c   as an optimal basic feasible solution.the set of all slip           
c   activity vectors is the set of convex linear combinations           
c   of the set of optimal basic feasible solutions.                     
c     fabric diagrams are produced according to one of 5 criteria       
c   which invoke random selections.                                     
c---lcrit=1        assume minimum number of slip systems activated.     
c            slip activity vector is one of basic feasible solutions    
c            chosen at random from the set of optimal basic solutions.  
c---lcrit=2        assume that slip activity vector is a convex         
c            linear combination of the optimal basic feasible solution  
c            slip activity vectors,chosen at random.                    
c---lcrit=3        assume slip activity vector chosen according to 1.   
c            from this point yield streses for each system are increased
c            stepwise in proportion to yield stress and decreased       
c            stepwise in proportion to previous slip activity on system.
c---lcrit=4        assume slip activity vector chosen according to 1.   
c            from this point yield streses for each system are increased
c            stepwise in proportion to yield stress and increased       
c            stepwise in proportion to previous slip activity on system.
c---lcrit=5        set of yield values peturbed randomly,removing       
c            degeneracy in general.                                     
c-----------------------------------------------------------------------
card3 alpha,beta,gamma,a,b,c -specification of crystallographic axes.   
c                                          -----format 6f10.0----       
card4 type of point group as below.        -----format i4--------       
c       0001 holosymmetric cubic                                        
c       0002 trigonal point groups with 60 degree primitive segment.    
c       0003 trigonal point groups with 120 degree segment.             
card5 number of operations in the point group                           
cards following                            -----format i2,3i2,i3-----   
c       information to generate point group rotation operations.        
card following  lists number of symmetry sets and number of first in set
c                                              --------format 21 i 3 ---
cards following  number nrep+2----title and mechanisms identified.      
c     nrep is the number of symmetry sets                               
cards following                        ----------format 20 i3---------- 
c     information concerning shear mechanisms                           
c     blank card terminates entry                                       
cards following-blank card terminates                                   
c    itype=1  orientation of uniaxial strain axes-format specified      
c    itype=2  no cards to be input here                                 
c    itype=3  no cards to be input here                                 
cards following                                                         
c     specification of critical resolved shear stresses for yield       
c     on shear mechanisms.  default=1.0e10                              
c        col 1-3   number of the yield surface configuration            
c        8(i3,f6.2)       number of symmetry set-yield value            
c        default 1.0e10                                                 
c        blank card terminates                                          
cards following                                  -----  i3,9f6.4  ----- 
c        specification of the deformation gradient for each deformation 
c        step.at each change in the deformation path a gradient must be 
c        input.                                                         
c----successive yield configurations may follow if itype=1.             
c---input/output  devices ----------------------------------------------
c-------logical unit ntemp=11                                           
c-----used as a temporary output device if an inverse pole figure is    
c-----to be prepared.  otherwise it can be used as an input device      
c-----if a pre-existing orientation distribution is to be deformed.     
c-----in the case of normal usage..itype=3...it can be a dummy unit.    
c-------logical unit ndata=12                                           
c-----stores a duplicate data matrix,and the results of the analysis.   
c-----holds rotation matrices,work done,active systems,activities.      
c-------logical unit nread=12                                           
c-----used for input device.                                            
c-------logical unit nlist=6                                            
c-------logical unit nread2=5
c-------logical unit nelle=17                                           
c-----------------------------------------------------------------------
c-----the taylor analysis is set up as the primal lp problem.           
c---the solution of the bishop hill analysis is stored in               
c   the pi vector and is the solution of the dual lp problem            
c---linear programming is carried out using a package supplied          
c   by the anu computer centre - courtesy of david anderson             
c   this package was translated from algolw by anderson and made        
c   operational on the 360/50 and univac 1108 systems. the original     
c   subroutines were written by mark saunders,stanford university,and   
c   are documented in stanford university technical report.             
c   anu documentation-in preparation.                                   
c   a great number of modifications have been made to the package for   
c   usage by this program.                                              
c-----------------------------------------------------------------------
c                                                                       
c                                                                       
c                                                                       
      implicit real * 8 (d),logical (q)                                 
      real *8 etensor(4), ermap(9)
      character*32 fnxlinfo,fncrss
      real * 8 cost,matrix,rhs,yields,pivec,slips                       
      real   defpaf                                                     
      dimension dstryn (3,3),axis(3),daxis(3),dpole(3),dline(3)         
      dimension iline(8),value(8),icard(20)                             
      dimension card(20),dmap(3,3)
      integer map(102,102)
      real mob
      character*32 ifn,lfn,mfn,efn,ofn
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      common /logic/ qd,qt,qr,qdata,qbasis                              
      common /crystl/ nrep,irepre(20),itypof,rotat(103,3,3)             
c-----zmap describes orientation of frame after the deformation step.   
c-----rmap describes orientation of frame prior to the deformation step.
c-----ngrain=grain number                                               
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)            
      common /path/ defpaf(400,9),alltau(2,103),nconf,nchang(2)         
      common /xtalog/ dalpha,dbeta,dgamma,da,db,dc,dlat(3,3),drecl(3,3) 
      common /storer/ iactiv(15),nactiv,zactiv(10),ngrain,iterat,work   
      common /degen/ nbasic,redcst(103),holds(65,103)                   
      common /param/ itype,kcycle,lcycle,lcrit,number,nom(22,4)         
      common pi,root2,root3,pi90,pifac,jrand                            
      common /lp1/ matrix(6,103),rhs(6),yields(103),slips(103),pivec(6),
     1     cost,qerror                                                  
      common /lp2/ num(103),qbasic(103)                                 
      common /lp3/ nrow,ncol,nlimit,nsys,nartif
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
      common /newpar/ itdef,rrate,hrdrate
      common /tbhdrx/ xlmap(3,4,30000)
      
      
      
c------in the case of wanting to produce multiple inverse pole figures  
      if(.not.qfirst) go to 470                                         
c                                                                       
c---initialization------------------------------------------------------
c                                                                       

  920 format(a32)
      
      if(jtrans .eq. 1) then
        open(unit=15,file=fnxlinfo,status='old',iostat=ios)
        if (ios.ne.0) then
          print *,'Cannot open file ',fnxlinfo
        endif
        open(unit=16,file=fncrss,status='old',iostat=ios)
        if (ios.ne.0) then
          print *,'Cannot open file ',fncrss
        endif
        rewind 15
        rewind 16
        open(unit=19,file='tbhellexl',status='old',iostat=ios)
        if (ios.ne.0) open(unit=19,file='tbhellexl',status='new')
      else
        close(unit=15)
        read(16,920) ifn
        open(unit=15,file=ifn,status='old')                                          
      endif
                                           
      nread=15                                                          
      nread2=15                                                         
      nlist=19                                                          
      ntemp=18                                                          
      ndata=19
      nelle=17                                                          
      nout=20

c      read (nelle,100) ((xlmap(kk,jj,1),jj=1,3),kk=1,3)                       
c100   format (9f8.5)                                                   
      
      jinc=0
      do 100 kk=1,3
      do 100 jj=1,3
         jinc=jinc+1
         xlmap(kk,jj,1)=ermap(jinc)
  100 continue

      xlmap(1,4,1)=1                               


      qparam=.false.                                                    
      ngrain=0                                                          
      root3=sqrt(3.0)                                                   
      root2=sqrt(2.0)                                                   
      pi=dasin(1.0d0)*2.0d0                                             
      pifac=pi/180.0                                                    
      nrow=5                                                            
      nlimit=103                                                        
      maxcyc=400                                                        
      nset=20 
      qdata=.true.
c-----zero stores                                                       
      do 5 j=1,nlimit                                                   
      yields(j)=1.0e10                                                  
      num(j)=0                                                          
      slips(j)=0.00                                                     
      qbasic(j)=.false.                                                 
      do 5 k=1,nrow                                                     
      matrix(k,j)=0.00                                                  
    5 continue                                                          
c---specification of options. 
      read(nread,11) qt,qr,qd
      if (qr) write (ndata,11) qt,qr,qd
 11   format(3L1)                                                      
C11   format(8l1)                                                      
c---method of tbh program operation.                                        
      read (nread,21) itype,kcycle,lcycle,ntotal,nchang                 
      if (qr) write (ndata,21) itype,kcycle,lcycle,ntotal,nchang                
 21   format (8i4)                                                      
c---method of drx program operation.
      if(jtrans.eq.1) nxl=ntotal
      porb=0.5
      read (nread,22) mob,pnuke,angnuke,noex,iex,gbemob                 
      if (qr) write (ndata,22) mob,pnuke,angnuke,noex,iex,gbemob                
22    format (f5.1,1x,f5.3,1x,f5.1,1x,i1,1x,i4,1x,f5.2)                                                      
      read (nread,23) itdef,rrate,hrdrate                
      if (qr) write (ndata,23) itdef,rrate,hrdrate                
23    format (i4,1x,f5.3,1x,f5.3)                                                      
      do 2 j=1,2                                                        
2     if(nchang(j).eq.0) nchang(j)=1000                                 
c-----lcrit fixed at five in this version                               
      lcrit=5                                                           
      if(itype.eq.0) lcrit=0                                            
      if (kcycle.gt.maxcyc) go to 690                                   
      if(jtrans.eq.1) number=ntotal
      
      if(jtrans.eq.1) then
          if(number.eq.0) then
              number=1
        ntotal=1
          endif
      endif
      
      if (itype) 10,30,20                                               
10    itype=3                                                           
      go to 40                                                          
20    itype=2                                                           
      go to 40                                                          
30    itype=1                                                           
      kcycle=1                                                          
      lcycle=1                                                          
      lcrit=0                                                           
      qparam=.false.                                                    
40    continue                                                          
                                                       
c-----specify date and time                                             
      if (qr) write(nlist,50)                                                   
   50 format (10(/),9x,'lp program-----leiden version three')           
      call daytim(idate,itime)                                          
c---initialize random number generator                                  
      jset=mod(itime,2000)+1                                               
      do 633 j=1,jset                                                     
      xxr=randz(xr)                                                       
633   continue                                                          
c---specify crystallographic information.                               
      read (nread,129) dalpha,dbeta,dgamma,da,db,dc                     
      if (qr) write (ndata,129) dalpha,dbeta,dgamma,da,db,dc                    
129   format (6f10.4)                                                   
      call begina                                                       
      read (nread,120) itypof                                           
      if (qr) write (ndata,120) itypof                                          
120   format (i4)                                                       
c-----dummy read of point group                                         
      read (nread,130) nops                                             
      if (qr) write (ndata,130) nops                                            
      do 101 j=1,nops                                                   
      read (nread,125) la,lb,lc,ld,le                                   
      if (qr) write (ndata,125) la,lb,lc,ld,le                                  
125   format (4i2,i3)                                                   
101   continue                                                          
c---specification of representative mechanisms.                         
      read (nread,130) nrep,irepre                                      
      if (qr) write (ndata,130) nrep,irepre                                     
130   format (21i3)                                                     
      if (nrep.gt.nset-1)  go to 716                                    
c-----read title and the text for mechanism identification.             
      never=nrep+1                                                      
      do 102 j=1,never                                                  
      read (nread,103) (nom(j,ko),ko=1,4)                               
      if (qr) write (ndata,103) (nom(j,ko),ko=1,4)                              
  103 format (20a4)                                                     
  102 continue                                                          
c---generate strain tensors associated with shear mechanisms            
c---shear plane in reciprocal lattice coordinates                       
c---shear direction in lattice coordinates                              
      if (qt.and..not.qd) write (nlist,140)                             
  140 format(10x,40(1h-)//10x,'crystallographic shear mechanisms.')     
      if (qd) write (nlist,141)                                         
  141 format(10x,'crystallographic shear mechanisms.'/                  
     1    8x,'system',6x,'plane',6x,'direction')                        
      nsys=0                                                            
150   continue                                                          
      read (nread,160,end=290) dpole,dline                              
      if (qr) write (ndata,160) dpole,dline                             
160   format (6f3.0)                                                    
      z=dabs(dpole(1))+dabs(dpole(2))+dabs(dpole(3))                    
      if (z.lt.0.000001) go to 290                                      
      nsys=nsys+1                                                       
      if (nsys.gt.nlimit) go to 710                                     
      if (qd) write (nlist,170) nsys,dpole,dline                        
170   format (//10x,'system number',i4/10x,'shear plane',5x,'pole',3f7.2
     1,5x,'direction',3f7.2)                                            
      if (qt.and..not.qd) write (nlist,171) nsys,dpole,dline            
171   format (3x,i5,5x,3f5.1,5x,3f5.1)                                  
      call dchang (dpole,dpole,drecl)                                   
      call dchang (dline,dline,dlat)                                    
      dlenp=dsqrt(dpole(1)**2+dpole(2)**2+dpole(3)**2)                  
      dlenl=dsqrt(dline(1)**2+dline(2)**2+dline(3)**2)                  
      do 220 il=1,3                                                     
      dpole(il)=dpole(il)/dlenp                                         
220   dline(il)=dline(il)/dlenl                                         
      if (qd) write (nlist,230) dpole,dline                             
230   format (10x,'bond coords',5x,'pole',3f7.4,5x,'direction',3f7.4)   
      if (qd) write (nlist,240) nsys                                    
240   format (/6x,'row',i3,3x,'rotation tensor',24x,'strain tensor')    
      do 270 j=1,3                                                      
      do 250 k=1,3                                                      
      dstryn (j,k)=0.5d0*(dpole(k)*dline(j)+dpole(j)*dline(k))          
      rotat(nsys,j,k)=0.5d0*(dpole(k)*dline(j)-dpole(j)*dline(k))       
250   continue                                                          
260   format (10x,3f10.5,10x,3f10.5)                                    
      if (qd) write (nlist,260) (rotat(nsys,j,kj),kj=1,3),              
     1     (dstryn(j,kj),kj=1,3)                                        
270   continue                                                          
      z=dstryn (1,1)+dstryn (2,2)+dstryn (3,3)                          
      if (abs(z).gt.0.000001) go to 749                                 
c---generate the lp matrix                                              
      kl=0                                                              
      do 280 k=1,3                                                      
      do 280 l=k,3                                                      
      kl=kl+1                                                           
      matrix(kl,nsys)=dstryn (k,l)                                      
280   continue                                                          
      go to 150                                                         
290   continue                                                          
      ncol=nsys                                                         
      nrep=nrep+1                                                       
      if (nrep.gt.20) go to 716                                         
      irepre(nrep)=nsys+1                                               
      if (qd) write (nlist,300) (kj,(matrix(ko,kj),ko=1,6),kj=1,nsys)   
300   format (10x,40(1h=)/10x,'lp matrix'/(10x,i4,5x,6f12.6))           
      if (qr) write (nlist,310) nsys                                            
310   format (/10x,i4,3x,'crystallo-plastic mechanisms')                
      ncol=nsys                                                         
      iam=nrep-1                                                        
      if (qr) write (nlist,320)                                                 
320   format (10x,40(1h=))                                              
      if (itype.ne.1) go to 460                                         
c-----input of orientations in terms of inverse pole figure             
c-----specifications of the uniaxial principal axis of strain.          
c          orientation data in two forms as desired.                    
c          if first 12 cols blank orientation of axis can               
c          be specified in lattice coords  20x,3f10.4 format            
c          otherwise specify phi,rho in 2f10.4 format,phi anticlockwise 
c          from south,rho angle from vertical axis.                     
c          terminate data with end of file or blank card                
      if (qr) write (nlist,370)                                         
370   format (10x,40(1h-))                                              
380   read (nread,110,end=450) phi,rho,axis                             
      if (qr) write (ndata,110) phi,rho,axis                                    
110   format (6f10.0)                                                   
      if (abs(phi)+abs(rho).gt.0.00001)  go to 430                      
      z=abs(axis(1))+abs(axis(2))+abs(axis(3))                          
      if (z.lt.0.00001) go to 450                                       
      do 400 jk=1,3                                                     
400   daxis(jk)=axis(jk)                                                
      call dchang (daxis,daxis,dlat)                                    
      do 410 jk=1,3                                                     
410   axis(jk)=daxis(jk)                                                
      call phirho (axis,phi,rho)                                        
      go to 435                                                         
430   continue                                                          
      phi=phi*pifac                                                     
      rho=rho*pifac                                                     
  435 continue                                                          
      daxis(1)=cos(phi+pi/2.0)                                          
      daxis(2)=sin (phi+pi/2.00)                                        
      daxis(3)=0.00d0                                                   
      dro=rho                                                           
      call drotat (dmap,daxis,dro)                                      
      do 439 j=1,3                                                      
      do 439 k=1,3                                                      
      rmap(j,k)=dmap(j,k)                                               
  439 zmap(j,k)=rmap(j,k)                                               
      rho1=rho/pifac                                                    
      phi1=phi/pifac                                                    
      number=number+1                                                   
      if (qr) write (nlist,390) number,phi1,rho1,axis                   
390   format (10x,'initial orientation----grain',i4/                    
     1     10x,'phi',f7.3,6x,'rho',f7.3,6x,'axis',3f8.4)                
      if (qd) write (nlist,440) ((rmap(ko,kj),kj=1,3),ko=1,3)           
440   format (10x,'rotation matrix'/(20x,3f12.5))                       
      write (ntemp,445) ((rmap(k,j),j=1,3),k=1,3)                       
      write (ntemp,445) ((zmap(k,j),j=1,3),k=1,3)                       
      iterat=1                                                          
      write (ntemp,446) number,iterat                                   
445   format(9f8.5)                                                     
446   format(i4,i3)                                                     
      go to 380                                                         
  450 end file ntemp                                                    
      rewind ntemp                                                      
460   continue                                                          
c---initialization of objective function--------------------------------
      do 480 j=1,2                                                      
      do 480 k=1,103                                                    
480   alltau(j,k)=1.0e10                                                
      if (qr) write (nlist,481)                                                 
481   format (10x,'critical shear stresses')                            
470   continue                                                          
      ngrain=0                                                          
      qset=.true.                                                       
      if (qfirst) go to 409                                             
      i99=-99                                                           
      if (qr) write (ndata,487) i99                                             
487   format (//4x,i3)                                                  
c-----here writing previous yield values.                               
      if (qr) write (ndata,481)                                                 
      if (qr) write (ndata,560) (nom(1,ko),ko=1,4)                              
      never=nrep-1                                                      
      do 408 j=1,never                                                  
      iam=j+2                                                           
      jam=irepre(j)                                                     
      if (qr) write (ndata,560) (nom(iam,ko),ko=1,4),                           
     1     (alltau(ko,jam),ko=1,2)                                      
408   continue                                                          
409   continue                                                          
      if (itype.eq.1) go to 482                                         
      read (16,411,end=490) nconf,                                  
     1     (iline(jk),value(jk),jk=1,8)                                 
      if (qr) write (ndata,411) nconf,(iline(jk),value(jk),jk=1,8)              
411   format (i3,8(i3,f6.2))                                            
      go to 483                                                         
482   continue                                                          
      read (16,411,end=660)  nconf,                                 
     1     (iline(jk),value(jk),jk=1,8)                                 
483   continue                                                          
      if(nconf.eq.0) go to 490                                          
      if(nconf.gt.2) go to 728                                          
      do 412 k=1,8                                                      
      if (iline(k).eq.0) go to 412                                      
      neddy=iline(k)                                                    
      if (neddy.gt.nrep-1) go to 726                                    
      jam=irepre(neddy)                                                 
      kam=irepre(neddy+1)-1                                             
      qset=.false.                                                      
      do 510 j=jam,kam                                                  
      z=value(k)                                                        
      if (z.lt.1.00e-5) z=1.0e9                                         
      alltau(nconf,j)=z                                                 
510   continue                                                          
412   continue                                                          
      go to 409                                                         
490   continue                                                          
c-----terminate execution by end of file or two blank cards.            
      if (qset) go to 660                                               
c-----print out mechanism yield values on representative symmetry sets. 
c     never=nrep+1                                                      
      do 570 j=1,nrep                                                   
      if (j.ge.2) go to 550                                             
      if (qr) write (nlist,560)  (nom(j,ko),ko=1,4)                             
      go to 570                                                         
550   continue                                                          
      iam=irepre(j-1)                                                   
      if(qr)write (nlist,560)(nom(j,ko),ko=1,4),(alltau(ko,iam),ko=1,2)     
  560 format (5x,4a4,3x,f10.4,5x,f10.4)                                 
570   continue                                                          
      if (itype.eq.1) ntotal=number                                     
      if (qr) write (nlist,320)                                                 
      if (qfirst.and.qr) write (nlist,580) number,itype,kcycle,lcycle,lcrit    
     1      ,ntotal                                                     
580   format (10x,40(1h-)/10x,i4,3x,'data sets'/10x,'mode of operation',
     1i4/10x,i4,3x,'deformation steps'/10x,'frequency of data output'   
     2,i4/10x,'criterion for resolution of degeneracy',i4/              
     3     10x,'total number of grains to be examined',i5)              
c---read incremental deformation gradient-------------------------------
c        program is designed so that reference axial system is x south  
c        y east,and z up.                                               
c        deformation gradient read in by rows.                          
c----- 'maxcyc' increments allowed.                                     
      qflag=.false.                                                     
      if (qfirst.and.qr) write (nlist,591)                                     
  591 format (10x,'deformation path------------------------'/           
     1     10x,'at iteration',20x,'deformation gradient ')              
      if (itype.eq.1)  go to 601                                        
  590 continue                                                          
      nincr=1                
c      read (nelle,600,end=630)(defpaf(maxcyc,jk),jk=1,9)

      defpaf(maxcyc,1)=etensor(1)
      defpaf(maxcyc,2)=etensor(2)
      defpaf(maxcyc,3)=0
      defpaf(maxcyc,4)=etensor(3)
      defpaf(maxcyc,5)=etensor(4)
      defpaf(maxcyc,6)=0
      defpaf(maxcyc,7)=0
      defpaf(maxcyc,8)=0
      defpaf(maxcyc,9)=1


        
      if (qr) write (ndata,606)nincr,(defpaf(maxcyc,jk),jk=1,9)
600   format(9f6.4) 
606   format (i3,9f6.4) 
      xgamma=defpaf(maxcyc,2)
      if (nincr.eq.0.and.qflag)  go to 630                              
      if (nincr.eq.1) qflag=.true.                                      
      if (.not.qflag) go to 724                                         
      if (nincr.lt.1.or.nincr.gt.kcycle)  go to 721                     
      volchg=defpaf(maxcyc,1)+defpaf(maxcyc,5)+defpaf(maxcyc,9)         
      volchg=volchg-3.00
C      volchg= defpaf(maxcyc,1)* defpaf(maxcyc,5)
C      volchg=volchg-(defpaf(maxcyc,2)* defpaf(maxcyc,4))
C      volchg=volchg-1.0
      if (abs(volchg).gt. 0.00001)  go to 747                           
      go to 603                                                         
601   continue                                                          
c-----for i.r.d. diagrams only                                          
      nincr=1                                                           
      do 602 jk=1,9                                                     
      defpaf(maxcyc,jk)=0.00d0                                          
602   continue                                                          
      defpaf(maxcyc,1)=1.05d0                                           
      defpaf(maxcyc,5)=1.05d0                                           
      defpaf(maxcyc,9)=0.90d0                                           
603   continue                                                          
      do 640 j=nincr,maxcyc                                             
      do 640 k=1,9                                                      
      defpaf(j,k)=defpaf(maxcyc,k)                                      
  640 continue                                                          
      if(qfirst.and.qr)write(nlist,592) nincr,(defpaf(maxcyc,ko),ko=1,9)    
  592 format (10x,i5,25x,3(f6.4,3x)/(40x,3(f6.4,3x)))                   




      if(itype.eq.3) go to 630




      if (itype.ne.1) go to 590                                         
  630 continue                                                          
c-----if sufficient grains deformed value of number changed.            
      if (ngrain.ge.ntotal) go to 757                                   
      if (ngrain+number.gt.ntotal) number=ntotal-ngrain                 
c-----set random number generator                                       
      ngr=ngrain                                                        
      if (ngr.eq.0) ngr=1                                               
      do 685 j=1,ngrain                                                 
      xr=randz(xr)                                                       
685   continue                                                          
c-----subsidiary programs recognize blanks in echo data set.            
      i99=-99                                                           
      if (qr) write (ndata,999) i99                                             
999   format (64x,3hend,i3)                                             
      return                                                            
c-----terminate execution for itype=1                                   
660   continue                                                          
      end file ndata                                                    
      rewind ntemp                                                      
      rewind ndata                                                      
      if (qr) write (nlist,680)                                                 
680   format (10x,'normal termination')                                 
      call daytim(idate,itime)                                          
      if (qr) write (nlist,320)                                                 
      stop                                                              
c---error exits-------------------------------------------------------- 
690   write (nlist,700) kcycle                                          
700   format (10x,'modify program so that',i4,2x,'cycles allowed')      
      write (nlist,801)                                                 
801   format (10x,'--------- error exit from sbr setup ---------')      
      stop                                                              
710   write (nlist,720)                                                 
720   format (10x,'not enough columns in matrix')                       
      write (nlist,801)                                                 
      stop                                                              
  724 write (nlist,725)                                                 
  725 format (10x,'first deformation step not specified')               
      write (nlist,801)                                                 
      stop                                                              
  716 write (nlist,727)                                                 
  727 format (10x,' too many mechanism symmetry sets')                  
      write (nlist,801)                                                 
      stop                                                              
721   write (nlist,777)                                                 
777   format(10x,'error in deformation increment data')                 
      write (nlist,801)                                                 
      stop                                                              
  726 write(nlist,717)                                                  
  717 format(10x,'invalid symmetry set specified--too high')            
      write (nlist,801)                                                 
      stop                                                              
  728 write (nlist,729)                                                 
  729 format (10x,' only two yield surface configurations allowed')     
      write (nlist,801)                                                 
      stop                                                              
  747 write (nlist,748) nincr,volchg,(defpaf(maxcyc,jk),jk=1,9)         
  748 format (10x,'deformation increment',i4,2x,                        
     1     'implies volume change',f10.5/                               
     2     10x,'deformation gradient'/(20x,3f12.7))                     
      write (nlist,801)                                                 
      stop                                                              
  749 write (nlist,750) nsys,dpole,dline                                
  750 format (10x,'glide system',i4,4x,'pole',3f7.4,5x,'direction',     
     1     3f7.4/10x,'has been specified to imply a volume change')     
      write (nlist,801)                                                 
      stop                                                              
757   if (qr) write (nlist,760) ntotal                                          
760   format (10x,'normal termination',i15,5x,'grains deformed'/        
     1     (10x,40(1h=)))                                               
      stop                                                              
      end                                                               
                                                                        
      subroutine degena                                                 
c---routine examines all combinations of yield                          
c   surfaces from the set of all yield surfaces defining                
c   the optimal basic feasible vertex on the yield surface.             
c   these are perturbed towards the origin with result that             
c   degeneracy in the choice of basic feasible slip activity            
c   vectors is removed.                                                 
c   it is assumed that vertices are defined at least by five            
c   independent slip systems. minor changes allow other assumptions.    
c-----numper=number of yield surfaces defining vertex.                  
c-----nbasic=number of degenerate slip activity vectors                 
c     that can be specified.                                            
c-----holds is array of degenerate slip activity vectors.               
c-----iactiv is an array flagging the yield surfaces defining           
c     the optimal basic feasible solution.                              
c-----------------------------------------------------------------------
      implicit real * 8 (d),logical (q)                                 
      logical qerror,qbasis                                             
      real * 8 cost,matrix,rhs,yields,pivec,slips,petbn                 
      real defpaf,deviat,dummy                                          
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      common /logic/ qd,qt,qr,qdata,qbasis                              
      common /degen/ nbasic,redcst(103),holds(65,103)                   
      common /crystl/ nrep,irepre(20),itypof,rotat(103,3,3)             
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)            
      common /storer/ iactiv(15),nactiv,zactiv(10),ngrain,iterat,work   
      common /path/ defpaf(400,9),alltau(2,103),nconf,nchang(2)         
      common /param/ itype,kcycle,lcycle,lcrit,number,nom(22,4)         
      common /lp1/ matrix(6,103),rhs(6),yields(103),slips(103),pivec(6),
     1cost,qerror                                                       
      common /lp2/ num(103),qbasic(103)                                 
      common /lp3/ nrow,ncol,nlimit,nsys,nartif                         
      common pi,root2,root3,pi90,pifac,jrand                            
c-----perturbation                                                      
      dxmag=10000.00d0                                                  
      xmag=dxmag                                                        
      npetbn=65                                                         
      petbn=0.003d0                                                     
      nbasic=0                                                          
      kam=nactiv-4                                                      
      if (kam.lt.1) go to 90                                            
      do 60 jam=1,kam                                                   
      jbm=jam+1                                                         
      kbm=kam+1                                                         
      do 60 jcm=jbm,kbm                                                 
      jdm=jcm+1                                                         
      kdm=kbm+1                                                         
      do 60 jem=jdm,kdm                                                 
      jfm=jem+1                                                         
      kfm=kdm+1                                                         
      do 60 jgm=jfm,kfm                                                 
      jhm=jgm+1                                                         
      khm=kfm+1                                                         
      do 60 jim=jhm,khm                                                 
      lam=iactiv(jam)                                                   
      yields(lam)=yields(lam)-petbn                                     
      lcm=iactiv(jcm)                                                   
      yields(lcm)=yields(lcm)-petbn                                     
      lem=iactiv(jem)                                                   
      yields(lem)=yields(lem)-petbn                                     
      lgm=iactiv(jgm)                                                   
      yields(lgm)=yields(lgm)-petbn                                     
      lim=iactiv(jim)                                                   
      yields(lim)=yields(lim)-petbn                                     
      qbasis=.true.                                                     
      call lp                                                           
      if (qerror) return                                                
c-------store optimal basic feasible solutions                          
      if (nbasic.eq.0) go to 30                                         
      do 20 ilk=1,nbasic                                                
      do 10 il=1,nsys                                                   
      if (num(il) .lt. 0)  go to 10                                     
c-----this clumsy device avoids underflow                               
      ihold=holds(ilk,il)*xmag                                          
      islip=slips(il)*dxmag                                             
      qtell=ihold.eq.islip                                              
      if  (.not.qtell)  go to 20                                        
10    continue                                                          
      go to 50                                                          
20    continue                                                          
30    nbasic=nbasic+1                                                   
      if (nbasic.gt.npetbn) go to 70                                    
      do 40 im=1,nsys                                                   
40    holds(nbasic,im)=slips(im)                                        
50    continue                                                          
      yields(lam)=yields(lam)+petbn                                     
      yields(lcm)=yields(lcm)+petbn                                     
      yields(lem)=yields(lem)+petbn                                     
      yields(lgm)=yields(lgm)+petbn                                     
      yields(lim)=yields(lim)+petbn                                     
60    continue                                                          
      if (.not.qd.and.itype.ne.1) go to 300                             
c------calculate lattice rotations                                      
      do 255 jk=1,nbasic                                                
      do 254 kl=1,nsys                                                  
      slips (kl) = holds (jk,kl)                                        
254   continue                                                          
      call zipzap                                                       
      if (itype.ne.1) go to 255                                         
      call mult (zmap,rmap,zmap)                                        
      work=cost                                                         
      call drawit                                                       
255   continue                                                          
300   continue                                                          
      return                                                            
c---error exits.                                                        
70    continue                                                          
      write (nlist,80) npetbn                                           
   80 format(10x,'number of possible slip activity vectors too great'/  
     1     10x,'degeneracy is greater than',i5)                         
      write (nlist,801)                                                 
  801 format (10x,'---------  error exit from sbr degena ---------')    
      return                                                            
90    write (nlist,100)                                                 
  100 format(10x,'less than five glide systems active')                 
      write (nlist,801)                                                 
      return                                                            
      end                                                               
      subroutine flag                                                   
c-----routine calculates reduced cost and flags activated systems.      
      implicit real * 8 (d),logical (q)                                 
      real * 8 cost,matrix,rhs,yields,pivec,slips                       
      real defpaf,dummy                                                 
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      common /logic/ qd,qt,qr,qdata,qbasis                              
      common /crystl/ nrep,irepre(20),itypof,rotat(103,3,3)             
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)            
      common /path/ defpaf(400,9),alltau(2,103),nconf,nchang(2)         
      common /degen/ nbasic,redcst(103),holds(65,103)                   
      common /storer/ iactiv(15),nactiv,zactiv(10),ngrain,iterat,work   
      common /lp1/ matrix(6,103),rhs(6),yields(103),slips(103),pivec(6),
     1     cost,qerror                                                  
      common /lp2/ num(103),qbasic(103)                                 
      common /lp3/ nrow,ncol,nlimit,nsys,nartif                         
c-------calculate reduced cost.                                         
      pivec(6)=0.00d0                                                   
      errtol=0.001                                                      
      do 20 ik=1,nsys                                                   
      z=0.0                                                             
      do 10 il=1,nrow                                                   
      zz=matrix(il,ik)                                                  
      zzz=pivec(il)                                                     
      z=z+zz*zzz                                                        
10    continue                                                          
      redcst(ik)=yields(ik)                                             
      redcst(ik)=redcst(ik)-z                                           
   20 continue                                                          
c-------calculation of reduced cost complete-flag activated systems.    
      nactiv=0                                                          
      do 50 il=1,nsys                                                   
      z=yields(il)                                                      
      if (num(il).lt.0) go to 50                                        
      if (redcst(il).gt.errtol) go to 50                                
      nactiv=nactiv+1                                                   
      if (nactiv.gt.15) go to 140                                       
      iactiv(nactiv)=il                                                 
50    continue                                                          
      if (qd) go to 100                                                 
      return                                                            
c-----tabulation of situation on each glide system                      
  100 continue                                                          
      if (qr) write (nlist,110) nrow,ncol,nlimit,nsys,errtol                    
  110 format (10x,40(1h-)/10x,'dump of data in common blocks'/          
     1     10x,'from subroutine flag'///                                
     2     10x,'nrow    ncol    nlimit  nsys    errtol'/                
     3     10x,4i8,f12.9)                                               
      if (qr) write (nlist,120)                                                 
  120 format (/3x,'flags',3x,'qbasic',4x,'num',11x,'yields',9x,         
     1     'res.shear stress',3x,'reduced cost',5x,'activity',          
     2     9x,'error tolerance')                                        
      do 400 ik=1,nlimit                                                
      if (num(ik).lt.1.and..not.qbasic(ik))  go to 400                  
c-----to avoid underflow                                                
      z=yields(ik)                                                      
      dummy=z-redcst(ik)                                                
      qtell=abs(redcst(ik)).lt.errtol                                   
      qart=num(ik).lt.0                                                 
      qprob=qbasic(ik).and. .not.qtell                                  
      qbase=qbasic(ik)                                                  
      if (qr) write (nlist,717) qbasic(ik),num(ik),                             
     1     yields(ik),dummy,redcst(ik),slips(ik)                        
     2     ,errtol                                                      
  717 format (11x,3x,l1,4x,i5,2x,5(3x,f15.9))                           
      if (qart) write (nlist,200)                                       
      if(qtell) write (nlist,210)                                       
      if(qprob) write (nlist,220)                                       
      if(qbase) write (nlist,230)                                       
200   format (1h+,1x,'artif')                                           
210   format (1h+,7x,'active')                                          
220   format (1h+,1x,13(1h-))                                           
230   format (1h+,1x,7(1h=))                                            
  400 continue                                                          
      return                                                            
c-----error exits.                                                      
140   write (nlist,150) nactiv                                          
150   format (10x,'sbr flag- more than',i3,3x,'systems activated')      
      stop                                                              
      end                                                               
c---calculates relative rotation of the reference frame occurring as    
c   a result of deformation by the specified slip activity vector.      
c-----rmap describes orientation of grain prior to deformation step.    
c-----zmap describes orientation of grain after  deformation step.      
c-----matrices specified with respect to the external frame.            
c-----------------------------------------------------------------------
      subroutine zipzap                                                 
      implicit real * 8 (d),logical (q)                                 
      real * 8 cost,matrix,rhs,yields,pivec,slips                       
      real defpaf                                                       
      dimension axis(3),ikey(15)                                        
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      common /logic/ qd,qt,qr,qdata,qbasis                              
      common /crystl/ nrep,irepre(20),itypof,rotat(103,3,3)             
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)            
      common /path/ defpaf(400,9),alltau(2,103),nconf,nchang(2)         
      common /storer/ iactiv(15),nactiv,zactiv(10),ngrain,iterat,work   
      common /degen/ nbasic,redcst(103),holds(65,103)                   
      common /param/ itype,kcycle,lcycle,lcrit,number,nom(22,4)         
      common pi,root2,root3,pi90,pifac,jrand                            
      common /lp1/ matrix(6,103),rhs(6),yields(103),slips(103),pivec(6),
     1     cost,qerror                                                  
      common /lp3/ nrow,ncol,nlimit,nsys,nartif                         
c-----calculation of the infinitesimal rigid body rotation relative     
c-----displacement tensor with respect to crystal coordinates.          
      do 10 ilk=1,3                                                     
      do 10 ilm=1,3                                                     
10    zmap(ilk,ilm)=0.00                                                
      do 20 ilk=1,nsys                                                  
      do 20 ilm=1,3                                                     
      do 20 iln=1,3                                                     
      z=slips(ilk)                                                      
      zmap(ilm,iln)=zmap(ilm,iln)+rotat(ilk,ilm,iln)*z                  
20    continue                                                          
c-----imposed vorticity (wrt external frame) -equals-                   
c-----crystal axis rotation rate(wrt external frame)+incremental        
c-----strain axis rotation rate (wrt crystal axes)                      
      do 30 j=1,3                                                       
      do 30 k=1,3                                                       
30    zmap(j,k)=vort(j,k)-zmap(j,k)                                     
c-----calculate curl of displacement field.                             
c----- axis is   +w12   -w13   +w23                                     
      axis(1)=+zmap(2,3)                                                
      axis(2)=-zmap(1,3)                                                
      axis(3)=+zmap(1,2)                                                
      theta=sqrt(axis(1)**2+axis(2)**2+axis(3)**2)                      
      if (theta.gt.0.00001) go to 40                                    
      za=1.0/root3                                                      
      axis(1)=za                                                        
      axis(2)=za                                                        
      axis(3)=za                                                        
      go to 60                                                          
40    continue                                                          
      do 50 ilm=1,3                                                     
50    axis(ilm)=axis(ilm)/theta                                         
60    continue                                                          
c-----for this increment calculate for the grain the relative           
c-----rotation matrix,with respect to the external frame.in other       
c-----words  zmap now maps the old bond axes into their new position.   
      call rotate (zmap,axis,theta)                                     
      if (.not.qr.and..not.qdata)  return                               
c-----initialize for output.                                            
      nactiv=0                                                          
      do 70 kl=1,nsys                                                   
      tivity=dabs(slips(kl))                                            
      if (tivity.lt.0.0001) go to 70                                    
      nactiv=nactiv+1                                                   
      if (nactiv.gt.10)  go to 727                                      
      zactiv(nactiv)=tivity                                             
      iactiv(nactiv)=kl                                                 
70    continue                                                          
c-----optional listing                                                  
      if (.not.qr) return                                               
      axis(1) = 0.00                                                    
      axis(2)= 0.0                                                      
      axis(3)= 1.00                                                     
      call change (axis,axis,rmap)                                      
      call firo (axis,phi,rho)                                          
      phi=phi/pifac                                                     
      rho=rho/pifac                                                     
      if (qr) write (nlist,79) phi,rho                                          
79    format(10x,'orientation (0 0 1)      phi=',f8.3,5x,               
     1     'rho=',f8.3)                                                 
      work=cost                                                         
      if (qr) write (nlist,80) work,nactiv                                      
      if (qr) write (nlist,81) (iactiv(ko),ko=1,nactiv)                         
      do 83 j=1,nactiv                                                  
      key=iactiv(j)                                                     
      ikey(j)=iset(key)                                                 
83    continue                                                          
      if (qr) write (nlist,82) (ikey(ko),ko=1,nactiv)                           
      if (qr) write (nlist,84) (zactiv(ko),ko=1,nactiv)                         
80    format (5x,4hwork,f12.6,10x,i5,3x,6hactive)                       
81    format (5x,6hsystem,2x,12i9)                                      
82    format (5x,3hset,5x,12i9)                                         
84    format (5x,8hactivity,12f9.5)                                     
      if (.not.qd) return                                               
      if (qr) write (nlist,90)                                                  
      do 92 k=1,3                                                       
      if (qr) write (nlist,91) (zmap(k,l),l=1,3),(rmap(k,l),l=1,3)              
92    continue                                                          
   90 format (10x,'incremental rotation',28x,                           
     1     'initial orientation of bond frame')                         
91    format (10x,3f12.5,12x,3f12.5)                                    
      ztheta=theta/pifac                                                
      if (qr) write (nlist,100) axis,ztheta                                     
100   format (10x,'axis of rotation',3f12.4,10x,'theta',f12.4,' deg.')  
      return                                                            
  727 if (qr) write (nlist,737) nactiv                                          
  737 format (10x,'nactiv is exceeding limit',i10/                      
     1     10x,'error exit from sbr zipzap')                            
      stop                                                              
      end                                                               
      subroutine drawit                                                 
c---routine outputs data for production of fabric diagrams              
c-----zmap describes orientation of frame after the deformation step.   
c-----rmap describes orientation of frame prior to deformation step.    
c-----ngrain=grain number                                               
c-----original and new orientation matrices output row wise with        
c-----columns indicating orientation of bond axes wrt reference frame.  
c-----------------------------------------------------------------------
      implicit logical (q)                                 
      common /logic/ qd,qt,qr,qdata,qbasis                              
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)            
      common /storer/ iactiv(15),nactiv,zactiv(10),ngrain,iterat,work
      common /print/nvals,linc
      common /tbhdrx/ xlmap(3,4,30000)
      
      ndata=19
      nout=20                                                          
      if (qr) write (ndata,100) ((rmap(k,j),j=1,3),k=1,3)                       
      if (qr) write (ndata,100) ((zmap(k,j),j=1,3),k=1,3)                       
c      write (nout,100) ((zmap(k,j),j=1,3),k=1,3)                       
100   format (10f8.5)                                                   
      call sort                                                         
      if (nactiv.ge.5)  go to 500                                       
      nact=nactiv+1                                                     
      do 499 j=nact,5                                                   
      iactiv(j)=0                                                       
      zactiv(j)=0.00                                                    
  499 continue                                                          
  500 continue                                                          
      if (qr) write (ndata,200) nvals,linc,work,nactiv,                      
     1     (iactiv(l),l=1,5),(zactiv(l),l=1,5),xlmap(1,4,nvals)                          
  200 format (i4,i3,f12.6,6i3,5f8.5,f7.0)
c      write(nout,201)work
  201 format(f12.6)
      return                                                            
      end                                                               
      subroutine outit(nmbr)                                                 
c---routine outputs data for production of fabric diagrams              
c-----zmap describes orientation of frame after the deformation step.   
c-----rmap describes orientation of frame prior to deformation step.    
c-----ngrain=grain number                                               
c-----original and new orientation matrices output row wise with        
c-----columns indicating orientation of bond axes wrt reference frame.  
c-----------------------------------------------------------------------
      implicit logical (q)                                 
      common /logic/ qd,qt,qr,qdata,qbasis                              
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)            
      common /storer/ iactiv(15),nactiv,zactiv(10),ngrain,iterat,work
      common /print/nvals,linc
      common /tbhdrx/ xlmap(3,4,30000)
      
      ndata=19
      
c      call reqx(nmbr)
c      write (ndata,100) ((rmap(k,j),j=1,3),k=1,3)                       
      if (qr) write (ndata,100) ((zmap(k,j),j=1,3),k=1,3)                       
100   format (10f8.5)                                                   
      if (qr) write (ndata,200) nmbr,work                          
  200 format (i4,5x,f8.5)
  
      return                                                            
      end 
      
      function iset (n)                                                 
      common /crystl/ nrep,irepre(20),itypof,rotat(103,3,3)             
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      do 10 j=1,nrep                                                    
      if (irepre(j).le.n) go to 10                                      
      iset=j-1                                                          
      return                                                            
10    continue                                                          
      write (nlist,20)                                                  
20    format (10x,'error exit-iset')                                    
      stop                                                              
      end                                                               
      subroutine sort                                                   
      common /storer/ isys(15),nact,sysact(10),ngrain,iterat,work       
      dimension ix(10),x(10)                                            
      do 100 j=1,nact                                                   
      x(j)=sysact (j)                                                   
      ix (j)=isys(j)                                                    
  100 continue                                                          
      do 200 j=1,nact                                                   
      z=-1.0e10                                                         
      do 210 k=1,nact                                                   
      if (x(k).le.z)  go to 210                                         
      iam=k                                                             
      z=x(k)                                                            
  210 continue                                                          
      sysact(j)=x(iam)                                                  
      isys  (j)=ix(iam)                                                 
      x(iam)=-2.0e10                                                    
  200 continue                                                          
      return                                                            
      end                                                               
      subroutine rotate (rmap,axis,theta)                               
c---calculates rotation matrix *rmap* given *axis* of rotation          
c   and *theta* is the angle of rotation (radians).  sense convention   
c   is that,if thumb points in the direction *axis* away from the       
c   origin then fingers of the right hand curl in a positive sense.     
c---coordinate system is crystallographic here.                         
      dimension rmap(3,3),axis(3),coord(3,3),scrat(3,3)                 
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      zz=sqrt(axis(1)**2+axis(2)**2+axis(3)**2)                         
      if (abs(zz).lt.0.000001)  go to 50                                
      do 5 j=1,3                                                        
      do 5 k=1,3                                                        
5     rmap(j,k)=0.00                                                    
      do 10 j=1,3                                                       
      coord(j,1)=axis(j)/zz                                             
      coord(j,2)=0.00                                                   
10    continue                                                          
      inot=2                                                            
      do 30 j=1,3                                                       
      dummy=abs(axis(j))                                                
      if (dummy.lt.0.01)  inot=j                                        
   30 continue                                                          
      jnot=inot+1                                                       
      knot=inot-1                                                       
      if (jnot.eq.4)  jnot=1                                            
      if (knot.eq.0)  knot=3                                            
      coord (inot,2)=0.00                                               
      coord (jnot,2)= axis(knot)                                        
      coord (knot,2)=-axis(jnot)                                        
      zav=sqrt(axis(jnot)**2+axis(knot)**2)                             
      coord(1,3)=coord(2,1)*coord(3,2)-coord(3,1)*coord(2,2)            
      coord(2,3)=coord(3,1)*coord(1,2)-coord(1,1)*coord(3,2)            
      coord(3,3)=coord(1,1)*coord(2,2)-coord(2,1)*coord(1,2)            
      zaa= sqrt(coord(1,3)**2+coord(2,3)**2+coord(3,3)**2)              
      if (zav.lt.0.00001) go to 50                                      
      do 32 k=1,3                                                       
      coord(k,3)=coord(k,3)/zaa                                         
   32 coord(k,2)=coord(k,2)/zav                                         
      rmap(1,1)=1.00                                                    
      rmap(2,2)=cos(theta)                                              
      rmap(3,2)=sin (theta)                                             
      rmap(2,3)=-rmap(3,2)                                              
      rmap(3,3)=+rmap(2,2)                                              
      call mult (coord,rmap,rmap)                                       
      do 31 j=1,3                                                       
      do 31 k=j,3                                                       
      z=coord(j,k)                                                      
      coord(j,k)=coord(k,j)                                             
      coord(k,j)=z                                                      
   31 continue                                                          
      call mult (rmap,coord,rmap)                                       
      return                                                            
50    write(nlist,60)                                                   
60    format(10x,'error exit-axis null vector in call to rotate')       
      stop                                                              
      end                                                               
      subroutine mult (a,b,c)                                           
      dimension a(3,3), b(3,3), c(3,3), d(3,3)                          
      do 20 i=1,3                                                       
      do 20 j=1,3                                                       
      z=0.0                                                             
      do 10 k=1,3                                                       
      z=z+a(i,k)*b(k,j)                                                 
10    continue                                                          
      d(i,j)=z                                                          
20    continue                                                          
      do 30 i=1,3                                                       
      do 30 j=1,3                                                       
30    c(i,j)=d(i,j)                                                     
      return                                                            
      end                                                               
      subroutine dmult (da,db,dc)                                       
      implicit real * 8 (a-h,o-p,r-z)                                   
      dimension da(3,3),db(3,3),dc(3,3),d(3,3)                          
      do 20 i=1,3                                                       
      do 20 j=1,3                                                       
      dz=0.0d0                                                          
      do 10 k=1,3                                                       
      dz=dz+da(i,k)*db(k,j)                                             
10    continue                                                          
      d(i,j)=dz                                                         
20    continue                                                          
      do 30 i=1,3                                                       
      do 30 j=1,3                                                       
      dc(i,j)=d(i,j)                                                    
30    continue                                                          
      return                                                            
      end                                                               
      subroutine trans (a,b)                                            
      dimension a(3,3),b(3,3)                                           
      do 10 i=1,3                                                       
      do 10 j=i,3                                                       
      z=a(i,j)                                                          
      b(i,j)=a(j,i)                                                     
      b(j,i)=z                                                          
10    continue                                                          
      return                                                            
      end                                                               
      subroutine drotat (rmap,axis,theta)                               
c---calculates rotation matrix *rmap* given *axis* of rotation          
c   and *theta* is the angle of rotation (radians).  sense convention   
c   is that,if thumb points in the direction *axis* away from the       
c   origin fingers of the right hand curl in a positive sense.          
c---coordinate system is crystallographic here.                         
      implicit logical (q),real * 8 (a-h,o-p,r-z)                       
      dimension rmap(3,3),axis(3),coord(3,3),scrat(3,3)                 
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      zz=dsqrt(axis(1)**2+axis(2)**2+axis(3)**2)                        
      if (dabs(zz).lt.0.000001d0)  go to 50                             
      do 5 j=1,3                                                        
      do 5 k=1,3                                                        
5     rmap(j,k)=0.00d0                                                  
      do 10 j=1,3                                                       
      coord(j,1)=axis(j)/zz                                             
      coord(j,2)=0.00d0                                                 
10    continue                                                          
      inot=2                                                            
      do 30 j=1,3                                                       
      dummy=dabs(axis(j))                                               
      if (dummy.lt.0.01d0)  inot=j                                      
   30 continue                                                          
      jnot=inot+1                                                       
      knot=inot-1                                                       
      if (jnot.eq.4)  jnot=1                                            
      if (knot.eq.0)  knot=3                                            
      coord (inot,2)=0.00d0                                             
      coord (jnot,2)= axis(knot)                                        
      coord (knot,2)=-axis(jnot)                                        
      zav=dsqrt(axis(jnot)**2+axis(knot)**2)                            
      coord(1,3)=coord(2,1)*coord(3,2)-coord(3,1)*coord(2,2)            
      coord(2,3)=coord(3,1)*coord(1,2)-coord(1,1)*coord(3,2)            
      coord(3,3)=coord(1,1)*coord(2,2)-coord(2,1)*coord(1,2)            
      zaa=dsqrt(coord(1,3)**2+coord(2,3)**2+coord(3,3)**2)              
      if (zav.lt.0.00001d0)    go to 50                                 
      do 32 k=1,3                                                       
      coord(k,3)=coord(k,3)/zaa                                         
   32 coord(k,2)=coord(k,2)/zav                                         
      rmap(1,1)=1.00d0                                                  
      rmap(2,2)=dcos(theta)                                             
      rmap(3,2)=dsin(theta)                                             
      rmap(2,3)=-rmap(3,2)                                              
      rmap(3,3)=+rmap(2,2)                                              
      call dmult (coord,rmap,rmap)                                      
      do 11 j=1,3                                                       
      do 11 k=j,3                                                       
      z=coord(j,k)                                                      
      coord(j,k)=coord(k,j)                                             
      coord(k,j)=z                                                      
   11 continue                                                          
      call dmult (rmap,coord,rmap)                                      
      return                                                            
50    write(nlist,60)                                                   
60    format(10x,'error exit-axis null vector in call to rotate')       
      stop                                                              
      end                                                               
      subroutine begina                                                 
c------d(3,3) transforms lattice vectors to bond cartesian axes.        
c------e(3,3) transforms reciprocal lattice vectors to bond axes.       
c------det(3,3) bond determinant.                                       
c     note that these transformations are the basis for crystallographic
c     calculations throughout this series of programs.                  
c---z axis is normal to a and b crystallographic axes.                  
c---y axis is the b crystallographic axis.                              
c---z axis is normal x y axes,set up system is right handed.            
c-----------------------------------------------------------------------
      implicit  real * 8 (a-h,o-p,r-z)                                  
      logical qd,qt,qr,qdata,qbasis                                     
      real * 8 al,be,ga,a,b,c,d,e                                       
      dimension det(3,3)                                                
      common /xtalog/ al,be,ga,a,b,c,d(3,3),e(3,3)                      
      common /logic/ qd,qt,qr,qdata,qbasis                              
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      if (qr) write (nlist,10) al,be,ga,a,b,c                                   
10    format (///9x,' alpha,beta,gamma,a,b,c are'/10x,6f12.4)           
      pi=dasin (1.0d0)*2.0d0                                            
      pifac=pi/180.00d0                                                 
      alpha=al                                                          
      beta=be                                                           
      gamma=ga                                                          
      alpha=alpha*pifac                                                 
      beta=beta*pifac                                                   
      gamma=gamma*pifac                                                 
      cal=dcos(alpha)                                                   
      cbe=dcos(beta)                                                    
      cga=dcos(gamma)                                                   
      sbe=dsin(beta)                                                    
      detval=1.0d0-cal**2-cbe**2-cga**2+2.0d0*cal*cbe*cga               
      aa=dsqrt(detval)                                                  
      volcel=a*b*c*aa                                                   
      cosgma=aa/sbe                                                     
      cosrho=(cga-cbe*cal)/sbe                                          
      erchek=cosrho**2+cosgma**2+cal**2-1.0d0                           
      if (qd) write (nlist,20) detval,volcel,cosgma,cosrho,erchek       
20    format (//10x,'detval',f14.5,6x,'volcel',f14.5/10x,'cosgma',f14.5 
     1  ,6x,'cosrho',f14.5/10x,'error check in difference from unity',  
     2   g15.2)                                                         
      d(1,1)=a*sbe                                                      
      d(1,2)=b*cosrho                                                   
      d(1,3)=0.0d0                                                      
      d(2,1)=0.0d0                                                      
      d(2,2)=b*cosgma                                                   
      d(2,3)=0.0d0                                                      
      d(3,1)=a*cbe                                                      
      d(3,2)=b*cal                                                      
      d(3,3)=c                                                          
      e(1,1)=1.0d0/a/sbe                                                
      e(1,2)=0.0d0                                                      
      cotbe=0.00d0                                                      
      if (dabs(be)-90.00d0.gt.1.00d-7) cotbe=dtan(beta)                 
      e(1,3)=-1.0d0*cotbe/c                                             
      e(2,1)=-cosrho/a/cosgma/sbe                                       
      e(2,2)=1.0d0/b/cosgma                                             
      e(2,3)=((cosrho*cotbe)-cal)/c/cosgma                              
      e(3,1)=0.0d0                                                      
      e(3,2)=0.0d0                                                      
      e(3,3)=1.0d0/c                                                    
      det(1,1)=1.0d0                                                    
      det(2,2)=1.0d0                                                    
      det(3,3)=1.0d0                                                    
      det(2,1)=cga                                                      
      det(1,2)=cga                                                      
      det(1,3)=cbe                                                      
      det(3,1)=cbe                                                      
      det(2,3)=cal                                                      
      det(3,2)=cal                                                      
      if (qd) write (nlist,30) ((d(j,i),i=1,3),j=1,3)                   
      if (qd) write (nlist,40) ((e(j,i),i=1,3),j=1,3)                   
      if (qd) write (nlist,50) ((det(j,i),i=1,3),j=1,3)                 
30    format (//10x,'bond matrix for lattice rows'/(3f12.4))            
40    format (//10x,'bond matrix for reciprocal lattice rows'/(3f12.4)) 
50    format (//10x,'bond determinant'/(3f12.4))                        
      return                                                            
      end                                                               
      subroutine euler (rmap,a,b,c)                                     
c---rotation matrix calculated given eulerian angles(radians)           
c---sense convention-thumb away from origin in the direction            
c---.axis. and fingers of right hand curl in a positive sense.          
c---eulerian angles taken about x3,x2,x3 in turn.                       
      dimension rmap(3,3),am(3,3),bm(3,3),cm(3,3)                       
      do 20 i=1,3                                                       
      do 10 j=1,3                                                       
      am(i,j)=0.0                                                       
      bm(i,j)=0.0                                                       
10    cm(i,j)=0.0                                                       
      am(i,i)=cos(a)                                                    
      bm(i,i)=cos(b)                                                    
      cm(i,i)=cos(c)                                                    
20    continue                                                          
      am(3,3)=1.0                                                       
      bm(2,2)=1.0                                                       
      cm(3,3)=1.0                                                       
      am(1,2)=-sin (a)                                                  
      am(2,1)=+sin (a)                                                  
      bm(1,3)=+sin (b)                                                  
      bm(3,1)=-sin (b)                                                  
      cm(1,2)=-sin (c)                                                  
      cm(2,1)=+sin (c)                                                  
      call mult (bm,am,rmap)                                            
      call mult (cm,rmap,rmap)                                          
      return                                                            
      end                                                               
      subroutine phirho (axis,phi,rho)                                  
      dimension axis(3)                                                 
      if (axis(3).gt.0.00)  go to 15                                    
      do 10 i=1,3                                                       
      axis(i)=-axis(i)                                                  
   10 continue                                                          
   15 continue                                                          
      z=sqrt(axis(1)**2+axis(2)**2)                                     
      zz=sqrt(axis(1)**2+axis(2)**2+axis(3)**2)                         
      if (zz.lt.0.0001) go to 40                                        
      if (z.lt.0.00001) go to 20                                        
      phi = abs (acos (axis(1)/z))                                      
      go to 30                                                          
20    phi=0.0                                                           
30    if(axis(2).lt.0.00)phi=-phi                                       
      rho = acos (abs (axis(3)/zz))                                     
      return                                                            
40    phi=0.00                                                          
      rho=0.00                                                          
      return                                                            
      end                                                               
      subroutine firo (axis,phi,rho)                                    
c-----conversion from cartesian coordinates to spherical polar coordinat
c---axes  x-south,y-east,z-up                                           
c---phi is trend anticlockwise from x,rho is angle                      
c---made with z-axis---in radians.                                      
      dimension axis(3)                                                 
      z=sqrt(axis(1)**2+axis(2)**2)                                     
      zz=sqrt(axis(1)**2+axis(2)**2+axis(3)**2)                         
      if(zz.lt.0.0001) go to 15                                         
      if(z.lt.0.00001) go to 10                                         
      phi=abs(acos(axis(1)/z))                                          
      go to 11                                                          
   10 phi=0.0                                                           
   11 if(axis(2).lt.0.00) phi=-phi                                      
      rho=acos(axis(3)/zz)                                              
      return                                                            
   15 phi=0.00                                                          
      rho=0.00                                                          
      return                                                            
      end                                                               
      subroutine dchang (h,hh,hhh)                                      
      implicit real  * 8 (a-h,o-p,q-z)                                  
c      vector hh is the transform of vector h by matrix hhh             
      dimension h(3),hh(3),hhh(3,3),x(3)                                
      do 20 i=1,3                                                       
      a=0.00                                                            
      do 10 j=1,3                                                       
10    a=a+hhh(i,j)*h(j)                                                 
20    x(i)=a                                                            
      do 30 i=1,3                                                       
30    hh(i)=x(i)                                                        
      return                                                            
      end                                                               
      subroutine change (h,hh,hhh)                                      
c      vector hh is the transform of vector h by matrix hhh             
      dimension h(3),hh(3),hhh(3,3),x(3)                                
      do 20 i=1,3                                                       
      a=0.00                                                            
      do 10 j=1,3                                                       
10    a=a+hhh(i,j)*h(j)                                                 
20    x(i)=a                                                            
      do 30 i=1,3                                                       
30    hh(i)=x(i)                                                        
      return                                                            
      end                                                               
                                                                        
      subroutine lp                                                     
      implicit logical (q),real * 8 (a-h,o-p,r-z)                       
      logical qerror,basic,qbasis                                       
      real * 8 l
      real*4 xr,randz                                                        
      real rand,pii,pi90,pifac,root2,root3                              
      common /lp1/ a(6,103),b(6),c(103),x(103),pi(6),cost,qerror        
      common /lp2/ num(103),basic(103)                                  
      common /lp3/ m,n,nlimit,nsys,nartif                               
      common /lp4/ dx(6),d(6),bc(6),l(6,6),cs,ialpha,ibeta,itn          
      common /lp5/ u(6),v(6)                                            
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      common /logic/ qd,qt,qr,qdata,qbasis                              
      common pii,root2,root3,pi90,pifac,jrand                           
      data ierror/0/,irecov/0/                                          
c     ------------------------------------------------------------------
c                                                                       
c     the subroutine attempts to solve the usual lp problem-            
c                   minimize      c*x                                   
c           subject to             ax = b    ,     x.ge.0               
c     a        is m*n , with m.lt.n                                     
c     x        is an n-vector , to hold the solution.                   
c     pi       is an m- vector , containing the lagrange multipliers.   
c     basic    can point to an initial basis if *qbasis* is .true.      
c     num      is an n-vector of column numbers, assumed positive.      
c              if num(j).lt.0 column j is artificial                    
c     qerror=.false. on exit iff an optimal solution was found          
c     itmax     should be set equal to the maximum number of            
c               iterations to be tolerated.                             
c                                                                       
c     ------------------------------------------------------------------
      irecov=0                                                          
10    qerror=.false.                                                    
      b(6)=-b(1)-b(4)                                                   
      if (qbasis) call setl                                             
      if (qerror.or..not.qbasis) call coldst                            
      call iter                                                         
      if (ialpha.ne.0) go to 30                                         
      if (nartif.gt.0) go to 70                                         
      return                                                            
30    qerror=.true.                                                     
      if (ibeta.eq.0) go to 50                                          
      write (nlist,40)                                                  
40    format (' lp----error termination.')                              
      go to 130                                                         
50    write (nlist,60)                                                  
60    format (' lp----problem unbounded')                               
      go to 130                                                         
70    do 90 j=1,n                                                       
      if (.not.basic(j).or.num(j).gt.0) go to 90                        
      if (x(j).lt.1.0d-5) go to 90                                      
c-----one recovery attempt allowed                                      
      qbasis=.false.                                                    
      if (irecov.eq.-1)  go to 140                                      
      call coldst                                                       
      irecov=-1                                                         
      go to 10                                                          
140   continue                                                          
      write (nlist,80)                                                  
80    format (10x,'error exit-lp problem not feasible')                 
      qerror=.true.                                                     
      return                                                            
90    continue                                                          
      nallow=1                                                          
      if (nartif.le.nallow) go to 120                                   
      do 100 j=1,m                                                      
      pet=randz(xr)*1.00e-3                                              
      b(j)=b(j)-pet                                                     
100   continue                                                          
      b(6)=-b(1)-b(4)                                                   
      ierror=ierror+1                                                   
      irecov=irecov+1                                                   
      write (nlist,110) ierror,irecov                                   
110   format (3x,'inadequate specification of vertex-error  number',i4, 
     1     5x,'recovery attempt',i4)                                    
      if (ierror.eq.10) stop                                            
      qbasis=.false.                                                    
      go to 10                                                          
120   continue                                                          
      qerror=.false.                                                    
130   continue                                                          
      return                                                            
      end                                                               
      subroutine iter                                                   
      implicit logical (q),real * 8 (a-h,o-p,r-z)                       
      logical qerror,basic                                              
      real * 8 l                                                        
      common /lp1/ a(6,103),b(6),c(103),x(103),pi(6),cost,qerror        
      common /lp2/ num(103),basic(103)                                  
      common /lp3/ m,n,nlimit,nsys,nartif                               
      common /lp4/ dx(6),d(6),bc(6),l(6,6),cs,ialpha,ibeta,itn          
      common /lp5/ u(6),v(6)                                            
      common /lp6/ dummya,dummyz                                        
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      itmax=10*m                                                        
      itn=0                                                             
10    if (itn.ge.itmax) go to 80                                        
      nerr=0                                                            
20    call findal                                                       
      if (ialpha.eq.0) return                                           
      itn=itn+1                                                         
      do 30 i=1,m                                                       
      v(i)=a(i,ialpha)                                                  
30    u(i)=v(i)                                                         
c-----(ldl*)u  =  u    ,    dx  =  a*u                                  
      call bksolv                                                       
      k=0                                                               
      do 50 j=1,n                                                       
      if (.not.basic(j)) go to 50                                       
      k=k+1                                                             
      s=0.0d0                                                           
      do 40 i=1,m                                                       
40    s=s+a(i,j)*u(i)                                                   
      dx(k)=s                                                           
50    continue                                                          
      call fndbta                                                       
      if (ibeta.eq.0) return                                            
      dummya=1.0d0                                                      
      dummyz=c(ialpha)                                                  
      call rank1                                                        
      do 60 i=1,m                                                       
60    v(i)=a(i,ibeta)                                                   
      dummya=-1.0d0                                                     
      dummyz=c(ibeta)                                                   
      call rank1                                                        
      if (.not.qerror) go to 70                                         
      nerr=nerr+1                                                       
      if (nerr.gt.1) return                                             
      qerror=.false.                                                    
      call setl                                                         
      if (qerror) return                                                
      go to 20                                                          
70    basic(ialpha)=.true.                                              
      basic(ibeta)=.false.                                              
      call pisolv                                                       
      go to 10                                                          
80    continue                                                          
c-----document problem                                                  
      if (qd) call flag                                                 
      write (nlist,180)                                                 
180   format (10x,'-----no exact solution for this grain-----')         
      return                                                            
      end                                                               
      subroutine coldst                                                 
      implicit real * 8 (a-h,o-p,r-z),logical (q)                       
      logical qerror,basic,q                                            
      real * 8 l                                                        
      common /lp1/ a(6,103),b(6),c(103),x(103),pi(6),cost,qerror        
      common /lp2/ num(103),basic(103)                                  
      common /lp3/ m,n,nlimit,nsys,nartif                               
      common /lp4/ dx(6),d(6),bc(6),l(6,6),cs,ialpha,ibeta,itn          
      common /lp5/ u(6),v(6)                                            
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      common /logic/ qd,qt,qr,qdata,qbasis                              
      n=nsys                                                            
      if (qd) write (nlist,10)                                          
10    format ('0cold start.')                                           
      s=0.0d0                                                           
      do 20 j=1,n                                                       
      basic(j)=.false.                                                  
      x(j)=0.0d0                                                        
      t=dabs(c(j))                                                      
      if(num(j).gt.0.and.s.lt.t)s=t                                     
20    continue                                                          
      art=1.0e3*s                                                       
      nartif=m                                                          
c-----add as many artificial variables as required                      
      do 60 k=1,m                                                       
      if (n.ne.nlimit) go to 40                                         
      write (nlist,30)                                                  
30    format (' too many columns in matrix.')                           
      stop                                                              
40    n=n+1                                                             
      num(n)=-k                                                         
      c(n)=art                                                          
      basic(n)=.true.                                                   
      do 50 i=1,m                                                       
50    a(i,n)=0.0d0                                                      
      t=-1.0d0                                                          
      if(b(k).ge.0.0d0)t=1.0d0                                          
      a(k,n)=t                                                          
      x(n)=dabs(b(k))                                                   
      bc(k)=t*art                                                       
      pi(k)=bc(k)
c      write(nlist,33)pi(k),k
33    format(10x,f12.4,10x,i4)                                                       
60    continue                                                          
      s=0.0d0                                                           
      do 70 i=1,m                                                       
70    s=s+dabs(b(i))                                                    
      cost=art*s                                                        
      do 90 i=1,m                                                       
      d(i)=1.0d0                                                        
      if (i.eq.1) go to 90                                              
      ii=i-1                                                            
      do 80 j=1,ii                                                      
80    l(i,j)=0.0d0                                                      
90    continue                                                          
      return                                                            
      end                                                               
      subroutine setx                                                   
      implicit logical (q),real * 8 (a-h,o-p,r-z)                       
      logical qerror,basic                                              
      real * 8 l                                                        
      common /lp1/ a(6,103),b(6),c(103),x(103),pi(6),cost,qerror        
      common /lp2/ num(103),basic(103)                                  
      common /lp3/ m,n,nlimit,nsys,nartif                               
      common /lp4/ dx(6),d(6),bc(6),l(6,6),cs,ialpha,ibeta,itn          
      common /lp5/ u(6),v(6)                                            
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      do 10 i=1,m                                                       
10    u(i)=b(i)                                                         
c-----(ldl*)u = u ,v = a*u                                              
      call bksolv                                                       
      k=0                                                               
      do 30 j=1,n                                                       
      if (.not.basic(j)) go to 30                                       
      k=k+1                                                             
      s=0.0d0                                                           
      do 20 i=1,m                                                       
20    s=s+a(i,j)*u(i)                                                   
      v(k)=s                                                            
30    continue                                                          
      k=0                                                               
      cost=0.0d0                                                        
      do 70 j=1,n                                                       
      if (.not.basic(j)) go to 60                                       
      k=k+1                                                             
      t=v(k)                                                            
      if (t.ge.-1.0d-6) go to 50                                        
      qerror=.true.                                                     
      return                                                            
50    if(t.le.0.0d0)t=1.0d-20                                           
      x(j)=t                                                            
      cost=cost+c(j)*t                                                  
      go to 70                                                          
60    x(j)=0.0d0                                                        
70    continue                                                          
      return                                                            
      end                                                               
      subroutine setl                                                   
      implicit logical (q),real * 8 (a-h,o-p,r-z)                       
      logical qerror,basic                                              
      real * 8 l                                                        
      common /lp1/ a(6,103),b(6),c(103),x(103),pi(6),cost,qerror        
      common /lp2/ num(103),basic(103)                                  
      common /lp3/ m,n,nlimit,nsys,nartif                               
      common /lp4/ dx(6),d(6),bc(6),l(6,6),cs,ialpha,ibeta,itn          
      common /lp5/ u(6),v(6)                                            
      common /lp6/ dummya,dummyz                                        
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      qerror=.false.                                                    
      do 40 i=1,m                                                       
      bc(i)=0.0d0                                                       
      d(i)=1.0d0                                                        
      if (i.eq.1) go to 40                                              
      ii=i-1                                                            
      do 30 j=1,ii                                                      
30    l(i,j)=0.0d0                                                      
40    continue                                                          
      do 60 j=1,n                                                       
      if (.not.basic(j)) go to 60                                       
      do 50 i=1,m                                                       
50    v(i)=a(i,j)                                                       
      dummya=1.0d0                                                      
      dummyz=c(j)                                                       
      call rank1                                                        
60    continue                                                          
      do 80 j=1,m                                                       
      do 70 i=1,m                                                       
70    v(i)=0.0d0                                                        
      v(j)=1.0d0                                                        
      dummya=-1.0d0                                                     
      dummyz=0.0d0                                                      
80    call rank1                                                        
      if (qerror) return                                                
      call pisolv                                                       
      call setx                                                         
      return                                                            
      end                                                               
      subroutine bksolv                                                 
      implicit logical (q),real * 8 (a-h,o-p,r-z)                       
      logical qerror,basic                                              
      real * 8 l                                                        
      real*8 v(6)                                                    
      common /lp1/ a(6,103),b(6),c(103),x(103),pi(6),cost,qerror        
      common /lp2/ num(103),basic(103)                                  
      common /lp3/ m,n,nlimit,nsys,nartif                               
      common /lp4/ dx(6),d(6),bc(6),l(6,6),cs,ialpha,ibeta,itn          
      common /lp5/ u(6),vdummy(6)                                                 
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
      do 5 j=1,m                                                        
5     v(j)=u(j)                                                         
c-----solve  (ldl*)v  =  u                                              
      v(1)=u(1)                                                         
      if (m.gt.1) go to 10                                              
      v(m)=v(m)/d(m)                                                    
      return                                                            
10    do 30 i=2,m                                                       
      s=0.0d0                                                           
      ii=i-1                                                            
      do 20 j=1,ii                                                      
20    s=s+l(i,j)*v(j)                                                   
30    v(i)=u(i)-s                                                       
      v(m)=v(m)/d(m)                                                    
      mm=m-1                                                            
      do 50 i=1,mm                                                      
      k=m-i                                                             
      ii=k+1                                                            
      s=0.0d0                                                           
      do 40 j=ii,m                                                      
40    s=s+l(j,k)*v(j)                                                   
50    v(k)=v(k)/d(k)-s                                                  
      do 60 j=1,m                                                       
60    u(j)=v(j)                                                         
      return                                                            
      end                                                               
      subroutine pisolv                                                 
      implicit logical (q),real * 8 (a-h,o-p,r-z)                       
      logical qerror,basic                                              
      real * 8 l                                                        
      common /lp1/ a(6,103),b(6),c(103),x(103),pi(6),cost,qerror        
      common /lp2/ num(103),basic(103)                                  
      common /lp3/ m,n,nlimit,nsys,nartif                               
      common /lp4/ dx(6),d(6),bc(6),l(6,6),cs,ialpha,ibeta,itn          
      common /lp5/ u(6),v(6)                                            
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
c-----solve  l*pi  =  bc                                                
      pi(m)=bc(m)                                                       
c      write(nlist,334)pi(m),m
334   format('pisolv1',10x,f12.4,10x,i4)                                                       
      mm=m-1                                                            
      if (m.eq.1) return                                                
      do 20 i=1,mm                                                      
      k=m-i                                                             
      ii=k+1                                                            
      s=0.0d0                                                           
      do 10 j=ii,m                                                      
c      write(nlist,335)s,l(j,k),pi(j),j
335   format('pisolv2',10x,f12.4,10x,f12.4,10x,f12.4,10x,i4)
10    s=s+l(j,k)*pi(j)                                                  
20    pi(k)=bc(k)-s                                                     
c      write(nlist,33)pi(k),k
33    format('pisolv3',10x,f12.4,10x,i6)                                                       
      return                                                            
      end                                                               
      subroutine findal                                                 
      implicit logical (q),real * 8 (a-h,o-p,r-z)                       
      logical qerror,basic                                              
      real * 8 l                                                        
      common /lp1/ a(6,103),b(6),c(103),x(103),pi(6),cost,qerror        
      common /lp2/ num(103),basic(103)                                  
      common /lp3/ m,n,nlimit,nsys,nartif                               
      common /lp4/ dx(6),d(6),bc(6),l(6,6),cs,ialpha,ibeta,itn          
      common /lp5/ u(6),v(6)                                            
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
c-----find a variable to enter the basis                                
      ialpha=0                                                          
      cs=-1.0d-8                                                        
c-----calculate reduced cost                                            
      do 20 j=1,n                                                       
      if (basic(j).or.num(j).lt.0) go to 20                             
      s=0.0d0                                                           
      do 10 i=1,m                                                       
10    s=s+a(i,j)*pi(i)                                                  
      s=c(j)-s                                                          
      if (cs.le.s) go to 20                                             
      cs=s                                                              
      ialpha=j                                                          
20    continue                                                          
      return                                                            
      end                                                               
      subroutine fndbta                                                 
      implicit logical (q),real * 8 (a-h,o-p,r-z)                       
      logical qerror, basic                                             
      real * 8 l                                                        
      common /lp1/ a(6,103),b(6),c(103),x(103),pi(6),cost,qerror        
      common /lp2/ num(103),basic(103)                                  
      common /lp3/ m,n,nlimit,nsys,nartif                               
      common /lp4/ dx(6),d(6),bc(6),l(6,6),cs,ialpha,ibeta,itn          
      common /lp5/ u(6),v(6)                                            
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
c-----find which variable leaves the basis                              
      ibeta=0                                                           
      k=0                                                               
      xnew=1.0e30                                                       
      do 10 j=1,n                                                       
      if (.not.basic(j)) go to 10                                       
      k=k+1                                                             
      dxk=dx(k)                                                         
      if (dxk.le.1.0d-6) go to 10                                       
      t=x(j)/dxk                                                        
      if (xnew.le.t) go to 10                                           
      xnew=t                                                            
      ibeta=j                                                           
10    continue                                                          
      if (ibeta.eq.0) return                                            
c-----update activity vector and cost.                                  
      k=0                                                               
      do 30 j=1,n                                                       
      if (.not.basic(j)) go to 30                                       
      k=k+1                                                             
      t=x(j)-dx(k)*xnew                                                 
      if (t.lt.0.0d0) go to 20                                          
      x(j)=t                                                            
      go to 30                                                          
20    x(j)=1.0d-20                                                      
30    continue                                                          
      x(ialpha)=xnew                                                    
      cost=cost+cs*xnew                                                 
      if(num(ibeta).lt.0)nartif=nartif-1                                
      return                                                            
      end                                                               
      subroutine rank1                                                  
      implicit logical (q),real * 8 (a-h,o-p,r-z)                       
      logical qerror,basic                                              
      real * 8 l                                                        
      common /lp1/ a(6,103),b(6),c(103),x(103),pi(6),cost,qerror        
      common /lp2/ num(103),basic(103)                                  
      common /lp3/ m,n,nlimit,nsys,nartif                               
      common /lp4/ dx(6),d(6),bc(6),l(6,6),cs,ialpha,ibeta,itn          
      common /lp5/ u(6),v(6)                                            
      common /lp6/ dummya,dummyz                                        
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                           
c     modifies lower triangular l and diagonal d so that                
c                   ldl*  =   ldl*  +  alpha * vv*                      
c     assuming the modified decomposition exists.                       
c-----also applies the transformation to vector bc.                     
      alpha=dummya                                                      
      z=dummyz                                                          
      do 50 k=1,m                                                       
      p=v(k)                                                            
      gamma=alpha*p                                                     
      t=d(k)                                                            
      delta=t+gamma*p                                                   
      if (alpha.ge.0.0d0.or.delta.ge.1.0d-12) go to 30                  
      write (nlist,10)                                                  
10    format (' ldl''will not be positive definite.')                   
      write (nlist,20) t,delta,k                                        
20    format (' old and new diagonal elements-',2(1pg20.11),i9)         
      qerror=.true.                                                     
      return                                                            
30    d(k)=delta                                                        
      beta=gamma/delta                                                  
c      write(nlist,334)gamma,delta
334   format('rank1',10x,f12.4,10x,f12.4)
      alpha=alpha*t/delta                                               
      t=bc(k)                                                           
      z=z-t*p                                                           
      bc(k)=t+beta*z                                                    
      kk=k+1                                                            
      if (kk.gt.m) go to 50                                             
      do 40 i=kk,m                                                      
      t=l(i,k)                                                          
      v(i)=v(i)-t*p
c      write(nlist,33)t,beta,v(i),i
33    format('rank1',10x,f12.4,10x,f12.4,10x,f12.4,i6)                                                     
40    l(i,k)=t+beta*v(i)                                                
50    continue                                                          
      return                                                            
      end                                                               
      subroutine daytim(ix,iy)                                    
      implicit logical (q)                                 
      common /logic/ qd,qt,qr,qdata,qbasis                              
      integer*4 iday(3)
      common /iounit/ nlist,nread,nelle,nout,ndata,ntemp                          
 
      call datime(iday,isec,0)
      ithour=isec/3600
      itmin=isec/60-60*ithour
      itsec=isec-3600*ithour-60*itmin
 
      if (qr) write (nlist,1) iday(1),iday(2),iday(3)                                          
1     format (10x,6hdate: ,i2,1h/,i2,1h/,i4)                          
      if (qr) write (nlist,2) ithour,itmin,itsec                                   
2     format (10x,'time:',i5,2x,'hours',5x,i5,'  minutes',5x,i5         
     1     ,'  seconds after midnight'/10x,40('-'))
       ix=iday(1)
       iy=isec                  
      return                                                            
      end                                                               
c      real function oldrand(xr)                                            
c-----the value jrand is transmitted via common                         
c      common pi,root2,root3,pi90,pifac,jrand                            
c      integer*4 toolbx,randm
c      include :include files:quickdraw.inc
c      jrand=4
c      randm= toolbx(random)
c      rand=(randm+32768.0)/65535.0
 
c      return                                                            
c      end                                                               

c -----------------------------------------------------------

      subroutine init 
      integer map(102,102)
      real mob
      integer*2 age(102,102)
      
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
      common /tbhdrx/ xlmap(3,4,30000)
      common pi,root2,root3,pi90,pifac,jrand
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)      
      common /agecom/ age

      xlno=float(nxl)**0.5
      nxs=int(xlno+1.0)**2.0
      test = aint(xlno)
      if(test.eq.xlno) nxs=nxl
      
      if(nxl.eq.1) then
         indgn=322
      elseif(nxl.eq.2) then
         indgn=2
      elseif(nxl.eq.0) then
         indgn=10000
      else
         indgn=nxs
      endif           
      
      do 10 j=1,102,101
         do 20 k=1,102
           map(j,k)=0
           map(k,j)=0
   20    continue
   10 continue

      do 5 j=1,102
         do 6 k=1,102
           age(j,k)=0
   6     continue
   5  continue

      do 30 i=1,indgn       
555   ang1=2.0*randz(xr)-1.0                                             
      ang2=2.0*randz(xr)-1.0                                             
      ang3=2.0*randz(xr)-1.0                                             
      zmag=sqrt(ang1**2+ang2**2+ang3**2)                                
      if (zmag.lt.0.00001.or.zmag.gt.1.00) go to 555                     
      ang1=randz(xr)*2.0*pi                                              
      ang2=acos(ang3/zmag)                                              
      ang3=randz(xr)*2.0*pi                                              
      a1=ang1/pifac                                                     
      a2=ang2/pifac                                                     
      a3=ang3/pifac                                                     
      call euler (rmap,ang1,ang2,ang3)
      call xeqr(i)
      xlmap(1,4,i)=0.0
      xlmap(2,4,i)=0.0
   30 continue
      
      if(nxl.eq.1) then
         call hexset
      elseif(nxl.eq.2) then
         xlmap(1,4,1)=5000
         xlmap(1,4,2)=5000
         do 40 n=2,101
           do 50 m=2,101
             if(m.gt.75.or.m.lt.25) then
                map(n,m)=1
             else
                map(n,m)=2
             endif
   50      continue
   40    continue
      elseif(nxl.eq.0) then
         indx=1
      do 80 n=2,101
        do 90 m=2,101
           map(n,m)=indx
           indx=indx+1
           xlmap(1,4,map(n,m))=1
   90       continue
   80    continue         
      else
         it2=int(100.0/xlno)+1
      do 60 n=2,101
        do 70 m=2,101
           it=int((float(n)-1.0)/it2)
           it3=int((float(m)-1.0)/it2)
           map(n,m)=int(it*(int(xlno)+1))+it3+1
           xlmap(1,4,map(n,m))=xlmap(1,4,map(n,m))+1
   70       continue
   60    continue
      endif

      return
      end

c -----------------------------------------------------------

      subroutine strain
      integer map(102,102)
      real mob
      integer strip(100),i(100),gamma
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex

     
      gamma=int(xgamma*100.0)     
      if(gamma.eq.0.0) return

      i(1)=2
      i(gamma+2)=102
      space=100.0/float(gamma)
      igam=gamma+1

      do 10 j=2,gamma+1
         ilow=int(((float(j)-2.0)*space))+2
         ihigh=int(((float(j)-1.0)*space))+1
         if(j.eq.igam) ihigh=101
         idel=ihigh-ilow
c         call ran
         i(j)=int(randz(xr)*(idel+1))+ilow
   10 continue

      do 20 l2=2,gamma+1
         do 30 m=i(l2),i(l2+1)-1

            do 40 n=1,100
               strip(n)=map(m,n+1)
   40       continue

            do 50 n2=2,101
               inc=n2+l2-2
               if(inc.eq.100) then
                  map(m,n2)=strip(100)
               else
                  map(m,n2)=strip(mod(inc,100))
               endif
   50       continue

   30    continue
   20 continue
    
      return
      end

c -----------------------------------------------------------

      subroutine drx 
      integer map(102,102),trials
      real mob                
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
      common /newpar/ itdef,rrate,hrdrate
      
      if(l.le.itdef) then
         demob=mob
      else
         demob=mob*(rrate**(l-itdef))
      endif

      trials=int((demob+gbemob)*10000)
      if(trials.eq.0) return

      do 10 lsec=1,trials
          
c         call ran
         j=int(randz(xr)*99.9999)+2
c         call ran
         k=int(randz(xr)*99.9999)+2
c         call ran
      delgbe=gbemob/(demob+gbemob)
      if(randz(xr).gt.delgbe) then
     
       f=(mod(j,2)*2)-1
c       call ran
       g=int(randz(xr)*6.0)+1
       
       if(g.eq.1) then
          j2=j+1
          k2=k+0
       elseif(g.eq.2) then
          j2=j+1
          k2=k+f
       elseif(g.eq.3) then
          j2=j+0
          k2=k-1
       elseif(g.eq.4) then
          j2=j+0
          k2=k+1
       elseif(g.eq.5) then
          j2=j-1
          k2=k+0
       elseif(g.eq.6) then
          j2=j-1
          k2=k+f
       endif
  
       if(map(j2,k2).eq.0) then
          if(k2.eq.1) then
         k2=k+99
          elseif(k2.eq.102) then
         k2=k-99
          endif
       endif
  
       if(map(j,k).ne.map(j2,k2)) then
          if(map(j2,k2).ne.0) then
         call gbm_rr
          endif
       endif
       
         else
        call gbe
      endif
   10 continue

      return
      end
      
c -----------------------------------------------------------

      subroutine gbe

      integer map(102,102),f,f2,g
      integer*2 age(102,102)
      real mob
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
      common /tbhdrx/ xlmap(3,4,30000)
      common /agecom/ age

c      call ran
      j=int(randz(xr)*99.99999)+2
c      call ran
      k=int(randz(xr)*99.99999)+2
      
      nprob = 1
      f=(mod(j,2)*2)-1
      nonbrs=6
      
      do 10 g=1,6
      
         if(g.eq.1) then
        j2=j+1
        k2=k+0
         elseif(g.eq.2) then
        j2=j+1
        k2=k+f
         elseif(g.eq.3) then
        j2=j+0
        k2=k-1
         elseif(g.eq.4) then
        j2=j+0
        k2=k+1
         elseif(g.eq.5) then
        j2=j-1
        k2=k+0
         elseif(g.eq.6) then
        j2=j-1
        k2=k+f
         endif
    
         if(map(j2,k2).eq.0) then
        if(k2.eq.1) then
           k2=k+99
        elseif(k2.eq.102) then
           k2=k-99
        else
           nonbrs=nonbrs-1
        endif
         endif
    
    
      
         if(map(j2,k2).gt.0) then
        if(map(j,k).ne.map(j2,k2)) then
            nprob=nprob+1
        else
            nprob=nprob-1
        endif
         endif
     
   10     continue
      
      ncore = nprob+nonbrs
      if(ncore.eq.1) goto 100
      
      if(nonbrs .eq.0) then
          prob=1.0
      else
          prob=float(nprob)/float(nonbrs)
      endif
      
c      call ran
      
      if(randz(xr).lt.prob) then
  456        f=(mod(j,2)*2)-1
c         call ran
         g=int(randz(xr)*5.9)+1
         
         if(g.eq.1) then
        j2=j+1
        k2=k+0
         elseif(g.eq.2) then
        j2=j+1
        k2=k+f
         elseif(g.eq.3) then
        j2=j+0
        k2=k-1
         elseif(g.eq.4) then
        j2=j+0
        k2=k+1
         elseif(g.eq.5) then
        j2=j-1
        k2=k+0
         elseif(g.eq.6) then
        j2=j-1
        k2=k+f
         endif
         
           
         if(map(j2,k2).eq.0) goto 456
         if(map(j,k).eq.map(j2,k2)) then
                continue
c            goto 456
         else
            xlmap(1,4,map(j2,k2))=xlmap(1,4,map(j2,k2))+1                   
            xlmap(1,4,map(j,k))=xlmap(1,4,map(j,k))-1                       
            map(j,k)=map(j2,k2)
            age(j,k)=l-1
         endif
      
      endif
  100 continue      
  
      return
      
      end
            
c -----------------------------------------------------------

      subroutine gbm_rr
      integer map(102,102)
      integer*2 age(102,102)
      real mob
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
      common /tbhdrx/ xlmap(3,4,30000)
      common /agecom/ age
      common /newpar/ itdef,rrate,hrdrate
      
      pi=3.1415927
      con=pi/180.0
      pow = 3.0
      
      
      age1=float(l-1-age(j,k))
      age2=float(l-1-age(j2,k2))

      age1=aint((age1*xgamma)/0.1)
      age2=aint((age2*xgamma)/0.1)
      
      if (age1.gt.6.0) age1=6.0
      if (age2.gt.6.0) age2=6.0
      
      es1=xlmap(3,4,map(j,k))                                                       
      es2=xlmap(3,4,map(j2,k2)) 
      
      es1=es1*(1.0-(hrdrate**age1))
      es2=es2*(1.0-(hrdrate**age2))
      
c      call ran
     
      if(randz(xr).lt.pnuke.and.l.le.itdef) then
        if(xlmap(1,4,map(j,k)).gt.20.0.and.
     +      xlmap(3,4,map(j,k)).gt.0.20) then
          call nuke
        endif
      else             
         diff=es2-es1
         call mig(diff)
      endif

      return
      end

c -----------------------------------------------------------

      subroutine mig(diff)
      integer map(102,102)
      integer*2 age(102,102)
      real mob
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
      common /tbhdrx/ xlmap(3,4,30000)
      common /agecom/ age
      
c      call ran
      test=randz(xr)
      
      if(diff.gt.0) then                                                
         if(test.lt.diff) then                                          
            xlmap(1,4,map(j2,k2))=xlmap(1,4,map(j2,k2))-1                   
            xlmap(1,4,map(j,k))=xlmap(1,4,map(j,k))+1                       
            map(j2,k2)=map(j,k)                                         
            age(j2,k2)=l-1                                                            
         endif                                                          
      else                                                              
         if(test.lt.-diff) then                                         
            xlmap(1,4,map(j2,k2))=xlmap(1,4,map(j2,k2))+1                   
            xlmap(1,4,map(j,k))=xlmap(1,4,map(j,k))-1                       
            map(j,k)=map(j2,k2)                                         
            age(j,k)=l-1                                                                                                                                    
         endif                                                          
      endif

      return
      end

c -----------------------------------------------------------

      subroutine nuke
      integer map(102,102),orig
      real mob,nmap(3,3),axis(3),nmap1(3,3),rmap2(3,3)
      real omap(3)
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
      common /tbhdrx/ xlmap(3,4,30000)
      common pi,root2,root3,pi90,pifac,jrand
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)
      
      range=(angnuke/180.0)*pi
      axis(1)=0.0
      axis(2)=0.0
      axis(3)=1.0
      
      indgn=indgn+1
      
      ind=map(j,k)
      call reqx(ind)
      call change(axis,omap,rmap)
      call firo(omap,phi,rho)
      
555   ang1=2.0*randz(xr)-1.0                                             
      ang2=2.0*randz(xr)-1.0                                             
      ang3=2.0*randz(xr)-1.0                                             
      zmag=sqrt(ang1**2+ang2**2+ang3**2)                                
      if (zmag.lt.0.00001.or.zmag.gt.1.00) go to 555                     
      ang1=randz(xr)*2.0*pi                                              
      ang2=acos(ang3/zmag)                                              
      ang3=randz(xr)*2.0*pi                                              
      a1=ang1/pifac                                                     
      a2=ang2/pifac                                                     
      a3=ang3/pifac                                                     
      call euler (rmap,ang1,ang2,ang3)
      

      call change(axis,omap,rmap)
      call firo(omap,phi1,rho1)
      
      dphi=abs(phi1-phi)
      drho=abs(rho1-rho)
      
      if(dphi.gt.range.or.drho.gt.range) go to 555

      call xeqr(indgn)
      
      xlmap(1,4,indgn)=0                                                  
      xlmap(2,4,indgn)=float(l-1)
      
      test=float(mod(j,2))         

      if(test.eq.0.0) then
         ibit=-1
      else
         ibit=0
      endif
      
      it=0
      orig=map(j,k)

      do 10 ir=j-2,j+2

         it=it+1
         if(it.eq.1.or.it.eq.5) then
            il=k-1
            ih=k+1
         elseif(it.eq.2.or.it.eq.4) then
            il=k-1+ibit
            ih=k+2+ibit
         else
            il=k-2
            ih=k+2
         endif
 
         do 20 ic=il,ih
            call set2(ir,ic,orig)
   20    continue
   10 continue

      return
      end

c -----------------------------------------------------------

      subroutine set2(ir,ic,orig)
      integer*2 age(102,102)
      integer map(102,102),orig
      real mob
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
      common /tbhdrx/ xlmap(3,4,30000)                                   
      common /agecom/ age                
                                                                        
      if(ir.lt.2.or.ir.gt.101.or.ic.lt.2.or.ic.gt.101) return             
      if(map(ir,ic).ne.orig) return                                     
                                                                        
      xlmap(1,4,orig)=xlmap(1,4,orig)-1                         
      map(ir,ic) = indgn                                                
      xlmap(1,4,indgn)=xlmap(1,4,indgn)+1
      age(ir,ic)=l-1
      return                                      


      end

c -----------------------------------------------------------

      subroutine hexset
      integer map(102,102),odd,even,total,p,r,c
      real mob
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
      common /tbhdrx/ xlmap(3,4,30000)

      open(unit=20,file='gsfab.in',status='old')                                              
      
      do 10 m=2,101
         read(20,20)(map(m,mm),mm=2,51)
   20    format(50i4)
   10 continue
      do 30 m=2,101
         read(20,20)(map(m,mm),mm=52,101)
   30 continue
    
      close(20)

      do 40 r=2,101
        do 50 c=2,101
          total=map(r,c)       
          xlmap(1,4,total)=xlmap(1,4,total)+1
   50   continue
   40 continue
   
      return
      end

c -----------------------------------------------------------

      subroutine hex(ir,ic,total)
      integer map(102,102),total,good,r,c
      real mob
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex

      good=0  
      inc=mod(ir,2)
      it=0

      do 10 r=ir-3,ir+3
        
         it=it+1
         if(it.eq.1.or.it.eq.7) then
            il=ic-2+inc
            ih=ic+1+inc
         elseif(it.eq.2.or.it.eq.6) then
            il=ic-2
            ih=ic+2
         elseif(it.eq.3.or.it.eq.5) then
            il=ic-3+inc
            ih=ic+2+inc
         elseif(it.eq.4) then
            il=ic-3
            ih=ic+3
         endif
        
         do 20 c=il,ih

            call set(r,c,total,good)

   20    continue

   10 continue
      
      if(good.eq.0) total=total-1

      return
      end

c -----------------------------------------------------------

      subroutine set(r,c,total,good)
      integer map(102,102),total,good,r,c
      real mob
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
      common /tbhdrx/ xlmap(3,4,30000)
      
      if(r.lt.2.or.r.gt.101.or.c.lt.2.or.c.gt.101) return   

      good=1

      map(r,c)=total 
      
      xlmap(1,4,total)=xlmap(1,4,total)+1


      return
      end

c -----------------------------------------------------------           
      subroutine plotmap                                                
                                                                        
                                                                        
c      include :include files:quickdraw.inc
      integer map(102,102),p,q                                            
      real caxis(3),mob,lam,well(3),aaxis(3),axis(3),xmap(3,3)                                       
      integer*2 center(2)                                               
      integer   size                                                    
      integer*2 oval(4),rect(4)                                         
      integer*4 picture,region,grafptr                                  
c      integer*4 toolbx                                                  
      integer*4 xl1,yl1                                                 
      character*32 string                                              
      character*32 fmt                                                  

      
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
      common /tbhdrx/ xlmap(3,4,30000)                                   
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)                                                                  
                                                                        
      pi=asin(1.0)*2.0
      pifac=pi/180.0
      deg89=89.9*pifac
      angle=0.0*pifac                                                                                                                     
      con=pi/180.0                                                      
      root2=sqrt(2.0)
      call euler(xmap,angle,0.0,0.0)
      
      caxis(1)=0.0
      caxis(2)=0.0
      caxis(3)=1.0
      
      aaxis(1)=1.0
      aaxis(2)=0.0
      aaxis(3)=0.0
      
                                                                        
c      call toolbx (setrect,rect,0,0,800,600)                         
c      call toolbx (cliprect,rect)                                    
                                     
c      call toolbx (setrect,rect,0,0,100,100)                         
c      picture=toolbx(openpicture,rect)                               
                                     
      if(l.eq.1) then                                                
c     call toolbx(setrect,rect,0,0,500,400)                       
c     call toolbx(eraserect,rect)                                 
      endif                                                          
                                                                        
c      call toolbx(textsize,6)                                           
c      call toolbx(textfont,4)                                           
                                                                        
                                                                        
                                                                        
                                                                        
                                                                        
      do 30 p=2,101                                                      
                                                                        
         do 40 q=2,101                                                   
                                                                        
            ip=p+1                                                      
            oval(1)=int((ip-1)*1.3)
        oval(2)=int(((q-1)*1.5)+(mod(p,2)*0.75))
        oval(3)=oval(1)+3
        oval(4)=oval(2)+3                                           
            n=map(p,q) 
            axis(1)= xlmap(3,1,n)                                                    
            axis(2)= xlmap(3,2,n)                                                     
            axis(3)= xlmap(3,3,n) 
c        call reqx(n)
c           call change (axis,axis,rmap)                                      
            call firo (axis,phi,rho)                                                                                           
        
                                                                        
                                                                        
   40    continue                                                       
                                                                        
   30 continue                                                          
                                                                        
                                                                        
      if(l.gt.1.and.xgamma.gt.0.0) then                                 
         gam=(float(l)-1)*xgamma                                        
         lam=((gam**2.0)+2.0+(gam*(((gam**2.0)+4.0)**0.5)))/2.0         
         tantheta=gam/(1.0+(gam**2.0)-(1.0/lam))                        
         theta=atan(tantheta)                                           
                                                                        
         xline=cos(theta)                                               
         yline=sin(theta)                                               
                                                                        
         center(1)=200                                                  
         center(2)=35                                                   
                                                                        
         xl1=int(center(1)+(xline*30.0))                                
         yl1=int(center(2)-(yline*30.0))                                
                                                                     
      endif                                                             
                                                                        
      oval(1)=5                                                         
      oval(2)=170                                                       
      oval(3)=65                                                        
      oval(4)=230                                                       
            
                                                 
                                    
      fmt=' (4hinc=,i3)'
      xnumero=float(l-1)
      call numword(fmt,xnumero,1,1,150)                                    
                                                        
      fmt=' (4hmob=,f5.2)'
      call numword(fmt,mob,2,77,150)                                    
                                                        
      fmt=' (6hpnuke=,f5.4)'
      call numword(fmt,pnuke,2,115,150)                                    
                                                        
      fmt=' (4hang=,f5.1)'
      call numword(fmt,angnuke,2,162,150)                                    
                                                        
      fmt=' (6hgamma=,f5.2)'
      xnumero=float(l-1)*xgamma
      call numword(fmt,xnumero,2,205,150)                                    
                                                                       
                                                                        
                                                                        
                                                                        
      return                                                            
      end                                                               
c -----------------------------------------------------------

      subroutine numword(fmt,xnumero,iorf,ix,iy)
      
c      include :include files:quickdraw.inc
      
c     character*256 str255
      character*32 string                                             
      character*32 fmt
      
      if(iorf.eq.1) then
        numero=int(xnumero)
        write(string,fmt) numero 
      else
        write(string,fmt) xnumero
      endif
      
       
      return
      end
      
      
c -----------------------------------------------------------

c      subroutine ran
c      integer map(102,102)
c      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob      
c      integer*4 toolbx,randm
c      include :include files:quickdraw.inc
      
c      randm= toolbx(random)
c      rnd=(randm+32768.0)/65535.0
c      rnd=rand()
c      return
c      end
 
c -----------------------------------------------------------
      subroutine savewds


c      include :include files:quickdraw.inc
      integer map(102,102),p,q
      real caxis(3),mob                                                                                   
      integer   size                                                                              
c      integer*4 toolbx                                                                       
      character*32 string                                               
      character*20 fmt                                                  
      
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /p2/ nincs,mob,tbhrate,porb,pnuke,angnuke,xgamma,iex
    
    
      
                           
      fmt=' (4hinc=,i3)'
      xnumero=float(l-1)
      call numword(fmt,xnumero,1,2,300)                                    
                                                        
      fmt=' (4hmob=,f5.2)'
      call numword(fmt,mob,2,144,300)                                    
                                                        
      fmt=' (6hpnuke=,f5.4)'
      call numword(fmt,pnuke,2,230,300)                                    
                                                        
      fmt=' (4hang=,f5.1)'
      call numword(fmt,angnuke,2,324,300)                                    
                                                        
      fmt=' (6hgamma=,f5.2)'
      xnumero=float(l-1)*xgamma
      call numword(fmt,xnumero,2,410,300)                                    

  
      return
      
      end


      
c -----------------------------------------------------------           
                                                                        
      subroutine xeqr(i)                                                
      common /tbhdrx/ xlmap(3,4,30000)                                   
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)            
                                        
                                                                        
      do 10 jxl=1,3                                                     
        do 20 kxl=1,3                                                   
           xlmap(jxl,kxl,i)=rmap(jxl,kxl)                               
   20   continue                                                        
   10 continue                                                          
                                                                        
      return                                                            
      end                                                               
c -----------------------------------------------------------           
                                                                        
      subroutine reqx(i)                                                
      common /tbhdrx/ xlmap(3,4,30000)                                   
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)            
                                                                        
      do 10 jxl=1,3                                                     
        do 20 kxl=1,3                                                   
           rmap(jxl,kxl)=xlmap(jxl,kxl,i)                               
   20   continue                                                        
   10 continue                                                          
                                                                        
      return                                                            
      end                                                               
c -----------------------------------------------------------           
                                                                        
      subroutine rtrx(i)                                                
      common /tbhdrx/ xlmap(3,4,30000)                                   
      common /grain/ grad(3,3),vort(3,3),rmap(3,3),zmap(3,3)            
                                                                        
      do 10 jxl=1,3                                                     
        do 20 kxl=1,3                                                   
           rmap(jxl,kxl)=xlmap(kxl,jxl,i)                               
   20   continue                                                        
   10 continue                                                          
                                                                        
      return                                                            
      end                                                               
c -----------------------------------------------------------

      subroutine cpy(a,b)                                            
      dimension a(3,3),b(3,3)                                           
      do 10 i=1,3                                                       
      do 10 j=i,3                                                                                                                 
      b(i,j)=a(i,j)                                                                                                              
10    continue                                                          
      return                                                            
      end
      
c -----------------------------------------------------------
      
      subroutine splot(rmap,axis,ix,iy,rad,size)                                             
                                                                                                                                                                                                                        
      real axis(3),rmap(3,3),xmap(3,3),well(3)
      integer*2 center(2)                                               
      integer   size                                                    
      integer*2 oval(4),rect(4)                                         
      integer*4 picture,region,grafptr                                  
c      integer*4 toolbx
c      include :include files:quickdraw.inc
          
      pi=asin(1.0)*2.0
      pifac=pi/180.0
      deg89=89.9*pifac
      angle=0.0*pifac
      root2=sqrt(2.0)
      
      call euler(xmap,angle,0.0,0.0)
      
      call change(axis,well,rmap)
      call change(well,well,xmap)
      call firo(well,phi,rho)
      if(rho.lt.deg89) goto 930
      rho=pi-rho
      radius=root2*sin(rho/2.0)*rad
      x=cos(phi)*radius
      y=sin(phi)*radius
      if(rho.lt.deg89) goto 940
      rho=pi-rho
  930 phi=phi+pi
      radius=root2*sin(rho/2.0)*rad
      x=cos(phi)*radius
      y=sin(phi)*radius
 
      
      
  940 oval(1)=iy-int(y)-(size/2)                                                      
      oval(2)=int(x)+ix-(size/2)                                                       
      oval(3)=oval(1)+size                                                    
      oval(4)=oval(2)+size                                                      
c      call toolbx(frameoval,oval)
      
      
      return                                                            
      end                                                                        

c -----------------------------------------------------------

c str255: converts a fortran string to a pascal lstring
c

      character*256 function str255(string)

      character*(*) string
    
c      str255 = char(len(trim(string)))//string
    
      end
 
c -----------------------------------------------------------
      subroutine mapout
      
      real axis(3)
      integer map(102,102),p,q,row(100)
      common /drxpar/ map,l,j,k,j2,k2,indgn,rnd,noex,nxl,gbemob
      common /tbhdrx/ xlmap(3,4,30000)                                   
      
      write(18,60)l-1
   60 format(i4)
   
      do 10 m=2,101
         do 50 i=2,101
              n=map(m,i) 
            axis(1)= xlmap(3,1,n)                                                    
            axis(2)= xlmap(3,2,n)                                                     
            axis(3)= xlmap(3,3,n) 
            call firo (axis,phi,rho)
        phi2=phi*180.0/3.14159
        phi2=abs(amod(phi2,180.0))
        if(phi2.gt.90.0) phi2=180.0-phi2
        phi2=phi2+5
        iphi2=int(phi2/10.0)
        rho2=abs(rho*180.0/3.14159)
        if(rho2.gt.90.0) rho2=180.0-rho2
        rho2=90.0-rho2
        irho2=int(rho2/10.0)
        it=(irho2*10)+iphi2
        row(i-1)=it
   50    continue         

         write(18,20)(row(mm),mm=1,100)
   20    format(100i2.2)
   10 continue
   
      return
      end
c======================================================================== 
c this random number generator originally appeared in "toward a universal 
c random number generator" by george marsaglia and arif zaman. 
c florida state university report: fsu-scri-87-50 (1987)
c 
c it was later modified by f. james and published in "a review of pseudo-
c random number generators" 
c 
c this is the best known random number generator available.
c       (however, a newly discovered technique can yield 
c         a period of 10^600. but that is still in the development stage.)
c
c it passes all of the tests for random number generators and has a period 
c   of 2^144, is completely portable (gives bit identical results on all 
c   machines with at least 24-bit mantissas in the floating point 
c   representation). 
c 
c the algorithm is a combination of a fibonacci sequence (with lags of 97
c   and 33, and operation "subtraction plus one, modulo one") and an 
c   "arithmetic sequence" (using subtraction).
c
c on a vax 11/780, this random number generator can produce a number in 
c    13 microseconds. 
c======================================================================== 

      real function randz(xr)
      real temp(1)
      integer len
      
      len = 1
      call ranmar(temp,len)
      
      randz=temp(1)
      return
      end
 
      subroutine rmarin(ij,kl)
c this is the initialization routine for the random number generator ranmar()
c note: the seed variables can have values between:    0 <= ij <= 31328
c                                                      0 <= kl <= 30081
c the random number sequences created by these two seeds are of sufficient 
c length to complete an entire calculation with. for example, if sveral 
c different groups are working on different parts of the same calculation,
c each group could be assigned its own ij seed. this would leave each group
c with 30000 choices for the second seed. that is to say, this random 
c number generator can create 900 million different subsequences -- with 
c each subsequence having a length of approximately 10^30.
c 
c use ij = 1802 & kl = 9373 to test the random number generator. the
c subroutine ranmar should be used to generate 20000 random numbers.
c then display the next six random numbers generated multiplied by 4096*4096
c if the random number generator is working properly, the random numbers
c should be:
c           6533892.0  14220222.0  7275067.0
c           6172232.0  8354498.0   10633180.0


      real u(97), c, cd, cm
      integer i97, j97
      logical test
      common /raset1/ u, c, cd, cm, i97, j97, test

c      data test /.false./

      if( ij .lt. 0  .or.  ij .gt. 31328  .or.
     *    kl .lt. 0  .or.  kl .gt. 30081 ) then
          print '(a)', ' the first random number seed must have a value 
     *between 0 and 31328'
          print '(a)',' the second seed must have a value between 0 and         
     *30081'
            stop
      endif

      i = mod(ij/177, 177) + 2
      j = mod(ij    , 177) + 2
      k = mod(kl/169, 178) + 1
      l = mod(kl,     169) 

      do 2 ii = 1, 97
         s = 0.0
         t = 0.5
         do 3 jj = 1, 24
            m = mod(mod(i*j, 179)*k, 179)
            i = j
            j = k
            k = m
            l = mod(53*l+1, 169)
            if (mod(l*m, 64) .ge. 32) then
               s = s + t
            endif
            t = 0.5 * t
3        continue
         u(ii) = s
2     continue

      c = 362436.0 / 16777216.0
      cd = 7654321.0 / 16777216.0
      cm = 16777213.0 /16777216.0

      i97 = 97
      j97 = 33

      test = .true.
      return
      end

      subroutine ranmar(rvec, len)
c this is the random number generator proposed by george marsaglia in 
c florida state university report: fsu-scri-87-50
c it was slightly modified by f. james to produce an array of pseudorandom
c numbers.
      real rvec(1)
c changed from 
c      real rvec(*)
      real u(97), c, cd, cm
      integer i97, j97
      logical test
      common /raset1/ u, c, cd, cm, i97, j97, test
 
      integer ivec
 
      if( .not. test ) then
         print '(a)',' call the init routine (rmarin) before calling ran        
     *mar'  
         stop
      endif

      do 100 ivec = 1, len
         uni = u(i97) - u(j97)
         if( uni .lt. 0.0 ) uni = uni + 1.0
         u(i97) = uni
         i97 = i97 - 1
         if(i97 .eq. 0) i97 = 97
         j97 = j97 - 1
         if(j97 .eq. 0) j97 = 97
         c = c - cd
         if( c .lt. 0.0 ) c = c + cm
         uni = uni - c
         if( uni .lt. 0.0 ) uni = uni + 1.0
         rvec(ivec) = uni
100   continue
      return
      end

c--------------------------------------------------------
      subroutine toolbx()
      return
      end
