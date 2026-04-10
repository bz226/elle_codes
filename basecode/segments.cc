#include "segmentsP.h"
#include "segments.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "unodesP.h"
#include "unodes.h"
#include "nodes.h"
#include "interface.h"
#include "file.h"
#include "attribarray.h"
#include "error.h"
#include "general.h"
#include "runopts.h"
#include "polygon.h"
#include "tripoly.h"
#include "polyutils.h"
#include "../utilities/gpc/gpcclip.h"
#include "check.h"
#include "convert.h"
#include "log.h"
/*#include "timefn.h"*/

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

using std::ios;
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
using std::vector;
using std::list;
using std::pair;

using std::cout;
using std::endl;

Segment::Segment(int id, int id_StartNode, int id_EndNode)
{
    _id = id;
    SetStartNode(id_StartNode);
    SetEndNode(id_EndNode);
}

Segment::~Segment()
{
}

Unode *Segment::GetStartNode()
{
	return Segment::StartNode;
}

void Segment::SetStartNode(Unode *unode)
{
    Segment::StartNode = unode;
}

void Segment::SetStartNode(int idval)
{
    Segment::StartNode = ElleGetParticleUnode(idval);
}

Unode *Segment::GetEndNode()
{
	return Segment::EndNode;
}

void Segment::SetEndNode(Unode *unode)
{
    Segment::EndNode = unode;
}

void Segment::SetEndNode(int idval)
{
    Segment::EndNode = ElleGetParticleUnode(idval);
}
		
double Segment::GetLength()
{
    return Segment::length;
}

void Segment::SetLength()
{
    Coords xy_start, xy_end;

    if (StartNode && EndNode)
    {
        ElleGetUnodePosition(StartNode->id(), &xy_start);
        ElleGetUnodePosition(EndNode->id(), &xy_end);
    }
    else
    {
        cout << "Tried to Set Segmentlength without a Start/Endnode" << endl;
        exit (0);
    }

    double xdiff = xy_start.x - xy_end.x;
    double ydiff = xy_start.y - xy_end.y;

    length = sqrt(xdiff*xdiff + ydiff*ydiff);
}

void Segment::SetLength(double length)
{
        Segment::length = length;
}

Coords Segment::GetEndNodePos()
{
    Coords xy;
    Segment::EndNode->getPosition(&xy);
    return xy;
}

Coords Segment::GetStartNodePos()
{
    Coords xy;
    Segment::StartNode->getPosition(&xy);
    return xy;
}

//==============================================================================================

vector<Segment> Segments;

using namespace std;
AttributeArray DfltSegmentAttrib(0);

int ElleMaxSegments()
{
    return (Segments.size());
}

int SegmentAttributeRange(int attr,double *min,double *max)
{
    unsigned char set=0;
    double val=0.0, b=0.0, c=0.0;

    if (Segments.size() > 0)
        DfltSegmentAttrib.getAttribute(attr,&val);

    *min = *max = val;

    for (vector<Segment>::iterator it=Segments.begin(); it!=Segments.end(); it++)
    {
        if (it->hasAttribute(attr))
            it->getAttribute(attr,&val);

        if (val<*min) *min=val;
        if (val>*max) *max=val;
    }
    return(0);
}

int ElleReadSegmentAttribData(FILE *fp, char str[], int attr_id)
{

    int err=0, i, j, num, found=0, last=0;

    int max;

    double val=0, val_3[3]={0,0,0};

    max = ElleMaxSegments();

    ElleInitSegmentAttribute(attr_id);

    while (!feof(fp) && !err) {

        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);

        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%lf\n", &val)) != 1)
                return(READ_ERR);
            DfltSegmentAttrib.setAttribute(attr_id,val);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            i = atoi(str);
            if ((num = fscanf(fp,"%lf", &val))!=1)
                return(READ_ERR);
            if (i<max) Segments[i].setAttribute(attr_id,val);
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

void ElleInitSegmentAttribute(int id)
{
// should this also check if Unodes are active ??
    if (!DfltSegmentAttrib.hasAttribute(id))
        DfltSegmentAttrib.initAttribute(id);
}

int ElleReadSegmentData(FILE *fp, char str[])
{

    Segments.clear();

    int err=0, i, num, found=0, last=0, id_start_unode, id_end_unode, test;
    double val=0;

    while (!feof(fp) && !err) {

        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);

        if (str[0] == '#') dump_comments( fp );
        else if (str[0]<'0' || str[0]>'9') return(0);

        else
        {
            i = atoi(str);

            if ((num = fscanf(fp,"%i %i", &id_start_unode, &id_end_unode))!=2)
                return(READ_ERR);
            Segment tmp(i, id_start_unode, id_end_unode);

            Segments.push_back(tmp);

            tmp.SetStartNode(id_start_unode);
            tmp.SetEndNode(id_end_unode);
            tmp.SetLength();

            cout << id_start_unode << " " << id_end_unode << endl;
            cout << tmp.id() << " " << tmp.GetStartNode()->id() << " " << tmp.GetEndNode()->id() << endl;
            cout << tmp.GetLength() << " " << tmp.GetStartNodePos().x << " " << tmp.GetEndNodePos().x << endl << endl;

        }
        num = fscanf(fp,"\n");
    }
    return(err);
}

int ElleWriteSegmentConnections(ofstream &outf)
{
    char label[20];
    int err=0;
    int oldp;
    vector<Segment>::iterator it;

    if (Segments.size() > 0) {
        if (!id_match(FileKeys,SEGMENTS,label)) err=KEY_ERR;
        else outf << label << endl;
        if (!err) {
            oldp = outf.precision(8);
            for (it=Segments.begin();it!=Segments.end() && outf;it++)
                outf << (*it) << endl;
            if (!outf)  err=WRITE_ERR;
            outf.precision(oldp);
        }
    }
    return(err);
}

int ElleWriteSegmentAttributeData(ofstream &outf,int *keys,int count)
{
    int err=0, i, j=0;
    int file_id=NO_VAL, cnt=0, write=0;

    vector<Segment>::iterator it;

    if (Segments.size() > 0) {

        double dflts[MAX_VALS];
        double vals[MAX_VALS];

        for(j=0;j<count;j++)
            DfltSegmentAttrib.getAttribute(keys[j],&dflts[j]);

        outf << SET_ALL;

        for(j=0;j<count;j++) outf << ' ' << dflts[j];
        outf << endl;

        for (it=Segments.begin(); it!=Segments.end() && outf;it++) {
            for(j=0;j<count;j++) {
                vals[j]=dflts[j];
                if (it->hasAttribute(keys[j]))
                    it->getAttribute(keys[j],&vals[j]);
            }

            // shouldn't have attribute if equal to dflt but
            // setAttribute doesn't check

            for(j=0,write=0;j<count;j++)
                if (vals[j]!=dflts[j])
                    write=1;

            if (write) {
                outf << it->id();
                for(j=0;j<count;j++)
                    outf << ' ' << vals[j];
                outf << endl;
            }
        }
        if (!outf)  err=WRITE_ERR;
    }
    return(err);
}

int ElleWriteSegmentData(const char *fname)
{

    char label[20];
    int err=0, i, j;
    int oldp;
    int file_id=NO_VAL, cnt=0;
    double val, val_3[3];
    double dfltval, dfltval_3[3];

    vector<Segment>::iterator it;

    if (Segments.size() > 0) {
        ofstream outf(fname,ios::out|ios::app);
        if (!outf) return(OPEN_ERR);

        cout << "HI! I'M IN ELLEWRITESEGMENTDATA! " << Segments.size() << endl;

        if (err=ElleWriteSegmentConnections(outf)) return(err);

        /* this should work on dflt attrib list which is a
           member of unodearray class */

        int maxa = DfltSegmentAttrib.numAttributes();

        cout << "maxa: " << maxa << endl;

        int *attr_ids = new int[maxa];

        DfltSegmentAttrib.getList(attr_ids, maxa);

        oldp = outf.precision(8);
        
        outf.setf(ios::scientific,ios::floatfield);
        
        for (i=0; i<maxa && !err;i++) {
            if (!id_match(FileKeys,attr_ids[i],label)) {
                cerr << "Ignoring unknown segment attribute " <<
                        attr_ids[i] << endl;
            }
            else {
                outf << label << endl;
                err=ElleWriteSegmentAttributeData(outf,&attr_ids[i],1);
            }
        }

//            switch(attr_ids[i]) {
//            case  CONC_A:
//              file_id=U_CONC_A;
//              if (!id_match(FileKeys,file_id,label)) err = KEY_ERR;
//              outf << label << endl;
//              err=ElleWriteUnodeAttributeData(outf,&attr_ids[i],1);
//              break;
//            case  START_S_X:
//            case  START_S_Y:
//            case  PREV_S_X:
//            case  PREV_S_Y:
//            case  CURR_S_X:
//            case  CURR_S_Y:
//              file_id=U_FINITE_STRAIN;
//              if (!id_match(FileKeys,file_id,label)) err = KEY_ERR;
//              if (!err) outf << label;
//              cnt=0;
//              j=i;
//              while(j<maxa && cnt<NUM_FINITE_STRAIN_VALS &&
//                    id_match(FiniteStrainKeys,attr_ids[j],label)){
//                  outf << " " << label;
//                  j++;
//                  cnt++;
//              }
//              outf << endl;
//              err = ElleWriteUnodeAttributeData(outf,&attr_ids[i],cnt);
//              i=j-1; // for loop will incr i
//              break;
//            case  E_XX:
//            case  E_YY:
//            case  E_XY:
//            case  E_YX:
//            case  E_ZZ:
//            case  INCR_S:
//            case  BULK_S:
//            case  F_INCR_S:
//            case  F_BULK_S:
//              file_id=U_STRAIN;
//              if (!id_match(FileKeys,file_id,label)) err = KEY_ERR;
//              if (!err) outf << label;
//              cnt=0;
//              j=i;
//              while(j<maxa && cnt<NUM_FLYNN_STRAIN_VALS &&
//                    id_match(FlynnStrainKeys,attr_ids[j],label)){
//                  outf << " " << label;
//                  j++;
//                  cnt++;
//              }
//              outf << endl;
//              err = ElleWriteUnodeAttributeData(outf,&attr_ids[i],cnt);
//              i=j-1; // for loop will incr i
//              break;
//            case EULER_3:
//            case E3_ALPHA:
//              file_id=U_EULER_3;
//              if (!id_match(FileKeys,file_id,label)) err = KEY_ERR;
//              if (!err) outf << label;
//              cnt=3;
//              outf << endl;
//              err = ElleWriteUnodeAttributeData(outf,&attr_ids[i],cnt);
//              i += cnt-1; // for loop will incr i
//              break;
//            case CAXIS:
//            case CAXIS_X:
//              file_id=U_CAXIS;
//              if (!id_match(FileKeys,file_id,label)) err = KEY_ERR;
//              if (!err) outf << label;
//              cnt=1;
//              outf << endl;
//              err = ElleWriteUnodeAttributeData(outf,&attr_ids[i],cnt);
//              i += 2; // allow for 3 attrib in dflt list,for loop will incr i
//              break;
//            case DISLOCDEN:
//              file_id=U_DISLOCDEN;
//              // FALL THROUGH
//            default:
//              if (!id_match(FileKeys,attr_ids[i],label)) {
//                  cerr << "Ignoring unknown unode attribute " <<
//                         attr_ids[i] << endl;
//              }
//              else {
//                  outf << label << endl;
//                  err=ElleWriteUnodeAttributeData(outf,&attr_ids[i],1);
//              }
//              break;
//            }

        if (!outf)  err=WRITE_ERR;
        if (!err) {
                outf.precision(oldp);
                outf.setf(ios::fixed,ios::floatfield);
        }
        delete [] attr_ids;
    }
    return(err);
}

bool ElleGetSegmentPosition(int id, Coords *segment_pos)
{
    Segment s;
    Coords start_xy, end_xy;

    if (Segments.size()>0)
    {
        s = Segments[id];

        end_xy = s.GetEndNodePos();
        start_xy = s.GetStartNodePos();

        segment_pos->x = (start_xy.x + end_xy.y) / 2.0;
        segment_pos->y = (start_xy.y + end_xy.y) / 2.0;

        return true;
    }
    else
        return false;
}

int ElleGetSegmentAttribute( int id, double *val, const int attr_id)
{
    int err=0;
    Segment s;

    if (Segments.size()>0)
    {
        s = Segments[id];
        if (s.hasAttribute(attr_id))
            err=s.getAttribute(attr_id,val);
        else if (err=DfltSegmentAttrib.getAttribute(attr_id,val)) 
            err = ATTRIBID_ERR;
    }
    return( err );
}

void SegmentAttributeList(int **attr_ids, int *maxa)
{
    *attr_ids = 0; *maxa = 0;
    if (Segments.size()>0) {
        *maxa = DfltSegmentAttrib.numAttributes();
        if (*maxa>0)  {
            if ((*attr_ids = (int *)malloc(*maxa * sizeof(int)))==0)
                OnError("SegmentAttributeList",MALLOC_ERR);
            DfltSegmentAttrib.getList(*attr_ids,*maxa);
        }
    }
}
