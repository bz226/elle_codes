 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <iostream>
#include <vector>
#include <algorithm>
#include "attribarray.h"
#include "attrib.h"
#include "file.h"
#include "errnum.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

using std::vector;
using std::ostream;
using std::cout;
using std::cerr;
using std::endl;

const int GRN_FILL_COL = 20;

void AttributeArray::initAttribute( const int idval )
{
    Attribute tmp;
    int index=0, i;
    bool spare;

    while (index<(int)aa.size() && aa[index].getType()!=NO_NB) index++;
    switch(idval) {
    case COLOUR: tmp.setType(idval);
                 tmp.setValue(GRN_FILL_COL);
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 break;
    case EXPAND: tmp.setType(idval);
                 tmp.setValue(0);
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 break;
    case SPLIT  : tmp.setType(idval);
                 tmp.setValue(0);
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 break;
    case GRAIN  : tmp.setType(idval);
                 tmp.setValue(NO_NB);
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 break;
    case MINERAL: tmp.setType(idval);
                 tmp.setValue(QUARTZ);
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 break;
    case U_S_EXPONENT: 
    case S_EXPONENT: tmp.setType(idval);
    case U_VISCOSITY: 
    case VISCOSITY: tmp.setType(idval);
                 tmp.setValue(1);
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 break;
    case CAXIS:  tmp.setType(idval);
                 tmp.setValue(1); // initialise to cartesian for angle 0
                 if (index<(int)aa.size()) {
                     for (spare=1,i=index;i<index+3 && i<(int)aa.size();i++)
                         if (aa[i].getType()!=NO_NB) spare=0;
                     if (!spare) index=i;
                 }       
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 index++;
                 tmp.setType(CAXIS_Y);
                 tmp.setValue(0);
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 index++;
                 tmp.setType(CAXIS_Z);
                 tmp.setValue(0);
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 break;
    case EULER_3:tmp.setType(E3_ALPHA);
                 tmp.setValue(0);
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 while (index<(int)aa.size() && aa[index].getType()!=NO_NB)
                     index++;
                 tmp.setType(E3_BETA);
                 tmp.setValue(0);
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 while (index<(int)aa.size() && aa[index].getType()!=NO_NB)
                     index++;
                 tmp.setType(E3_GAMMA);
                 tmp.setValue(0);
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 break;
    case U_FINITE_STRAIN : for (i=0;i<NUM_FINITE_STRAIN_VALS;i++,index++) {
                    tmp.setType(FiniteStrainKeys[i].id);
                    tmp.setValue(0);
                    if (index<(int)aa.size()) aa[index]=tmp;
                    else aa.push_back(tmp);
                    }
                 break;
    case U_STRAIN : for (i=0;i<NUM_FLYNN_STRAIN_VALS;i++,index++) {
                    tmp.setType(FlynnStrainKeys[i].id);
                    tmp.setValue(0);
                    if (index<(int)aa.size()) aa[index]=tmp;
                    else aa.push_back(tmp);
                    }
                 break;
    case CONC_A: 
    case F_INCR_S: 
    case F_BULK_S: 
    case E_XX  : 
    case E_XY  :
    case E_YX  :
    case E_YY  :
    case E_ZZ  :
    case INCR_S: 
    case BULK_S: 
    case TAU_XX: 
    case TAU_YY: 
    case TAU_ZZ: 
    case TAU_XY: 
    case TAU_1 : 
    case PRESSURE: 
    case ATTRIB_A: 
    case ATTRIB_B: 
    case ATTRIB_C: 
    case ATTRIB_D: 
    case ATTRIB_E: 
    case ATTRIB_F: 
    case U_ATTRIB_A: 
    case U_ATTRIB_B: 
    case U_ATTRIB_C: 
    case U_ATTRIB_D: 
    case U_ATTRIB_E: 
    case U_ATTRIB_F: 
    case U_FRACTURES: 
    case U_PHASE: 
    case U_TEMPERATURE: 
    case U_DIF_STRESS: 
    case U_MEAN_STRESS: 
    case U_DENSITY: 
    case U_YOUNGSMODULUS: 
    case CYCLE:
    case AGE:
    case U_DISLOCDEN: 
    case DISLOCDEN:
    case U_ENERGY: 
    case ENERGY:
    case F_ATTRIB_A:
    case F_ATTRIB_B:
    case F_ATTRIB_C:
    case F_ATTRIB_D:
    case F_ATTRIB_E:
    case F_ATTRIB_F:
    case F_ATTRIB_I:
    case F_ATTRIB_J:
    case F_ATTRIB_K:
    case E3_ALPHA: 
    case E3_BETA: 
    case E3_GAMMA: 
    case CAXIS_Y: 
    case CAXIS_Z: 
    case START_S_X:
    case START_S_Y:
    case PREV_S_X:
    case PREV_S_Y:
    case CURR_S_X:
    case CURR_S_Y:
    case S_FLUID_FLUX:
    case S_PERMEABILITY:
                 tmp.setType(idval);
                 tmp.setValue(0);
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 break;
    case CAXIS_X: 
                 tmp.setType(idval);
                 tmp.setValue(1);//cartesian for angle 0
                 if (index<(int)aa.size()) aa[index]=tmp;
                 else aa.push_back(tmp);
                 break;
    default:     cerr << "Unknown attribute" << endl;
                 break;
    }
}

void AttributeArray::removeAttribute( const int idval )
{
    int i, idtmp;

    switch (idval) {
    case U_STRAIN : for (i=0;i<NUM_FLYNN_STRAIN_VALS;i++) {
                      idtmp = FlynnStrainKeys[i].id;
                      it_aa = aa.begin();
                      while (it_aa!=aa.end() && it_aa->getType()!=idtmp)
                         it_aa++;
                      if (it_aa!=aa.end()) it_aa->remove();
                    }
                 break;
    case U_FINITE_STRAIN : for (i=0;i<NUM_FINITE_STRAIN_VALS;i++) {
                      idtmp = FiniteStrainKeys[i].id;
                      it_aa = aa.begin();
                      while (it_aa!=aa.end() && it_aa->getType()!=idtmp)
                         it_aa++;
                      if (it_aa!=aa.end()) it_aa->remove();
                    }
                 break;
    case EULER_3:   while (it_aa!=aa.end() && it_aa->getType()!=E3_ALPHA)
                             it_aa++;
                    if (it_aa!=aa.end()) it_aa->remove();
                    while (it_aa!=aa.end() && it_aa->getType()!=E3_BETA)
                             it_aa++;
                    if (it_aa!=aa.end()) it_aa->remove();
                    while (it_aa!=aa.end() && it_aa->getType()!=E3_GAMMA)
                             it_aa++;
                    if (it_aa!=aa.end()) it_aa->remove();
                 break;
    default:     it_aa = aa.begin();
                 while (it_aa!=aa.end() && it_aa->getType()!=idval) it_aa++;
                 if (it_aa!=aa.end()) {
                     aa.erase(it_aa);
                     if (idval==CAXIS) {
                         while (it_aa!=aa.end() && it_aa->getType()!=CAXIS_Y)
                             it_aa++;
                         if (it_aa!=aa.end()) it_aa->remove();
                         while (it_aa!=aa.end() && it_aa->getType()!=CAXIS_Z)
                             it_aa++;
                         if (it_aa!=aa.end()) it_aa->remove();
                     }
                 }
                 break;
    }
}

int AttributeArray::numActive()
{
    int cnt=0;
    it_aa = aa.begin();
    while (it_aa!=aa.end()) {
        if ((*it_aa).getType()!=NO_NB) cnt++;
        it_aa++;
    }
    return(cnt);
}

void AttributeArray::setAttribute( const int idval,
                                    const int avalue)
{
    it_aa = aa.begin();
    while (it_aa!=aa.end() && (*it_aa).getType()!=idval) it_aa++;
    // reset existing attribute
    if (it_aa!=aa.end()) (*it_aa).setValue((double)avalue);
    else {
        it_aa = aa.begin();
        while (it_aa!=aa.end() && (*it_aa).getType()!=NO_NB) it_aa++;
    // set vacant attribute
        if (it_aa!=aa.end()) {
            (*it_aa).setType(idval);
            (*it_aa).setValue((double)avalue);
        }
        else {
    // add attribute
            Attribute tmp(idval,(double)avalue);
            aa.push_back(tmp);
        }
    }
}

int AttributeArray::getAttribute( const int idval,
                                    int *valptr)
{
    int err = 0;
    it_aa = aa.begin();
    while (it_aa!=aa.end() && (*it_aa).getType()!=idval) it_aa++;
    if (it_aa!=aa.end()) *valptr = (int)((*it_aa).getValue());
    else err = ATTRIBID_ERR;
    return(err);
}

void AttributeArray::setAttribute( const int idval,
                                    const double avalue)
{
    it_aa = aa.begin();
    while (it_aa!=aa.end() && (*it_aa).getType()!=idval) it_aa++;
    // reset existing attribute
    if (it_aa!=aa.end()) (*it_aa).setValue(avalue);
    else {
        it_aa = aa.begin();
        while (it_aa!=aa.end() && (*it_aa).getType()!=NO_NB) it_aa++;
    // set vacant attribute
        if (it_aa!=aa.end()) {
            (*it_aa).setType(idval);
            (*it_aa).setValue(avalue);
        }
        else {
    // add attribute
            Attribute tmp(idval,avalue);
            aa.push_back(tmp);
        }
    }
}

int AttributeArray::getAttribute( const int idval,
                                    double *valptr)
{
    int err = 0;
    it_aa = aa.begin();
    while (it_aa!=aa.end() && (*it_aa).getType()!=idval) it_aa++;
    if (it_aa!=aa.end()) *valptr = (*it_aa).getValue();
    else err = ATTRIBID_ERR;
    return(err);
}

void AttributeArray::setAttribute( const int idval,
                                    const Coords_3D *avalue)
{
    it_aa = aa.begin();
    while (it_aa!=aa.end() && (*it_aa).getType()!=idval) it_aa++;
    if (it_aa!=aa.end()) {
        (*it_aa).setValue(avalue->x); it_aa++;
        (*it_aa).setValue(avalue->y); it_aa++;
        (*it_aa).setValue(avalue->z);
    }
    else {
        Attribute tmp(idval,avalue->x);
        aa.push_back(tmp);
        tmp.setType(CAXIS_Y);
        tmp.setValue(avalue->y);
        aa.push_back(tmp);
        tmp.setType(CAXIS_Z);
        tmp.setValue(avalue->z);
        aa.push_back(tmp);
    }
}

int AttributeArray::getAttribute( const int idval,
                                    Coords_3D *valptr)
{
    int err=0;

    it_aa = aa.begin();
    while (it_aa!=aa.end() && (*it_aa).getType()!=idval) it_aa++;
    if (it_aa!=aa.end()) {
        valptr->x = (*it_aa).getValue(); it_aa++;
        if ((*it_aa).getType()==CAXIS_Y) {
            valptr->y = (*it_aa).getValue(); it_aa++;
        }
        else err=ATTRIBID_ERR;
        if (!err && ((*it_aa).getType()==CAXIS_Z)) {
            valptr->z = (*it_aa).getValue(); it_aa++;
        }
        else err=ATTRIBID_ERR;
    }
    else err=ATTRIBID_ERR;
    return(err);
}

bool AttributeArray::hasAttribute( const int idval )
{ 
    int tmpid;

    it_aa = aa.begin();
    while (it_aa!=aa.end() && (*it_aa).getType()!=idval) it_aa++;
    // this check only necessary til code fixed to store attribid
    // EULER_3 at start of euler group cf CAXIS
    if (it_aa==aa.end() && idval==EULER_3) {
        tmpid = E3_ALPHA;
        it_aa = aa.begin();
        while (it_aa!=aa.end() && it_aa->getType()!=tmpid) it_aa++;
    }
    else if (it_aa==aa.end() && idval==U_FINITE_STRAIN) {
        tmpid = START_S_X;
        it_aa = aa.begin();
        while (it_aa!=aa.end() && it_aa->getType()!=tmpid) it_aa++;
    }
    else if (it_aa==aa.end() && idval==U_STRAIN) {
        tmpid = E_XX;
        it_aa = aa.begin();
        while (it_aa!=aa.end() && it_aa->getType()!=tmpid) it_aa++;
    }
    return(it_aa!=aa.end());
}

void AttributeArray::getIntList(vector<int> &attr)
{
    it_aa = aa.begin();
    for (it_aa = aa.begin(); it_aa!=aa.end(); it_aa++) 
        if ((*it_aa).isIntAttribute()) 
            attr.push_back((*it_aa).getType());
}

vector<double> * AttributeArray::getAttributeValues()
{
	vector<double> *vals = new vector<double>;
    it_aa = aa.begin();
    for (it_aa = aa.begin(); it_aa!=aa.end(); it_aa++)
        vals->push_back((*it_aa).getValue());
    return(vals);
}

void AttributeArray::getList(int *const list,const int max)
{
    int *ptr;

    int cnt=0;
    ptr = list;
    it_aa = aa.begin();
    while (it_aa!=aa.end() && cnt < max) {
        *ptr = (*it_aa).getType();
        it_aa++; ptr++;
        cnt++;
    }
}

//Write object to an ostream reference
ostream & operator<< (ostream &os, /*const*/ AttributeArray &t)
{
    /*copy(t.aa.begin(),t.aa.end(),ostream_iterator<Attribute>(os));*/
	vector<Attribute> :: iterator it;
    for(it=t.aa.begin(); it!=t.aa.end(); it++)
    os << *(it);
    return os;
}
