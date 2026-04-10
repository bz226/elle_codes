 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_attribarray_h
#define _E_attribarray_h

#include <iostream>
#include <vector>
#include "attrib.h"
#include "attribute.h"

const int INIT_ATTRIB_NUM = 2;

class AttributeArray {
	std::vector<Attribute> aa;
	std::vector<Attribute> :: iterator it_aa;

public:
    AttributeArray(): aa(INIT_ATTRIB_NUM) { }
    AttributeArray(int size): aa(size) { }
    int numAttributes() { return( aa.size() ); }
    int numActive();
    bool hasAttribute( const int idval );
    void initAttribute(const int idval );
    void removeAttribute( const int idval );
    void setAttribute( const int idval,const int avalue);
    void setAttribute( const int idval,const double avalue);
    void setAttribute( const int idval,const Coords_3D *avalue);
    int getAttribute( const int idval,int *valptr);
    int getAttribute( const int idval,double *valptr);
    int getAttribute( const int idval,Coords_3D *valptr);
    void getList(int *const list,const int max);
    void getIntList(std::vector<int> &attr);
	std::vector<double> *getAttributeValues();
    void clean() { aa.erase(aa.begin(),aa.end()); }
    const AttributeArray & operator=(const AttributeArray &t) {
                          aa = t.aa;
                          it_aa = t.it_aa;
                          return t; }
    friend std::ostream & operator<< (std::ostream &, /*const*/ AttributeArray
&);
};

#endif
