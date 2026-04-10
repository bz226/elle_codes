 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_attribute_h
#define _E_attribute_h
#include <iostream>
#include <vector>
#include "attrib.h"

// storage class for attribute values

class Attribute {
    int id;
    double value;
public:
    Attribute(): id(NO_NB),value(0.0) { }
    Attribute(int idval): id(idval),value(0) { }
    Attribute(int idval, double avalue): id(idval),value(avalue) { }
    void setType(int idval) { id = idval; }
    void setValue(double avalue) { value = avalue; }
    int getType() { return(id); }
    double getValue() { return(value); }
    void remove() { id = NO_NB; value = 0.0; }
    const Attribute & operator=(const Attribute &t) { id=t.id; value=t.value;
                                                      return t; }
    bool isIntAttribute();
    bool isRealAttribute();
    bool isFlynnAttribute();
    friend std::ostream & operator<< (std::ostream &, const Attribute &);
};

#endif
