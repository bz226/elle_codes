 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_pointdata_h
#define _E_pointdata_h

#include <iostream>

// point class
class PointData {
    double x,y;
    int id,bnd;
public:
    PointData(): x(0.0),y(0.0),id(0),bnd(0) { }
    PointData(double xval,double yval,int idval,int bndval):
        x(xval),y(yval),id(idval),bnd(bndval) { }
    void setvalues(double xval,double yval,int idval,int bndval)
        { x = xval; y = yval; id = idval, bnd = bndval; }
    int idvalue() { return id; }
    int boundary() { return bnd; }
    double xvalue() { return x; }
    double yvalue() { return y; }
    unsigned char gt(PointData &t);
    bool id_match(const int idval) { return(idval==id); }
    friend std::ostream & operator<< (std::ostream &, const PointData &);
};

#endif
