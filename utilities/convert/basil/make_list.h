#include <iostream>

using std::cout;
using std::ostream;

// node class
class TNodeData {
    float x,y;
    int id,bnd;
public:
    TNodeData(): x(0.0),y(0.0),id(0),bnd(0) { }
    TNodeData(float xval,float yval,int idval,int bndval):
        x(xval),y(yval),id(idval),bnd(bndval) { }
    int idvalue() { return id; }
    int boundary() { return bnd; }
    float xvalue() { return x; }
    float yvalue() { return y; }
    unsigned char gt(TNodeData &t);
    friend ostream & operator<< (ostream &, TNodeData &);
};

//Write object to an ostream reference
ostream & operator<< (ostream &os, TNodeData &t)
{
    cout << t.x << ", "<< t.y << ", " << t.id <<", "<< t.bnd << '\n';
    return os;
}

unsigned char TNodeData::gt(TNodeData &t)
{
    unsigned char gt=0;

    if (bnd > t.boundary()) return(1);
    if (bnd == t.boundary())
        switch (bnd) {
        case 1: if (x > t.x) gt = 1;
                break;
        case 2: if (y > t.y) gt = 1;
                break;
        case 3: if (x < t.x) gt = 1;
                break;
        case 4: if (y < t.y) gt = 1;
                break;
        }
    return(gt);
}
