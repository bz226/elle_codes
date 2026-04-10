#ifndef _E_segmentsP_h
#define _E_segmentsP_h

#include <vector>
#include <iostream>

#include "unodesP.h"
#include "attrib.h"

using namespace std;

class Segment
{
	private:
		
                int _id;
		double length;	// update each timestep
		Unode *StartNode;
		Unode *EndNode;

        AttributeArray _attributes;

	public:
		
        Segment():_id(0),_attributes(0),StartNode(NULL), EndNode(NULL),length(0.0) { }
        Segment(int id, int StartNode, int EndNode);
        Segment(int id, Unode* Start, Unode* End):_id(id),_attributes(0),StartNode(Start), EndNode(End),length(0.0) { }
        Segment(int id, Unode* Start, Unode* End, double length):_id(id),_attributes(0),StartNode(Start), EndNode(End),length(length) { }

        ~Segment();
		
		Unode *GetStartNode();
		void SetStartNode(Unode* node);
        void SetStartNode(int idval);
		
		Unode *GetEndNode();
		void SetEndNode(Unode* node);
        void SetEndNode(int idval);
		
        Coords GetEndNodePos();
        Coords GetStartNodePos();
		
		double GetLength();
		void SetLength(double length);
        void SetLength();

        int setAttribute(const int id, int val) {
            _attributes.setAttribute(id,val); return(0); }
        int setAttribute(const int id, double val) {
            _attributes.setAttribute(id,val); return(0); }
        int setAttribute(const int id, Coords_3D *val) {
            _attributes.setAttribute(id,val); return(0); }

        int id() {
            return _id; }

        bool hasAttribute(const int id) {
            return(_attributes.hasAttribute(id)); }

        int getAttribute(const int id, int *val) {
            return(_attributes.getAttribute(id,val)); }

        int getAttribute(const int id, double *val) {
            return(_attributes.getAttribute(id,val)); }


        friend std::ostream & operator<< (std::ostream &os, /*const*/ Segment &t)
        {
            os << t.id() << " ";
            os << t.StartNode->id() << " ";
            os << t.EndNode->id() << " ";

            return os;
        }
		
};

#endif
