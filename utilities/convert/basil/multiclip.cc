#include <stdio.h>
#include <vector>
#include "attrib.h"
#include "polyarea.h"
#include "multiclip.h"
#include "errnum.h"
#include "error.h"

int multiclip(std::vector<Coords> &pv2, std::vector<Coords> &pv1,
				std::vector<std::vector<Coords> > &pclip )
{
    int cnt, i, err=0;
    int op=PBO_ISECT;
    Coords pos;
    POLYAREA *PA1=0, *PA2=0, *PAres=0, *p=0;
    PLINE *poly1=0, *poly2=0, *cntr;
    Vector v;

    v[2] = 0.0;
    for (i=0; i<pv1.size(); i++) {
        pos = pv1[i];
        v[0] = pos.x;
        v[1] = pos.y;
        if (poly1==0) {
            if ((poly1 = poly_NewContour(v)) == 0)
                OnError("",MALLOC_ERR);
        }
        else
            poly_InclVertex(poly1->head.prev, poly_CreateNode(v));
    }
    poly_PreContour(poly1, TRUE);
    if ( (poly1->Flags & PLF_ORIENT)
        != (unsigned int)(0 ? PLF_INV : PLF_DIR) )
        poly_InvContour(poly1);
    if (p == NULL)
        p = poly_Create();
    if (p == NULL)
        OnError("",MALLOC_ERR);
    poly_InclContour(p, poly1);
    if (!poly_Valid(p))
        OnError("polygon not valid",0);
    PA1 = 0;
    poly_M_Incl(&PA1, p);
    /*
     * create poly for cell
     */
    for (i=0; i<pv2.size(); i++) {
        pos = pv2[i];
        v[0] = pos.x;
        v[1] = pos.y;
        if (poly2==0) {
            if ((poly2 = poly_NewContour(v)) == 0)
                OnError("",MALLOC_ERR);
        }
        else
            poly_InclVertex(poly2->head.prev, poly_CreateNode(v));
    }
    poly_PreContour(poly2, TRUE);
    if ( (poly2->Flags & PLF_ORIENT)
        != (unsigned int)(0 ? PLF_INV : PLF_DIR) )
        poly_InvContour(poly2);
    p =  NULL;
    if (p == NULL)
        p = poly_Create();
    if (p == NULL)
        OnError("",MALLOC_ERR);
    poly_InclContour(p, poly2);
    if (!poly_Valid(p))
        OnError("polygon not valid",0);
    PA2 = 0;
    poly_M_Incl(&PA2, p);
    /*
     * create the new polygons
     */
    if ( !(poly_Valid(PA1) && poly_Valid(PA2)) )
    {
        poly_Free(&PA1);
        poly_Free(&PA2);
        OnError("One of polygons is not valid",0);
        return -1;
    }
    if (poly_Boolean(PA1, PA2, &PAres, op) != err_ok)
    {
        poly_Free(&PA1);
        poly_Free(&PA2);
        poly_Free(&PAres);
        OnError("Error while clipping",0);
        return -1;
    }
    if (PAres != NULL) {
		std::vector<Coords> pv;
        i=0;  POLYAREA *curpa = PAres;
        do { i++; } while ((curpa = curpa->f) != PAres);
        do {
            for (cntr = curpa->contours;cntr != 0;cntr = cntr->next) {
                VNODE *cur = &cntr->head;
                do {
                    pos.x = cur->point[0];
                    pos.y = cur->point[1];
                    pv.push_back(pos);
                } while (( cur = cur->next ) != &cntr->head);
            }
            pclip.push_back(pv);
        } while ((curpa = curpa->f) != PAres);
    }
    /*else*/
        /*err = EMPTYPOLY_ERR;*/

    poly_Free(&PA1);
    poly_Free(&PA2);
    poly_Free(&PAres);
    return(err);
}

