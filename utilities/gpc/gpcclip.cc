#include <stdio.h>
#include <vector>
#include "attrib.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "gpc.h"
#ifdef __cplusplus
}
#endif
#include "gpcclip.h"
#include "errnum.h"
#include "error.h"

#define MALLOC(p, b, t, s) {p= (t)malloc(b); if (!(p)) { \
                        fprintf(stderr, "gpc malloc failure: %s\n", s); \
                        exit(0);}}
int gpcclip(std::vector<Coords> &pv2, std::vector<Coords> &pv1,
				std::vector<std::vector<Coords> > &pclip , gpc_op op)
{
    int cnt, i, err=0;
    /*gpc_op op=GPC_INT;*/
    Coords pos;
    gpc_polygon poly1, poly2, res;

    poly1.hole = 0;
    poly1.num_contours = 1;
    MALLOC(poly1.contour,poly1.num_contours
         * sizeof(gpc_vertex_list), gpc_vertex_list *, "contour creation");
    poly1.contour[0].num_vertices = pv1.size();
    MALLOC(poly1.contour[0].vertex, poly1.contour[0].num_vertices
           * sizeof(gpc_vertex), gpc_vertex *, "vertex creation");
    for (i=0; i<pv1.size(); i++) {
        pos = pv1[i];
        poly1.contour[0].vertex[i].x = pos.x;
        poly1.contour[0].vertex[i].y = pos.y;
    }
    /*
     * create poly for cell
     */
    poly2.hole = 0;
    poly2.num_contours = 1;
    MALLOC(poly2.contour,poly2.num_contours
         * sizeof(gpc_vertex_list), gpc_vertex_list *, "contour creation");
    poly2.contour[0].num_vertices = pv2.size();
    MALLOC(poly2.contour[0].vertex, poly2.contour[0].num_vertices
           * sizeof(gpc_vertex), gpc_vertex *, "vertex creation");
    for (i=0; i<pv2.size(); i++) {
        pos = pv2[i];
        poly2.contour[0].vertex[i].x = pos.x;
        poly2.contour[0].vertex[i].y = pos.y;
    }
    gpc_polygon_clip(op,&poly1, &poly2, &res);

    /*
     * copy result polygon
     */
    for (cnt=0;cnt<res.num_contours;cnt++) {
		std::vector<Coords> pv;
        for (i= 0; i < res.contour[cnt].num_vertices; i++) {
            pos.x = res.contour[cnt].vertex[i].x;
            pos.y = res.contour[cnt].vertex[i].y;
            pv.push_back(pos);
        }
        pclip.push_back(pv);
    }
    gpc_free_polygon(&poly1);
    gpc_free_polygon(&poly2);
    gpc_free_polygon(&res);
    return(err);
}

